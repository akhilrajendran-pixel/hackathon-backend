"""
LLM agent layer: prompt construction, Bedrock Converse API call, citation parsing, intent detection.
Uses Amazon Bedrock (Claude Sonnet) for LLM inference.
"""
import re
import json
import asyncio
import logging
from typing import Dict, List, Tuple, Optional

import boto3
from botocore.exceptions import ClientError

import config
import retriever
import session_manager
import guardrails

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a Sales Co-Pilot for our company. Your purpose is to help sales executives, account managers, and pre-sales consultants find relevant information from our internal knowledge base of proposals, case studies, and whitepapers.

STRICT RULES:
1. Answer ONLY using the provided document chunks below. Do NOT use any external knowledge.
2. Every factual claim MUST cite its source in this exact format: [Source: filename, Page X]
3. If the provided chunks don't contain relevant information for the query, say: "I don't have relevant information in our knowledge base for this query." Do NOT fabricate or guess.
4. NEVER fabricate client names, project names, metrics, dollar amounts, percentages, or outcomes not present in the chunks.
5. Use business language, not technical jargon. Be concise and actionable.
6. Structure answers with a brief summary first, then supporting details with citations.
7. When multiple documents are relevant, synthesize insights across them and cite each source.
8. For follow-up questions, use the conversation history to understand context (pronouns like "that proposal", "the same client", refinements like "narrow to 2023").

RESPONSE FORMAT:
- First line must be: [INTENT: category]
  Where category is one of: retrieve_similar_work, summarize_experience, compare_offerings, extract_metrics, general_query
- Then a 1-2 sentence summary
- Follow with bullet points or structured details
- Each factual point must have a [Source: filename, Page X] citation
- End with any caveats if information is limited"""


_bedrock_client = None


def _get_bedrock_client():
    """Return a cached boto3 bedrock-runtime client."""
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=config.AWS_REGION,
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
        )
    return _bedrock_client


def _format_chunks_for_prompt(chunks: List[Dict]) -> str:
    """Format retrieved chunks as context for the LLM."""
    if not chunks:
        return "No relevant document chunks were found for this query."

    lines = ["=== RETRIEVED DOCUMENT CHUNKS ===\n"]
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"--- Chunk {i} ---")
        lines.append(f"Source: {chunk['filename']}, Page {chunk['page']}")
        lines.append(f"Type: {chunk['doc_type']}, Year: {chunk.get('year', 'N/A')}")
        lines.append(f"Relevance Score: {chunk['relevance_score']}")
        lines.append(f"Text:\n{chunk['chunk_text']}")
        lines.append("")
    return "\n".join(lines)


def _build_bedrock_messages(
    query: str,
    chunks: List[Dict],
    conversation_history: List[Dict],
) -> Tuple[List[Dict], List[Dict]]:
    """Build the message list and system prompt for the Bedrock Converse API.

    Returns (system, messages) where system is the Converse system param
    and messages is a list of {role, content: [{text}]} dicts.
    """
    system = [{"text": SYSTEM_PROMPT}]
    messages = []

    # Add conversation history (prior turns)
    for turn in conversation_history:
        messages.append({
            "role": turn["role"],
            "content": [{"text": turn["content"]}],
        })

    # Build current user message with context
    context = _format_chunks_for_prompt(chunks)
    user_message = f"{context}\n\n=== USER QUERY ===\n{query}"
    messages.append({"role": "user", "content": [{"text": user_message}]})

    return system, messages


async def _call_bedrock(system: List[Dict], messages: List[Dict]) -> str:
    """Call Bedrock Converse API (sync boto3 wrapped in run_in_executor)."""
    client = _get_bedrock_client()

    def _invoke():
        response = client.converse(
            modelId=config.BEDROCK_MODEL_ID,
            system=system,
            messages=messages,
            inferenceConfig={
                "temperature": 0.3,
                "maxTokens": 1024,
            },
        )
        return response["output"]["message"]["content"][0]["text"]

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _invoke)


def _parse_intent(response_text: str) -> Tuple[str, str]:
    """
    Extract intent classification from response.
    Returns (intent, cleaned_response).
    """
    intent = "general_query"
    cleaned = response_text

    match = re.search(r'\[INTENT:\s*(\w+)\]', response_text)
    if match:
        intent = match.group(1)
        cleaned = response_text[match.end():].strip()

    return intent, cleaned


def _parse_citations(response_text: str, retrieved_chunks: List[Dict]) -> List[Dict]:
    """
    Extract [Source: filename, Page X] citations from response
    and match them back to retrieved chunk metadata.
    """
    citation_pattern = r'\[Source:\s*([^,\]]+?)(?:,\s*Page\s*(\d+))?\]'
    found_citations = re.findall(citation_pattern, response_text)

    seen = set()
    citations = []

    for cited_filename, cited_page in found_citations:
        cited_filename = cited_filename.strip()
        key = f"{cited_filename}::{cited_page}"
        if key in seen:
            continue
        seen.add(key)

        best_match = None
        for chunk in retrieved_chunks:
            if cited_filename in chunk["filename"] or chunk["filename"] in cited_filename:
                if cited_page and str(chunk["page"]) == cited_page:
                    best_match = chunk
                    break
                elif not best_match:
                    best_match = chunk

        if best_match:
            citations.append({
                "document": best_match["filename"],
                "doc_type": best_match["doc_type"],
                "year": best_match.get("year"),
                "page": best_match["page"],
                "chunk_text": best_match["chunk_text"][:300] + "..." if len(best_match["chunk_text"]) > 300 else best_match["chunk_text"],
                "relevance_score": best_match["relevance_score"],
                "drive_link": best_match.get("drive_link"),
            })
        else:
            citations.append({
                "document": cited_filename,
                "doc_type": "unknown",
                "year": None,
                "page": int(cited_page) if cited_page else None,
                "chunk_text": None,
                "relevance_score": None,
                "drive_link": None,
            })

    return citations


async def process_query(
    session_id: str,
    query: str,
) -> Dict:
    """
    Full agent pipeline:
    1. Check input guardrails
    2. Retrieve relevant chunks
    3. Compute confidence
    4. Build prompt with history
    5. Call Bedrock
    6. Parse intent + citations
    7. Check output guardrails
    8. Store turn in session
    9. Return structured response
    """
    # 1. Input guardrails
    guardrail_result = guardrails.check_input(query)
    if guardrail_result:
        return {
            "session_id": session_id,
            "answer": None,
            "citations": [],
            "confidence": None,
            "confidence_score": None,
            "intent": None,
            "guardrail_triggered": guardrail_result,
        }

    # 2. Retrieve
    chunks = retriever.retrieve(query)

    # 3. Confidence
    confidence_level, confidence_score = retriever.compute_confidence(chunks)

    # Check for no-answer scenario
    if not chunks or confidence_score < config.NO_ANSWER_THRESHOLD:
        no_answer_msg = (
            "I don't have relevant information in our knowledge base for this query. "
            "The available proposals, case studies, and whitepapers don't appear to "
            "cover this topic. Please try a different question or check with the source team."
        )
        session_manager.add_turn(session_id, "user", query)
        session_manager.add_turn(session_id, "assistant", no_answer_msg, [])

        return {
            "session_id": session_id,
            "answer": no_answer_msg,
            "citations": [],
            "confidence": "low",
            "confidence_score": confidence_score,
            "intent": "general_query",
            "guardrail_triggered": None,
        }

    # 4. Build messages with conversation history
    history = session_manager.get_history_for_llm(session_id)
    system, messages = _build_bedrock_messages(query, chunks, history)

    # 5. Call Bedrock
    try:
        raw_answer = await _call_bedrock(system, messages)
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        logger.error("Bedrock API error (%s): %s", error_code, e)
        return {
            "session_id": session_id,
            "answer": f"LLM service error: {error_code}. Check AWS credentials and Bedrock model access.",
            "citations": [],
            "confidence": None,
            "confidence_score": None,
            "intent": None,
            "guardrail_triggered": {"type": "system_error", "message": f"Bedrock {error_code}: {e}"},
        }
    except Exception as e:
        import traceback
        logger.error("Bedrock API call failed: %s\n%s", e, traceback.format_exc())
        return {
            "session_id": session_id,
            "answer": f"An error occurred while processing your query: {type(e).__name__}: {e}",
            "citations": [],
            "confidence": None,
            "confidence_score": None,
            "intent": None,
            "guardrail_triggered": {"type": "system_error", "message": str(e)},
        }

    # 6. Parse intent and citations
    intent, cleaned_answer = _parse_intent(raw_answer)
    citations = _parse_citations(raw_answer, chunks)

    # If no citations were parsed but we have chunks, add them as implicit citations
    if not citations and chunks:
        for chunk in chunks[:3]:
            citations.append({
                "document": chunk["filename"],
                "doc_type": chunk["doc_type"],
                "year": chunk.get("year"),
                "page": chunk["page"],
                "chunk_text": chunk["chunk_text"][:300] + "..." if len(chunk["chunk_text"]) > 300 else chunk["chunk_text"],
                "relevance_score": chunk["relevance_score"],
                "drive_link": chunk.get("drive_link"),
            })

    # 7. Output guardrails
    retrieved_filenames = [c["filename"] for c in chunks]
    output_guard = guardrails.check_output(cleaned_answer, retrieved_filenames)

    # Add low-confidence disclaimer
    if confidence_level == "low":
        cleaned_answer = guardrails.add_low_confidence_disclaimer(cleaned_answer)

    # 8. Store turn in session
    session_manager.add_turn(session_id, "user", query)
    session_manager.add_turn(session_id, "assistant", cleaned_answer, citations)

    # 9. Return response
    return {
        "session_id": session_id,
        "answer": cleaned_answer,
        "citations": citations,
        "confidence": confidence_level,
        "confidence_score": confidence_score,
        "intent": intent,
        "guardrail_triggered": output_guard,
    }

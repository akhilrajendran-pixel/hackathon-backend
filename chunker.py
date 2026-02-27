"""
Text chunking with metadata extraction.

Chunk size: 500-700 tokens (~600 target) with 100-token overlap.
Uses sentence boundaries.
"""
import re
import logging
import hashlib
from typing import Dict, List, Optional

import tiktoken

import config

logger = logging.getLogger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")

# ── Region keyword map ──────────────────────────────────────────────────────
REGION_KEYWORDS = {
    "south india": [
        "tamil nadu", "chennai", "karnataka", "bangalore", "bengaluru",
        "kerala", "kochi", "thiruvananthapuram", "andhra pradesh",
        "hyderabad", "telangana", "south india", "coimbatore", "mysore",
        "madurai", "visakhapatnam", "vijayawada",
    ],
    "north india": [
        "delhi", "ncr", "uttar pradesh", "haryana", "gurgaon", "gurugram",
        "noida", "punjab", "rajasthan", "north india", "lucknow", "jaipur",
        "chandigarh",
    ],
    "west india": [
        "mumbai", "maharashtra", "pune", "gujarat", "ahmedabad",
        "surat", "goa", "west india",
    ],
    "east india": [
        "kolkata", "west bengal", "odisha", "bhubaneswar",
        "bihar", "jharkhand", "east india",
    ],
}


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _split_sentences(text: str) -> List[str]:
    """Split text on sentence-ending punctuation, keeping the delimiter."""
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in parts if s.strip()]


def _extract_doc_type(filename: str) -> str:
    """Classify document type from filename keywords."""
    fn = filename.lower()
    if any(kw in fn for kw in ("case study", "case-study", "case_study", "case studies",
                                "success story", "success stories", " ss ", "ss_",
                                "gadgeon ss", "gadgeon_ss")):
        return "case_study"
    if any(kw in fn for kw in ("whitepaper", "white paper", "white-paper")):
        return "whitepaper"
    if any(kw in fn for kw in ("proposal",)):
        return "proposal"
    if any(kw in fn for kw in ("pitch", "deck")):
        return "pitch_deck"
    if any(kw in fn for kw in ("offering", "services", "overview", "corp overview",
                                "engineering stack", "delivery model",
                                "execution", "governance")):
        return "service_presentation"
    # Check file extension for whitepapers (docx files are typically whitepapers in this corpus)
    if fn.endswith(".docx"):
        return "whitepaper"
    return "unknown"


def _extract_year(filename: str, text_prefix: str = "") -> Optional[str]:
    """Extract year from filename first, then from first 500 chars of text."""
    m = re.search(r'20[1-2]\d', filename)
    if m:
        return m.group()
    m = re.search(r'20[1-2]\d', text_prefix[:500])
    if m:
        return m.group()
    return None


def _extract_regions(text: str) -> List[str]:
    """Scan text for region identifiers."""
    lower = text.lower()
    found = set()
    for region, keywords in REGION_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                found.add(region)
                break
    return sorted(found)


def chunk_document(doc: Dict) -> List[Dict]:
    """
    Chunk a document extracted by extractor.py.

    Args:
        doc: {"filename": str, "pages": [{"page": int, "text": str}], "full_text": str}

    Returns:
        List of chunk dicts with metadata.
    """
    filename = doc["filename"]
    pages = doc["pages"]
    full_text = doc["full_text"]

    doc_type = _extract_doc_type(filename)
    year = _extract_year(filename, full_text)
    regions = _extract_regions(full_text)

    chunks: List[Dict] = []

    for page_info in pages:
        page_num = page_info["page"]
        page_text = page_info["text"]
        sentences = _split_sentences(page_text)

        current_chunk_sentences: List[str] = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = _count_tokens(sentence)

            if current_tokens + sent_tokens > config.CHUNK_SIZE_TOKENS and current_chunk_sentences:
                # Emit chunk
                chunk_text = " ".join(current_chunk_sentences)
                chunk_id = _make_chunk_id(filename, page_num, len(chunks))
                chunks.append({
                    "chunk_id": chunk_id,
                    "chunk_text": chunk_text,
                    "filename": filename,
                    "doc_type": doc_type,
                    "year": year,
                    "page": page_num,
                    "regions": regions,
                    "token_count": _count_tokens(chunk_text),
                })

                # Overlap: keep last sentences up to CHUNK_OVERLAP_TOKENS
                overlap_sentences: List[str] = []
                overlap_tokens = 0
                for s in reversed(current_chunk_sentences):
                    st = _count_tokens(s)
                    if overlap_tokens + st > config.CHUNK_OVERLAP_TOKENS:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_tokens += st

                current_chunk_sentences = overlap_sentences
                current_tokens = overlap_tokens

            current_chunk_sentences.append(sentence)
            current_tokens += sent_tokens

        # Emit remaining text
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunk_id = _make_chunk_id(filename, page_num, len(chunks))
            chunks.append({
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,
                "filename": filename,
                "doc_type": doc_type,
                "year": year,
                "page": page_num,
                "regions": regions,
                "token_count": _count_tokens(chunk_text),
            })

    logger.info("Chunked %s → %d chunks (type=%s, year=%s)", filename, len(chunks), doc_type, year)
    return chunks


def _make_chunk_id(filename: str, page: int, idx: int) -> str:
    raw = f"{filename}::p{page}::c{idx}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

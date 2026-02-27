"""
Input validation + output safety checks.
"""
import re
import logging
from typing import Optional, Dict

import config

logger = logging.getLogger(__name__)

# ── Off-topic patterns ──────────────────────────────────────────────────────
OFF_TOPIC_PATTERNS = [
    r"\b(weather|temperature|forecast)\b",
    r"\b(joke|funny|humor|laugh)\b",
    r"\b(poem|poetry|rhyme|haiku|limerick)\b",
    r"\b(recipe|cook|bake|ingredient)\b",
    r"\b(write me|compose|create a story|write a)\b",
    r"\b(movie|film|song|music|lyrics)\b",
    r"\b(sports? scores?|game results?|who won|cricket|football|basketball|soccer)\b",
    r"\b(news today|current events|stock price|stock market)\b",
    r"\b(translate|translation)\b",
    r"\b(code|program|debug|function|algorithm|python|javascript)\b.*\b(write|create|build|fix)\b",
    r"\b(math|calculate|solve|equation)\b",
    r"\b(who is|what is the capital|when was .* born)\b",
    r"\b(play|game|trivia|quiz)\b",
    r"\b(horoscope|zodiac|astrology)\b",
    r"\b(homework|assignment|exam|test prep)\b",
    r"\bhow\s+(do|can|to)\s+.*(cook|make food|bake|pasta|recipe)\b",
    r"\b(diet|fitness|workout|exercise|weight loss)\b",
    r"\b(relationship|dating|love advice)\b",
    r"\b(travel|vacation|hotel|flight|booking)\b",
    r"\b(personal|life advice|self.help)\b",
]

# ── Prompt injection patterns ───────────────────────────────────────────────
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above",
    r"forget\s+(all\s+)?previous",
    r"you\s+are\s+now",
    r"act\s+as\s+(?!a\s+sales)",  # "act as" unless "act as a sales..."
    r"system\s*prompt",
    r"new\s+instructions",
    r"override\s+instructions",
    r"pretend\s+you",
    r"role[\s-]*play",
    r"jailbreak",
    r"DAN\b",
    r"do\s+anything\s+now",
    r"disregard\s+(all\s+)?prior",
    r"\[system\]",
    r"<\s*system\s*>",
]

# ── PII patterns ────────────────────────────────────────────────────────────
PII_PATTERNS = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone": r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b",
    "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "pan": r"\b[A-Z]{5}\d{4}[A-Z]\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
}

# ── Toxicity keywords (minimal set for a sales tool) ────────────────────────
TOXICITY_KEYWORDS = [
    "kill", "murder", "attack", "bomb", "terrorist", "weapon",
    "hate", "racist", "sexist", "slur",
]


# ═══════════════════════════════════════════════════════════════════════════
# INPUT GUARDRAILS
# ═══════════════════════════════════════════════════════════════════════════

def check_input(query: str) -> Optional[Dict]:
    """
    Run all input guardrails. Returns None if clean,
    or a guardrail dict {"type": ..., "message": ...} if blocked.
    """
    # 1. Query length
    if len(query) > config.MAX_QUERY_LENGTH:
        return {
            "type": "query_too_long",
            "message": f"Query exceeds the maximum length of {config.MAX_QUERY_LENGTH} characters. Please shorten your question.",
        }

    if not query.strip():
        return {
            "type": "empty_query",
            "message": "Please enter a question about our proposals, case studies, or whitepapers.",
        }

    # 2. Prompt injection
    q_lower = query.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, q_lower):
            logger.warning("Prompt injection detected: %s", query[:100])
            return {
                "type": "prompt_injection",
                "message": "This query appears to contain instructions that could compromise the system. Please rephrase your sales-related question.",
            }

    # 3. PII detection
    pii_found = []
    for pii_type, pattern in PII_PATTERNS.items():
        if re.search(pattern, query):
            pii_found.append(pii_type)
    if pii_found:
        logger.warning("PII detected in query (%s): %s", pii_found, query[:50])
        return {
            "type": "pii_detected",
            "message": f"Your query appears to contain sensitive personal information ({', '.join(pii_found)}). Please remove it and try again.",
        }

    # 4. Off-topic detection
    if _is_off_topic(query):
        return {
            "type": "off_topic",
            "message": "This question is outside the scope of the Sales Co-Pilot. I can only answer questions related to our proposals, case studies, and whitepapers.",
        }

    return None


def _is_off_topic(query: str) -> bool:
    """Check if query matches off-topic patterns."""
    q_lower = query.lower()

    # Allow sales-adjacent keywords to pass even if they partially match off-topic
    # Uses word boundaries to avoid false positives (e.g. "oem" inside "poem")
    sales_patterns = [
        r"\bproposal", r"\bcase stud", r"\bwhitepaper", r"\bclient\b", r"\bcustomer\b",
        r"\bsales\b", r"\bproject\b", r"\bengagement\b", r"\bdelivery\b", r"\bsolution\b",
        r"\bmanufactur", r"\bdigital\b", r"\btransformation\b", r"\bimplementation\b",
        r"\bexperience\b", r"\bindustr", r"\brevenue\b", r"\bcost\b", r"\broi\b",
        r"\bdifferentiator", r"\bcompetitor", r"\boutcome\b", r"\bmetric",
        r"\bpitch\b", r"\bdeck\b", r"\besg\b", r"\bsustainab", r"\bmes\b", r"\bscada\b",
        r"\bfactory\b", r"\boem\b", r"\bsupply chain\b", r"\banalytics\b",
        r"\biot\b", r"\bembedded\b", r"\boffering", r"\bcapabilit",
        r"\bengineering\b", r"\bhealthcare\b", r"\bpharma", r"\bautomotive\b",
        r"\btelehealth\b", r"\be-mobility\b", r"\bedge\b", r"\bcloud\b",
    ]
    if any(re.search(p, q_lower) for p in sales_patterns):
        return False

    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, q_lower):
            return True

    return False


def strip_pii(text: str) -> str:
    """Remove PII from text (for pre-processing before sending to LLM)."""
    for pii_type, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"[{pii_type.upper()}_REDACTED]", text)
    return text


# ═══════════════════════════════════════════════════════════════════════════
# OUTPUT GUARDRAILS
# ═══════════════════════════════════════════════════════════════════════════

def check_output(answer: str, retrieved_filenames: list) -> Optional[Dict]:
    """
    Run output guardrails on the LLM's answer.
    Returns None if clean, or a guardrail dict if issues found.
    """
    # 1. Toxicity check
    answer_lower = answer.lower()
    for kw in TOXICITY_KEYWORDS:
        # Only flag if the word appears in a non-business context
        if kw in answer_lower and not _is_business_context(answer_lower, kw):
            logger.warning("Toxicity keyword detected in output: %s", kw)
            return {
                "type": "toxic_content",
                "message": "The generated response was flagged for potentially inappropriate content. Please rephrase your question.",
            }

    # 2. Citation verification — check that [Source: ...] references match retrieved docs
    cited_sources = re.findall(r'\[Source:\s*([^,\]]+)', answer)
    if cited_sources:
        unmatched = []
        for src in cited_sources:
            src_clean = src.strip()
            if not any(src_clean in fn for fn in retrieved_filenames):
                unmatched.append(src_clean)
        if unmatched:
            logger.warning("Unmatched citations found: %s", unmatched)
            # Don't block, but flag
            return {
                "type": "citation_warning",
                "message": f"Some citations could not be verified against retrieved documents: {unmatched}",
            }

    return None


def _is_business_context(text: str, keyword: str) -> bool:
    """Check if a toxicity keyword is used in business context."""
    # Common sales/business phrases
    business_phrases = [
        "cost kill", "kill rate", "attack the market", "market attack",
        "competitive attack", "supply chain attack",
    ]
    return any(phrase in text for phrase in business_phrases)


def add_low_confidence_disclaimer(answer: str) -> str:
    """Prepend disclaimer for low-confidence answers."""
    disclaimer = (
        "Limited matching content found. "
        "The following answer may be incomplete - verify with the source team.\n\n"
    )
    return disclaimer + answer

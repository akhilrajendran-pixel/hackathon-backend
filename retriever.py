"""
Hybrid search: vector (ChromaDB) + keyword (BM25) with Reciprocal Rank Fusion.
Includes metadata pre-filtering and confidence scoring.
"""
import re
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import config
import indexer

logger = logging.getLogger(__name__)


# ── Metadata filter extraction ──────────────────────────────────────────────

def _extract_metadata_filters(query: str) -> Dict:
    """
    Parse the query for explicit metadata cues (year, doc_type, region).
    Returns a dict of ChromaDB where-filters.
    """
    filters = {}
    q_lower = query.lower()

    # Year detection
    year_match = re.search(r'\b(20[1-2]\d)\b', query)
    if year_match:
        filters["year"] = year_match.group(1)

    # Doc type detection
    if any(kw in q_lower for kw in ("case study", "case studies", "case-study", "success story", "success stories")):
        filters["doc_type"] = "case_study"
    elif any(kw in q_lower for kw in ("whitepaper", "white paper", "whitepapers", "white papers")):
        filters["doc_type"] = "whitepaper"
    elif any(kw in q_lower for kw in ("proposal", "proposals")):
        filters["doc_type"] = "proposal"
    elif any(kw in q_lower for kw in ("pitch", "deck", "pitch deck")):
        filters["doc_type"] = "pitch_deck"
    elif any(kw in q_lower for kw in ("service presentation", "offerings", "service overview")):
        filters["doc_type"] = "service_presentation"

    # Region detection
    from chunker import REGION_KEYWORDS
    for region, keywords in REGION_KEYWORDS.items():
        for kw in keywords:
            if kw in q_lower:
                filters["region"] = region
                break
        if "region" in filters:
            break

    return filters


def _build_chroma_where(filters: Dict) -> Optional[Dict]:
    """Convert our filter dict into ChromaDB where clause."""
    conditions = []
    if "year" in filters:
        conditions.append({"year": {"$eq": filters["year"]}})
    if "doc_type" in filters:
        conditions.append({"doc_type": {"$eq": filters["doc_type"]}})
    if "region" in filters:
        conditions.append({"regions": {"$contains": filters["region"]}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ── Vector search ───────────────────────────────────────────────────────────

def _vector_search(query: str, where: Optional[Dict], top_k: int) -> List[Tuple[str, float]]:
    """
    Query ChromaDB. Returns list of (chunk_id, distance_score).
    ChromaDB cosine distance: 0 = identical, 2 = opposite.
    We convert to similarity: sim = 1 - dist/2.
    """
    collection = indexer.get_collection()
    if collection is None:
        return []

    # Use query_texts so ChromaDB auto-embeds with its default function
    kwargs = {
        "query_texts": [query],
        "n_results": top_k,
        "include": ["distances"],
    }
    if where:
        kwargs["where"] = where

    try:
        results = collection.query(**kwargs)
    except Exception as e:
        logger.warning("ChromaDB query with filter failed (%s), retrying without filter", e)
        kwargs.pop("where", None)
        results = collection.query(**kwargs)

    ids = results["ids"][0] if results["ids"] else []
    distances = results["distances"][0] if results["distances"] else []

    scored = []
    for cid, dist in zip(ids, distances):
        similarity = 1.0 - dist / 2.0  # cosine distance → similarity
        scored.append((cid, similarity))

    return scored


# ── BM25 search ─────────────────────────────────────────────────────────────

def _bm25_search(query: str, filters: Dict, top_k: int) -> List[Tuple[str, float]]:
    """
    Query BM25 index. Returns list of (chunk_id, bm25_score).
    Scores are normalized to [0, 1] range.
    """
    bm25, chunks = indexer.get_bm25()
    if bm25 is None or not chunks:
        return []

    tokenized_query = query.lower().split()
    raw_scores = bm25.get_scores(tokenized_query)

    # Apply metadata filters manually
    scored = []
    for i, score in enumerate(raw_scores):
        chunk = chunks[i]
        if "year" in filters and chunk.get("year") != filters["year"]:
            continue
        if "doc_type" in filters and chunk.get("doc_type") != filters["doc_type"]:
            continue
        if "region" in filters:
            if filters["region"] not in chunk.get("regions", []):
                continue
        scored.append((chunk["chunk_id"], score))

    # Normalize
    if scored:
        max_score = max(s for _, s in scored) or 1.0
        scored = [(cid, s / max_score) for cid, s in scored]

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ── Reciprocal Rank Fusion ──────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    vector_results: List[Tuple[str, float]],
    bm25_results: List[Tuple[str, float]],
    k: int = config.RRF_K,
) -> List[Tuple[str, float]]:
    """
    Merge two ranked lists using RRF: score = sum(1 / (k + rank)).
    """
    rrf_scores: Dict[str, float] = defaultdict(float)

    for rank, (cid, _) in enumerate(vector_results):
        rrf_scores[cid] += 1.0 / (k + rank + 1)

    for rank, (cid, _) in enumerate(bm25_results):
        rrf_scores[cid] += 1.0 / (k + rank + 1)

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return merged


# ── Main retrieval function ─────────────────────────────────────────────────

def retrieve(query: str, top_k: int = config.FINAL_TOP_K) -> List[Dict]:
    """
    Full hybrid retrieval pipeline:
    1. Extract metadata filters from query
    2. Vector search (ChromaDB)
    3. BM25 keyword search
    4. RRF fusion
    5. Return top-k results with full metadata

    Returns list of dicts with: chunk_id, chunk_text, filename, doc_type,
    year, page, relevance_score, drive_link, regions
    """
    # Step 1: metadata pre-filter
    filters = _extract_metadata_filters(query)
    chroma_where = _build_chroma_where(filters)
    logger.info("Query filters: %s", filters)

    # Step 2: vector search
    vector_results = _vector_search(query, chroma_where, config.VECTOR_TOP_K)

    # Step 3: BM25 search
    bm25_results = _bm25_search(query, filters, config.BM25_TOP_K)

    # Step 4: RRF fusion
    fused = _reciprocal_rank_fusion(vector_results, bm25_results)

    # Build chunk lookup
    all_chunks = indexer.get_all_chunks()
    chunk_map = {c["chunk_id"]: c for c in all_chunks}

    # Also get vector similarity scores for confidence calculation
    vector_score_map = {cid: score for cid, score in vector_results}

    # Step 5: assemble top-k results
    results = []
    for chunk_id, rrf_score in fused[:top_k]:
        chunk = chunk_map.get(chunk_id)
        if not chunk:
            continue

        # Use vector similarity as the relevance score (more interpretable)
        vec_score = vector_score_map.get(chunk_id, 0.0)

        results.append({
            "chunk_id": chunk_id,
            "chunk_text": chunk["chunk_text"],
            "filename": chunk["filename"],
            "doc_type": chunk["doc_type"],
            "year": chunk.get("year"),
            "page": chunk["page"],
            "regions": chunk.get("regions", []),
            "relevance_score": round(vec_score, 4),
            "rrf_score": round(rrf_score, 4),
            "drive_link": indexer.get_drive_link(chunk["filename"]),
        })

    logger.info("Retrieved %d chunks for query (filters=%s)", len(results), filters)
    return results


def compute_confidence(results: List[Dict]) -> Tuple[str, float]:
    """
    Compute confidence level from top-3 retrieval scores.
    Returns (level_str, numeric_score).
    """
    if not results:
        return "low", 0.0

    top_scores = [r["relevance_score"] for r in results[:3]]
    avg_score = sum(top_scores) / len(top_scores)

    if avg_score >= config.HIGH_CONFIDENCE_THRESHOLD:
        level = "high"
    elif avg_score >= config.MEDIUM_CONFIDENCE_THRESHOLD:
        level = "medium"
    else:
        level = "low"

    return level, round(avg_score, 4)

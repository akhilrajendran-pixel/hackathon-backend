"""
Hybrid search: vector (OpenSearch kNN) + keyword (OpenSearch BM25) with Reciprocal Rank Fusion.
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
    Returns a dict of filter values.
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


def _build_opensearch_filter(filters: Dict) -> Optional[Dict]:
    """Convert our filter dict into an OpenSearch bool filter clause."""
    conditions = []

    if "year" in filters:
        conditions.append({"term": {"year": filters["year"]}})
    if "doc_type" in filters:
        conditions.append({"term": {"doc_type": filters["doc_type"]}})
    if "region" in filters:
        conditions.append({"term": {"regions": filters["region"]}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"bool": {"must": conditions}}


# ── Vector search ───────────────────────────────────────────────────────────

def _vector_search(query: str, os_filter: Optional[Dict], top_k: int) -> List[Tuple[str, float]]:
    """
    OpenSearch kNN search. Returns list of (chunk_id, similarity_score).
    Cosinesimil scores are already in [0, 1] range.
    """
    client = indexer.get_collection()
    if client is None:
        return []

    query_vector = indexer.embed_query(query)

    knn_body = {
        "size": top_k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": top_k,
                }
            }
        },
        "_source": ["chunk_id"],
    }

    # Add filter if present
    if os_filter:
        knn_body["query"]["knn"]["embedding"]["filter"] = os_filter

    try:
        results = client.search(index=config.OPENSEARCH_INDEX_NAME, body=knn_body)
    except Exception as e:
        logger.warning("OpenSearch kNN query with filter failed (%s), retrying without filter", e)
        knn_body["query"]["knn"]["embedding"].pop("filter", None)
        results = client.search(index=config.OPENSEARCH_INDEX_NAME, body=knn_body)

    scored = []
    for hit in results["hits"]["hits"]:
        chunk_id = hit["_source"]["chunk_id"]
        score = hit["_score"]  # cosinesimil: higher is more similar
        scored.append((chunk_id, score))

    return scored


# ── BM25 search ─────────────────────────────────────────────────────────────

def _bm25_search(query: str, os_filter: Optional[Dict], top_k: int) -> List[Tuple[str, float]]:
    """
    OpenSearch BM25 keyword search on the 'text' field.
    Scores are normalized to [0, 1] range.
    """
    client = indexer.get_collection()
    if client is None:
        return []

    # Build query body
    if os_filter:
        search_body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": {"match": {"text": query}},
                    "filter": os_filter if isinstance(os_filter, list) else [os_filter],
                }
            },
            "_source": ["chunk_id"],
        }
    else:
        search_body = {
            "size": top_k,
            "query": {
                "match": {"text": query}
            },
            "_source": ["chunk_id"],
        }

    try:
        results = client.search(index=config.OPENSEARCH_INDEX_NAME, body=search_body)
    except Exception as e:
        logger.warning("OpenSearch BM25 query failed (%s), retrying without filter", e)
        search_body = {
            "size": top_k,
            "query": {"match": {"text": query}},
            "_source": ["chunk_id"],
        }
        results = client.search(index=config.OPENSEARCH_INDEX_NAME, body=search_body)

    hits = results["hits"]["hits"]
    if not hits:
        return []

    # Normalize scores to [0, 1]
    max_score = hits[0]["_score"] if hits[0]["_score"] > 0 else 1.0
    scored = []
    for hit in hits:
        chunk_id = hit["_source"]["chunk_id"]
        normalized = hit["_score"] / max_score
        scored.append((chunk_id, normalized))

    return scored


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
    2. Vector search (OpenSearch kNN)
    3. BM25 keyword search (OpenSearch match)
    4. RRF fusion
    5. Return top-k results with full metadata

    Returns list of dicts with: chunk_id, chunk_text, filename, doc_type,
    year, page, relevance_score, drive_link, regions
    """
    # Step 1: metadata pre-filter
    filters = _extract_metadata_filters(query)
    os_filter = _build_opensearch_filter(filters)
    logger.info("Query filters: %s", filters)

    # Step 2: vector search
    vector_results = _vector_search(query, os_filter, config.VECTOR_TOP_K)

    # Step 3: BM25 search
    bm25_results = _bm25_search(query, os_filter, config.BM25_TOP_K)

    # Fallback: if filters returned nothing, retry without filters
    if not vector_results and not bm25_results and os_filter:
        logger.info("Filtered search returned 0 results, retrying without filters")
        vector_results = _vector_search(query, None, config.VECTOR_TOP_K)
        bm25_results = _bm25_search(query, None, config.BM25_TOP_K)

    # Step 4: RRF fusion
    fused = _reciprocal_rank_fusion(vector_results, bm25_results)

    # Build score lookup from vector results for confidence
    vector_score_map = {cid: score for cid, score in vector_results}

    # Fetch full documents for the fused top-k chunk IDs
    client = indexer.get_collection()
    if client is None:
        return []

    # Get top candidate IDs
    top_ids = [cid for cid, _ in fused[:top_k * 2]]  # fetch extra in case some miss

    # Batch fetch from OpenSearch using terms query on chunk_id field
    results = []
    doc_map = {}
    if top_ids:
        try:
            fetch_response = client.search(
                index=config.OPENSEARCH_INDEX_NAME,
                body={
                    "query": {"terms": {"chunk_id": top_ids}},
                    "_source": {"excludes": ["embedding"]},
                    "size": len(top_ids),
                },
            )
            for hit in fetch_response["hits"]["hits"]:
                src = hit["_source"]
                doc_map[src["chunk_id"]] = src
        except Exception as e:
            logger.warning("Batch fetch failed: %s", e)
            doc_map = {}

    # Step 5: assemble top-k results
    for chunk_id, rrf_score in fused[:top_k]:
        src = doc_map.get(chunk_id)
        if not src:
            continue

        vec_score = vector_score_map.get(chunk_id, 0.0)

        results.append({
            "chunk_id": chunk_id,
            "chunk_text": src.get("text", ""),
            "filename": src["filename"],
            "doc_type": src["doc_type"],
            "year": src.get("year"),
            "page": src.get("page", 0),
            "regions": src.get("regions", []),
            "relevance_score": round(vec_score, 4),
            "rrf_score": round(rrf_score, 4),
            "drive_link": indexer.get_drive_link(src["filename"]),
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

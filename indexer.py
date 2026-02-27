"""
Amazon OpenSearch Serverless vector index with Bedrock Titan Embed v2.
Handles both kNN vector search and BM25 keyword search in a single index.
"""
import json
import time
import logging
from typing import List, Dict, Optional

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, helpers
from requests_aws4auth import AWS4Auth

import config

logger = logging.getLogger(__name__)

# Module-level cached clients
_opensearch_client: Optional[OpenSearch] = None
_bedrock_client = None

# Drive link map: filename → webViewLink (in-memory cache)
_drive_links: Dict[str, str] = {}


# ── AWS Clients ─────────────────────────────────────────────────────────────

def _get_opensearch_client() -> OpenSearch:
    """Get or create a cached OpenSearch Serverless client with AWS V4 auth."""
    global _opensearch_client
    if _opensearch_client is None:
        if not config.OPENSEARCH_ENDPOINT:
            raise RuntimeError("OPENSEARCH_ENDPOINT is not set in .env")

        session = boto3.Session(
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
            region_name=config.AWS_REGION,
        )
        credentials = session.get_credentials()
        awsauth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            config.AWS_REGION,
            "aoss",  # OpenSearch Serverless service name
            session_token=credentials.token,
        )

        # Extract host from endpoint URL
        host = config.OPENSEARCH_ENDPOINT.replace("https://", "").replace("http://", "").rstrip("/")

        _opensearch_client = OpenSearch(
            hosts=[{"host": host, "port": 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=60,
        )
    return _opensearch_client


def _get_bedrock_embed_client():
    """Get or create a cached Bedrock runtime client for embeddings."""
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=config.AWS_REGION,
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY,
        )
    return _bedrock_client


# ── Embedding ────────────────────────────────────────────────────────────────

def _embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of texts using Bedrock Titan Embed v2.
    Returns a list of embedding vectors.
    """
    client = _get_bedrock_embed_client()
    vectors = []
    for text in texts:
        body = json.dumps({
            "inputText": text,
            "dimensions": config.EMBEDDING_DIMENSIONS,
        })
        response = client.invoke_model(
            modelId=config.BEDROCK_EMBED_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        result = json.loads(response["body"].read())
        vectors.append(result["embedding"])
    return vectors


def embed_query(query: str) -> List[float]:
    """Embed a single query string (called by retriever for kNN search)."""
    return _embed_texts([query])[0]


# ── Index Management ─────────────────────────────────────────────────────────

def _ensure_index():
    """Create the OpenSearch index with kNN mapping if it doesn't exist."""
    client = _get_opensearch_client()
    index = config.OPENSEARCH_INDEX_NAME

    if client.indices.exists(index=index):
        return

    body = {
        "settings": {
            "index": {
                "knn": True,
            }
        },
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": config.EMBEDDING_DIMENSIONS,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "faiss",
                    },
                },
                "text": {"type": "text"},
                "chunk_id": {"type": "keyword"},
                "filename": {"type": "keyword"},
                "doc_type": {"type": "keyword"},
                "year": {"type": "keyword"},
                "page": {"type": "integer"},
                "regions": {"type": "keyword"},  # stored as array of keywords
                "drive_link": {"type": "keyword", "index": False},
            }
        },
    }

    client.indices.create(index=index, body=body)
    logger.info("Created OpenSearch index '%s' with kNN mapping, waiting for it to become active...", index)
    time.sleep(5)  # Give OpenSearch Serverless time to initialize the index


# ── Build Index ──────────────────────────────────────────────────────────────

def build_index(chunks: List[Dict]):
    """
    Delete existing docs, embed all chunks, and bulk-index into OpenSearch.
    """
    if not chunks:
        logger.warning("No chunks to index")
        return

    client = _get_opensearch_client()
    index = config.OPENSEARCH_INDEX_NAME

    # Delete and recreate index to clear all existing documents
    try:
        if client.indices.exists(index=index):
            client.indices.delete(index=index)
            logger.info("Deleted existing index '%s'", index)
    except Exception as e:
        logger.warning("Could not delete index (may not exist): %s", e)

    _ensure_index()

    # Embed and index in batches of 25
    batch_size = 25
    total = len(chunks)

    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["chunk_text"] for c in batch]

        # Embed batch
        vectors = _embed_texts(texts)

        # Prepare bulk actions
        actions = []
        for chunk, vector in zip(batch, vectors):
            regions = chunk.get("regions", [])

            doc = {
                "embedding": vector,
                "text": chunk["chunk_text"],
                "chunk_id": chunk["chunk_id"],
                "filename": chunk["filename"],
                "doc_type": chunk["doc_type"],
                "year": chunk.get("year") or "unknown",
                "page": chunk["page"],
                "regions": regions,  # stored as keyword array directly
                "drive_link": _drive_links.get(chunk["filename"], ""),
            }

            actions.append({
                "_index": index,
                "_source": doc,
            })

        # Bulk index — raise_on_error=False so partial failures don't crash ingestion
        success, errors = helpers.bulk(client, actions, refresh=False, raise_on_error=False)
        logger.info("  Indexed batch %d-%d / %d (%d success)", i, min(i + batch_size, total), total, success)
        if errors:
            logger.warning("  Bulk index errors in batch: %d failed", len(errors))

    # OpenSearch Serverless handles refresh automatically (no explicit refresh API)
    logger.info("OpenSearch index built with %d vectors", total)


# ── Query Helpers ────────────────────────────────────────────────────────────

def get_chunk_count() -> int:
    """Return the number of indexed documents. Works after restart."""
    try:
        client = _get_opensearch_client()
        result = client.count(index=config.OPENSEARCH_INDEX_NAME)
        return result["count"]
    except Exception:
        return 0


def get_all_chunks() -> List[Dict]:
    """
    Paginate through all documents using search_after (for /admin/pipeline).
    OpenSearch Serverless does not support the scroll API.
    """
    try:
        client = _get_opensearch_client()
        index = config.OPENSEARCH_INDEX_NAME

        chunks = []
        body = {
            "query": {"match_all": {}},
            "_source": {"excludes": ["embedding"]},
            "size": 500,
            "sort": [{"_doc": "asc"}],
        }

        while True:
            response = client.search(index=index, body=body)
            hits = response["hits"]["hits"]
            if not hits:
                break

            for hit in hits:
                src = hit["_source"]
                chunks.append({
                    "chunk_id": src["chunk_id"],
                    "chunk_text": src.get("text", ""),
                    "filename": src["filename"],
                    "doc_type": src["doc_type"],
                    "year": src.get("year"),
                    "page": src.get("page", 0),
                    "regions": src.get("regions", []),
                })

            # Use last hit's sort value for next page
            body["search_after"] = hits[-1]["sort"]

        return chunks
    except Exception as e:
        logger.warning("get_all_chunks failed: %s", e)
        return []


# ── Drive Links ──────────────────────────────────────────────────────────────

def set_drive_links(links: Dict[str, str]):
    """Store filename → drive link mapping from ingestion."""
    global _drive_links
    _drive_links = links


def get_drive_link(filename: str) -> Optional[str]:
    """Get drive link from in-memory cache, fallback to OpenSearch."""
    # Check in-memory cache first
    if filename in _drive_links:
        return _drive_links[filename]

    # Fallback: query OpenSearch for one doc with this filename
    try:
        client = _get_opensearch_client()
        result = client.search(
            index=config.OPENSEARCH_INDEX_NAME,
            body={
                "query": {"term": {"filename": filename}},
                "_source": ["drive_link"],
                "size": 1,
            },
        )
        hits = result["hits"]["hits"]
        if hits:
            link = hits[0]["_source"].get("drive_link", "")
            if link:
                _drive_links[filename] = link  # cache it
                return link
    except Exception:
        pass
    return None


def get_collection():
    """
    Returns the OpenSearch client if the index has data.
    Backward-compatible check used by retriever.
    """
    try:
        if get_chunk_count() > 0:
            return _get_opensearch_client()
    except Exception:
        pass
    return None

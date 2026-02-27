"""
ChromaDB vector index + BM25 keyword index.
Uses ChromaDB's built-in default embedding function (no OpenAI needed).
"""
import logging
from typing import List, Dict, Optional

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

import config

logger = logging.getLogger(__name__)

# Module-level state
_chroma_client: Optional[chromadb.ClientAPI] = None
_collection = None
_bm25_index: Optional[BM25Okapi] = None
_bm25_chunks: List[Dict] = []
_all_chunks: List[Dict] = []

# Drive link map: filename → webViewLink
_drive_links: Dict[str, str] = {}


def set_drive_links(links: Dict[str, str]):
    """Store filename → drive link mapping from ingestion."""
    global _drive_links
    _drive_links = links


def get_drive_link(filename: str) -> Optional[str]:
    return _drive_links.get(filename)


def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            is_persistent=True,
            persist_directory=config.CHROMA_PERSIST_DIR,
        ))
        # Delete existing collection if re-indexing
        try:
            _chroma_client.delete_collection(config.CHROMA_COLLECTION_NAME)
        except Exception:
            pass
        # Use ChromaDB's default embedding function (all-MiniLM-L6-v2)
        _collection = _chroma_client.create_collection(
            name=config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def embed_query(query: str) -> List[float]:
    """Embed a single query using ChromaDB's default embedding function."""
    collection = _get_collection()
    # ChromaDB's default embedding function is accessible via the collection
    ef = collection._embedding_function
    return ef([query])[0]


def build_index(chunks: List[Dict]):
    """
    Build both ChromaDB vector index and BM25 keyword index from chunks.
    ChromaDB handles embedding automatically with its default function.
    """
    global _bm25_index, _bm25_chunks, _all_chunks

    if not chunks:
        logger.warning("No chunks to index")
        return

    _all_chunks = chunks
    collection = _get_collection()

    # Prepare data
    ids = [c["chunk_id"] for c in chunks]
    documents = [c["chunk_text"] for c in chunks]
    metadatas = []
    for c in chunks:
        metadatas.append({
            "filename": c["filename"],
            "doc_type": c["doc_type"],
            "year": c.get("year") or "unknown",
            "page": c["page"],
            "regions": ",".join(c.get("regions", [])),
        })

    # Add to ChromaDB in batches — ChromaDB auto-embeds using its default function
    logger.info("Indexing %d chunks into ChromaDB (auto-embedding)...", len(chunks))
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        end = i + batch_size
        collection.add(
            ids=ids[i:end],
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )
        logger.info("  Indexed batch %d-%d / %d", i, min(end, len(ids)), len(ids))
    logger.info("ChromaDB index built with %d vectors", len(ids))

    # Build BM25 index
    tokenized = [doc.lower().split() for doc in documents]
    _bm25_index = BM25Okapi(tokenized)
    _bm25_chunks = chunks
    logger.info("BM25 index built with %d documents", len(chunks))


def get_all_chunks() -> List[Dict]:
    return _all_chunks


def get_collection():
    return _get_collection() if _collection else None


def get_bm25():
    return _bm25_index, _bm25_chunks


def get_chunk_count() -> int:
    return len(_all_chunks)

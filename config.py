"""
Configuration — all constants, API keys, folder IDs.
"""
import os
from pathlib import Path

# Load .env file if present
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

# Google Drive (comma-separated folder IDs to pull from multiple folders)
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "1_1Fu2G7b4FXIoRUW1Nvo04kifbpdgYh2")
GOOGLE_DRIVE_FOLDER_IDS = [fid.strip() for fid in GOOGLE_DRIVE_FOLDER_ID.split(",") if fid.strip()]
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", "service_account.json")

# AWS credentials
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")

# Amazon Bedrock (LLM) — can use a different region than OpenSearch
BEDROCK_LLM_REGION = os.getenv("BEDROCK_LLM_REGION", "") or AWS_REGION
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-sonnet-20241022-v2:0")

# Amazon Bedrock (Embeddings)
BEDROCK_EMBED_MODEL_ID = os.getenv("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1024"))

# Amazon OpenSearch Serverless
OPENSEARCH_ENDPOINT = os.getenv("OPENSEARCH_ENDPOINT", "")
OPENSEARCH_INDEX_NAME = os.getenv("OPENSEARCH_INDEX_NAME", "sales-copilot")

# Chunking
CHUNK_SIZE_TOKENS = 600        # target 500-700
CHUNK_OVERLAP_TOKENS = 100
MAX_CHUNK_CHARS = 3000         # safety cap

# Retrieval
VECTOR_TOP_K = 15
BM25_TOP_K = 15
FINAL_TOP_K = 5
RRF_K = 60

# Confidence thresholds
HIGH_CONFIDENCE_THRESHOLD = 0.80
MEDIUM_CONFIDENCE_THRESHOLD = 0.55
NO_ANSWER_THRESHOLD = 0.40

# Session
SESSION_TTL_MINUTES = 30

# Guardrails
MAX_QUERY_LENGTH = 1000

# Concurrency
MAX_CONCURRENT_LLM_CALLS = 5

# Local fallback
LOCAL_DOCS_DIR = "local_docs"

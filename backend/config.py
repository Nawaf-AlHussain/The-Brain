import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Deployment mode detection
# ---------------------------------------------------------------------------
# Auto-detected on Vercel (VERCEL=1). Can also be forced with
# DEPLOYMENT_MODE=serverless for any other ephemeral-filesystem host.
IS_SERVERLESS = (
    os.getenv("VERCEL") == "1"
    or os.getenv("DEPLOYMENT_MODE", "").lower() in {"serverless", "vercel", "lite"}
)

# Engine selection
LLM_ENGINE = os.getenv("LLM_ENGINE", "ollama").lower()

# Base URLs
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LLAMA_CPP_BASE_URL = os.getenv("LLAMA_CPP_BASE_URL", "http://localhost:8080/v1")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
RERANKER_BASE_URL = os.getenv("RERANKER_BASE_URL", "")
# External rerankers (Cohere / Jina / SiliconFlow) may use a different key than
# the LLM provider. Falls back to OPENAI_API_KEY when not set.
RERANKER_API_KEY = os.getenv("RERANKER_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")

# Optional separate embedding endpoint — lets you mix providers, e.g. Groq for
# LLM + Gemini/OpenAI for embeddings. Falls back to the OpenAI settings.
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")

# External API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-no-key-required")

# Model names
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5:9b")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen2.5vl:latest")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:8b")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# Model settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP_SIZE", "100"))
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "32768"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "7200"))
LLM_MAX_ASYNC = int(os.getenv("LLM_MAX_ASYNC", "1"))
EMBEDDING_TIMEOUT = int(os.getenv("EMBEDDING_TIMEOUT", "300"))
EMBEDDING_MAX_ASYNC = int(os.getenv("EMBEDDING_MAX_ASYNC", "1"))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "4096"))
MAX_EMBED_TOKENS = int(os.getenv("MAX_EMBED_TOKENS", "8192"))

# Neo4j
# Leave NEO4J_URI empty (or unset on serverless) to run without a graph DB —
# the app falls back to in-memory NetworkX storage (demo mode, non-persistent).
NEO4J_URI = os.getenv("NEO4J_URI", "" if IS_SERVERLESS else "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Optional external vector DB (recommended on serverless so vectors survive
# instance recycling). Empty = local NanoVectorDB files in WORKING_DIR.
QDRANT_URL = os.getenv("QDRANT_URL", "")
# QDRANT_API_KEY is read directly by LightRAG from the environment.

# Filestructure
# On serverless platforms (Vercel) only /tmp is writable, so default there.
# NOTE: /tmp is ephemeral — vector indexes do not survive instance recycling.
if IS_SERVERLESS:
    _DEFAULT_WORKING_DIR = "/tmp/rag_storage"
    _DEFAULT_UPLOAD_DIR = "/tmp/uploads"
    _DEFAULT_OUTPUT_DIR = "/tmp/output"
else:
    _DEFAULT_WORKING_DIR = "/app/rag_storage"
    _DEFAULT_UPLOAD_DIR = "/app/uploads"
    _DEFAULT_OUTPUT_DIR = "/app/output"

WORKING_DIR = os.getenv("WORKING_DIR", _DEFAULT_WORKING_DIR)
UPLOAD_DIR = os.getenv("UPLOAD_DIR", _DEFAULT_UPLOAD_DIR)
OUTPUT_DIR = os.getenv("OUTPUT_DIR", _DEFAULT_OUTPUT_DIR)
PARSER = os.getenv("PARSER", "mineru")

# File types that can be ingested without the heavy MinerU parser
# (serverless/lite mode: only these can be inserted directly as text)
TEXT_ONLY_EXTENSIONS = {".txt", ".md", ".html", ".json", ".csv"}

# Persistent files
HIDDEN_TYPES_FILE = Path(WORKING_DIR) / "hidden_types.json"
CONV_FILE = Path(WORKING_DIR) / "conversations.json"
COMPLETED_LOG = Path(WORKING_DIR) / "completed_docs.json"

# Document settings
# Union with TEXT_ONLY_EXTENSIONS so lite mode can actually ingest every type
# its 503 message advertises (.txt/.md/.html/.json/.csv) — previously .html,
# .json and .csv were rejected by the ALLOWED_EXTENSIONS gate before the
# lite-mode branch was reached.
ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".txt",
    ".md",
    ".pptx",
    ".xlsx",
} | TEXT_ONLY_EXTENSIONS
# Serverless platforms cap request bodies (Vercel: ~4.5 MB) — default lower there
if os.getenv("MAX_UPLOAD_BYTES"):
    MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES"))
elif IS_SERVERLESS:
    MAX_UPLOAD_BYTES = 4 * 1024 * 1024  # 4 MB
else:
    MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB

# config.py
from dotenv import load_dotenv
import os

load_dotenv()

VERTICA_CONFIG = {
    "server": os.getenv("VERTICA_SERVER"),
    "database": os.getenv("VERTICA_DATABASE"),
    "uid": os.getenv("VERTICA_UID"),
    "pwd": os.getenv("VERTICA_PWD"),
}
# Optional Vertica port (commonly 5433). If set, include it in the config.
_vertica_port = os.getenv("VERTICA_PORT")
if _vertica_port:
    # Store as int when possible, otherwise keep string
    try:
        VERTICA_CONFIG["port"] = int(_vertica_port)
    except ValueError:
        VERTICA_CONFIG["port"] = _vertica_port

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Optional OpenAI-compatible base URL (e.g., vLLM at http://127.0.0.1:8000/v1)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
# Optional timeout (seconds) for model requests
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "30"))

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
# Backward-compatible alias used in some modules
MODEL_NAME = LLM_MODEL

SIM_CUTOFF = float(os.getenv("SIM_CUTOFF", 0.80))
TOP_K = int(os.getenv("TOP_K", 3))

# Optional: force local embeddings instead of hitting the embeddings API
USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "0").strip() in {"1", "true", "True"}

# Optional: reduce TensorFlow logging when using sentence-transformers
if os.getenv("TF_CPP_MIN_LOG_LEVEL") is None:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# KPI planner variability
KPI_TEMPERATURE = float(os.getenv("KPI_TEMPERATURE", "0.7"))

# Optional LLM sampling controls for local servers (vLLM/ollama). These are ignored by OpenAI if unsupported.
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "1.0"))
LLM_PRESENCE_PENALTY = float(os.getenv("LLM_PRESENCE_PENALTY", "0.0"))
LLM_FREQUENCY_PENALTY = float(os.getenv("LLM_FREQUENCY_PENALTY", "0.0"))
# Use a different name to avoid clashing with selector TOP_K
_llm_top_k = os.getenv("LLM_TOP_K")
LLM_SAMPLING_TOP_K = int(_llm_top_k) if (_llm_top_k or "").strip() else None
_rep_pen = os.getenv("LLM_REPETITION_PENALTY")
LLM_REPETITION_PENALTY = float(_rep_pen) if (_rep_pen or "").strip() else None

# Validation strictness toggle
VALIDATION_MODE = (os.getenv("VALIDATION_MODE", "lenient") or "").strip().lower()
STRICT_VALIDATION = VALIDATION_MODE in {"strict", "true", "1"}

# Debug directory (relative to module locations); agents will create if missing
DEBUG_DIR = os.getenv("DEBUG_DIR", "debugg")

import os
import sys
from dotenv import load_dotenv

load_dotenv()

_script_name: str = os.path.splitext(os.path.basename(sys.argv[0]))[0]

# --- API Gemini ---

GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
MODEL_NAME: str = "gemini-3.1-flash-lite"

# Limites do plano gratuito do Google AI Studio
MAX_REQUESTS_PER_MINUTE: int = 15
MAX_REQUESTS_PER_DAY: int = 500

# Backoff exponencial em caso de erro 429
BACKOFF_INITIAL_DELAY: float = 4.0    # segundos
BACKOFF_MAX_DELAY: float = 120.0      # máximo de 2 minutos
BACKOFF_MAX_RETRIES: int = 5

# --- Caminhos ---

BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CACHE_DIR: str = os.path.join(BASE_DIR, "cache")
QUOTA_FILE: str = os.path.join(BASE_DIR, "cache", "_quota.json")
LOG_FILE: str = os.path.join(BASE_DIR, "logs", f"{_script_name}.log")
PROMPTS_DIR: str = os.path.join(BASE_DIR, "prompts")
DATA_IMAGES_DIR: str = os.path.join(BASE_DIR, "data", "images")
RULES_DIR: str = os.path.join(BASE_DIR, "data", "rules")
INSPECTIONS_DIR: str = os.path.join(BASE_DIR, "data", "inspections")
REPORTS_DIR:     str = os.path.join(BASE_DIR, "data", "reports")

# --- Logging ---

LOG_LEVEL: str = "INFO"   # DEBUG | INFO | WARNING | ERROR

# --- Defaults de inspeção ---

DEFAULT_ZONE_ID: str = "Z_S1"
DEFAULT_STRATEGY: str = "B"   # A=zero-shot | B=chain-of-thought | C=few-shot

# --- RAG Memory ---

EMBED_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

VECTORSTORE_DIR: str = os.path.join(BASE_DIR, "vectorstore")

RAG_COLLECTION_HYBRID: str = "inspections_hybrid"
RAG_COLLECTION_FULL: str = "inspections_full_record"

RAG_DEFAULT_K: int = 3

# --- Report Generator ---

REPORT_MAX_TOKENS: int = 4096
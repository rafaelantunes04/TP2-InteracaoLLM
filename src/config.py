"""
config.py — Configuração central do Retail Vision Intelligence System
LIACD TP2

Todos os parâmetros configuráveis do sistema estão aqui.
As variáveis de ambiente (API keys, etc.) são lidas do ficheiro .env.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

# Carrega o .env uma única vez aqui — os outros módulos não precisam de o fazer
load_dotenv()

# ---------------------------------------------------------------------------
# API Gemini
# ---------------------------------------------------------------------------

GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
MODEL_NAME: str = "gemini-3.1-flash-lite"

# Limites do plano gratuito do Google AI Studio
MAX_REQUESTS_PER_MINUTE: int = 15
MAX_REQUESTS_PER_DAY: int = 500

# Backoff exponencial em caso de erro 429
BACKOFF_INITIAL_DELAY: float = 4.0    # segundos
BACKOFF_MAX_DELAY: float = 120.0      # máximo de 2 minutos
BACKOFF_MAX_RETRIES: int = 5

# ---------------------------------------------------------------------------
# Cache local
# ---------------------------------------------------------------------------

CACHE_DIR: Path = Path("inspection_cache")
QUOTA_FILE: Path = CACHE_DIR / "_quota.json"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FILE: str = "logs/shelf_inspector.log"
LOG_LEVEL: str = "INFO"   # DEBUG | INFO | WARNING | ERROR

# ---------------------------------------------------------------------------
# Defaults de inspeção
# ---------------------------------------------------------------------------

DEFAULT_ZONE_ID: str = "Z_S1"
DEFAULT_STRATEGY: str = "B"   # A=zero-shot | B=chain-of-thought | C=few-shot
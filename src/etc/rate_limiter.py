"""
rate_limiter.py — Controlo de taxa de pedidos e backoff exponencial
Retail Vision Intelligence System — LIACD TP2

Responsabilidades:
  - Impedir que o número de pedidos por minuto exceda MAX_REQUESTS_PER_MINUTE
  - Retentar chamadas à API com backoff exponencial em caso de erro 429

Uso típico (pelo shelf_inspector):
    from rate_limiter import chamar_com_backoff
    texto = chamar_com_backoff(cliente, conteudo)
"""

from __future__ import annotations

import logging
import time

import config
from google.api_core.exceptions import ResourceExhausted, TooManyRequests
from google import genai

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Janela deslizante de timestamps (pedidos do último minuto)
# ---------------------------------------------------------------------------

_timestamps_pedidos: list[float] = []


def aguardar_limite_taxa() -> None:
    """Bloqueia se o número de pedidos no último minuto atingiu o limite."""
    agora = time.monotonic()
    _timestamps_pedidos[:] = [t for t in _timestamps_pedidos if agora - t < 60.0]

    if len(_timestamps_pedidos) >= config.MAX_REQUESTS_PER_MINUTE:
        espera = 60.0 - (agora - _timestamps_pedidos[0]) + 0.5
        logger.warning(f"Rate limit atingido — a aguardar {espera:.1f}s …")
        time.sleep(max(espera, 0))

    _timestamps_pedidos.append(time.monotonic())


def chamar_com_backoff(
    cliente: genai.Client,
    conteudo: list,
) -> str:
    """
    Envia o pedido à API Gemini com backoff exponencial em caso de erro 429.

    Parâmetros
    ----------
    cliente  : instância autenticada de genai.Client
    conteudo : lista de parts (imagem + prompt) a enviar

    Devolve
    -------
    O texto bruto da resposta da API.

    Raises
    ------
    TooManyRequests | ResourceExhausted
        Se o limite de tentativas for atingido sem sucesso.
    """
    atraso = config.BACKOFF_INITIAL_DELAY

    for tentativa in range(1, config.BACKOFF_MAX_RETRIES + 1):
        try:
            aguardar_limite_taxa()
            resposta = cliente.models.generate_content(
                model=config.MODEL_NAME,
                contents=conteudo,
            )
            return resposta.text

        except (TooManyRequests, ResourceExhausted):
            if tentativa == config.BACKOFF_MAX_RETRIES:
                raise
            logger.warning(
                f"Erro 429 — tentativa {tentativa}/{config.BACKOFF_MAX_RETRIES}. "
                f"Backoff: {atraso:.0f}s …"
            )
            time.sleep(atraso)
            atraso = min(atraso * 2, config.BACKOFF_MAX_DELAY)

        except Exception:
            raise
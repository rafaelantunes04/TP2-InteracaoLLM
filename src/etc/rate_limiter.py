from __future__ import annotations

import logging
import time

import config
from google.genai import errors
from google import genai

logger = logging.getLogger(__name__)

# Janela deslizante com os timestamps dos pedidos feitos no ultimo minuto
_timestamps_pedidos: list[float] = []


def aguardar_limite_taxa() -> None:
    """Bloqueia a execução se o número de pedidos no último minuto já atingiu o limite."""
    agora = time.monotonic()
    _timestamps_pedidos[:] = [t for t in _timestamps_pedidos if agora - t < 60.0]

    if len(_timestamps_pedidos) >= config.MAX_REQUESTS_PER_MINUTE:
        espera = 60.0 - (agora - _timestamps_pedidos[0]) + 0.5
        logger.warning(f"Rate limit atingido — a aguardar {espera:.1f}s …")
        time.sleep(max(espera, 0))

    _timestamps_pedidos.append(time.monotonic())


def chamar_com_backoff(cliente: genai.Client, conteudo: list) -> str:
    """
    Envia o pedido à API Gemini, repetindo com backoff exponencial sempre
     que a API responde com erro 429 (demasiados pedidos).

    Devolve o texto bruto da resposta. Se as tentativas se esgotarem sem
     sucesso, a excepção da API (TooManyRequests ou ResourceExhausted) é
     propagada.
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

        except errors.ClientError as e:
            if e.code != 429:
                raise

            if tentativa == config.BACKOFF_MAX_RETRIES:
                raise

            logger.warning(
                f"Erro 429 — tentativa {tentativa}/{config.BACKOFF_MAX_RETRIES}. "
                f"Backoff: {atraso:.0f}s …"
            )
            time.sleep(atraso)
            atraso = min(atraso * 2, config.BACKOFF_MAX_DELAY)
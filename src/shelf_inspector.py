"""
shelf_inspector.py — Componente 1: Inspetor de Prateleiras
Retail Vision Intelligence System — LIACD TP2

Analisa imagens de prateleiras usando Google Gemini 3.1 Flash Lite e devolve
um JSON estruturado conforme o schema da Secção 4.2.

python src/shelf_inspector.py --random
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal

import config
from etc.cache_manager import CacheManager
from etc.rate_limiter import chamar_com_backoff
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# Resolução de Caminhos
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_log_path = BASE_DIR / config.LOG_FILE
_log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=_log_path,
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

EstrategiaPrompting = Literal["A", "B", "C"]

# ---------------------------------------------------------------------------
# Instância partilhada de CacheManager
# ---------------------------------------------------------------------------

_cache = CacheManager(
    cache_dir=config.CACHE_DIR,
    quota_file=config.QUOTA_FILE,
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def _carregar_prompt(estrategia: EstrategiaPrompting, caminho_imagem: str, id_zona: str) -> str:
    """Lê o template da estratégia indicada e substitui as variáveis."""
    schema = (BASE_DIR / "prompts" / "schema_reminder.txt").read_text(encoding="utf-8").strip()
    template = (BASE_DIR / "prompts" / f"strategy_{estrategia.lower()}.txt").read_text(encoding="utf-8")

    return (
        template
        .replace("{zone_id}", id_zona)
        .replace("{image_path}", caminho_imagem)
        .replace("{schema}", schema)
    )


@lru_cache(maxsize=1)
def _get_client() -> genai.Client:
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY não definida. Verifica o ficheiro .env.")
    return genai.Client(api_key=config.GEMINI_API_KEY)


# ---------------------------------------------------------------------------
# Parser de JSON
# ---------------------------------------------------------------------------

def _extrair_json(texto: str) -> dict:
    """Extrai o JSON da resposta removendo blocos markdown se existirem."""
    texto = texto.strip()

    if "```" not in texto:
        return json.loads(texto)

    for parte in texto.split("```"):
        parte = parte.strip().removeprefix("json").strip()
        try:
            return json.loads(parte)
        except json.JSONDecodeError:
            continue

    raise json.JSONDecodeError("Nenhum bloco JSON válido encontrado", texto, 0)


# ---------------------------------------------------------------------------
# Construção do resultado de erro de parse
# ---------------------------------------------------------------------------

def _resultado_erro_parse(erro: Exception, texto_bruto: str, caminho_imagem: str, id_zona: str) -> dict:
    """Constrói um resultado estruturado quando o parse da resposta falha."""
    momento = datetime.now(timezone.utc)
    return {
        "inspection_id": f"INS_{momento.strftime('%Y%m%d_%H%M%S')}_ERR",
        "timestamp": momento.isoformat(),
        "image_path": str(caminho_imagem),
        "zone_id": id_zona,
        "overall_status": "critical",
        "issues": [{
            "issue_id": "ISS_PARSE_ERR",
            "type": "other",
            "location": "N/A",
            "severity": "high",
            "description": f"Erro de parse: {erro}",
            "confidence": 0.0,
            "affected_area_pct": 0.0,
        }],
        "shelf_fill_rate": 0.0,
        "products_detected": [],
        "model_reasoning": f"ERRO DE PARSE — Resposta bruta:\n{texto_bruto[:1000]}",
    }


# ---------------------------------------------------------------------------
# Função principal de inspeção
# ---------------------------------------------------------------------------

def inspecionar_prateleira(
    caminho_imagem: str,
    id_zona: str = config.DEFAULT_ZONE_ID,
    estrategia: EstrategiaPrompting = config.DEFAULT_STRATEGY,
) -> dict:
    if not Path(caminho_imagem).exists():
        raise FileNotFoundError(f"Imagem não encontrada: {caminho_imagem}")

    if estrategia not in ("A", "B", "C"):
        raise ValueError(f"Estratégia inválida: {estrategia!r}. Usa 'A', 'B' ou 'C'.")

    chave = _cache.chave(caminho_imagem, estrategia)
    resultado_em_cache = _cache.get(chave)
    if resultado_em_cache is not None:
        return resultado_em_cache

    if _cache.quota_esgotada():
        raise RuntimeError(
            f"⚠️ Quota diária esgotada ({config.MAX_REQUESTS_PER_DAY} req/dia). "
            "Novas inspeções só serão possíveis amanhã (UTC)."
        )

    logger.info(
        f"A processar: {Path(caminho_imagem).name} "
        f"| zona: {id_zona} | estratégia: {estrategia}"
    )

    conteudo = [
        types.Part.from_bytes(
            data=Path(caminho_imagem).read_bytes(),
            mime_type="image/jpeg",
        ),
        _carregar_prompt(estrategia, caminho_imagem, id_zona),
    ]

    texto_bruto = chamar_com_backoff(_get_client(), conteudo)
    _cache.incrementar_quota()
    logger.info("Resposta recebida da API Gemini.")

    resultado = _processar_resposta(texto_bruto, caminho_imagem, id_zona)
    _cache.set(chave, resultado)
    return resultado


def _processar_resposta(texto_bruto: str, caminho_imagem: str, id_zona: str) -> dict:
    """Faz parse do JSON devolvido pela API e preenche os campos obrigatórios."""
    try:
        dados = _extrair_json(texto_bruto)
    except (ValueError, json.JSONDecodeError) as erro:
        logger.error(f"Falha ao processar JSON: {erro}")
        return _resultado_erro_parse(erro, texto_bruto, caminho_imagem, id_zona)

    momento = datetime.now(timezone.utc)
    dados.setdefault("inspection_id", f"INS_{momento.strftime('%Y%m%d_%H%M%S')}_001")
    dados.setdefault("timestamp", momento.isoformat())
    dados.setdefault("overall_status", "warning")
    dados.setdefault("issues", [])
    dados.setdefault("shelf_fill_rate", 0.0)
    dados.setdefault("products_detected", [])
    dados.setdefault("model_reasoning", "")
    dados["image_path"] = str(caminho_imagem)
    dados["zone_id"] = id_zona
    return dados


def obter_estado_quota() -> dict:
    return _cache.estado()


def limpar_cache() -> int:
    return _cache.limpar()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspetor de prateleiras — Retail Vision Intelligence System",
    )
    parser.add_argument(
        "imagem",
        nargs="?",
        help="Caminho para a imagem da prateleira (JPEG)",
    )
    parser.add_argument(
        "--zona",
        default=config.DEFAULT_ZONE_ID,
        help="ID da zona (ex: Z_S3). Default: %(default)s",
    )
    parser.add_argument(
        "--estrategia",
        choices=["A", "B", "C"],
        default=config.DEFAULT_STRATEGY,
        help="Estratégia de prompting. Default: %(default)s",
    )
    parser.add_argument(
        "--quota",
        action="store_true",
        help="Mostra o estado actual da quota diária e sai.",
    )
    parser.add_argument(
        "--limpar-cache",
        action="store_true",
        help="Remove todas as entradas de cache e sai.",
    )
    parser.add_argument(
        "--random",
        metavar="DIRETORIO",
        nargs="?",
        const=str(BASE_DIR / "data" / "images"),
        help="Escolhe uma imagem aleatória do diretório indicado (padrão: data/images).",
    )
    args = parser.parse_args()

    if args.quota:
        print(json.dumps(obter_estado_quota(), ensure_ascii=False, indent=2))
        sys.exit(0)

    if args.limpar_cache:
        n = limpar_cache()
        print(f"Cache limpa: {n} entrada(s) removida(s).")
        sys.exit(0)

    if args.random:
        dir_path = Path(args.random)
        if not dir_path.exists():
            print(f"Erro: O diretório '{dir_path}' não existe.", file=sys.stderr)
            sys.exit(1)

        imagens_encontradas = [
            p for p in dir_path.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg")
        ]

        if not imagens_encontradas:
            print(f"Erro: Nenhuma imagem JPEG encontrada em '{dir_path}'.", file=sys.stderr)
            sys.exit(1)

        args.imagem = str(random.choice(imagens_encontradas))
        print(f"🎲 Imagem sorteada: {args.imagem}\n")

    if not args.imagem:
        parser.print_help()
        sys.exit(1)

    try:
        resultado = inspecionar_prateleira(
            args.imagem,
            id_zona=args.zona,
            estrategia=args.estrategia,
        )
        print(json.dumps(resultado, ensure_ascii=False, indent=2))

    except RuntimeError as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Erro: {e}", file=sys.stderr)
        sys.exit(1)
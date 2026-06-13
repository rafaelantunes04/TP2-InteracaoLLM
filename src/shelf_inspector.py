import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from typing import Literal

import config
from etc.cache_manager import CacheManager
from etc.rate_limiter import chamar_com_backoff
from google import genai
from google.genai import types

# python src/shelf_inspector.py --random

# --- Logging ---

os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=config.LOG_FILE,
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

EstrategiaPrompting = Literal["A", "B", "C"]

# Instancia partilhada de cache + quota
_cache = CacheManager()

_MAPA_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}


# --- Prompts ---

def _carregar_prompt(estrategia: EstrategiaPrompting, caminho_imagem: str, id_zona: str) -> str:
    """Lê o template da estratégia indicada e substitui o schema, a imagem e a zona."""
    schema_path = os.path.join(config.PROMPTS_DIR, "schema_reminder.txt")
    strategy_path = os.path.join(config.PROMPTS_DIR, f"strategy_{estrategia.lower()}.txt")

    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Ficheiro de schema não encontrado em: {schema_path}")
    if not os.path.exists(strategy_path):
        raise FileNotFoundError(f"Ficheiro de prompt não encontrado em: {strategy_path}")

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = f.read().strip()
    with open(strategy_path, "r", encoding="utf-8") as f:
        template = f.read()

    return (
        template
        .replace("{zone_id}", id_zona)
        .replace("{image_path}", caminho_imagem)
        .replace("{schema}", schema)
    )


def _get_client() -> genai.Client:
    """Cria o cliente da API Gemini a partir da chave definida no .env."""
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY não definida. Verifica o ficheiro .env.")
    return genai.Client(api_key=config.GEMINI_API_KEY)


# --- Parser de JSON ---

def _extrair_json(texto: str) -> dict:
    """Extrai o JSON da resposta, removendo blocos markdown se existirem."""
    texto = texto.strip()

    if "```" not in texto:
        return json.loads(texto)

    for parte in texto.split("```"):
        parte = parte.strip()
        if parte.startswith("json"):
            parte = parte[4:].strip()
        try:
            return json.loads(parte)
        except json.JSONDecodeError:
            continue

    raise json.JSONDecodeError("Nenhum bloco JSON válido encontrado", texto, 0)


# --- Persistência ---

def _guardar_inspecao(resultado: dict) -> str:
    """Persiste o resultado da inspeção em config.INSPECTIONS_DIR como JSON. Devolve o caminho do ficheiro."""
    os.makedirs(config.INSPECTIONS_DIR, exist_ok=True)
    inspection_id = resultado.get("inspection_id", "INS_UNKNOWN")
    caminho = os.path.join(config.INSPECTIONS_DIR, f"{inspection_id}.json")
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    logger.info(f"Inspeção guardada em: {caminho}")
    return caminho


def _resultado_erro_parse(erro: Exception, texto_bruto: str, caminho_imagem: str, id_zona: str) -> dict:
    """Constrói um resultado estruturado quando o parse da resposta da API falha."""
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


# --- Inspecao ---

def inspecionar_prateleira(
    caminho_imagem: str,
    id_zona: str = config.DEFAULT_ZONE_ID,
    estrategia: EstrategiaPrompting = config.DEFAULT_STRATEGY,
) -> dict:
    """
    Analisa a imagem da prateleira indicada com a estratégia de prompting
     escolhida e devolve o JSON estruturado da inspeção (Secção 4.2).

    Antes de chamar a API verifica se já existe resultado em cache para a
     mesma imagem + estratégia, e só chama a API se a quota diária ainda
     não estiver esgotada.
    """
    if not os.path.exists(caminho_imagem):
        raise FileNotFoundError(f"Imagem não encontrada: {caminho_imagem}")

    if estrategia not in ("A", "B", "C"):
        raise ValueError(f"Estratégia inválida: {estrategia!r}. Usa 'A', 'B' ou 'C'.")

    # Tenta devolver resultado em cache antes de qualquer chamada à API
    chave = _cache.chave(caminho_imagem, estrategia)
    resultado_em_cache = _cache.get(chave)
    if resultado_em_cache is not None:
        _guardar_inspecao(resultado_em_cache)
        return resultado_em_cache

    if _cache.quota_esgotada():
        raise RuntimeError(
            f"Quota diária esgotada ({config.MAX_REQUESTS_PER_DAY} req/dia). "
            "Novas inspeções só serão possíveis amanhã (UTC)."
        )

    cliente = _get_client()
    logger.info(
        f"A processar: {os.path.basename(caminho_imagem)} "
        f"| zona: {id_zona} | estratégia: {estrategia}"
    )

    _, extensao = os.path.splitext(caminho_imagem.lower())
    tipo_mime = _MAPA_MIME.get(extensao, "image/jpeg")

    with open(caminho_imagem, "rb") as f:
        dados_imagem = f.read()

    conteudo = [
        types.Part.from_bytes(
            data=dados_imagem,
            mime_type=tipo_mime,
        ),
        _carregar_prompt(estrategia, caminho_imagem, id_zona),
    ]

    texto_bruto = chamar_com_backoff(cliente, conteudo)
    _cache.incrementar_quota()
    logger.info("Resposta recebida da API Gemini.")

    resultado = _processar_resposta(texto_bruto, caminho_imagem, id_zona)
    _cache.set(chave, resultado)
    _guardar_inspecao(resultado)
    return resultado


def _processar_resposta(texto_bruto: str, caminho_imagem: str, id_zona: str) -> dict:
    """Faz parse do JSON devolvido pela API e preenche os campos obrigatórios em falta."""
    try:
        dados = _extrair_json(texto_bruto)
    except (ValueError, json.JSONDecodeError) as erro:
        logger.error(f"Falha ao processar JSON: {erro}")
        return _resultado_erro_parse(erro, texto_bruto, caminho_imagem, id_zona)

    momento = datetime.now(timezone.utc)
    
    dados["inspection_id"] = f"INS_{momento.strftime('%Y%m%d_%H%M%S')}_001"
    dados["timestamp"] = momento.isoformat()

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


# --- CLI ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "imagem",
        nargs="?",
        help="Caminho para a imagem da prateleira (JPEG, PNG, WEBP, …)",
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
        const=config.DATA_IMAGES_DIR,
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
        dir_path = args.random
        if not os.path.exists(dir_path):
            print(f"Erro: O diretório '{dir_path}' não existe.", file=sys.stderr)
            sys.exit(1)

        imagens_encontradas = [
            os.path.join(dir_path, nome)
            for nome in os.listdir(dir_path)
            if os.path.splitext(nome.lower())[1] in _MAPA_MIME
        ]

        if not imagens_encontradas:
            print(f"Erro: Nenhuma imagem válida encontrada em '{dir_path}'.", file=sys.stderr)
            sys.exit(1)

        args.imagem = random.choice(imagens_encontradas)
        print(f"Imagem sorteada: {args.imagem}\n")

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
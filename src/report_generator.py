import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone

import config
from etc.cache_manager import CacheManager
from etc.rate_limiter import chamar_com_backoff
from google import genai

# python src/report_generator.py --inspecao data/inspections/INS_001.json
# python src/report_generator.py --sessao today
# python src/report_generator.py --sessao today --zona Z_S1
# python src/report_generator.py --sessao today --sem-rag

# --- Logging ---

os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=config.LOG_FILE,
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

_cache = CacheManager()


# --- Cliente API ---

def _get_client() -> genai.Client:
    """Cria o cliente da API Gemini a partir da chave definida no .env."""
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY não definida. Verifica o ficheiro .env.")
    return genai.Client(api_key=config.GEMINI_API_KEY)


# --- Carregamento de dados ---

def _carregar_inspecao(caminho: str) -> dict:
    """Lê um ficheiro JSON de inspeção do disco."""
    with open(caminho, "r", encoding="utf-8") as f:
        return json.load(f)


def _inspecoes_da_sessao(sessao: str) -> list[dict]:
    """Devolve todas as inspeções de uma sessão ('today' ou 'YYYY-MM-DD')."""
    if not os.path.isdir(config.INSPECTIONS_DIR):
        return []

    prefixo = (
        datetime.now(timezone.utc).date().isoformat()
        if sessao == "today"
        else sessao
    )

    inspecoes = []
    for fname in sorted(os.listdir(config.INSPECTIONS_DIR)):
        if not fname.endswith(".json"):
            continue
        caminho = os.path.join(config.INSPECTIONS_DIR, fname)
        try:
            with open(caminho, "r", encoding="utf-8") as f:
                dados = json.load(f)
        except Exception as e:
            logger.warning(f"Ficheiro de inspeção inválido ({fname}): {e}")
            continue
        if dados.get("timestamp", "").startswith(prefixo):
            inspecoes.append(dados)

    return inspecoes


def _stats_sessao(inspecoes: list[dict]) -> dict:
    """Agrega estatísticas de alto nível de uma lista de inspeções."""
    zonas      = {i.get("zone_id", "?") for i in inspecoes}
    criticos   = sum(
        sum(1 for iss in i.get("issues", []) if iss.get("severity") == "high")
        for i in inspecoes
    )
    warnings   = sum(
        sum(1 for iss in i.get("issues", []) if iss.get("severity") == "medium")
        for i in inspecoes
    )
    fill_rates = [i.get("shelf_fill_rate", 0.0) for i in inspecoes if "shelf_fill_rate" in i]
    fill_medio = sum(fill_rates) / len(fill_rates) if fill_rates else 0.0

    return {
        "zonas":     sorted(zonas),
        "criticos":  criticos,
        "warnings":  warnings,
        "fill_medio": fill_medio,
    }


# --- Formatação do contexto para o prompt ---

def _formatar_inspecao(insp: dict) -> str:
    """Converte um dict de inspeção num bloco de texto legível para o prompt."""
    issues = insp.get("issues", [])
    notifs = insp.get("rule_engine_notifications", [])
    audit  = insp.get("rule_engine_audit", [])

    issues_txt = "\n".join(
        f"  • [{iss.get('severity', '?').upper()}] {iss.get('type', '?')} @ {iss.get('location', '?')}: "
        f"{iss.get('description', '')[:150]} "
        f"(área={iss.get('affected_area_pct', 0):.0f}%, confiança={iss.get('confidence', 0):.0%})"
        for iss in issues
    ) or "  (nenhum issue detectado)"

    regras_txt = "\n".join(
        f"  • {n.get('rule_id')} [{n.get('alert_level', '?').upper()}]: {n.get('message', '')[:200]}"
        for n in notifs
    ) or "  (nenhuma regra disparada)"

    auditoria_txt = "\n".join(
        f"  • {e.get('rule_id')}: {'DISPAROU' if e.get('fired') else 'não disparou'}"
        + (f" → {e.get('notification', '')[:120]}" if e.get("fired") else "")
        for e in audit
    ) or "  (auditoria vazia)"

    return (
        f"Inspection ID : {insp.get('inspection_id', 'N/A')}\n"
        f"Timestamp     : {insp.get('timestamp', 'N/A')}\n"
        f"Zona          : {insp.get('zone_id', 'N/A')}\n"
        f"Overall Status: {insp.get('overall_status', 'N/A')}\n"
        f"Fill Rate     : {insp.get('shelf_fill_rate', 0):.1%}\n"
        f"Issues ({len(issues)}):\n{issues_txt}\n"
        f"Regras disparadas:\n{regras_txt}\n"
        f"Auditoria completa:\n{auditoria_txt}"
    )


def _contexto_rag(inspecoes: list[dict], rag_memory) -> str:
    """Consulta o RAG por zona e devolve o contexto histórico formatado."""
    if rag_memory is None:
        return "RAG não disponível nesta execução."

    zonas_com_issues = sorted({
        i.get("zone_id", "?")
        for i in inspecoes
        if i.get("issues")
    })

    if not zonas_com_issues:
        return "Nenhuma zona com issues — consulta histórica não necessária."

    blocos = []
    for zona in zonas_com_issues:
        try:
            resultado    = rag_memory.consultar(f"problemas históricos recorrentes na zona {zona}", k=config.RAG_DEFAULT_K)
            answer       = resultado.get("answer", "(sem resposta)")
            recuperados  = resultado.get("retrieved_inspections", [])
            refs_str     = "; ".join(
                f"{r.get('inspection_id', '?')} ({r.get('date', r.get('timestamp', '?'))[:10]})"
                for r in recuperados
            ) or "N/A"
        except Exception as e:
            logger.warning(f"RAG falhou para zona {zona}: {e}")
            answer, refs_str = "(erro ao consultar RAG)", "N/A"

        blocos.append(f"### Zona {zona}\n{answer}\nReferências: {refs_str}")

    return "\n\n".join(blocos)


# --- Prompt ---

def _carregar_prompt(inspecoes: list[dict], stats: dict, contexto_historico: str) -> list:
    """Lê o template de prompts/report_generator.txt e substitui os placeholders."""
    blocos_inspecoes = "\n\n---\n\n".join(_formatar_inspecao(i) for i in inspecoes)
    sessao_data      = inspecoes[0].get("timestamp", "")[:10]

    prompt_path = os.path.join(config.PROMPTS_DIR, "report_generator.txt")
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            template = f.read()
        texto = (
            template
            .replace("{sessao_data}",        sessao_data)
            .replace("{n_zonas}",            str(len(stats["zonas"])))
            .replace("{zonas}",              ", ".join(stats["zonas"]))
            .replace("{n_criticos}",         str(stats["criticos"]))
            .replace("{n_warnings}",         str(stats["warnings"]))
            .replace("{fill_medio}",         f"{stats['fill_medio']:.1%}")
            .replace("{blocos_inspecoes}",   blocos_inspecoes)
            .replace("{contexto_historico}", contexto_historico)
        )
        return [texto]

    # Fallback se o ficheiro de prompt não existir
    return [f"""És um sistema especializado em análise de prateleiras de retalho.
Gera um Inspection Report em Markdown com base nos dados abaixo.

════════════════════════════════════════
DADOS DA SESSÃO
Data            : {sessao_data}
Zonas           : {", ".join(stats["zonas"])} ({len(stats["zonas"])} zona(s))
Issues críticos : {stats["criticos"]}
Warnings        : {stats["warnings"]}
Fill rate médio : {stats['fill_medio']:.1%}

════════════════════════════════════════
INSPEÇÕES DETALHADAS

{blocos_inspecoes}

════════════════════════════════════════
CONTEXTO HISTÓRICO (RAG)

{contexto_historico}

════════════════════════════════════════
Gera o relatório com EXACTAMENTE estas secções:

# Inspection Report — {sessao_data}

## 1. Sumário Executivo
Máximo 150 palavras. Estado geral: zonas, issues críticos, warnings, fill rate médio.
Tom directo e accionável. Nível de urgência global (normal / atenção / crítico).

## 2. Problemas por Zona
Para cada zona com issues (### Zona X): lista de problemas com severidade, fill rate
actual vs histórico do RAG, e padrões recorrentes identificados.

## 3. Regras Disparadas
Para cada regra que disparou: ID, nível de alerta, dados que activaram a condição e
acção gerada. Se nenhuma disparou, indica explicitamente.

## 4. Contexto Histórico Relevante
Padrões passados do RAG relevantes para esta sessão. Cada referência com inspection_id
e data no formato [INS_XXX — YYYY-MM-DD]. Se sem contexto útil, indica explicitamente.

## 5. Recomendações
No máximo 5 acções concretas ordenadas por urgência (1 = mais urgente), no formato:
**Acção X:** [quem] deve [o quê] em [onde] [quando/prazo].

Responde APENAS com o relatório em Markdown. Sem comentários nem texto adicional.
"""]


# --- Geração ---

def gerar_relatorio(inspecoes: list[dict], rag_memory=None, guardar: bool = True) -> str:
    """Gera um Inspection Report em Markdown para a lista de inspeções fornecida."""
    if not inspecoes:
        raise ValueError("Nenhuma inspeção fornecida para gerar relatório.")

    # Deduplicar e ordenar cronologicamente
    vistas, unicas = set(), []
    for i in inspecoes:
        iid = i.get("inspection_id", "")
        if iid not in vistas:
            vistas.add(iid)
            unicas.append(i)
    unicas.sort(key=lambda x: x.get("timestamp", ""))

    # Cache
    ids_concat  = "_".join(i.get("inspection_id", "") for i in unicas)
    cache_chave = "report_" + hashlib.md5(ids_concat.encode()).hexdigest()[:12]

    hit = _cache.get(cache_chave)
    if hit and isinstance(hit.get("report"), str):
        logger.info(f"Cache hit — relatório: {cache_chave}")
        print("(resultado do cache)", file=sys.stderr)
        return hit["report"]

    stats              = _stats_sessao(unicas)
    contexto_historico = _contexto_rag(unicas, rag_memory)
    conteudo           = _carregar_prompt(unicas, stats, contexto_historico)

    logger.info(f"A gerar relatório para {len(unicas)} inspeção(ões): {ids_concat[:80]}")
    relatorio = chamar_com_backoff(_get_client(), conteudo)

    if relatorio and not relatorio.strip().startswith("#"):
        relatorio = f"# Inspection Report — {unicas[0].get('timestamp', '')[:10]}\n\n" + relatorio

    _cache.set(cache_chave, {"report": relatorio})

    if guardar:
        _guardar_relatorio(relatorio, unicas)

    return relatorio


# --- Persistência ---

def _guardar_relatorio(relatorio: str, inspecoes: list[dict]) -> str:
    """Persiste o relatório em config.REPORTS_DIR e devolve o caminho do ficheiro."""
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    data_sessao   = inspecoes[0].get("timestamp", datetime.now(timezone.utc).isoformat())[:10]
    zonas_str     = "_".join(sorted({i.get("zone_id", "") for i in inspecoes}))
    caminho       = os.path.join(config.REPORTS_DIR, f"report_{data_sessao}_{zonas_str}.md")

    with open(caminho, "w", encoding="utf-8") as f:
        f.write(relatorio)

    logger.info(f"Relatório guardado em: {caminho}")
    print(f"Relatório guardado em: {caminho}", file=sys.stderr)
    return caminho


# --- CLI ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--inspecao",
        metavar="CAMINHO",
        help="Caminho para um ficheiro JSON de inspeção.",
    )
    group.add_argument(
        "--sessao",
        metavar="SESSAO",
        help="Gera relatório para todas as inspeções da sessão ('today' ou YYYY-MM-DD).",
    )
    parser.add_argument(
        "--zona",
        metavar="ZONE_ID",
        help="Filtrar por zona (combina com --sessao).",
    )
    parser.add_argument(
        "--sem-rag",
        action="store_true",
        help="Desactiva a consulta ao RAG.",
    )
    parser.add_argument(
        "--output",
        metavar="FICHEIRO",
        help="Caminho de output (por omissão guarda em data/reports/).",
    )

    args = parser.parse_args()

    if args.inspecao:
        if not os.path.isfile(args.inspecao):
            print(f"Erro: ficheiro não encontrado: {args.inspecao}", file=sys.stderr)
            sys.exit(1)
        lista_inspecoes = [_carregar_inspecao(args.inspecao)]
    else:
        lista_inspecoes = _inspecoes_da_sessao(args.sessao)
        if args.zona:
            lista_inspecoes = [i for i in lista_inspecoes if i.get("zone_id") == args.zona]
        if not lista_inspecoes:
            print(f"Nenhuma inspeção encontrada para sessão='{args.sessao}'" + (f" zona='{args.zona}'" if args.zona else ""), file=sys.stderr)
            sys.exit(1)

    rag = None
    if not args.sem_rag:
        try:
            from rag_memory import RAGMemory
            rag = RAGMemory()
        except Exception as e:
            logger.warning(f"RAGMemory não disponível: {e}")
            print(f"Aviso: RAG não disponível ({e}). A continuar sem contexto histórico.", file=sys.stderr)

    relatorio = gerar_relatorio(lista_inspecoes, rag_memory=rag, guardar=(args.output is None))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(relatorio)
        print(f"Relatório guardado em: {args.output}", file=sys.stderr)

    print(relatorio)
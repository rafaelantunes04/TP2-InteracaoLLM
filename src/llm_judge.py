import json
import logging
import os
import time

import config
from etc.rate_limiter import chamar_com_backoff
from google import genai
from google.genai import types

# python llm_judge.py --amostras judge_samples.json --output judge_evaluation.json

# --- Logging ---

logging.basicConfig(
    filename=config.LOG_FILE,
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cliente Gemini
# ---------------------------------------------------------------------------

def _get_client() -> genai.Client:
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY não definida. Verifica o ficheiro .env.")
    return genai.Client(api_key=config.GEMINI_API_KEY)


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def _extrair_json(texto: str) -> dict:
    """Remove blocos markdown e faz parse do JSON devolvido pelo juiz."""
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


def _parsear_resposta_juiz(texto: str) -> dict:
    """
    Extrai {"score": float, "justification": str} da resposta do modelo.
    Fallback para score=0.5 se o parse falhar.
    """
    try:
        dados = _extrair_json(texto)
        score = max(0.0, min(1.0, float(dados.get("score", 0.5))))
        return {"score": score, "justification": dados.get("justification", "")}
    except (json.JSONDecodeError, ValueError):
        # Heurística: tentar encontrar um número na linha com "score"
        for linha in texto.splitlines():
            if "score" in linha.lower():
                try:
                    score = float("".join(c for c in linha if c.isdigit() or c == "."))
                    score = score / 10 if score > 1 else score
                    return {"score": max(0.0, min(1.0, score)), "justification": texto}
                except ValueError:
                    pass
        return {"score": 0.5, "justification": texto}


def _chamar_juiz(prompt: str) -> dict:
    """Envia o prompt ao Gemini Flash (temperature=0) e devolve o resultado parseado."""
    cliente = _get_client()
    conteudo = [types.Part.from_text(text=prompt)]

    try:
        texto = chamar_com_backoff(cliente, conteudo, temperature=0)
        logger.info("Resposta do juiz recebida.")
        return _parsear_resposta_juiz(texto)
    except Exception as erro:
        logger.error(f"Falha na chamada ao juiz: {erro}")
        return {"score": 0.5, "justification": f"Erro: {erro}"}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_PROMPT_ALUCINACAO = """\
És um avaliador especializado em visão por computador para ambientes de retalho.

Tarefa: avalia se as descrições de problemas reportadas pelo sistema são verificáveis
com base no raciocínio explícito do modelo. Uma afirmação é uma alucinação se não for
suportada por evidência visual descrita no raciocínio.

Raciocínio do modelo:
{raciocinio}

Descrições de issues reportadas:
{descricoes}

Responde APENAS com JSON válido, sem texto adicional:
{{
  "score": <float 0.0–1.0, onde 1.0 = sem alucinações, 0.0 = tudo alucinado>,
  "justification": "<explicação em português, máx. 3 frases>"
}}"""

_PROMPT_FIDELIDADE = """\
És um avaliador de sistemas RAG (Retrieval-Augmented Generation).

Tarefa: verifica se todas as afirmações factuais na resposta estão suportadas pelo
contexto recuperado. Afirmações não suportadas pelo contexto são infidelidades.

Contexto recuperado:
{contexto}

Resposta do sistema:
{resposta}

Responde APENAS com JSON válido, sem texto adicional:
{{
  "score": <float 0.0–1.0, onde 1.0 = totalmente fiel ao contexto>,
  "justification": "<explicação em português, máx. 3 frases>"
}}"""

_PROMPT_RELEVANCIA = """\
És um avaliador de qualidade de respostas num sistema de monitorização de prateleiras.

Tarefa: avalia se a resposta aborda efectivamente a pergunta do utilizador.

Critérios:
1. A resposta responde directamente à pergunta?
2. É específica (menciona zonas, datas, inspection_ids quando relevante)?
3. Evita generalidades vagas?

Pergunta: {query}
Resposta: {resposta}

Responde APENAS com JSON válido, sem texto adicional:
{{
  "score": <float 0.0–1.0, onde 1.0 = completamente relevante>,
  "justification": "<explicação em português, máx. 3 frases>"
}}"""

_PROMPT_QUALIDADE_RELATORIO = """\
És um avaliador de relatórios operacionais para gestores de lojas de retalho.

Critério de avaliação: {criterio}

Relatório:
{relatorio}

Responde APENAS com JSON válido, sem texto adicional:
{{
  "score": <float 0.0–1.0>,
  "justification": "<explicação em português, máx. 4 frases>",
  "pontos_fortes": ["<ponto 1>", "<ponto 2>"],
  "pontos_fracos": ["<ponto 1>", "<ponto 2>"]
}}"""

_PROMPT_META_ANALISE = """\
És um meta-avaliador. Compara a avaliação humana com a avaliação automática (LLM-as-Judge)
da mesma amostra e identifica pontos de concordância, discordância e possíveis enviesamentos.

Avaliação humana:
- Score: {score_humano}
- Notas: {notas_humanas}

Avaliação LLM-as-Judge:
- Score: {score_juiz}
- Justificação: {justificacao_juiz}

Responde APENAS com JSON válido, sem texto adicional:
{{
  "concordam": <true|false>,
  "delta": <diferença absoluta entre scores>,
  "analise": "<análise em português, máx. 5 frases>",
  "enviesamento_juiz": "<enviesamento identificado ou 'nenhum'>"
}}"""


# ---------------------------------------------------------------------------
# Funções de avaliação
# ---------------------------------------------------------------------------

def avaliar_alucinacao(raciocinio: str, descricoes: list[str]) -> float:
    """
    Avalia se as descrições de issues são suportadas pelo raciocínio do modelo.
    Devolve score [0, 1] — 1.0 significa sem alucinações.
    """
    texto_descricoes = "\n".join(f"- {d}" for d in descricoes) if descricoes else "(nenhuma)"
    prompt = _PROMPT_ALUCINACAO.format(
        raciocinio=raciocinio or "(sem raciocínio)",
        descricoes=texto_descricoes,
    )
    resultado = _chamar_juiz(prompt)
    logger.info(f"Alucinação — score={resultado['score']:.2f} | {resultado['justification'][:80]}")
    return resultado["score"]


def avaliar_fidelidade(resposta: str, contexto: str) -> float:
    """
    Avalia se a resposta RAG é fiel aos chunks recuperados.
    Devolve score [0, 1].
    """
    prompt = _PROMPT_FIDELIDADE.format(
        contexto=contexto or "(contexto vazio)",
        resposta=resposta or "(resposta vazia)",
    )
    resultado = _chamar_juiz(prompt)
    logger.info(f"Fidelidade — score={resultado['score']:.2f} | {resultado['justification'][:80]}")
    return resultado["score"]


def avaliar_relevancia(query: str, resposta: str) -> float:
    """
    Avalia se a resposta RAG responde à query do utilizador.
    Devolve score [0, 1].
    """
    prompt = _PROMPT_RELEVANCIA.format(
        query=query or "(query vazia)",
        resposta=resposta or "(resposta vazia)",
    )
    resultado = _chamar_juiz(prompt)
    logger.info(f"Relevância — score={resultado['score']:.2f} | {resultado['justification'][:80]}")
    return resultado["score"]


def avaliar_relatorio(relatorio_md: str, criterio: str = "clareza, accionabilidade e completude") -> dict:
    """
    Avalia a qualidade de um relatório de inspeção em Markdown.
    Devolve dict com score, justificação, pontos_fortes e pontos_fracos.
    """
    prompt = _PROMPT_QUALIDADE_RELATORIO.format(
        criterio=criterio,
        relatorio=relatorio_md[:4000],
    )
    try:
        texto = chamar_com_backoff(_get_client(), [types.Part.from_text(text=prompt)], temperature=0)
        dados = _extrair_json(texto)
        dados["score"] = max(0.0, min(1.0, float(dados.get("score", 0.5))))
        return dados
    except Exception as erro:
        logger.error(f"Falha ao avaliar relatório: {erro}")
        return {"score": 0.5, "justification": str(erro), "pontos_fortes": [], "pontos_fracos": []}


def _meta_analisar(score_humano: float, notas_humanas: str,
                   score_juiz: float, justificacao_juiz: str) -> dict:
    """
    Compara a avaliação humana com a do juiz e identifica enviesamentos.
    Usada internamente por avaliar_amostras().
    """
    prompt = _PROMPT_META_ANALISE.format(
        score_humano=score_humano,
        notas_humanas=notas_humanas or "(sem notas)",
        score_juiz=score_juiz,
        justificacao_juiz=justificacao_juiz or "(sem justificação)",
    )
    try:
        texto = chamar_com_backoff(_get_client(), [types.Part.from_text(text=prompt)], temperature=0)
        return _extrair_json(texto)
    except Exception as erro:
        logger.warning(f"Falha na meta-análise: {erro}")
        return {
            "concordam": abs(score_humano - score_juiz) < 0.2,
            "delta": round(abs(score_humano - score_juiz), 4),
            "analise": f"Meta-análise indisponível: {erro}",
            "enviesamento_juiz": "desconhecido",
        }


# ---------------------------------------------------------------------------
# Avaliação completa com meta-análise (Secção 9.3)
# ---------------------------------------------------------------------------

def avaliar_amostras(caminho_amostras: str, caminho_output: str = "judge_evaluation.json") -> dict:
    """
    Executa o LLM-as-Judge sobre um conjunto de amostras com anotação humana
    e produz a meta-análise de concordância juiz/humano.

    Formato esperado do ficheiro de amostras (JSON):
    [
        {
            "tipo": "alucinacao" | "fidelidade" | "relevancia" | "relatorio",
            "input": { ... campos específicos do tipo ... },
            "score_humano": 0.8,
            "notas_humanas": "..."
        },
        ...
    ]
    """
    if not os.path.exists(caminho_amostras):
        raise FileNotFoundError(f"Ficheiro de amostras não encontrado: {caminho_amostras}")

    with open(caminho_amostras, encoding="utf-8") as f:
        amostras: list[dict] = json.load(f)

    resultados = []
    deltas = []

    print(f"\n[LLM-as-Judge] {len(amostras)} amostras a avaliar...\n")

    for i, amostra in enumerate(amostras):
        tipo = amostra.get("tipo", "desconhecido")
        inp  = amostra.get("input", {})
        score_humano  = float(amostra.get("score_humano", 0.5))
        notas_humanas = amostra.get("notas_humanas", "")

        resultado_juiz: dict = {}

        if tipo == "alucinacao":
            score = avaliar_alucinacao(
                raciocinio=inp.get("raciocinio", ""),
                descricoes=inp.get("descricoes", []),
            )
            resultado_juiz = {"score": score, "justification": "(ver log)"}

        elif tipo == "fidelidade":
            score = avaliar_fidelidade(
                resposta=inp.get("resposta", ""),
                contexto=inp.get("contexto", ""),
            )
            resultado_juiz = {"score": score, "justification": "(ver log)"}

        elif tipo == "relevancia":
            score = avaliar_relevancia(
                query=inp.get("query", ""),
                resposta=inp.get("resposta", ""),
            )
            resultado_juiz = {"score": score, "justification": "(ver log)"}

        elif tipo == "relatorio":
            resultado_juiz = avaliar_relatorio(
                relatorio_md=inp.get("relatorio", ""),
                criterio=inp.get("criterio", "clareza, accionabilidade e completude"),
            )
            score = resultado_juiz.get("score", 0.5)

        else:
            logger.warning(f"Tipo desconhecido na amostra {i}: {tipo!r}")
            continue

        meta = _meta_analisar(
            score_humano=score_humano,
            notas_humanas=notas_humanas,
            score_juiz=resultado_juiz.get("score", score),
            justificacao_juiz=resultado_juiz.get("justification", ""),
        )

        delta = abs(score_humano - resultado_juiz.get("score", score))
        deltas.append(delta)

        resultados.append({
            "indice":         i,
            "tipo":           tipo,
            "score_humano":   score_humano,
            "score_juiz":     round(resultado_juiz.get("score", score), 4),
            "delta":          round(delta, 4),
            "justificacao":   resultado_juiz.get("justification", ""),
            "meta_analise":   meta,
        })

        print(f"  [{i+1}/{len(amostras)}] {tipo} — humano={score_humano:.2f} | juiz={resultado_juiz.get('score', score):.2f}")
        time.sleep(0.5)   # respeitar rate limit

    taxa_concordancia = sum(1 for r in resultados if r["meta_analise"].get("concordam")) / max(len(resultados), 1)
    delta_medio       = sum(deltas) / max(len(deltas), 1)

    relatorio = {
        "resumo": {
            "total_amostras":    len(resultados),
            "taxa_concordancia": round(taxa_concordancia, 4),
            "delta_medio":       round(delta_medio, 4),
            "delta_maximo":      round(max(deltas, default=0), 4),
        },
        "amostras": resultados,
    }

    with open(caminho_output, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)

    print(f"\nRelatório LLM-as-Judge guardado em: {caminho_output}")
    print(f"Taxa de concordância: {taxa_concordancia * 100:.1f}% | Delta médio: {delta_medio:.3f}")
    logger.info(f"LLM-as-Judge concluído. Relatório: {caminho_output}")

    return relatorio


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="LLM-as-Judge — avaliação automática de qualidade (Secção 9.3)"
    )
    parser.add_argument(
        "--amostras",
        required=True,
        help="Ficheiro JSON com amostras anotadas humanamente.",
    )
    parser.add_argument(
        "--output",
        default="judge_evaluation.json",
        help="Ficheiro de saída com os resultados do juiz. Default: %(default)s",
    )
    args = parser.parse_args()

    try:
        resultado = avaliar_amostras(args.amostras, args.output)
        print(json.dumps(resultado["resumo"], ensure_ascii=False, indent=2))
    except FileNotFoundError as e:
        print(f"Erro: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Erro inesperado: {e}", file=sys.stderr)
        sys.exit(1)

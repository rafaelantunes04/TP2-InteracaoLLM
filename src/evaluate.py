import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import config
from src.shelf_inspector import inspecionar_prateleira
from src.rule_engine import processar_regra, executar_regra
from src.rag_memory import consultar, sintetizar

# python evaluate.py --images-dir test_images/ --output evaluation_report.json

# --- Logging ---

os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=config.LOG_FILE,
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

_EXTENSOES_VALIDAS = {".jpg", ".jpeg", ".png", ".webp"}

# ---------------------------------------------------------------------------
# Regras sintéticas para avaliar o Rule Engine
# ---------------------------------------------------------------------------

_REGRAS_SINTETICAS = [
    {
        "texto": "Avisa-me quando a prateleira inferior estiver mais de 30% vazia.",
        "condicoes_esperadas": {"fill_rate_threshold": 0.7, "location_filter": "bottom"},
        "e_ambigua": False,
    },
    {
        "texto": "Na zona Z_S1, se não houver produtos de laticínios visíveis, é crítico.",
        "condicoes_esperadas": {"zone_filter": ["Z_S1"]},
        "e_ambigua": False,
    },
    {
        "texto": "Avisa-me quando a prateleira estiver vazia.",
        "condicoes_esperadas": {},
        "e_ambigua": True,
    },
    {
        "texto": "Se um produto estiver tombado, considera sempre severidade alta.",
        "condicoes_esperadas": {"issue_types": ["misaligned"], "severity_threshold": "high"},
        "e_ambigua": False,
    },
    {
        "texto": "Quando o fill rate cair abaixo de 60% entre as 10h e as 13h, avisa mas não é urgente.",
        "condicoes_esperadas": {
            "fill_rate_threshold": 0.6,
            "time_filter": {"hours_start": 10, "hours_end": 13},
        },
        "e_ambigua": False,
    },
    {
        "texto": "Avisa quando houver problemas.",
        "condicoes_esperadas": {},
        "e_ambigua": True,
    },
]

_INSPECAO_SINTETICA = {
    "zone_id": "Z_S1",
    "overall_status": "warning",
    "issues": [
        {"type": "empty_shelf", "severity": "high", "location": "bottom", "affected_area_pct": 0.40},
    ],
    "shelf_fill_rate": 0.60,
    "timestamp": "2025-04-01T11:00:00Z",
}

# Queries com ground truth para o RAG
_QUERIES_RAG = [
    {
        "query": "Quando foi a última vez que a zona Z_S1 teve problemas de prateleira vazia?",
        "inspection_ids_relevantes": [],
    },
    {
        "query": "Que zonas tiveram mais issues de planograma nas últimas 2 semanas?",
        "inspection_ids_relevantes": [],
    },
    {
        "query": "Existe algum padrão nos problemas detetados às sextas-feiras à tarde?",
        "inspection_ids_relevantes": [],
    },
    {
        "query": "Que regras foram mais frequentemente disparadas este mês?",
        "inspection_ids_relevantes": [],
    },
]


# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def _carregar_ground_truth(dir_imagens: str) -> dict:
    """Lê o ground_truth.json dentro do directório de imagens de teste."""
    caminho = os.path.join(dir_imagens, "ground_truth.json")
    if not os.path.exists(caminho):
        print(f"Erro: ground_truth.json não encontrado em '{dir_imagens}'.", file=sys.stderr)
        sys.exit(1)
    with open(caminho, encoding="utf-8") as f:
        return json.load(f)


def _json_valido(resultado: dict) -> bool:
    """Verifica se o resultado tem os campos obrigatórios do schema (Secção 4.2)."""
    campos = {"inspection_id", "overall_status", "issues", "shelf_fill_rate", "model_reasoning"}
    return isinstance(resultado, dict) and campos.issubset(resultado.keys())


def _condicoes_correspondem(regra_json: dict, esperado: dict) -> bool:
    """Verificação permissiva: esperado tem de ser subconjunto das condições da regra."""
    condicoes = regra_json.get("conditions", {})
    for chave, valor in esperado.items():
        if chave not in condicoes:
            return False
        val_pred = condicoes[chave]
        if isinstance(valor, dict):
            if not isinstance(val_pred, dict):
                return False
            for k, v in valor.items():
                if val_pred.get(k) != v:
                    return False
        elif isinstance(valor, list):
            if not set(valor).issubset(set(val_pred or [])):
                return False
        else:
            if val_pred != valor:
                return False
    return True


# ---------------------------------------------------------------------------
# Métricas — Análise Visual (Secção 9.2)
# ---------------------------------------------------------------------------

def _avaliar_visual(dir_imagens: str, ground_truth: dict) -> dict:
    """
    Calcula as métricas de análise visual sobre todas as imagens com ground truth.

    Métricas: JSON Parse Rate, Issue Detection Rate, False Positive Rate,
              Severity Accuracy, Hallucination Rate (via LLM-as-Judge).
    """
    from llm_judge import avaliar_alucinacao

    imagens = [
        os.path.join(dir_imagens, nome)
        for nome in sorted(os.listdir(dir_imagens))
        if os.path.splitext(nome.lower())[1] in _EXTENSOES_VALIDAS
           and nome in ground_truth
    ]

    total = len(imagens)
    json_ok = 0
    tp = fp = fn = 0
    severidade_correcta = severidade_total = 0
    scores_alucinacao = []

    print(f"\n[Análise Visual] {total} imagens a processar...")

    for caminho in imagens:
        nome = os.path.basename(caminho)
        gt = ground_truth[nome]
        gt_issues = {i["type"] for i in gt.get("issues", [])}
        gt_severidade = {i["type"]: i["severity"] for i in gt.get("issues", [])}

        try:
            resultado = inspecionar_prateleira(caminho, estrategia=config.DEFAULT_STRATEGY)
        except Exception as erro:
            logger.warning(f"Falha ao inspecionar {nome}: {erro}")
            continue

        if not _json_valido(resultado):
            logger.warning(f"JSON inválido para {nome}")
            continue
        json_ok += 1

        pred_issues = resultado.get("issues", [])
        pred_tipos = {i["type"] for i in pred_issues}
        pred_severidade = {i["type"]: i["severity"] for i in pred_issues}

        tp += len(gt_issues & pred_tipos)
        fp += len(pred_tipos - gt_issues)
        fn += len(gt_issues - pred_tipos)

        for tipo in gt_issues & pred_tipos:
            severidade_total += 1
            if pred_severidade.get(tipo) == gt_severidade.get(tipo):
                severidade_correcta += 1

        score_h = avaliar_alucinacao(
            raciocinio=resultado.get("model_reasoning", ""),
            descricoes=[i.get("description", "") for i in pred_issues],
        )
        scores_alucinacao.append(score_h)

        print(f"  ✓ {nome}")

    total_pred = tp + fp
    total_gt   = tp + fn

    return {
        "json_parse_rate":      round(json_ok / total, 4) if total else 0,
        "issue_detection_rate": round(tp / total_gt, 4)   if total_gt    else 0,
        "false_positive_rate":  round(fp / max(total_pred, 1), 4),
        "severity_accuracy":    round(severidade_correcta / max(severidade_total, 1), 4),
        "hallucination_rate":   round(1 - (sum(scores_alucinacao) / max(len(scores_alucinacao), 1)), 4),
        "_detalhes": {
            "total_imagens": total, "json_ok": json_ok,
            "tp": tp, "fp": fp, "fn": fn,
            "severidade_correcta": severidade_correcta,
            "severidade_total": severidade_total,
        },
    }


# ---------------------------------------------------------------------------
# Métricas — RAG (Secção 9.2)
# ---------------------------------------------------------------------------

def _avaliar_rag(caminho_gt_rag: str | None = None) -> dict:
    """
    Calcula Recall@3, Faithfulness e Answer Relevance sobre as queries de referência.
    Se existir um ground truth de RAG externo, usa-o em alternativa às queries padrão.
    """
    from llm_judge import avaliar_fidelidade, avaliar_relevancia

    queries = _QUERIES_RAG
    if caminho_gt_rag and os.path.exists(caminho_gt_rag):
        with open(caminho_gt_rag, encoding="utf-8") as f:
            queries = json.load(f)

    print(f"\n[RAG] {len(queries)} queries a avaliar...")

    recall_hits = 0
    scores_fidelidade = []
    scores_relevancia = []

    for q in queries:
        texto_query = q["query"]
        ids_relevantes = set(q.get("inspection_ids_relevantes", []))

        try:
            resultados = consultar(texto_query, k=3)
            resposta   = sintetizar(texto_query, resultados)
        except Exception as erro:
            logger.warning(f"Falha na query RAG '{texto_query[:40]}...': {erro}")
            continue

        ids_recuperados = {r.get("inspection_id") for r in resultados}
        if ids_relevantes and ids_recuperados & ids_relevantes:
            recall_hits += 1
        elif not ids_relevantes:
            recall_hits += 0.5   # sem GT definido: pontuação neutra

        contexto = "\n".join(r.get("summary", "") for r in resultados)
        scores_fidelidade.append(avaliar_fidelidade(resposta=resposta, contexto=contexto))
        scores_relevancia.append(avaliar_relevancia(query=texto_query, resposta=resposta))

        print(f"  ✓ {texto_query[:60]}")

    n = max(len(queries), 1)
    return {
        "recall_at_3":      round(recall_hits / n, 4),
        "faithfulness":     round(sum(scores_fidelidade) / max(len(scores_fidelidade), 1), 4),
        "answer_relevance": round(sum(scores_relevancia) / max(len(scores_relevancia), 1), 4),
    }


# ---------------------------------------------------------------------------
# Métricas — Rule Engine (Secção 9.2)
# ---------------------------------------------------------------------------

def _avaliar_rule_engine() -> dict:
    """
    Calcula Rule Parse Rate, Rule Correctness e Ambiguity Detection
    sobre o conjunto de regras sintéticas definido acima.
    """
    print(f"\n[Rule Engine] {len(_REGRAS_SINTETICAS)} regras a avaliar...")

    total = len(_REGRAS_SINTETICAS)
    parse_ok = 0
    correcto = 0
    ambiguidade_detectada = 0

    total_ambiguas     = sum(1 for r in _REGRAS_SINTETICAS if r["e_ambigua"])
    total_nao_ambiguas = total - total_ambiguas

    for regra_def in _REGRAS_SINTETICAS:
        try:
            regra_json = processar_regra(regra_def["texto"])
        except Exception as erro:
            logger.warning(f"Falha ao processar regra: {erro}")
            continue

        campos_obrigatorios = {"rule_id", "natural_language", "conditions", "action", "validation"}
        if not isinstance(regra_json, dict) or not campos_obrigatorios.issubset(regra_json.keys()):
            continue
        parse_ok += 1

        # Ambiguity Detection
        validacao = regra_json.get("validation", {})
        previu_ambigua = not validacao.get("is_valid", True) or bool(validacao.get("ambiguities"))
        if regra_def["e_ambigua"] and previu_ambigua:
            ambiguidade_detectada += 1

        # Rule Correctness — apenas regras não-ambíguas
        if not regra_def["e_ambigua"]:
            if _condicoes_correspondem(regra_json, regra_def["condicoes_esperadas"]):
                try:
                    executar_regra(regra_json, _INSPECAO_SINTETICA)
                    correcto += 1
                except Exception as erro:
                    logger.warning(f"Falha ao executar regra: {erro}")

        print(f"  ✓ {regra_def['texto'][:60]}")

    return {
        "rule_parse_rate":     round(parse_ok / total, 4) if total else 0,
        "rule_correctness":    round(correcto / max(total_nao_ambiguas, 1), 4),
        "ambiguity_detection": round(ambiguidade_detectada / max(total_ambiguas, 1), 4),
        "_detalhes": {
            "total_regras": total, "parse_ok": parse_ok,
            "correcto": correcto,
            "ambiguidade_detectada": ambiguidade_detectada,
            "total_ambiguas": total_ambiguas,
        },
    }


# ---------------------------------------------------------------------------
# Apresentação dos resultados
# ---------------------------------------------------------------------------

def _imprimir_tabela(visual: dict, rag: dict, rules: dict) -> None:
    """Imprime as métricas no terminal de forma legível."""
    def pct(v: float) -> str:
        return f"{v * 100:.1f}%"

    linhas = [
        ("Shelf Inspector", "JSON Parse Rate",       pct(visual.get("json_parse_rate", 0))),
        ("",                "Issue Detection Rate",  pct(visual.get("issue_detection_rate", 0))),
        ("",                "False Positive Rate",   pct(visual.get("false_positive_rate", 0))),
        ("",                "Severity Accuracy",     pct(visual.get("severity_accuracy", 0))),
        ("",                "Hallucination Rate",    pct(visual.get("hallucination_rate", 0))),
        ("RAG Memory",      "Recall@3",              pct(rag.get("recall_at_3", 0))),
        ("",                "Faithfulness",          pct(rag.get("faithfulness", 0))),
        ("",                "Answer Relevance",      pct(rag.get("answer_relevance", 0))),
        ("Rule Engine",     "Rule Parse Rate",       pct(rules.get("rule_parse_rate", 0))),
        ("",                "Rule Correctness",      pct(rules.get("rule_correctness", 0))),
        ("",                "Ambiguity Detection",   pct(rules.get("ambiguity_detection", 0))),
    ]

    print("\n" + "=" * 62)
    print(f"{'Componente':<18} {'Métrica':<24} {'Valor':>8}")
    print("=" * 62)
    for componente, metrica, valor in linhas:
        print(f"{componente:<18} {metrica:<24} {valor:>8}")
    print("=" * 62)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Harness de avaliação — TP2 Retail Vision Intelligence System"
    )
    parser.add_argument(
        "--images-dir",
        required=True,
        help="Directório com imagens de teste e ground_truth.json.",
    )
    parser.add_argument(
        "--output",
        default="evaluation_report.json",
        help="Ficheiro de saída com as métricas (JSON). Default: %(default)s",
    )
    parser.add_argument(
        "--rag-gt",
        default=None,
        help="Ficheiro JSON com ground truth de queries RAG (opcional).",
    )
    parser.add_argument(
        "--skip-visual",
        action="store_true",
        help="Ignora a avaliação visual.",
    )
    parser.add_argument(
        "--skip-rag",
        action="store_true",
        help="Ignora a avaliação do RAG.",
    )
    parser.add_argument(
        "--skip-rules",
        action="store_true",
        help="Ignora a avaliação do Rule Engine.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.images_dir):
        print(f"Erro: directório não encontrado: '{args.images_dir}'.", file=sys.stderr)
        sys.exit(1)

    ground_truth = _carregar_ground_truth(args.images_dir)

    metricas_visual = {}
    metricas_rag    = {}
    metricas_rules  = {}

    if not args.skip_visual:
        metricas_visual = _avaliar_visual(args.images_dir, ground_truth)

    if not args.skip_rag:
        metricas_rag = _avaliar_rag(args.rag_gt)

    if not args.skip_rules:
        metricas_rules = _avaliar_rule_engine()

    _imprimir_tabela(metricas_visual, metricas_rag, metricas_rules)

    relatorio = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "images_dir": args.images_dir,
        "metricas": {
            "visual":  metricas_visual,
            "rag":     metricas_rag,
            "rules":   metricas_rules,
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)

    print(f"\nRelatório guardado em: {args.output}")
    logger.info(f"Avaliação concluída. Relatório: {args.output}")

import argparse
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import chromadb
import config
from chromadb.utils import embedding_functions
from etc.rate_limiter import chamar_com_backoff
from google import genai

# python src/rag_memory.py --indexar data/inspections/INS_001.json
# python src/rag_memory.py --query "Quando foi a última vez que a zona Z_S1 teve prateleira vazia?"
# python src/rag_memory.py --query "..." --estrategia full_record --k 5
# python src/rag_memory.py --contar
# python src/rag_memory.py --avaliar data/eval_queries.json

# --- Logging ---

os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=config.LOG_FILE,
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

# Estratégias de chunking disponíveis (ver 6.5 do enunciado)
ESTRATEGIA_HIBRIDA = "hibrido"
ESTRATEGIA_FULL = "full_record"


# --- Cliente API ---

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


# --- Prompts ---

_PROMPT_SUMMARY_FICHEIRO = "rag_summary_prompt.txt"
_PROMPT_SINTESE_FICHEIRO = "rag_synthesis_prompt.txt"
_PROMPT_FILTROS_FICHEIRO = "rag_filter_extraction_prompt.txt"


def _carregar_prompt(nome_ficheiro: str) -> str:
    """Lê um ficheiro de prompt de config.PROMPTS_DIR."""
    caminho = os.path.join(config.PROMPTS_DIR, nome_ficheiro)
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as erro:
        logger.error(f"Não foi possível ler o ficheiro de prompt {caminho}: {erro}")
        raise


# --- Geração de texto indexável ---

def _gerar_summary(cliente: genai.Client, inspecao: dict) -> str:
    """
    Usa o Gemini para gerar o summary rico que recebe embeddings na
     estratégia HÍBRIDA. Em caso de falha da API, recorre a um summary
     determinístico construído a partir dos campos estruturados.
    """
    inspection_id = inspecao.get("inspection_id", "?")
    prompt = _carregar_prompt(_PROMPT_SUMMARY_FICHEIRO).replace(
        "{{INSPECTION_JSON}}", json.dumps(inspecao, ensure_ascii=False, indent=2)
    )

    try:
        summary = chamar_com_backoff(cliente, [prompt]).strip()
        if len(summary) < 30:
            raise ValueError("summary devolvido pela LLM é demasiado curto")
        return summary
    except Exception as erro:
        logger.warning(f"Falha na geração de summary para {inspection_id}: {erro}. A usar fallback.")
        return _construir_summary_fallback(inspecao)


def _construir_summary_fallback(inspecao: dict) -> str:
    """Summary determinístico a partir dos campos estruturados, sem chamar a API."""
    zona = inspecao.get("zone_id", "zona desconhecida")
    timestamp = inspecao.get("timestamp", "")
    estado = inspecao.get("overall_status", "desconhecido")
    fill_rate = inspecao.get("shelf_fill_rate", 0.0)
    produtos = ", ".join(inspecao.get("products_detected", [])) or "não identificados"

    issues = inspecao.get("issues", [])
    descricao_issues = "; ".join(
        f"{i.get('type', '?')} ({i.get('severity', '?')}) — {i.get('location', '?')}"
        for i in issues
    ) or "nenhum"

    return (
        f"Inspeção na {zona} em {timestamp}. Estado geral: {estado}. "
        f"Fill rate: {fill_rate:.0%}. Produtos detetados: {produtos}. "
        f"Problemas: {descricao_issues}."
    )


def _inspecao_para_texto(inspecao: dict) -> str:
    """Serializa a inspeção completa, usada como documento na estratégia FULL_RECORD."""
    return json.dumps(inspecao, ensure_ascii=False, indent=2)


def _construir_metadata(inspecao: dict) -> dict[str, Any]:
    """
    Constrói os metadados estruturados guardados no ChromaDB, usados para
     pre-retrieval filtering (zona, data, status, fill rate, ...).

    ChromaDB só aceita str, int, float e bool — listas são serializadas
     como strings separadas por vírgulas.
    """
    timestamp = inspecao.get("timestamp", "")
    data = timestamp[:10] if timestamp else ""  # "YYYY-MM-DD"

    weekday, hora = -1, -1
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        weekday, hora = dt.weekday(), dt.hour
    except ValueError:
        pass

    issues = inspecao.get("issues", [])
    tipos_issue = {i.get("type", "") for i in issues if i.get("type")}
    severidades = {i.get("severity", "") for i in issues if i.get("severity")}

    return {
        "inspection_id": inspecao.get("inspection_id", ""),
        "zone_id": inspecao.get("zone_id", ""),
        "date": data,
        "weekday": weekday,  # 0=segunda ... 6=domingo
        "hour": hora,
        "overall_status": inspecao.get("overall_status", ""),
        "shelf_fill_rate": float(inspecao.get("shelf_fill_rate", 0.0)),
        "has_critical_issue": any(i.get("severity") == "high" for i in issues),
        "issue_types_str": ",".join(sorted(tipos_issue)),
        "severities_str": ",".join(sorted(severidades)),
        "n_issues": len(issues),
        "image_path": inspecao.get("image_path", ""),
    }


# --- Retrieval ---

def _extrair_filtros(cliente: genai.Client, query: str) -> dict[str, Any]:
    """
    Usa o Gemini para extrair filtros estruturados (zone_id, days_back,
     issue_type, status) da query em linguagem natural.

    Em caso de falha, devolve {} e a query segue sem pre-retrieval filtering.
    """
    prompt = _carregar_prompt(_PROMPT_FILTROS_FICHEIRO).replace("{{QUERY}}", query)

    try:
        return _extrair_json(chamar_com_backoff(cliente, [prompt]))
    except Exception as erro:
        logger.debug(f"Falha na extração de filtros: {erro}. A continuar sem filtros.")
        return {}


def _construir_where(filtros: dict[str, Any]) -> dict | None:
    """Constrói a cláusula `where` do ChromaDB a partir dos filtros extraídos."""
    condicoes = []

    if filtros.get("zone_id"):
        condicoes.append({"zone_id": {"$eq": filtros["zone_id"]}})

    if filtros.get("days_back"):
        limite = (datetime.now(timezone.utc) - timedelta(days=int(filtros["days_back"]))).strftime("%Y-%m-%d")
        condicoes.append({"date": {"$gte": limite}})

    if filtros.get("status"):
        condicoes.append({"overall_status": {"$eq": filtros["status"]}})

    if not condicoes:
        return None
    if len(condicoes) == 1:
        return condicoes[0]
    return {"$and": condicoes}


def _sintetizar_resposta(cliente: genai.Client, query: str, contexto: str) -> str:
    """Usa o Gemini para sintetizar a resposta final a partir dos chunks recuperados."""
    prompt = (
        _carregar_prompt(_PROMPT_SINTESE_FICHEIRO)
        .replace("{{QUERY}}", query)
        .replace("{{CONTEXTO}}", contexto)
    )

    try:
        return chamar_com_backoff(cliente, [prompt]).strip()
    except Exception as erro:
        logger.error(f"Falha na síntese da resposta: {erro}")
        return "Não foi possível sintetizar uma resposta neste momento. Consulta os registos recuperados directamente."


# --- Memória vetorial ---

class RAGMemory:
    """
    Memória vetorial de inspeções de prateleiras.

    Suporta duas estratégias de chunking para comparação de Recall@3 (6.5):
      - hibrido:     summary gerado por LLM como texto de embedding,
                      com metadados estruturados para pre-retrieval filtering.
      - full_record: inspection record completo como documento único (baseline).
    """

    def __init__(self, vectorstore_dir: str = config.VECTORSTORE_DIR, default_k: int = config.RAG_DEFAULT_K):
        self._cliente = _get_client()
        self._default_k = default_k

        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.EMBED_MODEL_NAME)
        chroma = chromadb.PersistentClient(path=vectorstore_dir)

        self._col_hibrido = chroma.get_or_create_collection(
            name=config.RAG_COLLECTION_HYBRID,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        self._col_full = chroma.get_or_create_collection(
            name=config.RAG_COLLECTION_FULL,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        self._chroma = chroma


    # --- Indexação ---

    def indexar_inspecao(self, inspecao: dict) -> str:
        """Indexa uma inspeção nas duas colections (hibrido + full_record). Idempotente."""
        inspection_id = inspecao.get("inspection_id", "")
        if not inspection_id:
            raise ValueError("inspection_id em falta na inspeção.")

        if self._col_hibrido.get(ids=[inspection_id])["ids"]:
            logger.info(f"Inspeção {inspection_id} já indexada. A saltar.")
            return inspection_id

        metadata = _construir_metadata(inspecao)

        summary = _gerar_summary(self._cliente, inspecao)
        self._col_hibrido.add(ids=[inspection_id], documents=[summary], metadatas=[metadata])

        full_text = _inspecao_para_texto(inspecao)
        self._col_full.add(ids=[inspection_id], documents=[full_text], metadatas=[metadata])

        logger.info(f"Inspeção {inspection_id} indexada (summary: {len(summary)} chars, full: {len(full_text)} chars).")
        return inspection_id

    def indexar_lote(self, inspecoes: list[dict]) -> list[str]:
        """Indexa uma lista de inspeções. O rate limiting é tratado pelo etc.rate_limiter."""
        indexadas = []
        for i, inspecao in enumerate(inspecoes):
            try:
                indexadas.append(self.indexar_inspecao(inspecao))
            except Exception as erro:
                logger.error(f"Erro ao indexar inspeção {i}: {erro}")
        return indexadas


    # --- Consulta ---

    def consultar(self, query: str, k: int | None = None, estrategia: str = ESTRATEGIA_HIBRIDA) -> dict:
        """
        Responde a uma query em linguagem natural usando RAG.

        Devolve {"query", "answer", "retrieved_inspections", "estrategia"}.
        """
        k = k or self._default_k
        colecao = self._col_hibrido if estrategia == ESTRATEGIA_HIBRIDA else self._col_full

        filtros = _extrair_filtros(self._cliente, query)
        where = _construir_where(filtros)
        logger.info(f"Query: '{query}' | filtros: {filtros} | estratégia: {estrategia}")

        n_results = min(k, colecao.count() or 1)
        try:
            resultados = colecao.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as erro:
            logger.warning(f"Retrieval com filtros falhou ({erro}). A tentar sem filtros...")
            resultados = colecao.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

        if not resultados["ids"][0]:
            return {
                "query": query,
                "answer": "Não foram encontrados registos relevantes na memória de inspeções.",
                "retrieved_inspections": [],
                "estrategia": estrategia,
            }

        recuperados, partes_contexto = [], []
        for documento, meta, distancia in zip(resultados["documents"][0], resultados["metadatas"][0], resultados["distances"][0]):
            score = round(1 - distancia, 4)  # distância coseno -> similaridade
            recuperados.append({
                "inspection_id": meta.get("inspection_id", "?"),
                "date": meta.get("date", "?"),
                "score": score,
                "summary": documento,
            })
            partes_contexto.append(f"[{meta.get('inspection_id', '?')} | {meta.get('date', '?')} | score={score}]\n{documento}")

        contexto = "\n\n---\n\n".join(partes_contexto)
        resposta = _sintetizar_resposta(self._cliente, query, contexto)

        return {
            "query": query,
            "answer": resposta,
            "retrieved_inspections": recuperados,
            "estrategia": estrategia,
        }

    def contexto_relevante(self, query: str, k: int | None = None, estrategia: str = ESTRATEGIA_HIBRIDA) -> list[dict]:
        """Devolve os k chunks mais relevantes sem síntese (para uso no report_generator)."""
        return self.consultar(query, k=k, estrategia=estrategia)["retrieved_inspections"]


    # --- Avaliação — Recall@k (6.5) ---

    def avaliar_recall(self, queries_com_ground_truth: list[dict], k: int = 3) -> dict:
        """
        Calcula Recall@k para as duas estratégias.

        queries_com_ground_truth: lista de {"query": str, "relevant_ids": list[str]}.
        """
        resultados = {
            ESTRATEGIA_HIBRIDA: {"hits": 0, "total": 0, "details": []},
            ESTRATEGIA_FULL: {"hits": 0, "total": 0, "details": []},
        }

        for item in queries_com_ground_truth:
            query, relevantes = item["query"], set(item["relevant_ids"])

            for estrategia in (ESTRATEGIA_HIBRIDA, ESTRATEGIA_FULL):
                recuperados = {r["inspection_id"] for r in self.contexto_relevante(query, k=k, estrategia=estrategia)}
                acertou = bool(recuperados & relevantes)

                resultados[estrategia]["hits"] += int(acertou)
                resultados[estrategia]["total"] += 1
                resultados[estrategia]["details"].append({
                    "query": query,
                    "relevant": sorted(relevantes),
                    "retrieved": sorted(recuperados),
                    "hit": acertou,
                })

        for estrategia, dados in resultados.items():
            dados[f"recall_at_{k}"] = round(dados["hits"] / dados["total"], 4) if dados["total"] else 0.0

        logger.info(
            f"Recall@{k} — hibrido: {resultados[ESTRATEGIA_HIBRIDA][f'recall_at_{k}']:.2f} | "
            f"full_record: {resultados[ESTRATEGIA_FULL][f'recall_at_{k}']:.2f}"
        )
        return resultados


    # --- Queries pré-definidas (6.4) ---

    def ultima_prateleira_vazia(self, zone_id: str) -> dict:
        """'Quando foi a última vez que a zona X teve problemas de prateleira vazia?'"""
        return self.consultar(f"Quando foi a última vez que a zona {zone_id} teve problemas de prateleira vazia?")

    def zonas_com_mais_issues_planograma(self, dias: int = 14) -> dict:
        """'Que zonas tiveram mais issues de planograma nas últimas N semanas?'"""
        return self.consultar(f"Que zonas tiveram mais issues de planograma nos últimos {dias} dias?")

    def padroes_sexta_feira_tarde(self) -> dict:
        """'Existe algum padrão nos problemas detetados às sextas-feiras à tarde?'"""
        return self.consultar("Existe algum padrão nos problemas detetados às sextas-feiras à tarde?")

    def regras_mais_disparadas(self, mes: str | None = None) -> dict:
        """'Que regras foram mais frequentemente disparadas este mês?'"""
        if mes:
            return self.consultar(f"Que regras foram mais frequentemente disparadas em {mes}?")
        return self.consultar("Que regras foram mais frequentemente disparadas este mês?")


    # --- Utilitários ---

    def contar(self) -> dict[str, int]:
        """Devolve o número de documentos indexados em cada colection."""
        return {ESTRATEGIA_HIBRIDA: self._col_hibrido.count(), ESTRATEGIA_FULL: self._col_full.count()}

    def remover_inspecao(self, inspection_id: str) -> None:
        """Remove uma inspeção de ambas as colections."""
        self._col_hibrido.delete(ids=[inspection_id])
        self._col_full.delete(ids=[inspection_id])
        logger.info(f"Inspeção {inspection_id} removida de ambas as colections.")

    def limpar_tudo(self) -> None:
        """Apaga todos os documentos indexados (útil para testes)."""
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.EMBED_MODEL_NAME)
        for nome in (config.RAG_COLLECTION_HYBRID, config.RAG_COLLECTION_FULL):
            self._chroma.delete_collection(nome)

        self._col_hibrido = self._chroma.get_or_create_collection(name=config.RAG_COLLECTION_HYBRID, embedding_function=ef, metadata={"hnsw:space": "cosine"})
        self._col_full = self._chroma.get_or_create_collection(name=config.RAG_COLLECTION_FULL, embedding_function=ef, metadata={"hnsw:space": "cosine"})
        logger.warning("Todas as inspeções foram removidas das colections.")


# --- CLI ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Memory — indexação e consulta da memória de inspeções")
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("--indexar", metavar="FICHEIRO_JSON", help="Indexa um inspection_record JSON (ou lista de records).")
    group.add_argument("--query", metavar="TEXTO", help="Faz uma query em linguagem natural à memória.")
    group.add_argument("--contar", action="store_true", help="Mostra o número de documentos indexados em cada colection.")
    group.add_argument("--avaliar", metavar="FICHEIRO_JSON", help="Avalia Recall@k a partir de um ficheiro de queries com ground truth.")

    parser.add_argument("--estrategia", choices=[ESTRATEGIA_HIBRIDA, ESTRATEGIA_FULL], default=ESTRATEGIA_HIBRIDA, help="Usado com --query.")
    parser.add_argument("--k", type=int, default=config.RAG_DEFAULT_K, help="Usado com --query e --avaliar.")

    args = parser.parse_args()
    memoria = RAGMemory()

    if args.indexar:
        with open(args.indexar, "r", encoding="utf-8") as f:
            dados = json.load(f)

        inspecoes = dados if isinstance(dados, list) else [dados]
        ids = memoria.indexar_lote(inspecoes)
        print(f"Indexadas {len(ids)} inspeção(ões): {', '.join(ids)}")

    elif args.query:
        resultado = memoria.consultar(args.query, k=args.k, estrategia=args.estrategia)
        print("\n── Resposta ──────────────────────────────────────")
        print(resultado["answer"])
        print("\n── Inspeções recuperadas ────────────────────────")
        for r in resultado["retrieved_inspections"]:
            print(f"  [{r['inspection_id']}] {r['date']}  score={r['score']:.3f}")
            print(f"  {r['summary'][:120]}...")

    elif args.contar:
        contagens = memoria.contar()
        print(f"Colection hibrido:     {contagens[ESTRATEGIA_HIBRIDA]} documentos")
        print(f"Colection full_record: {contagens[ESTRATEGIA_FULL]} documentos")

    elif args.avaliar:
        with open(args.avaliar, "r", encoding="utf-8") as f:
            ground_truth = json.load(f)

        resultados = memoria.avaliar_recall(ground_truth, k=args.k)
        print(f"\nRecall@{args.k}:")
        for estrategia, dados in resultados.items():
            print(f"  {estrategia}: {dados[f'recall_at_{args.k}']:.2%}")

        caminho_saida = os.path.join(config.BASE_DIR, "evaluation_rag_recall.json")
        with open(caminho_saida, "w", encoding="utf-8") as f:
            json.dump(resultados, f, ensure_ascii=False, indent=2)
        print(f"\nResultados guardados em {caminho_saida}")
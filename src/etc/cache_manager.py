import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import config

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Gere a cache local de resultados das inspeções e a quota diária da API Gemini.

    A chave de cache é o hash MD5 da imagem + estratégia de prompting, o que
     garante que mudar a imagem invalida o cache, mas que a mesma imagem
     analisada com a mesma estratégia reaproveita o resultado sem gastar quota.

    O contador de pedidos é guardado num ficheiro JSON e é reposto a zero
     automaticamente assim que muda o dia em UTC.
    """

    def __init__(self):
        self.cache_dir = config.CACHE_DIR
        self.quota_file = config.QUOTA_FILE
        self.max_req_dia = config.MAX_REQUESTS_PER_DAY
        self.max_req_minuto = config.MAX_REQUESTS_PER_MINUTE

    # --- Cache ---

    def chave(self, caminho_imagem: str, estrategia: str) -> str:
        """Gera a chave de cache a partir do MD5 da imagem e da estratégia usada."""
        return f"{self._md5(caminho_imagem)}_{estrategia}"

    def get(self, chave: str) -> Optional[dict]:
        """Devolve o resultado guardado em cache, ou None se não existir."""
        ficheiro = os.path.join(self.cache_dir, f"{chave}.json")
        if os.path.exists(ficheiro):
            logger.info("Cache HIT — a retornar resultado em cache.")
            with open(ficheiro, "r", encoding="utf-8") as f:
                return json.load(f)

        logger.debug(f"Cache MISS para chave: {chave}")
        return None

    def set(self, chave: str, resultado: dict) -> None:
        """Guarda o resultado de uma inspeção em disco."""
        os.makedirs(self.cache_dir, exist_ok=True)
        ficheiro = os.path.join(self.cache_dir, f"{chave}.json")
        with open(ficheiro, "w", encoding="utf-8") as f:
            json.dump(resultado, f, ensure_ascii=False, indent=2)
        logger.info(f"Resultado guardado em cache: {ficheiro}")

    def limpar(self) -> int:
        """
        Remove todos os ficheiros de cache, sem mexer no ficheiro de quota,
         e devolve o número de ficheiros removidos.
        """
        if not os.path.exists(self.cache_dir):
            return 0

        contagem = 0
        for nome in os.listdir(self.cache_dir):
            if nome.endswith(".json") and nome != "_quota.json":
                os.remove(os.path.join(self.cache_dir, nome))
                contagem += 1

        logger.info(f"Cache limpa: {contagem} ficheiro(s) removido(s).")
        return contagem

    # --- Quota diária ---

    def quota_esgotada(self) -> bool:
        """Devolve True se o limite diário de pedidos já foi atingido."""
        return self._ler_quota()["count"] >= self.max_req_dia

    def incrementar_quota(self) -> None:
        """Soma mais um pedido ao contador da quota diária."""
        quota = self._ler_quota()
        quota["count"] += 1
        self._escrever_quota(quota)
        logger.info(f"Quota diária: {quota['count']}/{self.max_req_dia} req usadas.")

    def estado(self) -> dict:
        """Devolve um dicionário com o estado actual da quota."""
        quota = self._ler_quota()
        return {
            "data": quota["date"],
            "pedidos_usados": quota["count"],
            "pedidos_restantes": self.max_req_dia - quota["count"],
            "limite_por_dia": self.max_req_dia,
            "limite_por_minuto": self.max_req_minuto,
        }

    # --- Auxiliares ---

    @staticmethod
    def _md5(caminho: str) -> str:
        """Calcula o MD5 do ficheiro lendo-o em blocos, para nao carregar o ficheiro todo para memória."""
        h = hashlib.md5()
        with open(caminho, "rb") as f:
            for bloco in iter(lambda: f.read(65536), b""):
                h.update(bloco)
        return h.hexdigest()

    def _ler_quota(self) -> dict:
        """Lê o ficheiro de quota, ou começa um registo novo se não existir ou for de outro dia."""
        hoje = datetime.now(timezone.utc).date().isoformat()

        if os.path.exists(self.quota_file):
            with open(self.quota_file, "r", encoding="utf-8") as f:
                dados = json.load(f)
            if dados.get("date") == hoje:
                return dados

        return {"date": hoje, "count": 0}

    def _escrever_quota(self, quota: dict) -> None:
        """Persiste o estado da quota em disco."""
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.quota_file, "w", encoding="utf-8") as f:
            json.dump(quota, f)
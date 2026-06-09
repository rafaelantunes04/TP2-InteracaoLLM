"""
cache_manager.py — Gestão de cache local e quota diária
Retail Vision Intelligence System — LIACD TP2

Responsabilidades:
  - Cache de resultados em disco por chave MD5 + estratégia (Secção 4.4)
  - Controlo e persistência da quota diária de pedidos à API (Secção 4.4)

Uso típico (pelo shelf_inspector):
    cache = CacheManager(config.CACHE_DIR, config.QUOTA_FILE)
    resultado = cache.get(chave)
    if resultado is None:
        resultado = chamar_api(...)
        cache.set(chave, resultado)
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import config

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Gere a cache local de resultados e a quota diária da API Gemini.

    Cache
    -----
    A chave é gerada a partir do hash MD5 da imagem + estratégia de prompting.
    Isto garante que qualquer alteração ao ficheiro de imagem invalida o cache,
    mas que imagens idênticas analisadas com a mesma estratégia reutilizam o
    resultado sem consumir quota.

    Quota
    -----
    O contador de pedidos é guardado num ficheiro JSON e reposto a zero
    automaticamente no início de cada dia UTC.
    """

    def __init__(self, cache_dir: Path, quota_file: Path):
        self.cache_dir = Path(cache_dir)
        self.quota_file = Path(quota_file)
        self.max_req_dia = config.MAX_REQUESTS_PER_DAY
        self.max_req_minuto = config.MAX_REQUESTS_PER_MINUTE

    # ------------------------------------------------------------------
    # Cache — interface pública
    # ------------------------------------------------------------------

    def chave(self, caminho_imagem: str, estrategia: str) -> str:
        """Gera a chave de cache: MD5 do ficheiro de imagem + estratégia."""
        return f"{self._md5(caminho_imagem)}_{estrategia}"

    def get(self, chave: str) -> dict | None:
        """Devolve o resultado guardado em cache, ou None se não existir."""
        ficheiro = self.cache_dir / f"{chave}.json"
        if ficheiro.exists():
            logger.info("Cache HIT — a retornar resultado em cache.")
            return json.loads(ficheiro.read_text(encoding="utf-8"))
        logger.debug(f"Cache MISS para chave: {chave}")
        return None

    def set(self, chave: str, resultado: dict) -> None:
        """Guarda o resultado em disco."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        ficheiro = self.cache_dir / f"{chave}.json"
        ficheiro.write_text(
            json.dumps(resultado, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Resultado guardado em cache: {ficheiro}")

    def limpar(self) -> int:
        """Remove todos os ficheiros de cache (sem apagar o ficheiro de quota).

        Devolve o número de ficheiros removidos.
        """
        if not self.cache_dir.exists():
            return 0

        ficheiros = [f for f in self.cache_dir.glob("*.json") if f.name != "_quota.json"]
        for ficheiro in ficheiros:
            ficheiro.unlink()

        logger.info(f"Cache limpa: {len(ficheiros)} ficheiro(s) removido(s).")
        return len(ficheiros)

    # ------------------------------------------------------------------
    # Quota diária — interface pública
    # ------------------------------------------------------------------

    def quota_esgotada(self) -> bool:
        """Devolve True se o limite diário de pedidos foi atingido."""
        return self._ler_quota()["count"] >= self.max_req_dia

    def incrementar_quota(self) -> None:
        """Regista mais um pedido consumido na quota diária."""
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

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    @staticmethod
    def _md5(caminho: str) -> str:
        """Calcula o hash MD5 do ficheiro em blocos (eficiente para ficheiros grandes)."""
        h = hashlib.md5()
        with open(caminho, "rb") as f:
            for bloco in iter(lambda: f.read(65536), b""):
                h.update(bloco)
        return h.hexdigest()

    def _ler_quota(self) -> dict:
        """Lê o ficheiro de quota, ou cria um registo novo se não existir ou for de outro dia."""
        hoje = datetime.now(timezone.utc).date().isoformat()
        if self.quota_file.exists():
            dados = json.loads(self.quota_file.read_text(encoding="utf-8"))
            if dados.get("date") == hoje:
                return dados
        return {"date": hoje, "count": 0}

    def _escrever_quota(self, quota: dict) -> None:
        """Persiste o estado da quota em disco."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.quota_file.write_text(json.dumps(quota), encoding="utf-8")
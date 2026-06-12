from __future__ import annotations

import copy
import json
import logging
import os
import sys
import argparse
from datetime import datetime, timezone

import config
from etc.rate_limiter import chamar_com_backoff
from google import genai

# python src/rule_engine.py --listar
# python src/rule_engine.py --adicionar "Avisa-me quando o fill rate cair abaixo de 60%"
# python src/rule_engine.py --eliminar RULE_20250317_143022
# python src/rule_engine.py --testar RULE_20250317_143022 --resultado resultado.json

# --- Logging ---

os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=config.LOG_FILE,
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)

_SEVERIDADE_ORDEM = {"low": 0, "medium": 1, "high": 2}


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


# --- Prompt de conversão ---

# Nomes dos ficheiros de prompt, dentro de config.PROMPTS_DIR (ex: tp2/prompts/)
_PROMPT_CONVERSAO_FICHEIRO = "rule_engine_prompt.txt"
_SCHEMA_CONVERSAO_FICHEIRO = "rule_engine_schema.txt"


def _carregar_prompt(nome_ficheiro: str) -> str:
    """Lê um ficheiro de prompt/schema de config.PROMPTS_DIR."""
    caminho = os.path.join(config.PROMPTS_DIR, nome_ficheiro)
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            return f.read()
    except OSError as erro:
        logger.error(f"Não foi possível ler o ficheiro de prompt {caminho}: {erro}")
        raise


def _construir_prompt_conversao(texto_natural: str) -> str:
    """
    Monta o prompt de conversão de regra a partir dos templates em
     config.PROMPTS_DIR, substituindo os placeholders {{TEXTO_NATURAL}}
     e {{SCHEMA_JSON}} pelo texto da regra e pelo schema JSON de exemplo.
    """
    template       = _carregar_prompt(_PROMPT_CONVERSAO_FICHEIRO)
    schema_exemplo = _carregar_prompt(_SCHEMA_CONVERSAO_FICHEIRO).strip()

    return (
        template
        .replace("{{TEXTO_NATURAL}}", texto_natural)
        .replace("{{SCHEMA_JSON}}", schema_exemplo)
    )


# --- Conversão via LLM ---

def _converter_regra_com_llm(texto_natural: str) -> dict:
    """Usa o LLM Gemini para converter uma regra em linguagem natural para o schema JSON."""
    cliente = _get_client()
    prompt  = _construir_prompt_conversao(texto_natural)

    logger.info(f"A converter regra: {texto_natural[:80]}…")
    texto_bruto = chamar_com_backoff(cliente, [prompt])

    try:
        regra = _extrair_json(texto_bruto)
    except (ValueError, json.JSONDecodeError) as erro:
        logger.error(f"Falha no parse da resposta do LLM: {erro}")
        raise ValueError(f"O LLM devolveu uma resposta inválida: {erro}") from erro

    momento = datetime.now(timezone.utc)
    regra["rule_id"] = f"RULE_{momento.strftime('%Y%m%d_%H%M%S')}"
    regra["created_at"] = momento.isoformat()
    regra["natural_language"] = texto_natural

    logger.info(f"Regra convertida: {regra['rule_id']}")
    return regra


# --- Persistência ---

def _caminho_regra(rule_id: str) -> str:
    return os.path.join(config.RULES_DIR, f"{rule_id}.json")


def _guardar_regra(regra: dict) -> str:
    """Persiste a regra em disco como JSON. Devolve o rule_id."""
    os.makedirs(config.RULES_DIR, exist_ok=True)
    rule_id = regra["rule_id"]
    caminho = _caminho_regra(rule_id)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(regra, f, ensure_ascii=False, indent=2)
    logger.info(f"Regra guardada em: {caminho}")
    return rule_id


def _carregar_todas_as_regras() -> list[dict]:
    """Carrega todas as regras JSON do directório de regras."""
    if not os.path.exists(config.RULES_DIR):
        return []

    regras = []
    for nome in sorted(os.listdir(config.RULES_DIR)):
        if not nome.endswith(".json"):
            continue
        caminho = os.path.join(config.RULES_DIR, nome)
        try:
            with open(caminho, "r", encoding="utf-8") as f:
                regras.append(json.load(f))
        except (json.JSONDecodeError, OSError) as erro:
            logger.warning(f"Não foi possível carregar {caminho}: {erro}")

    logger.info(f"{len(regras)} regra(s) carregada(s) de {config.RULES_DIR}")
    return regras


def _eliminar_regra(rule_id: str) -> bool:
    """Remove o ficheiro JSON de uma regra. Devolve True se apagou, False se não existia."""
    caminho = _caminho_regra(rule_id)
    if os.path.exists(caminho):
        os.remove(caminho)
        logger.info(f"Regra eliminada: {rule_id}")
        return True
    logger.warning(f"Regra não encontrada para eliminar: {rule_id}")
    return False


# --- Avaliação de condições ---

def _avaliar_conditions(conditions: dict, resultado: dict) -> bool:
    """Verifica se todas as condições de uma regra são satisfeitas pelo resultado da inspeção."""

    # Filtro de zona
    zone_filter = conditions.get("zone_filter") or []
    if zone_filter and resultado.get("zone_id") not in zone_filter:
        return False

    # Filtro temporal (hora UTC actual)
    time_filter = conditions.get("time_filter")
    if time_filter:
        hora_atual = datetime.now(timezone.utc).hour
        h_inicio   = time_filter.get("hours_start", 0)
        h_fim      = time_filter.get("hours_end", 23)
        if not (h_inicio <= hora_atual <= h_fim):
            return False

    # Filtro de fill rate
    fill_threshold = conditions.get("fill_rate_threshold")
    if fill_threshold is not None:
        if resultado.get("shelf_fill_rate", 1.0) >= fill_threshold:
            return False

    # Filtro de tipos de issue
    issue_types = conditions.get("issue_types") or []
    if issue_types:
        tipos_presentes = {i.get("type") for i in resultado.get("issues", [])}
        if not tipos_presentes.intersection(issue_types):
            return False

    # Filtro de severidade mínima
    sev_threshold = conditions.get("severity_threshold")
    if sev_threshold:
        nivel_minimo = _SEVERIDADE_ORDEM.get(sev_threshold, 0)
        tem_severidade = any(
            _SEVERIDADE_ORDEM.get(i.get("severity", "low"), 0) >= nivel_minimo
            for i in resultado.get("issues", [])
        )
        if not tem_severidade:
            return False

    # Filtro de localização
    loc_filter = conditions.get("location_filter", "any")
    if loc_filter and loc_filter != "any":
        tem_localizacao = any(
            loc_filter.lower() in i.get("location", "").lower()
            for i in resultado.get("issues", [])
        )
        if not tem_localizacao:
            return False

    return True


# --- Geração de notificações ---

def _gerar_notificacao(regra: dict, resultado: dict) -> str:
    """Preenche o template de notificação com os dados concretos da inspeção."""
    template = regra.get("action", {}).get(
        "notification_message",
        "ALERTA [{alert_level}] Regra {rule_id} disparou na zona {zone_id}. [{timestamp}]",
    )
    try:
        return template.format(
            rule_id        = regra.get("rule_id", "?"),
            alert_level    = regra.get("action", {}).get("alert_level", "info").upper(),
            zone_id        = resultado.get("zone_id", "?"),
            fill_rate      = f"{resultado.get('shelf_fill_rate', 0.0):.1%}",
            issue_count    = len(resultado.get("issues", [])),
            timestamp      = resultado.get("timestamp", ""),
            overall_status = resultado.get("overall_status", ""),
        )
    except KeyError as erro:
        logger.warning(f"Placeholder inválido no template: {erro}")
        return f"[Erro no template da regra {regra.get('rule_id', '?')}: placeholder {erro} não reconhecido]"


# --- Motor principal ---

class RuleEngine:

    def __init__(self, rules_dir: str = config.RULES_DIR):
        self._rules_dir = rules_dir
        os.makedirs(self._rules_dir, exist_ok=True)


    # --- Conversão e gestão ---

    def converter_regra(self, texto_natural: str) -> dict:
        """
        Converte uma regra em linguagem natural para o schema JSON via LLM.
         Não persiste — útil para pré-visualização antes de adicionar.
        """
        return _converter_regra_com_llm(texto_natural)

    def adicionar_regra(self, texto_natural: str) -> dict:
        """
        Converte e persiste uma nova regra. Devolve a regra criada,
         incluindo validation.ambiguities para o gestor resolver se necessário.
        """
        regra = _converter_regra_com_llm(texto_natural)
        _guardar_regra(regra)
        return regra

    def eliminar_regra(self, rule_id: str) -> bool:
        """Remove uma regra pelo ID. Devolve True se eliminada, False se não existia."""
        return _eliminar_regra(rule_id)

    def recarregar(self) -> list[dict]:
        """Recarrega e devolve todas as regras do disco."""
        return _carregar_todas_as_regras()


    # --- Aplicação pós-inspeção ---

    def aplicar(self, resultado_llm: dict) -> dict:
        """
        Avalia todas as regras persistidas contra o resultado de inspeção.
         Acrescenta rule_engine_audit e rule_engine_notifications ao resultado devolvido.
        """
        resultado    = copy.deepcopy(resultado_llm)
        regras       = _carregar_todas_as_regras()
        auditoria    = []
        notificacoes = []

        for regra in regras:
            rule_id    = regra.get("rule_id", "?")
            conditions = regra.get("conditions", {})

            disparou = _avaliar_conditions(conditions, resultado)
            logger.debug(f"{rule_id} → {'DISPAROU' if disparou else 'não disparou'}")

            entrada = {
                "rule_id":     rule_id,
                "description": regra.get("description", ""),
                "fired":       disparou,
            }

            if disparou:
                notificacao = _gerar_notificacao(regra, resultado)
                alert_level = regra.get("action", {}).get("alert_level", "info")

                entrada["alert_level"]  = alert_level
                entrada["notification"] = notificacao

                notificacoes.append({
                    "rule_id":     rule_id,
                    "alert_level": alert_level,
                    "message":     notificacao,
                })
                logger.info(f"{rule_id} DISPAROU [{alert_level.upper()}] → {notificacao}")

            auditoria.append(entrada)

        resultado["rule_engine_audit"]         = auditoria
        resultado["rule_engine_notifications"] = notificacoes

        ids_disparadas = [e["rule_id"] for e in auditoria if e["fired"]]
        if ids_disparadas:
            logger.info(f"{len(ids_disparadas)} regra(s) disparada(s): " + ", ".join(ids_disparadas))

        return resultado

    def testar_regra(self, rule_id: str, resultado: dict) -> dict:
        """
        Testa uma regra específica contra um resultado de inspeção sem a persistir.
         Útil para validar antes de activar.
        """
        regras = _carregar_todas_as_regras()
        regra  = next((r for r in regras if r.get("rule_id") == rule_id), None)

        if regra is None:
            raise ValueError(f"Regra não encontrada: {rule_id!r}")

        disparou    = _avaliar_conditions(regra.get("conditions", {}), resultado)
        notificacao = _gerar_notificacao(regra, resultado) if disparou else None

        return {
            "rule_id":      rule_id,
            "fired":        disparou,
            "alert_level":  regra.get("action", {}).get("alert_level") if disparou else None,
            "notification": notificacao,
        }


    # --- Utilitários ---

    def listar_regras(self) -> list[dict]:
        """Devolve um resumo de todas as regras persistidas."""
        return [
            {
                "rule_id":     r.get("rule_id"),
                "created_at":  r.get("created_at"),
                "description": r.get("description"),
                "alert_level": r.get("action", {}).get("alert_level"),
                "is_valid":    r.get("validation", {}).get("is_valid", True),
                "ambiguities": r.get("validation", {}).get("ambiguities", []),
            }
            for r in _carregar_todas_as_regras()
        ]


# --- CLI ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rule Engine — gestão de regras de inspeção")
    group  = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--adicionar",
        metavar="REGRA",
        help="Converte e persiste uma nova regra em linguagem natural.",
    )
    group.add_argument(
        "--listar",
        action="store_true",
        help="Lista todas as regras persistidas.",
    )
    group.add_argument(
        "--eliminar",
        metavar="RULE_ID",
        help="Elimina uma regra pelo ID.",
    )
    group.add_argument(
        "--testar",
        metavar="RULE_ID",
        help="Testa uma regra contra um resultado de inspeção (requer --resultado).",
    )
    parser.add_argument(
        "--resultado",
        metavar="FICHEIRO_JSON",
        help="Caminho para o ficheiro JSON do resultado de inspeção (usado com --testar).",
    )
    args   = parser.parse_args()
    engine = RuleEngine()

    if args.adicionar:
        try:
            regra = engine.adicionar_regra(args.adicionar)
            print(json.dumps(regra, ensure_ascii=False, indent=2))

            ambiguidades = regra.get("validation", {}).get("ambiguities", [])
            if ambiguidades:
                print("\nAmbiguidades detectadas — clarifica com o gestor antes de activar:")
                for a in ambiguidades:
                    print(f"  • {a}")

        except Exception as e:
            print(f"Erro: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.listar:
        regras = engine.listar_regras()
        if not regras:
            print("Nenhuma regra encontrada.")
        else:
            print(json.dumps(regras, ensure_ascii=False, indent=2))

    elif args.eliminar:
        eliminada = engine.eliminar_regra(args.eliminar)
        if eliminada:
            print(f"Regra {args.eliminar} eliminada com sucesso.")
        else:
            print(f"Regra {args.eliminar} não encontrada.", file=sys.stderr)
            sys.exit(1)

    elif args.testar:
        if not args.resultado:
            print("Erro: --testar requer --resultado <ficheiro_json>.", file=sys.stderr)
            sys.exit(1)
        if not os.path.exists(args.resultado):
            print(f"Erro: ficheiro não encontrado: {args.resultado}", file=sys.stderr)
            sys.exit(1)
        try:
            with open(args.resultado, "r", encoding="utf-8") as f:
                resultado = json.load(f)
            saida = engine.testar_regra(args.testar, resultado)
            print(json.dumps(saida, ensure_ascii=False, indent=2))
        except ValueError as e:
            print(f"Erro: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Erro inesperado: {e}", file=sys.stderr)
            sys.exit(1)
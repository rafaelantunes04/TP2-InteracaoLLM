import logging
import os
import re
import sys
import tempfile
from pathlib import Path

import streamlit as st

# streamlit run src/interface.py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

# --- Logging ---

os.makedirs(os.path.dirname(config.LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=config.LOG_FILE,
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger(__name__)


# --- Configuração da página ---

st.set_page_config(
    page_title="Retail Vision Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.status-ok   { color: #22c55e; font-weight: 700; }
.status-warn { color: #f59e0b; font-weight: 700; }
.status-crit { color: #ef4444; font-weight: 700; }
.sev-high    { color: #ef4444; }
.sev-medium  { color: #f59e0b; }
.sev-low     { color: #6b7280; }
</style>
""", unsafe_allow_html=True)


# --- Componentes partilhados ---

@st.cache_resource(show_spinner="A inicializar motor de regras…")
def _get_rule_engine():
    """Devolve uma instância partilhada do RuleEngine (inicializada uma única vez)."""
    from rule_engine import RuleEngine
    return RuleEngine()


@st.cache_resource(show_spinner="A inicializar memória RAG…")
def _get_rag():
    """Devolve uma instância partilhada do RAGMemory, ou None se indisponível."""
    try:
        from rag_memory import RAGMemory
        return RAGMemory()
    except Exception as e:
        logger.warning(f"RAGMemory indisponível: {e}")
        return None


# --- Estado de sessão ---

def _init_estado() -> None:
    """Garante que todas as chaves necessárias existem em st.session_state."""
    defaults = {
        "mensagens":        [],     # [{role, content}]
        "caminho_imagem":   None,   # imagem carregada pelo utilizador
        "ultima_inspecao":  None,   # último resultado de inspeção (dict)
    }
    for chave, valor in defaults.items():
        if chave not in st.session_state:
            st.session_state[chave] = valor


# --- Apresentação de resultados ---

def _badge_status(status: str) -> str:
    """Devolve HTML com o badge colorido do status da inspeção."""
    cls  = {"ok": "status-ok", "warning": "status-warn", "critical": "status-crit"}.get(status, "")
    icon = {"ok": "✅",        "warning": "⚠️",           "critical": "🔴"          }.get(status, "❓")
    return f'<span class="{cls}">{icon} {status.upper()}</span>'


def _mostrar_inspecao(inspecao: dict) -> None:
    """Apresenta o resultado de uma inspeção de forma estruturada."""
    status   = inspecao.get("overall_status", "?")
    iid      = inspecao.get("inspection_id", "N/A")
    zona     = inspecao.get("zone_id", "N/A")
    fill     = inspecao.get("shelf_fill_rate", 0.0)
    ts       = inspecao.get("timestamp", "")[:16].replace("T", " ")
    issues   = inspecao.get("issues", [])
    notifs   = inspecao.get("rule_engine_notifications", [])

    st.markdown(
        f"**Inspeção** `{iid}` &nbsp;|&nbsp; Zona **{zona}** &nbsp;|&nbsp; {ts}"
        f" &nbsp;|&nbsp; {_badge_status(status)}",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Fill Rate",         f"{fill:.0%}")
    col2.metric("Issues",            len(issues))
    col3.metric("Alertas de Regras", len(notifs))

    if issues:
        st.markdown("**Problemas detectados:**")
        for iss in issues:
            sev     = iss.get("severity", "low")
            sev_cls = {"high": "sev-high", "medium": "sev-medium"}.get(sev, "sev-low")
            icon    = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(sev, "⚪")
            st.markdown(
                f'{icon} <span class="{sev_cls}">[{sev.upper()}]</span> '
                f'`{iss.get("type","?")}` @ *{iss.get("location","?")}*  \n'
                f'{iss.get("description","")}',
                unsafe_allow_html=True,
            )

    if notifs:
        st.markdown("**Regras disparadas:**")
        for n in notifs:
            icon = {"high": "🚨", "medium": "⚠️"}.get(n.get("alert_level", ""), "ℹ️")
            st.markdown(f"{icon} `{n.get('rule_id','?')}` — {n.get('message','')}")

    raciocinio = inspecao.get("model_reasoning", "")
    if raciocinio:
        with st.expander("🧠 Raciocínio do modelo", expanded=False):
            st.markdown(raciocinio)


def _mostrar_regras(regras: list[dict]) -> None:
    """Apresenta a lista de regras como cards expansíveis."""
    if not regras:
        st.info("Nenhuma regra definida. Usa `add rule \"...\"` para criar uma.")
        return

    for r in regras:
        rid   = r.get("rule_id", "?")
        desc  = r.get("description", "?")
        nivel = r.get("action", {}).get("alert_level", "info")
        icon  = {"high": "🚨", "medium": "⚠️"}.get(nivel, "ℹ️")
        amb   = r.get("validation", {}).get("ambiguities", [])

        with st.expander(f"{icon} `{rid}` — {desc[:80]}", expanded=False):
            st.json(r)
            if amb:
                st.warning("⚠️ Ambiguidades: " + "; ".join(str(a) for a in amb))


# --- Parsing de comandos ---

# Padrões regex, do mais específico para o mais geral
_PADROES_CMD: list[tuple[str, str]] = [
    ("inspecionar_tudo",    r"^inspect\s+all\s+--images-dir\s+(.+)$"),
    ("inspecionar_imagem",  r"^inspect\s+(\S+)\s+--image\s+(.+)$"),
    ("inspecionar_zona",    r"^inspect\s+(\S+)$"),
    ("adicionar_regra",     r"^add\s+rule\s+[\"'](.+)[\"']$"),
    ("adicionar_regra",     r"^add\s+rule\s+(.+)$"),
    ("listar_regras",       r"^list\s+rules?$"),
    ("eliminar_regra",      r"^delete\s+rule\s+(\S+)$"),
    ("testar_regra_imagem", r"^test\s+rule\s+(\S+)\s+--image\s+(.+)$"),
    ("testar_regra",        r"^test\s+rule\s+(\S+)$"),
    ("historico",           r"^history\s+[\"'](.+)[\"']$"),
    ("historico",           r"^history\s+(.+)$"),
    ("comparar",            r"^compare\s+(\S+)\s+(\S+)(?:\s+--period\s+[\"'](.+)[\"'])?$"),
    ("relatorio_zona",      r"^report\s+--zone\s+(\S+)(?:\s+--period\s+[\"'](.+)[\"'])?$"),
    ("relatorio_sessao",    r"^report\s+--session\s+(\S+)(?:\s+--zone\s+(\S+))?$"),
    ("relatorio",           r"^report$"),
    ("quota",               r"^(?:quota|estado|status)$"),
    ("ajuda",               r"^(?:help|ajuda|\?)$"),
    ("limpar_cache",        r"^(?:limpar[_\s]cache|clear[_\s]cache)$"),
]


def _parsear(cmd: str) -> tuple[str, tuple]:
    """Devolve (ação, grupos) para o primeiro padrão que corresponda ao comando."""
    txt = cmd.strip()
    for acao, padrao in _PADROES_CMD:
        m = re.match(padrao, txt, re.IGNORECASE)
        if m:
            return acao, m.groups()
    return "consulta_livre", (txt,)


# --- Comandos ---

def _cmd_inspecionar(caminho: str, zona: str) -> None:
    """Inspecciona uma imagem e aplica as regras activas."""
    from shelf_inspector import inspecionar_prateleira

    if not os.path.isfile(caminho):
        st.error(f"Ficheiro não encontrado: `{caminho}`")
        return

    with st.spinner(f"A analisar `{os.path.basename(caminho)}`…"):
        try:
            resultado = inspecionar_prateleira(caminho, id_zona=zona)
        except RuntimeError as e:
            st.error(str(e))
            return
        except Exception as e:
            logger.exception("Erro em inspecionar_prateleira")
            st.error(f"Erro durante a inspeção: {e}")
            return

    # Aplica o rule engine e indexa no RAG
    try:
        resultado = _get_rule_engine().aplicar(resultado)
    except Exception as e:
        logger.warning(f"Rule engine falhou: {e}")

    try:
        rag = _get_rag()
        if rag:
            rag.indexar_inspecao(resultado)
    except Exception as e:
        logger.warning(f"Indexação RAG falhou: {e}")

    st.session_state.ultima_inspecao = resultado
    _mostrar_inspecao(resultado)


def _cmd_inspecionar_tudo(dir_imagens: str) -> None:
    """Inspecciona todas as imagens de um directório sequencialmente."""
    from shelf_inspector import inspecionar_prateleira

    if not os.path.isdir(dir_imagens):
        st.error(f"Directório não encontrado: `{dir_imagens}`")
        return

    exts    = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    imagens = sorted(p for p in Path(dir_imagens).iterdir() if p.suffix.lower() in exts)

    if not imagens:
        st.warning(f"Nenhuma imagem encontrada em `{dir_imagens}`.")
        return

    st.markdown(f"**{len(imagens)} imagem(ns) encontrada(s). A processar…**")
    progresso = st.progress(0, text="A iniciar…")
    re_engine = _get_rule_engine()
    rag       = _get_rag()
    erros     = 0

    for idx, caminho in enumerate(imagens):
        zona = f"Z_S{idx + 1}"
        progresso.progress((idx + 1) / len(imagens), text=f"[{idx+1}/{len(imagens)}] {caminho.name}")

        try:
            res = inspecionar_prateleira(str(caminho), id_zona=zona)
            res = re_engine.aplicar(res)
            if rag:
                rag.indexar_inspecao(res)
            st.session_state.ultima_inspecao = res
        except RuntimeError as e:
            st.warning(f"`{caminho.name}`: {e}")
            erros += 1
            continue
        except Exception as e:
            logger.exception(f"Erro ao processar {caminho.name}")
            st.warning(f"`{caminho.name}`: erro inesperado — {e}")
            erros += 1
            continue

        icone = "✅" if res.get("overall_status") == "ok" else "⚠️"
        with st.expander(f"{icone} {caminho.name} — {zona}", expanded=False):
            _mostrar_inspecao(res)

    progresso.empty()
    st.success(f"Concluído: {len(imagens) - erros} inspecionada(s), {erros} erro(s).")


def _cmd_adicionar_regra(texto: str) -> None:
    """Converte e persiste uma nova regra a partir de linguagem natural."""
    with st.spinner("A converter regra com LLM…"):
        try:
            regra = _get_rule_engine().adicionar_regra(texto)
        except Exception as e:
            logger.exception("Erro ao adicionar regra")
            st.error(f"Não foi possível criar a regra: {e}")
            return

    st.success(f"✅ Regra criada: `{regra.get('rule_id', '?')}`")

    amb = regra.get("validation", {}).get("ambiguities", [])
    if amb:
        st.warning("⚠️ O sistema detectou ambiguidades:\n" + "\n".join(f"- {a}" for a in amb))

    with st.expander("Detalhes da regra", expanded=False):
        st.json(regra)


def _cmd_listar_regras() -> None:
    """Lista todas as regras persistidas."""
    regras = _get_rule_engine().recarregar()
    st.markdown(f"**{len(regras)} regra(s) definida(s):**")
    _mostrar_regras(regras)


def _cmd_eliminar_regra(rule_id: str) -> None:
    """Remove uma regra pelo ID."""
    ok = _get_rule_engine().eliminar_regra(rule_id)
    if ok:
        st.success(f"Regra `{rule_id}` eliminada.")
    else:
        st.warning(f"Regra `{rule_id}` não encontrada.")


def _cmd_testar_regra(rule_id: str, caminho: str | None = None) -> None:
    """Testa uma regra contra a última inspeção ou contra uma imagem fornecida."""
    if caminho:
        from shelf_inspector import inspecionar_prateleira
        if not os.path.isfile(caminho):
            st.error(f"Ficheiro não encontrado: `{caminho}`")
            return
        with st.spinner(f"A inspecionar `{os.path.basename(caminho)}`…"):
            try:
                resultado = inspecionar_prateleira(caminho)
            except Exception as e:
                st.error(f"Erro na inspeção: {e}")
                return
    elif st.session_state.ultima_inspecao:
        resultado = st.session_state.ultima_inspecao
        st.info(f"A usar última inspeção: `{resultado.get('inspection_id','?')}`")
    else:
        st.warning("Sem inspeção disponível. Faz primeiro `inspect <zona> --image <caminho>` ou fornece `--image`.")
        return

    try:
        resultado_teste = _get_rule_engine().testar_regra(rule_id, resultado)
    except ValueError as e:
        st.error(str(e))
        return
    except Exception as e:
        logger.exception("Erro ao testar regra")
        st.error(f"Erro ao testar regra: {e}")
        return

    disparou = resultado_teste.get("fired", False)
    icon     = "🚨" if disparou else "✅"
    estado   = "DISPAROU" if disparou else "não disparou"
    st.markdown(f"{icon} **Regra `{rule_id}` {estado}**")

    notif = resultado_teste.get("notification")
    if notif:
        st.info(notif)

    with st.expander("Detalhes do teste", expanded=False):
        st.json(resultado_teste)


def _cmd_historico(query: str) -> None:
    """Consulta o histórico de inspeções via RAG."""
    rag = _get_rag()
    if rag is None:
        st.warning("RAG não disponível. Verifica a configuração do ChromaDB.")
        return

    with st.spinner("A pesquisar no histórico…"):
        try:
            res = rag.consultar(query)
        except Exception as e:
            logger.exception("Erro na consulta RAG")
            st.error(f"Erro ao consultar histórico: {e}")
            return

    st.markdown(res.get("answer", "(sem resposta)"))

    refs = res.get("retrieved_inspections", [])
    if refs:
        with st.expander(f"Fontes consultadas ({len(refs)})", expanded=False):
            for r in refs:
                data = str(r.get("date", r.get("timestamp", "?")))[:10]
                st.markdown(f"- `{r.get('inspection_id','?')}` — {data}")


def _cmd_comparar(zona1: str, zona2: str, periodo: str | None) -> None:
    """Compara duas zonas via consulta RAG."""
    rag = _get_rag()
    if rag is None:
        st.warning("RAG não disponível.")
        return

    periodo_str = f" no período '{periodo}'" if periodo else ""
    query = (
        f"Compara o desempenho da zona {zona1} com a zona {zona2}{periodo_str}. "
        "Resume os problemas mais frequentes, fill rates e diferenças principais entre as duas zonas."
    )

    with st.spinner(f"A comparar {zona1} ↔ {zona2}…"):
        try:
            res = rag.consultar(query, k=6)
        except Exception as e:
            st.error(f"Erro na comparação: {e}")
            return

    st.markdown(f"### Comparação {zona1} ↔ {zona2}")
    st.markdown(res.get("answer", "(sem dados suficientes)"))


def _cmd_relatorio(sessao: str = "today", zona: str | None = None) -> None:
    """Gera um relatório em Markdown para a sessão e zona indicadas."""
    from report_generator import gerar_relatorio, _inspecoes_da_sessao

    with st.spinner("A carregar inspeções…"):
        inspecoes = _inspecoes_da_sessao(sessao)

    if zona:
        inspecoes = [i for i in inspecoes if i.get("zone_id") == zona]

    if not inspecoes:
        label = f"sessão='{sessao}'" + (f", zona='{zona}'" if zona else "")
        st.warning(f"Nenhuma inspeção encontrada para {label}.")
        return

    with st.spinner(f"A gerar relatório para {len(inspecoes)} inspeção(ões)…"):
        try:
            relatorio = gerar_relatorio(inspecoes, rag_memory=_get_rag())
        except Exception as e:
            logger.exception("Erro ao gerar relatório")
            st.error(f"Erro ao gerar relatório: {e}")
            return

    st.markdown(relatorio)
    st.download_button(
        "⬇️ Descarregar relatório (.md)",
        data=relatorio,
        file_name=f"report_{sessao}{'_' + zona if zona else ''}.md",
        mime="text/markdown",
    )


def _cmd_quota() -> None:
    """Mostra o estado actual da quota da API Gemini."""
    from shelf_inspector import obter_estado_quota
    try:
        q = obter_estado_quota()
    except Exception:
        st.info("Estado da quota indisponível.")
        return

    usado     = q.get("pedidos_hoje", 0)
    maximo    = q.get("max_por_dia", config.MAX_REQUESTS_PER_DAY)
    restantes = max(0, maximo - usado)

    col1, col2, col3 = st.columns(3)
    col1.metric("Usados hoje", usado)
    col2.metric("Máximo diário", maximo)
    col3.metric("Restantes", restantes)

    if restantes == 0:
        st.error("⛔ Quota diária esgotada. Novas inspeções só amanhã (UTC).")
    elif restantes < maximo * 0.2:
        st.warning(f"⚠️ Quota quase esgotada ({restantes} pedido(s) restante(s)).")


def _cmd_ajuda() -> None:
    """Apresenta os comandos disponíveis."""
    st.markdown("""
### Comandos disponíveis

**Inspeção**
```
inspect <zona> --image <caminho>
inspect all --images-dir <directório>
```
**Regras**
```
add rule "<descrição em linguagem natural>"
list rules
delete rule <RULE_ID>
test rule <RULE_ID>
test rule <RULE_ID> --image <caminho>
```
**Histórico (RAG)**
```
history "<pergunta>"
compare <zona1> <zona2>
compare <zona1> <zona2> --period "<período>"
```
**Relatórios**
```
report
report --session today
report --session <YYYY-MM-DD>
report --zone <zona>
```
**Sistema**
```
quota
help
```
Também podes **carregar uma imagem** pela barra lateral e usar `inspect <zona>` para a analisar.
    """)


def _cmd_consulta_livre(texto: str) -> None:
    """Rota para o RAG qualquer input que não corresponda a um comando conhecido."""
    rag = _get_rag()
    if rag is None:
        st.info("Não reconheci esse comando. Tenta `help` para ver os comandos disponíveis.")
        return

    with st.spinner("A pesquisar…"):
        try:
            res = rag.consultar(texto)
        except Exception as e:
            logger.warning(f"Consulta livre falhou: {e}")
            st.info("Não reconheci esse comando. Tenta `help` para ver os comandos disponíveis.")
            return

    st.markdown(res.get("answer", "(sem resposta)"))


# --- Dispatcher ---

def _despachar(cmd: str, caminho_imagem: str | None) -> None:
    """Faz parse do comando e chama o handler correspondente."""
    acao, grupos = _parsear(cmd)
    logger.info(f"Comando: {acao!r} | args: {grupos}")

    if acao == "inspecionar_imagem":
        zona, caminho = grupos
        _cmd_inspecionar(caminho.strip(), zona.strip())

    elif acao == "inspecionar_zona":
        zona = grupos[0].strip()
        if caminho_imagem:
            _cmd_inspecionar(caminho_imagem, zona)
        else:
            st.warning("Fornece `--image <caminho>` ou carrega uma imagem pela barra lateral.")

    elif acao == "inspecionar_tudo":
        _cmd_inspecionar_tudo(grupos[0].strip())

    elif acao == "adicionar_regra":
        _cmd_adicionar_regra(grupos[0].strip())

    elif acao == "listar_regras":
        _cmd_listar_regras()

    elif acao == "eliminar_regra":
        _cmd_eliminar_regra(grupos[0].strip())

    elif acao == "testar_regra_imagem":
        rule_id, caminho = grupos
        _cmd_testar_regra(rule_id.strip(), caminho.strip())

    elif acao == "testar_regra":
        _cmd_testar_regra(grupos[0].strip(), caminho_imagem)

    elif acao == "historico":
        _cmd_historico(grupos[0].strip())

    elif acao == "comparar":
        zona1, zona2, periodo = grupos
        _cmd_comparar(zona1.strip(), zona2.strip(), periodo)

    elif acao == "relatorio_sessao":
        sessao, zona = grupos
        _cmd_relatorio(sessao=(sessao or "today").strip(), zona=(zona or "").strip() or None)

    elif acao == "relatorio_zona":
        zona, _ = grupos
        _cmd_relatorio(zona=zona.strip())

    elif acao == "relatorio":
        _cmd_relatorio()

    elif acao == "quota":
        _cmd_quota()

    elif acao == "ajuda":
        _cmd_ajuda()

    elif acao == "limpar_cache":
        from shelf_inspector import limpar_cache
        n = limpar_cache()
        st.success(f"Cache limpa: {n} ficheiro(s) removido(s).")

    else:
        _cmd_consulta_livre(grupos[0] if grupos else cmd)


# --- Sidebar ---

def _sidebar() -> str | None:
    """Renderiza a barra lateral e devolve o caminho da imagem carregada (ou None)."""
    st.sidebar.title("🔍 Retail Vision")
    st.sidebar.caption("Intelligent Shelf Inspection")
    st.sidebar.divider()

    # Upload de imagem de prateleira
    st.sidebar.subheader("📷 Carregar imagem")
    upload = st.sidebar.file_uploader(
        "Arrasta uma imagem de prateleira",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed",
    )

    caminho_upload: str | None = None

    if upload is not None:
        # Guarda em ficheiro temporário identificado pelo nome + tamanho
        chave_tmp = f"_tmp_{upload.name}_{upload.size}"
        if chave_tmp not in st.session_state:
            sufixo = Path(upload.name).suffix
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=sufixo)
            tmp.write(upload.read())
            tmp.flush()
            st.session_state[chave_tmp] = tmp.name

        caminho_upload = st.session_state[chave_tmp]
        st.sidebar.image(caminho_upload, caption=upload.name, use_container_width=True)
        st.sidebar.caption("Usa `inspect <zona>` para analisar esta imagem.")

    st.sidebar.divider()

    # Estado resumido da quota
    st.sidebar.subheader("📊 Quota da API")
    try:
        from shelf_inspector import obter_estado_quota
        q      = obter_estado_quota()
        usado  = q.get("pedidos_hoje", 0)
        maximo = q.get("max_por_dia", config.MAX_REQUESTS_PER_DAY)
        st.sidebar.progress(usado / maximo if maximo else 0, text=f"{usado}/{maximo} pedidos hoje")
    except Exception:
        st.sidebar.caption("Quota indisponível")

    st.sidebar.divider()

    # Acções rápidas
    st.sidebar.subheader("⚡ Acções rápidas")
    col_a, col_b = st.sidebar.columns(2)

    if col_a.button("📋 Regras",   use_container_width=True):
        st.session_state._acao_rapida = "list rules"
    if col_b.button("📝 Relatório", use_container_width=True):
        st.session_state._acao_rapida = "report"
    if st.sidebar.button("❓ Ajuda", use_container_width=True):
        st.session_state._acao_rapida = "help"

    st.sidebar.divider()
    st.sidebar.caption(f"Modelo: `{config.MODEL_NAME}`")

    return caminho_upload


# --- Interface principal ---

def _interface() -> None:
    """Loop principal da interface conversacional."""
    _init_estado()
    caminho_imagem = _sidebar()

    st.title("Retail Vision Intelligence System")
    st.caption("Inspeção de prateleiras com LLM multimodal · Componente 5")

    # Apresenta histórico de mensagens da sessão
    for msg in st.session_state.mensagens:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Processa acção rápida desencadeada pelos botões da sidebar
    acao_rapida = st.session_state.pop("_acao_rapida", None)
    if acao_rapida:
        st.session_state.mensagens.append({"role": "user", "content": f"`{acao_rapida}`"})
        with st.chat_message("user"):
            st.markdown(f"`{acao_rapida}`")
        with st.chat_message("assistant"):
            _despachar(acao_rapida, caminho_imagem)
        st.session_state.mensagens.append({"role": "assistant", "content": "↑"})

    # Input do utilizador
    prompt = st.chat_input("Inspeciona, define regras, consulta histórico…  (escreve 'help' para ver comandos)")

    if prompt:
        st.session_state.mensagens.append({"role": "user", "content": f"`{prompt}`"})
        with st.chat_message("user"):
            st.markdown(f"`{prompt}`")
        with st.chat_message("assistant"):
            _despachar(prompt.strip(), caminho_imagem)
        st.session_state.mensagens.append({"role": "assistant", "content": "↑"})


if __name__ == "__main__":
    _interface()
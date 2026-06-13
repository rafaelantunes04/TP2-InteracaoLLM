# Interação com Modelos de Larga Escala - Trabalho Prático #2
**Rafael Antunes Nº55336**

## Retail Vision Intelligence - From Shelf Images to Operational Decisions

[Link repositório](https://github.com/rafaelantunes04/TP2-InteracaoLLM)

---

## Contextualização

Este trabalho dá continuidade ao TP1, mas muda de paradigma: em vez de partir de
eventos de tracking já estruturados, parte-se directamente de **imagens de
prateleiras de retalho** e usa-se um LLM multimodal (Gemini) para as transformar
em informação estruturada e acionável. O objetivo é construir um sistema completo
de "Retail Vision Intelligence" que vai desde a análise visual de uma prateleira
até à geração de relatórios em linguagem natural, passando por um motor de regras
configurável por linguagem natural e uma memória histórica baseada em RAG.

O sistema está organizado em cinco componentes principais:

1. **Shelf Inspector** - analisa uma imagem de prateleira com o Gemini e devolve
   um JSON estruturado de inspeção (issues, fill rate, produtos detectados,
   raciocínio do modelo).
2. **Rule Engine** - converte regras de negócio escritas em linguagem natural
   ("Avisa-me quando o fill rate cair abaixo de 60%") num schema JSON executável,
   e aplica-as a cada nova inspeção.
3. **RAG Memory** - indexa o histórico de inspeções num vectorstore (ChromaDB) e
   responde a perguntas sobre padrões passados.
4. **Report Generator** - agrega inspeções de uma sessão, cruza-as com o histórico
   do RAG e gera um Inspection Report em Markdown para gestores de loja.
5. **Interface conversacional** - uma aplicação Streamlit que expõe todos os
   componentes anteriores através de uma linha de comandos em linguagem natural.

Por trás destes componentes existe ainda uma camada de infraestrutura partilhada
(`config.py`, `cache_manager.py`, `rate_limiter.py`) que gere o cache de
resultados, a quota diária da API Gemini (plano gratuito do Google AI Studio) e o
backoff exponencial em caso de erro 429.

---

## Shelf Inspector - Análise Visual Estruturada

O `shelf_inspector.py` é o ponto de entrada do pipeline: recebe uma imagem de
prateleira, um `zone_id` e uma estratégia de prompting (`A`, `B` ou `C`), e
devolve um `inspection_record` em JSON com os campos definidos na Secção 4.2 do
enunciado - `overall_status`, `issues` (tipo, severidade, localização,
descrição, confiança, área afectada), `shelf_fill_rate`, `products_detected` e
`model_reasoning`.

### Estratégias de Prompting

Tal como no TP1, cada estratégia corresponde a um ficheiro de template em
`prompts/strategy_<x>.txt`, montado com o `schema_reminder.txt` (o schema JSON
de saída) e a imagem:

- **Estratégia A - Zero-Shot**: envia apenas as instruções, o schema e a imagem,
  sem qualquer exemplo. Serve como baseline.
- **Estratégia B - Chain-of-Thought** *(default)*: pede ao modelo para verbalizar
  o seu raciocínio antes de produzir o JSON final, preenchendo o campo
  `model_reasoning`. Este raciocínio é depois reutilizado pelo LLM-as-Judge para
  verificar alucinações (ver secção de Avaliação).
- **Estratégia C - Few-Shot**: acrescenta exemplos de inspeções bem formatadas
  ao prompt, para reforçar a consistência do schema de saída.

A escolha de tornar o raciocínio (estratégia B) um campo persistido e não apenas
um passo descartável foi central: sem ele, não seria possível avaliar
objectivamente se as `issues` reportadas têm suporte visual ou se são alucinadas.

### Cache e Gestão de Quota

Como o plano gratuito do Google AI Studio impõe limites apertados (15 pedidos por
minuto, 500 por dia - `config.py`), o `CacheManager` calcula uma chave a partir do
**MD5 da imagem + estratégia** e guarda o resultado completo da inspeção em disco.
Isto significa que:

- a mesma imagem com a mesma estratégia nunca volta a gastar quota;
- mudar a estratégia (ou a imagem) invalida automaticamente o cache, sem ser
  necessário gerir invalidações manualmente.

Antes de qualquer chamada à API, `inspecionar_prateleira` verifica primeiro o
cache e só depois a quota diária (`quota_esgotada`), levantando um `RuntimeError`
explícito se o limite já tiver sido atingido - em vez de deixar a chamada falhar
de forma opaca.

### Rate Limiting e Backoff

O `rate_limiter.py` mantém uma janela deslizante (`_timestamps_pedidos`) com os
pedidos do último minuto e bloqueia a execução (`time.sleep`) sempre que o limite
de 15 pedidos/minuto está prestes a ser excedido. Em caso de erro `429`, o
`chamar_com_backoff` aplica backoff exponencial (4s → 8s → 16s … até 120s, com um
máximo de 5 tentativas), o que tornou o pipeline robusto a picos de utilização sem
intervenção manual.

### Parsing e Robustez

Tal como no TP1, todas as respostas do modelo passam por `_extrair_json`, que lida
com blocos markdown (` ```json ... ``` `) e tenta vários fragmentos antes de
desistir. Quando o parse falha, em vez de propagar a excepção,
`_resultado_erro_parse` constrói um `inspection_record` válido com
`overall_status="critical"` e um `issue` do tipo `other` que documenta o erro -
garantindo que o resto do pipeline (rule engine, RAG, relatórios) nunca recebe um
registo malformado.

---

## Rule Engine - Regras de Negócio em Linguagem Natural

O `rule_engine.py` resolve um problema concreto de UX: gestores de loja não vão
escrever JSON. Em vez disso, escrevem frases como:

> "Na zona Z_S1, se não houver produtos de laticínios visíveis, é crítico."
> "Quando o fill rate cair abaixo de 60% entre as 10h e as 13h, avisa mas não é
> urgente."

A conversão para o schema JSON executável (`conditions`, `action`,
`validation`) é feita por uma única chamada ao Gemini, usando templates
(`rule_engine_prompt.txt` + `rule_engine_schema.txt`). O resultado é enriquecido
com `rule_id`, `created_at` e o texto original (`natural_language`), e persistido
em `data/rules/`.

### Avaliação de Condições

A função `_avaliar_conditions` é o "motor" propriamente dito: recebe as
`conditions` de uma regra e o `inspection_record` mais recente, e verifica, de
forma puramente determinística (sem chamadas ao LLM), seis tipos de filtro:

- `zone_filter` - restringe a regra a zonas específicas;
- `time_filter` - janela horária (em UTC) em que a regra é avaliada;
- `fill_rate_threshold` - dispara se o fill rate estiver abaixo do limite;
- `issue_types` - exige a presença de pelo menos um tipo de issue específico;
- `severity_threshold` - exige severidade mínima entre as issues detectadas;
- `location_filter` - exige que pelo menos uma issue mencione uma localização
  (ex.: "bottom").

Esta separação é deliberada: o LLM só é usado para **traduzir intenção** (texto →
JSON), nunca para **decidir** se uma regra dispara. A decisão é sempre
determinística e auditável, o que é essencial para um sistema de alertas - não
faria sentido um alerta crítico depender da aleatoriedade de um LLM.

### Auditoria e Notificações

Sempre que `RuleEngine.aplicar()` é chamado sobre um `inspection_record`, todas as
regras persistidas são avaliadas e o resultado é enriquecido com dois campos:

- `rule_engine_audit` - lista completa, incluindo regras que **não** dispararam,
  para permitir depuração ("porque é que esta regra não disparou?");
- `rule_engine_notifications` - apenas as regras que dispararam, com a mensagem
  já formatada a partir do template `notification_message` (placeholders como
  `{rule_id}`, `{zone_id}`, `{fill_rate}`, `{timestamp}`, etc.).

### Detecção de Ambiguidade

Uma parte importante do design foi tratar a ambiguidade como cidadã de primeira
classe. Regras como "Avisa-me quando a prateleira estiver vazia." ou "Avisa quando
houver problemas." não especificam zona, limiar nem tipo de issue. O prompt de
conversão pede ao LLM para preencher um campo `validation` com `is_valid` e
`ambiguities`, e essas ambiguidades são apresentadas ao gestor (`add rule` na
interface) antes de a regra ficar activa - em vez de o sistema "adivinhar" valores
por omissão que poderiam gerar falsos alertas (ou silêncio) sistemáticos.

---

## RAG Memory - Memória Histórica de Inspeções

O `rag_memory.py` implementa a memória de longo prazo do sistema, sobre um
ChromaDB persistente (`vectorstore/`) com embeddings multilingues
(`paraphrase-multilingual-MiniLM-L12-v2`).

### Duas Estratégias de Chunking

Para a comparação de Recall@3 pedida na Secção 6.5, foram implementadas duas
collections em paralelo, indexadas em simultâneo para cada inspeção:

- **`hibrido`**: o texto que recebe embedding é um **summary gerado pelo próprio
  Gemini** (`rag_summary_prompt.txt`), enriquecido com metadados estruturados
  (`zone_id`, `date`, `weekday`, `hour`, `overall_status`, `shelf_fill_rate`,
  `has_critical_issue`, `issue_types_str`, `n_issues`, …) usados para
  *pre-retrieval filtering* no ChromaDB.
- **`full_record`**: o `inspection_record` completo, serializado em JSON, como
  baseline sem qualquer processamento adicional.

A hipótese de base é que um summary semanticamente denso, com metadados
filtráveis, recupera contexto mais relevante do que despejar o JSON completo -
que contém muito "ruído" estrutural (chaves, formatação) que não ajuda a
similaridade semântica.

Caso a chamada ao Gemini para gerar o summary falhe, existe um fallback
determinístico (`_construir_summary_fallback`) que constrói uma frase a partir
dos campos estruturados - garantindo que a indexação nunca bloqueia por completo
por causa de uma falha pontual da API.

### Pre-Retrieval Filtering

Antes de fazer a query vetorial, `_extrair_filtros` usa o Gemini para extrair da
pergunta em linguagem natural filtros estruturados - `zone_id`, `days_back`,
`status` - que são depois traduzidos para uma cláusula `where` do ChromaDB
(`_construir_where`). Por exemplo, "Que zonas tiveram mais problemas nas últimas
2 semanas?" é traduzido para um filtro `date >= hoje - 14 dias`, combinado com a
busca semântica. Se a extração de filtros falhar, ou se a query com filtros não
devolver resultados, o sistema recua silenciosamente para uma busca puramente
semântica - nunca devolve "sem resultados" só porque a extração de filtros falhou.

### Síntese da Resposta

Os *k* chunks recuperados (por omissão, `RAG_DEFAULT_K = 3`) são concatenados em
contexto com o respectivo `inspection_id`, data e score de similaridade, e
passados a `_sintetizar_resposta`, que pede ao Gemini para responder à query
**apenas** com base nesse contexto. Esta separação entre *retrieval* e *síntese*
é o que permite depois avaliar **Faithfulness** (a resposta é fiel ao contexto
recuperado?) e **Answer Relevance** (a resposta responde à pergunta?) de forma
independente.

### Queries Pré-definidas

Foram implementadas as quatro queries do enunciado (Secção 6.4) como métodos
dedicados - `ultima_prateleira_vazia`, `zonas_com_mais_issues_planograma`,
`padroes_sexta_feira_tarde`, `regras_mais_disparadas` - que não são mais do que
wrappers sobre `consultar` com a pergunta já formulada, o que mantém a interface
conversacional simples (o utilizador pode escrever a pergunta livremente e cair
no mesmo caminho de código).

---

## Report Generator - De Inspeções a Relatórios de Gestão

O `report_generator.py` é o componente que fecha o ciclo "deteção → decisão". A
partir de uma lista de inspeções (de um ficheiro, ou de todas as inspeções de uma
sessão/zona), gera um **Inspection Report em Markdown** com cinco secções fixas:

1. **Sumário Executivo** - estado geral, urgência (normal / atenção / crítico),
   máximo 150 palavras;
2. **Problemas por Zona** - issues, fill rate actual vs. histórico do RAG, padrões
   recorrentes;
3. **Regras Disparadas** - ID, nível de alerta, condição que disparou, acção
   gerada (ou indicação explícita de que nenhuma disparou);
4. **Contexto Histórico Relevante** - referências `[INS_XXX - YYYY-MM-DD]`
   recuperadas pelo RAG;
5. **Recomendações** - no máximo 5 acções concretas, ordenadas por urgência, no
   formato `**Acção X:** [quem] deve [o quê] em [onde] [quando/prazo]`.

### Agregação e Contexto Histórico

Antes de chamar o LLM, `_stats_sessao` agrega estatísticas de alto nível (zonas
envolvidas, nº de issues críticos/warnings, fill rate médio), e `_contexto_rag`
consulta o RAG **uma vez por zona com issues**, perguntando especificamente por
"problemas históricos recorrentes na zona X". Isto evita uma única query genérica
que misturaria o histórico de todas as zonas, e permite à secção 2 do relatório
comparar directamente o estado actual de cada zona com o seu padrão histórico.

### Cache de Relatórios

Tal como o Shelf Inspector, o gerador de relatórios usa o `CacheManager` - a chave
é o hash MD5 da concatenação ordenada dos `inspection_id`s envolvidos. Isto evita
regenerar (e voltar a gastar quota com) o mesmo relatório se o utilizador pedir
`report --session today` várias vezes sem que existam novas inspeções.

### Modo `--sem-rag`

A flag `--sem-rag` permite gerar relatórios sem qualquer consulta ao RAG, o que
foi essencial durante o desenvolvimento e testes - sem isto, cada teste do
report generator implicaria também testar o RAG (e gastar quota adicional em
sínteses), o que tornava a iteração lenta.

---

## Avaliação (`evaluate.py` e `llm_judge.py`)

A avaliação segue, tal como no TP1, a filosofia **LLM-as-a-Judge**: em vez de
heurísticas rígidas de correspondência de texto, usa-se o próprio Gemini no papel
de avaliador para julgar qualidade semântica. O `llm_judge.py` define quatro
avaliadores especializados, cada um com o seu prompt dedicado:

- `avaliar_alucinacao` - verifica se as `descriptions` das issues reportadas são
  suportadas pelo `model_reasoning` (estratégia B);
- `avaliar_fidelidade` - verifica se a resposta do RAG é fiel ao contexto
  recuperado;
- `avaliar_relevancia` - verifica se a resposta do RAG responde efectivamente à
  query;
- `avaliar_relatorio` - avalia clareza, acionabilidade e completude de um
  Inspection Report, devolvendo também `pontos_fortes` e `pontos_fracos`.

Todos partilham o mesmo padrão: prompt que exige **apenas JSON**, parsing
defensivo (`_extrair_json` + heurística de fallback que procura um número junto
da palavra "score"), e um valor neutro (`0.5`) em caso de falha total - para que
uma falha pontual da API nunca derrube a avaliação completa.

### Meta-Análise Humano vs. Juiz

A função `avaliar_amostras` (Secção 9.3) implementa um passo extra: para cada
amostra anotada por um humano (`score_humano`, `notas_humanas`), o resultado do
LLM-as-Judge é confrontado com a anotação humana através de uma terceira chamada
ao LLM - `_meta_analisar` - que identifica se concordam, qual o `delta` entre
scores, e tenta apontar **enviesamentos do juiz**. O relatório final agrega a
**taxa de concordância** e o **delta médio** sobre todas as amostras, o que dá
uma medida de quão fiável é usar o LLM como substituto de um avaliador humano
neste domínio.

### Harness `evaluate.py`

O harness corre três blocos de avaliação independentes, cada um com a opção de
ser ignorado (`--skip-visual`, `--skip-rag`, `--skip-rules`):

**Análise Visual** - para cada imagem com `ground_truth.json`, corre
`inspecionar_prateleira` e calcula:

- *JSON Parse Rate* - % de respostas que produzem um `inspection_record` válido;
- *Issue Detection Rate* - recall dos tipos de issue face ao ground truth;
- *False Positive Rate* - % de issues reportadas que não existem no ground truth;
- *Severity Accuracy* - % de issues correctamente detectadas cuja severidade
  também coincide;
- *Hallucination Rate* - `1 − média(avaliar_alucinacao)` sobre todas as imagens.

**RAG** - sobre um conjunto de queries de referência (com `inspection_ids
relevantes` opcionais), calcula *Recall@3*, *Faithfulness* e *Answer Relevance*.
Quando não existe ground truth de relevância para uma query, é atribuída uma
pontuação neutra de recall (0.5), para não penalizar nem premiar artificialmente
queries exploratórias.

**Rule Engine** - sobre um conjunto fixo de seis regras sintéticas (três
ambíguas, três não-ambíguas, ver `_REGRAS_SINTETICAS`), calcula:

- *Rule Parse Rate* - % de regras convertidas com sucesso para o schema completo;
- *Rule Correctness* - % de regras não-ambíguas cujas `conditions` correspondem
  às esperadas (`_condicoes_correspondem`, verificação por subconjunto) **e**
  cuja execução contra uma inspeção sintética não levanta erros;
- *Ambiguity Detection* - % das regras ambíguas para as quais o LLM efectivamente
  marcou `is_valid=False` ou preencheu `ambiguities`.

O resultado final é impresso numa tabela e persistido em `evaluation_report.json`,
de forma análoga ao `metrics.json` do TP1 - permitindo comparar execuções ao longo
do tempo (por exemplo, mudar de estratégia A → B → C e ver o impacto nas métricas
visuais).

---

## Interface Conversacional (Streamlit)

A `interface.py` expõe todos os componentes acima através de uma única caixa de
chat, com um pequeno parser de comandos baseado em expressões regulares
(`_PADROES_CMD`) que cobre:

- `inspect <zona> --image <caminho>` / `inspect all --images-dir <dir>`;
- `add rule "..."`, `list rules`, `delete rule <ID>`, `test rule <ID> [--image ...]`;
- `history "..."`, `compare <zona1> <zona2> [--period "..."]`;
- `report`, `report --session <...>`, `report --zone <...>`;
- `quota`, `help`, `clear cache`.

Qualquer entrada que não corresponda a nenhum destes padrões cai no caso
`consulta_livre`, que é encaminhado directamente para o RAG - o que significa que
um gestor pode simplesmente **fazer uma pergunta em português corrente** ("Houve
algum problema na zona dos lacticínios esta semana?") sem precisar de saber a
sintaxe de nenhum comando.

Cada inspeção feita pela interface passa automaticamente pelo `RuleEngine.aplicar`
e é indexada no RAG - ou seja, a interface não é apenas uma camada de
visualização, é também o ponto onde os três componentes (inspector, rule engine,
RAG) ficam encadeados num único pipeline em tempo real. O estado da quota da API é
sempre visível na barra lateral, com avisos visuais quando está perto do limite
diário.

---

## Limitações Conhecidas do Sistema

- **Dependência de prompts externos.** Quase toda a lógica de "inteligência"
  (schemas, instruções, exemplos few-shot) vive em ficheiros de texto em
  `prompts/`, fora do código Python. Isto facilita iteração sem alterar código,
  mas torna o sistema frágil a alterações acidentais nesses ficheiros - não há
  validação de que um template tem todos os placeholders esperados antes de ser
  usado em produção.

- **Quota do plano gratuito como limitador estrutural.** Com 15 pedidos/minuto e
  500/dia, qualquer avaliação mais extensa (muitas imagens, muitas queries RAG,
  muitas amostras no LLM-as-Judge) demora consideravelmente, e o
  `evaluate.py` pode facilmente esgotar a quota diária a meio de uma corrida
  completa, deixando o relatório de avaliação incompleto.

- **Fallback de summary do RAG pode degradar Recall.** Quando a geração de
  summary pela LLM falha e se recorre ao `_construir_summary_fallback`, o texto
  indexado na collection `hibrido` é estruturalmente diferente (mais telegráfico,
  menos "narrativo") dos summaries gerados pelo modelo - o que pode introduzir
  inconsistência na qualidade do embedding entre inspeções indexadas em momentos
  diferentes.

- **Ambiguidade na detecção de regras depende inteiramente do LLM.** Embora a
  avaliação de condições seja determinística, a **classificação** de uma regra
  como ambígua ou não é feita por um único LLM call, sem segunda verificação -
  uma regra que o modelo classifica incorrectamente como válida pode ficar activa
  e gerar notificações com base em condições mal especificadas.

---

## Conclusão

### O que funcionou

A separação clara entre **interpretação por LLM** (converter linguagem natural em
estruturas) e **execução determinística** (avaliar condições, calcular
estatísticas, aplicar filtros) revelou-se, tal como no TP1, o ponto mais sólido do
sistema. O Rule Engine em particular beneficia muito disto: o LLM só precisa de
"acertar uma vez" na tradução da regra, e a partir daí o disparo de alertas é
100% reprodutível e auditável via `rule_engine_audit`.

A combinação de cache + quota + backoff exponencial, herdada e expandida a partir
da infra-estrutura do TP1, permitiu desenvolver e testar o pipeline inteiro
(inspector → rules → RAG → report) repetidamente sem esgotar a quota gratuita do
Google AI Studio, o que seria impraticável de outra forma.

A estratégia de duas collections paralelas no RAG (`hibrido` vs `full_record`)
também funcionou bem como ferramenta de comparação directa - ter ambas indexadas
desde o início, em vez de re-indexar tudo a posteriori, simplificou bastante a
avaliação de Recall@3.

### O que não funcionou

A latência continua a ser o principal problema: cada inspeção visual, cada
conversão de regra e cada query RAG implica pelo menos uma chamada ao Gemini, e a
geração de relatórios pode implicar várias chamadas adicionais (uma síntese RAG
por zona, mais a geração do relatório final). Numa sessão com várias zonas e
issues, o tempo de resposta da interface conversacional torna-se notório, e o
`evaluate.py` completo é lento o suficiente para que correr com `--skip-*`
desactivados se torne pouco prático em iterações rápidas de desenvolvimento.

A meta-análise humano-vs-juiz (`_meta_analisar`) também é mais frágil do que
gostaria: depende de uma terceira chamada ao LLM para "explicar" a discordância,
o que por vezes produz análises genéricas que pouco acrescentam ao simples
`delta` numérico já calculado.

### O que faria de diferente

Teria investido mais tempo num **validador estático dos ficheiros de prompt**
(verificar que todos os placeholders existem antes de qualquer chamada à API),
para apanhar erros de template em desenvolvimento em vez de em runtime.

No RAG, teria experimentado uma terceira estratégia de chunking - por exemplo,
indexar separadamente cada `issue` em vez de a inspeção completa - para perceber
se a granularidade mais fina melhora ainda mais o Recall@3 face ao `hibrido`.

Por fim, dado o peso que a quota do plano gratuito teve no ritmo de
desenvolvimento, teria adicionado desde o início um **modo de simulação** (mock
do cliente Gemini com respostas pré-gravadas) para permitir testar a lógica de
negócio do Rule Engine, RAG e Report Generator sem depender da API - reservando
as chamadas reais apenas para a avaliação final.

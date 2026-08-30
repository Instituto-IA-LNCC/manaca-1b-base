# Corpus Manacá — Manual Técnico de Extração
## Fase 1: Construção do Corpus PT-BR

**Elaborado por:** Bruno Leonardo Santos Menezes  
**Coordenação:** Prof. Fábio Porto  
**Instituição:** LNCC / Instituto de IA  
**Data:** Abril de 2026 | **Versão:** 0.2.0

---

> 🐳 **Fork `manaca-1b` — execução em Docker.** Este fork roda em containers
> Docker, não em `conda` + `screen` + SLURM. Regra de tradução dos comandos
> abaixo: onde se lê `conda activate manaca-corpus && python <script>`, use
> `docker compose run --rm corpus python <script>`; onde se lê
> `screen -S <fonte> ...`, use `docker compose run -d --name <fonte> corpus ...`
> (ou os atalhos do `Makefile`: `make gigaverbo`, `make dedup`, `make validate`).
> Guia completo: [`../docs/environment/setup-guide-docker-pt.md`](../docs/environment/setup-guide-docker-pt.md).

---

## Sumário

1. [Visão Geral da Estratégia](#1-visão-geral-da-estratégia)
2. [Pré-requisitos](#2-pré-requisitos)
3. [Estrutura dos Scripts](#3-estrutura-dos-scripts)
4. [Script 00 — Verificação do Ambiente](#4-script-00--verificação-do-ambiente)
5. [Script 01 — GigaVerbo](#5-script-01--gigaverbo)
6. [Script 02 — MADLAD-400](#6-script-02--madlad-400)
7. [Script 03 — FineWeb-2](#7-script-03--fineweb-2)
8. [Script 04 — HPLT 2.0](#8-script-04--hplt-20)
9. [Script 05 — Wikipedia PT-BR](#9-script-05--wikipedia-pt-br)
10. [Script 06 — Ulysses Tesemõ](#10-script-06--ulysses-tesemõ)
11. [Script 07 — Common Crawl](#11-script-07--common-crawl)
12. [Script 08 — Deduplicação Global](#12-script-08--deduplicação-global)
13. [Script 09 — Validação e Estatísticas](#13-script-09--validação-e-estatísticas)
14. [Fluxo de Execução](#14-fluxo-de-execução)
15. [Monitoramento](#15-monitoramento)
16. [Referências](#16-referências)

---

## 1. Visão Geral da Estratégia

A construção do corpus Manacá segue a metodologia estabelecida por três
trabalhos de referência máxima na área:

- **FineWeb** (Penedo et al., 2024, arXiv:2406.17557) — pipeline de filtragem
  de alta qualidade para dados web; o framework datatrove usado como base deste
  projeto foi desenvolvido especificamente para reproduzir e estender este trabalho.
- **Dolma** (Soldaini et al., 2024, arXiv:2402.00159) — arquitetura de corpus
  multi-fonte com deduplicação em duas passagens (within-source + cross-source).
- **Tucano** (de Lucena et al., 2024, arXiv:2411.07854) — referência específica
  para PT-BR; demonstra que o GigaVerbo como base produz modelos superiores ao
  treinamento em corpora multilíngues não curados.

### 1.1 Meta da Fase 1

| Métrica | Meta |
|---------|------|
| Volume total | >= 1 trilhão de tokens (1 TB tokens) |
| Idioma | Português do Brasil (PT-BR) |
| Licença | Compatível com uso comercial e acadêmico aberto |
| Formato de saída | Apache Parquet + compressão Zstandard |
| Rastreabilidade | Manifesto JSON por fonte com checksums SHA-256 |

### 1.2 Priorização das Fontes

| Tier | Critério | Fontes | Volume est. |
|------|----------|--------|-------------|
| 1 | Sem autenticação · licença aberta | GigaVerbo, MADLAD-400, FineWeb-2, HPLT 2.0, Wikipedia PT-BR, Ulysses Tesemõ | ~410-440 B tokens |
| 2 | Requer aprovação HuggingFace | CulturaX, Jabuticaba | ~340 B tokens |
| 3 | Construção própria via datatrove | Common Crawl snapshots 2024-2025 | ilimitado |

### 1.3 Pipeline de Qualidade (todas as fontes)

```
Fonte bruta
    |
    v
[LangID]        GlotLID / fastText — manter somente PT
    |           Threshold: score >= 0.65
    v
[Heurísticas]   Filtros Gopher (Rae et al. 2021, arXiv:2112.11446):
    |           min/max palavras · razão alfabética >= 0.50
    |           razão de símbolos <= 0.10 · stop words PT-BR
    v
[Dedup local]   MinHash LSH within-source
    |           Lee et al. (2022, arXiv:2107.06499)
    |           128 permutações · Jaccard threshold >= 0.80
    v
[Parquet/Zstd]  Shards de 50.000 documentos
                Schema: text · source · id · lang · score
    |
    v
[Dedup global]  MinHash LSH cross-source (Script 08)
    |
    v
Corpus final >= 1 TB tokens
```

### 1.4 Infraestrutura

| Componente | Especificação |
|------------|--------------|
| Runtime | Docker (imagem `manaca-corpus`, `docker/Dockerfile.corpus`) |
| CPU | ≥ 8 threads recomendado (I/O-bound) |
| RAM | ≥ 32 GiB recomendado |
| Armazenamento corpus | volume `WORK_DIR` — bind mount `./data` (ou NFS via `DATA_DIR`) |
| Python | 3.11 · imagem Docker (dependências pinadas em `requirements/corpus.txt`) |
| Framework principal | datatrove 0.9.0 (Penedo et al., 2024) |

---

## 2. Pré-requisitos

### 2.1 Ambiente (imagem Docker)

```bash
cp .env.example .env       # ajuste DATA_DIR, HF_TOKEN, ...
make build-corpus          # constrói a imagem manaca-corpus
# Um shell no container, se quiser inspecionar:
docker compose run --rm corpus /bin/sh
```

### 2.2 Volume de trabalho (bind mount)

O corpus é gravado no volume `WORK_DIR` (`/workspace/manaca-corpus` no
container), montado a partir de `DATA_DIR` no host (padrão `./data`). Garanta
que o diretório existe e é gravável — o container o cria automaticamente ao
iniciar. Para apontar a um NFS: `DATA_DIR=/caminho/do/nfs/manaca-corpus`.

### 2.3 Verificação completa (obrigatória antes de qualquer script)

```bash
make verify
# equivale a: docker compose run --rm corpus python corpus/scripts/00_verify_env.py
# Todos os itens críticos devem mostrar OK
```

---

## 3. Estrutura dos Scripts

```
corpus/
├── README.md                        <- Este manual técnico
├── configs/
│   └── manaca_corpus.yaml           <- Configuração centralizada
└── scripts/
    ├── 00_verify_env.py             <- Verificação do ambiente · executar primeiro
    ├── 01_acquire_gigaverbo.py      <- Tier 1 · P1 · 200B tokens · Apache 2.0
    ├── 02_acquire_madlad400.py      <- Tier 1 · P2 · ~80B tokens · Apache 2.0
    ├── 03_acquire_fineweb2.py       <- Tier 1 · P3 · ~150B tokens · ODC-By
    ├── 04_acquire_hplt2.py          <- Tier 1 · P4 · ~60B tokens · CC0
    ├── 05_acquire_wikipedia.py      <- Tier 1 · P5 · ~1B tokens · CC BY-SA
    ├── 06_acquire_ulysses.py        <- Tier 1 · P6 · ~10B tokens · Público
    ├── 07_cc_pipeline.py            <- Tier 3 · Common Crawl via datatrove
    ├── 08_global_dedup.py           <- Deduplicação cross-source MinHash LSH
    └── 09_validate_corpus.py        <- Validação e relatório estatístico final
```

### 3.1 Convenções aplicadas em todos os scripts

**Idempotência:** Cada script verifica shards já escritos antes de iniciar.
Re-executar nunca sobrescreve dados — retoma automaticamente do ponto de falha.

**Schema Parquet fixo** (idêntico em todos os scripts da suite):

```python
pa.schema([
    pa.field("text",   pa.string()),   # texto limpo
    pa.field("source", pa.string()),   # identificador da fonte
    pa.field("id",     pa.string()),   # id único do documento
    pa.field("lang",   pa.string()),   # código de idioma (GlotLID/fastText)
    pa.field("score",  pa.float32()),  # score de qualidade [0.0-1.0]
])
```

**Manifesto JSON:** Cada shard gravado atualiza atomicamente
$WORK_DIR/checkpoints/acquisition_manifest.json com metadados completos,
garantindo rastreabilidade para publicação aberta dos artefatos.

**Execução em container destacado (para jobs longos):**

```bash
# Container destacado (equivale ao screen; sobrevive ao fechamento do terminal):
docker compose run -d --name <fonte> corpus python corpus/scripts/NN_acquire_<fonte>.py
docker logs -f <fonte>
# ou, com os atalhos do Makefile: make gigaverbo && make logs SRC=gigaverbo
```

---

## 4. Script 00 — Verificação do Ambiente

**Arquivo:** `corpus/scripts/00_verify_env.py`  
**Tempo de execução:** ~1 minuto  
**Status:** ✅ Implementado e verificado em um servidor GPU Linux

### O que verifica

| Verificação | Critério de sucesso |
|-------------|---------------------|
| Python 3.11 | sys.version_info == (3, 11, x) |
| Pacotes críticos | 16 pacotes · import sem erro · versão mínima satisfeita |
| Parquet + Zstd | Escrita e leitura com schema completo de 5 campos |
| MinHash LSH | Jaccard idêntico = 1.0 · Jaccard diferente < 0.10 |
| fastText LangID | Identificação de texto PT-BR |
| HuggingFace Hub | HTTP 200 · latência < 5s |
| Volume WORK_DIR | Existência e permissão de escrita (aviso, não erro crítico) |
| Workspace dirs | Existência de 7 subdiretórios em $WORK_DIR |
| Disco | >= 50 GB disponíveis |

### Execução

```bash
make verify
# equivale a:
docker compose run --rm corpus python corpus/scripts/00_verify_env.py
```

---

## 5. Script 01 — GigaVerbo

**Arquivo:** `corpus/scripts/01_acquire_gigaverbo.py`  
**Prioridade:** 1 — primeira fonte a ser extraída  
**Status:** ✅ Implementado

### Justificativa Científica

O GigaVerbo é o maior corpus público monolíngue PT-BR. De Lucena et al. (2024)
demonstraram que treinar o modelo Tucano-1.1B exclusivamente neste corpus produz
desempenho superior ao mC4 e CC-100 em todos os benchmarks PT-BR avaliados,
comprovando que qualidade e especificidade superam volume bruto.

Referência: de Lucena, R. et al. (2024). *Tucano*. arXiv:2411.07854.

### Especificações

| Atributo | Valor |
|----------|-------|
| HuggingFace repo | TucanoBR/GigaVerbo |
| Tokens estimados | 200 bilhões |
| Tamanho comprimido | ~80-100 GB (Parquet + Zstd) |
| Licença | Apache 2.0 |
| Autenticação | Não necessária |
| Filtros | Sanidade básica (corpus já curado pela equipe TucanoBR) |
| Tempo estimado (ha4) | 8-16 horas |

### Execução

```bash
make verify

# Container destacado (equivale ao screen):
make gigaverbo
make logs SRC=gigaverbo
# equivale a:
#   docker compose run -d --name gigaverbo corpus python corpus/scripts/01_acquire_gigaverbo.py
#   docker logs -f gigaverbo
```

---

## 6. Script 02 — MADLAD-400

**Arquivo:** `corpus/scripts/02_acquire_madlad400.py`  
**Status:** ✅ Implementado

### Justificativa Científica

O MADLAD-400 (Kudugunta et al., 2024) cobre 419 línguas com pipeline rigoroso
de deduplicação. A partição PT (~80B tokens) complementa o GigaVerbo com
domínios e períodos históricos distintos (2013-2023).

**Diferencial:** filtro de variante PT-BR vs PT-PT via fastText lid.176.bin
(threshold 0.65) com segunda camada de marcadores lexicais.

| Atributo | Valor |
|----------|-------|
| HuggingFace repo | allenai/MADLAD-400 · config=pt |
| Tokens PT-BR estimados | ~50-60 bilhões |
| Licença | Apache 2.0 |
| Tempo estimado (ha4) | 6-12 horas |

---

## 7. Script 03 — FineWeb-2

**Arquivo:** `corpus/scripts/03_acquire_fineweb2.py`  
**Status:** ✅ Implementado

### Justificativa Científica

O FineWeb-2 (Penedo et al., 2024) é o corpus web de maior qualidade disponível
publicamente, resultado do pipeline Gopher + C4 + FineWeb aplicado a 96 snapshots
CC (2013-2024). Incluí-lo evita semanas de processamento CC para chegar à mesma
qualidade. O Script 07 complementa com snapshots 2025 não cobertos.

**Diferencial:** SnapshotTracker rastreia os 96 snapshots CC cobertos para evitar
sobreposição com o Script 07 (salvo em snapshot_distribution.json).

| Atributo | Valor |
|----------|-------|
| HuggingFace repo | HuggingFaceFW/fineweb-2 · config=por_Latn |
| Tokens PT-BR estimados | ~100-120 bilhões |
| Licença | ODC-By 1.0 |
| Tempo estimado (ha4) | 12-20 horas |

---

## 8. Script 04 — HPLT 2.0

**Arquivo:** `corpus/scripts/04_acquire_hplt2.py`  
**Status:** ✅ Implementado

### Justificativa Científica

O HPLT 2.0 (de Gibert et al., 2024) tem licença CC0 (domínio público — a mais
permissiva possível) e cobertura temporal distinta (coleções CC 2013-2023 com
pipeline de filtragem diferente do FineWeb-2), aumentando a diversidade temporal
do corpus Manacá.

**Diferencial:** CollectionTracker documenta distribuição por coleção HPLT
(análogo ao SnapshotTracker do Script 03). LangID reprocessado com fastText
para uniformidade metodológica entre fontes.

| Atributo | Valor |
|----------|-------|
| HuggingFace repo | HPLT/HPLT2.0_cleaned · config=pt |
| Tokens PT-BR estimados | ~35-45 bilhões |
| Licença | CC0 (domínio público) |
| Tempo estimado (ha4) | 4-8 horas |

---

## 9. Script 05 — Wikipedia PT-BR

**Arquivo:** `corpus/scripts/05_acquire_wikipedia.py`  
**Status:** ✅ Implementado

### Justificativa Científica — Papel Duplo

**Papel 1 — Corpus de treinamento:** texto enciclopédico factual, revisado por
humanos, âncora de qualidade análoga ao LLM-jp-corpus v4.

**Papel 2 — Bootstrap do modelo KenLM:** os textos extraídos treinam um modelo
de linguagem 5-gramas (Heafield et al., 2013) usado como filtro de perplexidade
no Script 07 (Common Crawl). Técnica introduzida pelo CCNet (Wenzek et al., 2020).

**Diferencial:** sem filtro de variante PT-BR/PT-PT (Wikipedia é predominantemente
PT-BR). Limpeza específica via clean_wikipedia_text(). Prefixação de título
(prática GPT-3/T5). Coleta de distribuição de scores para benchmark de qualidade.

| Atributo | Valor |
|----------|-------|
| HuggingFace repo | wikimedia/wikipedia · config=20231101.pt |
| Tokens estimados | ~1 bilhão |
| Licença | CC BY-SA 4.0 |
| Tempo estimado (ha4) | 1-2 horas |

---

## 10. Script 06 — Ulysses Tesemõ

**Arquivo:** `corpus/scripts/06_acquire_ulysses.py`  
**Status:** ✅ Implementado

### Justificativa Científica

O Ulysses Tesemõ (Nascimento et al., 2023) é o maior corpus jurídico-legislativo
PT-BR: 3,5 milhões de arquivos, 30,7 GiB, 159 fontes governamentais (Câmara,
Senado, STF, STJ, TCU, ministérios). Documentos governamentais brasileiros são
de domínio público por mandato constitucional (Art. 216, CF/1988).

**Diferencial único — pipeline Git:** esta é a única fonte não-HuggingFace.
Pipeline: git clone --depth=1 → varredura de arquivos → clean_legal_text()
→ Parquet. Threshold alfabético 40% (vs 50% nas demais) para acomodar
texto jurídico com números de processo, artigos e valores monetários.

| Atributo | Valor |
|----------|-------|
| GitHub | ulysses-camara/ulysses-tesemo |
| Tokens estimados | ~10 bilhões |
| Licença | Domínio público |
| Tempo estimado (ha4) | 2-4h clone + 2-4h processamento |

---

## 11. Script 07 — Common Crawl

**Arquivo:** `corpus/scripts/07_cc_pipeline.py`  
**Status:** ✅ Implementado · requer SLURM para produção

### Justificativa Científica

Processa snapshots 2025 não cobertos pelo FineWeb-2 (cutoff: dez/2024) usando
o pipeline FineWeb completo via framework datatrove. Acesso via HTTPS direto
(data.commoncrawl.org) sem custo.

**Diferencial estrutural:** único script baseado no framework datatrove como
orquestrador. Dois modos de execução via CLI:
- `--executor local`: teste sem SLURM (4-8 tarefas)
- `--executor slurm`: produção no cluster DEXL/SDumont (64+ tarefas)

Pipeline: WARCReader → LanguageFilter (GlotLID) → GopherQualityFilter →
GopherRepetitionFilter → C4QualityFilter → FineWebQualityFilter →
MinhashDedupSignature → ParquetWriter

| Atributo | Valor |
|----------|-------|
| Snapshots | CC-MAIN-2024-51, CC-MAIN-2025-08, CC-MAIN-2025-18 |
| Licença | Domínio público |
| Recursos | SLURM recomendado · 64+ tarefas · 256 GB RAM/nó |

### Execução

```bash
# Teste local (sem SLURM, validação do pipeline):
python corpus/scripts/07_cc_pipeline.py \
    --snapshot CC-MAIN-2025-08 --num-tasks 4 --executor local

# Produção (após SLURM disponível):
python corpus/scripts/07_cc_pipeline.py \
    --executor slurm --num-tasks 64 --slurm-partition cpu

# Listar snapshots configurados:
python corpus/scripts/07_cc_pipeline.py --list-snapshots
```

---

## 12. Script 08 — Deduplicação Global

**Arquivo:** `corpus/scripts/08_global_dedup.py`  
**Status:** ✅ Implementado · executar após Scripts 01-06

### Justificativa Científica

Soldaini et al. (2024) demonstraram que deduplicação cross-source remove 15-30%
de conteúdo redundante não detectado pela deduplicação within-source, pois fontes
distintas frequentemente capturam os mesmos documentos web.

**Algoritmo — duas passagens:**
- Passagem 1: computa assinaturas MinHash (128 permutações) e constrói índice LSH
- Passagem 2: resolve clusters mantendo o doc com maior quality_score

**Prioridade de fonte em clusters:** gigaverbo > wikipedia > ulysses >
fineweb2 > madlad400 > hplt2 > common_crawl

**Parâmetros:** 128 permutações · threshold Jaccard 0.80 · 5-gramas de palavras

Para volumes >50M documentos: `--use-datatrove` com SlurmPipelineExecutor.

### Execução

```bash
# Deduplicar todas as fontes Tier 1:
python corpus/scripts/08_global_dedup.py

# Verificar status das fontes disponíveis:
python corpus/scripts/08_global_dedup.py --status

# Para corpus >50M docs (distribuído):
python corpus/scripts/08_global_dedup.py \
    --use-datatrove --executor slurm --slurm-partition cpu
```

---

## 13. Script 09 — Validação e Estatísticas

**Arquivo:** `corpus/scripts/09_validate_corpus.py`  
**Status:** ✅ Implementado · executar após Script 08

### O que produz

| Saída | Conteúdo |
|-------|----------|
| `stats/corpus_validation_report.json` | Relatório completo (JSON) |
| `stats/corpus_summary.md` | Sumário Markdown para README (formato The Pile/Dolma) |
| `stats/corpus_progress.txt` | Progresso vs meta 1 TB |

**Métricas calculadas:**
- Contagem precisa de tokens via tokenizador LLM-jp-3 (ou estimativa 4 chars/token)
- Composição por fonte: docs, tokens, share%, licença, domínio
- Distribuições de qualidade: score, alpha_ratio, TTR, stopword_ratio (p10/p50/p90)
- Integridade: shards corrompidos, schema, contagem de rows

### Execução

```bash
# Validação completa com tokenizador preciso (recomendado):
python corpus/scripts/09_validate_corpus.py \
    --tokenizer llm-jp/llm-jp-3-tokenizer-nightly

# Validação rápida (estimativa de tokens):
python corpus/scripts/09_validate_corpus.py

# Apenas integridade dos shards (rápido):
python corpus/scripts/09_validate_corpus.py --integrity-only
```

---

## 14. Fluxo de Execução

```
PRÉ-REQUISITO: volume WORK_DIR gravável (bind mount ./data) + imagem construída
               (make build-corpus)
    |
    v
Semana 1
  |- make verify                ~1 min    obrigatório antes de tudo
  \- make gigaverbo             8-16h    iniciar imediatamente

Semana 1-2 (após Script 01 concluído)
  |- make madlad                6-12h
  \- make fineweb2              12-20h

Semana 2
  |- make hplt2                 4-8h
  |- make wikipedia             1-2h
  \- make ulysses               2-4h (clone) + 2-4h (processamento)

Semana 2-3
  \- make cc-pipeline           48-96h   (executor local; SLURM opcional se houver cluster)

Após todas as fontes
  |- make dedup                 24-72h
  \- make validate              1-3h
```

**Estimativa Tier 1 completo:** ~410-440B tokens PT-BR (~41-44% da meta)  
**Tier 2 pendente:** CulturaX (~200B) + Jabuticaba (~139B) — acesso HuggingFace em andamento

---

## 15. Monitoramento

```bash
# Estado geral do manifesto de aquisição
python -c "
import json, os
from pathlib import Path
p = Path(os.environ.get('WORK_DIR', str(Path.home() / 'manaca-corpus')))
m_path = p / 'checkpoints' / 'acquisition_manifest.json'
if not m_path.exists():
    print('Nenhum download iniciado ainda.')
else:
    m = json.loads(m_path.read_text())
    total_b = 0
    for src, s in m.get('sources', {}).items():
        b = s.get('estimated_tokens_b', s.get('estimated_tokens', 0) / 1e9)
        total_b += b
        print(f'{src:<25} {s[\"status\"]:<14} {b:6.1f}B tokens')
    print(f'TOTAL: {total_b:.1f}B / 1000.0B tokens ({total_b/10:.1f}%)')
"

# Espaço em disco (no host, no volume DATA_DIR)
df -h ./data && du -sh ./data/raw/*/

# Containers ativos do projeto (equivale a screen -ls / squeue)
docker ps

# Log em tempo real de qualquer fonte
docker logs -f <fonte>
# ou diretamente no volume: tail -f ./data/raw/<fonte>/download.log
```

---

## 16. Referências

1. Penedo, G. et al. (2024). *The FineWeb Datasets*. arXiv:2406.17557.
2. Soldaini, L. et al. (2024). *Dolma*. arXiv:2402.00159.
3. de Lucena, R. et al. (2024). *Tucano*. arXiv:2411.07854.
4. Kudugunta, S. et al. (2024). *MADLAD-400*. arXiv:2309.04662.
5. Lee, K. et al. (2022). *Deduplicating Training Data*. ACL 2022. arXiv:2107.06499.
6. Rae, J. et al. (2021). *Scaling Language Models: Gopher*. arXiv:2112.11446.
7. Wenzek, G. et al. (2020). *CCNet*. arXiv:2011.00180.
8. de Gibert, O. et al. (2024). *HPLT 2.0*. arXiv:2403.05010.
9. Joulin, A. et al. (2016). *fastText*. arXiv:1607.01759.
10. Barbaresi, A. (2021). *Trafilatura*. ACL 2021.
11. Nascimento, F. et al. (2023). *Ulysses*. STIL 2023.
12. Heafield, K. et al. (2013). *Scalable Modified Kneser-Ney LM Estimation*. ACL 2013.
13. Gao, L. et al. (2020). *The Pile*. arXiv:2101.00027.
14. Laurençon, A. et al. (2022). *ROOTS*. arXiv:2303.03915.
15. Kargaran, A. et al. (2023). *GlotLID*. arXiv:2303.12463.

---

*Versão 0.2.0 — Abril de 2026 | Projeto Manacá — LNCC × NII/LLM-jp*  
*Instituto-IA-LNCC · `brunolsm@lncc.br`*

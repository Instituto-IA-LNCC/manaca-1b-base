# Corpus Manacá — Manual Técnico de Extração

**[🇧🇷 Português](#português)** · **[🇬🇧 English](#english)**

---

## Português

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

---

## English

## Phase 1: Building the PT-BR Corpus

**Prepared by:** Bruno Leonardo Santos Menezes  
**Coordination:** Prof. Fábio Porto  
**Institution:** LNCC / AI Institute  
**Date:** April 2026 | **Version:** 0.2.0

---

> 🐳 **Fork `manaca-1b` — Docker execution.** This fork runs in Docker
> containers, not on `conda` + `screen` + SLURM. Command-translation rule for
> the instructions below: where you read `conda activate manaca-corpus && python <script>`, use
> `docker compose run --rm corpus python <script>`; where you read
> `screen -S <fonte> ...`, use `docker compose run -d --name <fonte> corpus ...`
> (or the `Makefile` shortcuts: `make gigaverbo`, `make dedup`, `make validate`).
> Full guide: [`../docs/environment/setup-guide-docker-pt.md`](../docs/environment/setup-guide-docker-pt.md).

---

## Contents

1. [Strategy Overview](#1-strategy-overview)
2. [Prerequisites](#2-prerequisites)
3. [Script Structure](#3-script-structure)
4. [Script 00 — Environment Verification](#4-script-00--environment-verification)
5. [Script 01 — GigaVerbo](#5-script-01--gigaverbo)
6. [Script 02 — MADLAD-400](#6-script-02--madlad-400)
7. [Script 03 — FineWeb-2](#7-script-03--fineweb-2)
8. [Script 04 — HPLT 2.0](#8-script-04--hplt-20)
9. [Script 05 — Wikipedia PT-BR](#9-script-05--wikipedia-pt-br)
10. [Script 06 — Ulysses Tesemõ](#10-script-06--ulysses-tesemõ)
11. [Script 07 — Common Crawl](#11-script-07--common-crawl)
12. [Script 08 — Global Deduplication](#12-script-08--global-deduplication)
13. [Script 09 — Validation and Statistics](#13-script-09--validation-and-statistics)
14. [Execution Flow](#14-execution-flow)
15. [Monitoring](#15-monitoring)
16. [References](#16-references)

---

## 1. Strategy Overview

The construction of the Manacá corpus follows the methodology established by three
top reference works in the field:

- **FineWeb** (Penedo et al., 2024, arXiv:2406.17557) — high-quality filtering
  pipeline for web data; the datatrove framework used as the basis of this
  project was developed specifically to reproduce and extend this work.
- **Dolma** (Soldaini et al., 2024, arXiv:2402.00159) — multi-source corpus
  architecture with two-pass deduplication (within-source + cross-source).
- **Tucano** (de Lucena et al., 2024, arXiv:2411.07854) — specific reference
  for PT-BR; demonstrates that GigaVerbo as a base produces models superior to
  training on non-curated multilingual corpora.

### 1.1 Phase 1 Goal

| Metric | Goal |
|---------|------|
| Total volume | >= 1 trillion tokens (1 TB tokens) |
| Language | Brazilian Portuguese (PT-BR) |
| License | Compatible with commercial use and open academic use |
| Output format | Apache Parquet + Zstandard compression |
| Traceability | Per-source JSON manifest with SHA-256 checksums |

### 1.2 Source Prioritization

| Tier | Criterion | Sources | Est. volume |
|------|----------|--------|-------------|
| 1 | No authentication · open license | GigaVerbo, MADLAD-400, FineWeb-2, HPLT 2.0, Wikipedia PT-BR, Ulysses Tesemõ | ~410-440 B tokens |
| 2 | Requires HuggingFace approval | CulturaX, Jabuticaba | ~340 B tokens |
| 3 | Built in-house via datatrove | Common Crawl snapshots 2024-2025 | unlimited |

### 1.3 Quality Pipeline (all sources)

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

### 1.4 Infrastructure

| Component | Specification |
|------------|--------------|
| Runtime | Docker (image `manaca-corpus`, `docker/Dockerfile.corpus`) |
| CPU | ≥ 8 threads recommended (I/O-bound) |
| RAM | ≥ 32 GiB recommended |
| Corpus storage | `WORK_DIR` volume — bind mount `./data` (or NFS via `DATA_DIR`) |
| Python | 3.11 · Docker image (dependencies pinned in `requirements/corpus.txt`) |
| Main framework | datatrove 0.9.0 (Penedo et al., 2024) |

---

## 2. Prerequisites

### 2.1 Environment (Docker image)

```bash
cp .env.example .env       # ajuste DATA_DIR, HF_TOKEN, ...
make build-corpus          # constrói a imagem manaca-corpus
# Um shell no container, se quiser inspecionar:
docker compose run --rm corpus /bin/sh
```

### 2.2 Working volume (bind mount)

The corpus is written to the `WORK_DIR` volume (`/workspace/manaca-corpus` in the
container), mounted from `DATA_DIR` on the host (default `./data`). Make sure
the directory exists and is writable — the container creates it automatically on
startup. To point to an NFS: `DATA_DIR=/caminho/do/nfs/manaca-corpus`.

### 2.3 Full verification (mandatory before any script)

```bash
make verify
# equivale a: docker compose run --rm corpus python corpus/scripts/00_verify_env.py
# Todos os itens críticos devem mostrar OK
```

---

## 3. Script Structure

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

### 3.1 Conventions applied in all scripts

**Idempotency:** Each script checks already-written shards before starting.
Re-running never overwrites data — it automatically resumes from the point of failure.

**Fixed Parquet schema** (identical across all scripts in the suite):

```python
pa.schema([
    pa.field("text",   pa.string()),   # texto limpo
    pa.field("source", pa.string()),   # identificador da fonte
    pa.field("id",     pa.string()),   # id único do documento
    pa.field("lang",   pa.string()),   # código de idioma (GlotLID/fastText)
    pa.field("score",  pa.float32()),  # score de qualidade [0.0-1.0]
])
```

**JSON manifest:** Each written shard atomically updates
$WORK_DIR/checkpoints/acquisition_manifest.json with complete metadata,
ensuring traceability for open publication of the artifacts.

**Execution in a detached container (for long jobs):**

```bash
# Container destacado (equivale ao screen; sobrevive ao fechamento do terminal):
docker compose run -d --name <fonte> corpus python corpus/scripts/NN_acquire_<fonte>.py
docker logs -f <fonte>
# ou, com os atalhos do Makefile: make gigaverbo && make logs SRC=gigaverbo
```

---

## 4. Script 00 — Environment Verification

**File:** `corpus/scripts/00_verify_env.py`  
**Execution time:** ~1 minute  
**Status:** ✅ Implemented and verified on a Linux GPU server

### What it checks

| Check | Success criterion |
|-------------|---------------------|
| Python 3.11 | sys.version_info == (3, 11, x) |
| Critical packages | 16 packages · import without error · minimum version satisfied |
| Parquet + Zstd | Write and read with the full 5-field schema |
| MinHash LSH | Identical Jaccard = 1.0 · Different Jaccard < 0.10 |
| fastText LangID | Identification of PT-BR text |
| HuggingFace Hub | HTTP 200 · latency < 5s |
| WORK_DIR volume | Existence and write permission (warning, not critical error) |
| Workspace dirs | Existence of 7 subdirectories in $WORK_DIR |
| Disk | >= 50 GB available |

### Execution

```bash
make verify
# equivale a:
docker compose run --rm corpus python corpus/scripts/00_verify_env.py
```

---

## 5. Script 01 — GigaVerbo

**File:** `corpus/scripts/01_acquire_gigaverbo.py`  
**Priority:** 1 — first source to be extracted  
**Status:** ✅ Implemented

### Scientific Rationale

GigaVerbo is the largest public monolingual PT-BR corpus. De Lucena et al. (2024)
showed that training the Tucano-1.1B model exclusively on this corpus yields
performance superior to mC4 and CC-100 across all evaluated PT-BR benchmarks,
proving that quality and specificity outweigh raw volume.

Reference: de Lucena, R. et al. (2024). *Tucano*. arXiv:2411.07854.

### Specifications

| Attribute | Value |
|----------|-------|
| HuggingFace repo | TucanoBR/GigaVerbo |
| Estimated tokens | 200 billion |
| Compressed size | ~80-100 GB (Parquet + Zstd) |
| License | Apache 2.0 |
| Authentication | Not required |
| Filters | Basic sanity (corpus already curated by the TucanoBR team) |
| Estimated time (ha4) | 8-16 hours |

### Execution

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

**File:** `corpus/scripts/02_acquire_madlad400.py`  
**Status:** ✅ Implemented

### Scientific Rationale

MADLAD-400 (Kudugunta et al., 2024) covers 419 languages with a rigorous
deduplication pipeline. The PT partition (~80B tokens) complements GigaVerbo with
distinct domains and historical periods (2013-2023).

**Distinguishing feature:** PT-BR vs PT-PT variant filter via fastText lid.176.bin
(threshold 0.65) with a second layer of lexical markers.

| Attribute | Value |
|----------|-------|
| HuggingFace repo | allenai/MADLAD-400 · config=pt |
| Estimated PT-BR tokens | ~50-60 billion |
| License | Apache 2.0 |
| Estimated time (ha4) | 6-12 hours |

---

## 7. Script 03 — FineWeb-2

**File:** `corpus/scripts/03_acquire_fineweb2.py`  
**Status:** ✅ Implemented

### Scientific Rationale

FineWeb-2 (Penedo et al., 2024) is the highest-quality publicly available web
corpus, the result of the Gopher + C4 + FineWeb pipeline applied to 96 CC
snapshots (2013-2024). Including it avoids weeks of CC processing to reach the same
quality. Script 07 complements it with 2025 snapshots not covered here.

**Distinguishing feature:** SnapshotTracker tracks the 96 covered CC snapshots to avoid
overlap with Script 07 (saved to snapshot_distribution.json).

| Attribute | Value |
|----------|-------|
| HuggingFace repo | HuggingFaceFW/fineweb-2 · config=por_Latn |
| Estimated PT-BR tokens | ~100-120 billion |
| License | ODC-By 1.0 |
| Estimated time (ha4) | 12-20 hours |

---

## 8. Script 04 — HPLT 2.0

**File:** `corpus/scripts/04_acquire_hplt2.py`  
**Status:** ✅ Implemented

### Scientific Rationale

HPLT 2.0 (de Gibert et al., 2024) has a CC0 license (public domain — the most
permissive possible) and distinct temporal coverage (CC collections 2013-2023 with
a filtering pipeline different from FineWeb-2), increasing the temporal diversity
of the Manacá corpus.

**Distinguishing feature:** CollectionTracker documents the distribution per HPLT
collection (analogous to Script 03's SnapshotTracker). LangID reprocessed with fastText
for methodological uniformity across sources.

| Attribute | Value |
|----------|-------|
| HuggingFace repo | HPLT/HPLT2.0_cleaned · config=pt |
| Estimated PT-BR tokens | ~35-45 billion |
| License | CC0 (public domain) |
| Estimated time (ha4) | 4-8 hours |

---

## 9. Script 05 — Wikipedia PT-BR

**File:** `corpus/scripts/05_acquire_wikipedia.py`  
**Status:** ✅ Implemented

### Scientific Rationale — Dual Role

**Role 1 — Training corpus:** factual encyclopedic text, human-reviewed,
a quality anchor analogous to LLM-jp-corpus v4.

**Role 2 — KenLM model bootstrap:** the extracted texts train a 5-gram language
model (Heafield et al., 2013) used as a perplexity filter in
Script 07 (Common Crawl). Technique introduced by CCNet (Wenzek et al., 2020).

**Distinguishing feature:** no PT-BR/PT-PT variant filter (Wikipedia is predominantly
PT-BR). Specific cleaning via clean_wikipedia_text(). Title prefixing
(GPT-3/T5 practice). Collection of score distribution for a quality benchmark.

| Attribute | Value |
|----------|-------|
| HuggingFace repo | wikimedia/wikipedia · config=20231101.pt |
| Estimated tokens | ~1 billion |
| License | CC BY-SA 4.0 |
| Estimated time (ha4) | 1-2 hours |

---

## 10. Script 06 — Ulysses Tesemõ

**File:** `corpus/scripts/06_acquire_ulysses.py`  
**Status:** ✅ Implemented

### Scientific Rationale

Ulysses Tesemõ (Nascimento et al., 2023) is the largest legal-legislative PT-BR
corpus: 3.5 million files, 30.7 GiB, 159 government sources (Câmara,
Senado, STF, STJ, TCU, ministries). Brazilian government documents are
public domain by constitutional mandate (Art. 216, CF/1988).

**Unique distinguishing feature — Git pipeline:** this is the only non-HuggingFace source.
Pipeline: git clone --depth=1 → file scan → clean_legal_text()
→ Parquet. Alphabetic threshold 40% (vs 50% for the others) to accommodate
legal text with case numbers, articles, and monetary values.

| Attribute | Value |
|----------|-------|
| GitHub | ulysses-camara/ulysses-tesemo |
| Estimated tokens | ~10 billion |
| License | Public domain |
| Estimated time (ha4) | 2-4h clone + 2-4h processing |

---

## 11. Script 07 — Common Crawl

**File:** `corpus/scripts/07_cc_pipeline.py`  
**Status:** ✅ Implemented · requires SLURM for production

### Scientific Rationale

Processes 2025 snapshots not covered by FineWeb-2 (cutoff: Dec/2024) using
the full FineWeb pipeline via the datatrove framework. Access via direct HTTPS
(data.commoncrawl.org) at no cost.

**Structural distinguishing feature:** the only script based on the datatrove framework as
orchestrator. Two execution modes via CLI:
- `--executor local`: test without SLURM (4-8 tasks)
- `--executor slurm`: production on the DEXL/SDumont cluster (64+ tasks)

Pipeline: WARCReader → LanguageFilter (GlotLID) → GopherQualityFilter →
GopherRepetitionFilter → C4QualityFilter → FineWebQualityFilter →
MinhashDedupSignature → ParquetWriter

| Attribute | Value |
|----------|-------|
| Snapshots | CC-MAIN-2024-51, CC-MAIN-2025-08, CC-MAIN-2025-18 |
| License | Public domain |
| Resources | SLURM recommended · 64+ tasks · 256 GB RAM/node |

### Execution

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

## 12. Script 08 — Global Deduplication

**File:** `corpus/scripts/08_global_dedup.py`  
**Status:** ✅ Implemented · run after Scripts 01-06

### Scientific Rationale

Soldaini et al. (2024) showed that cross-source deduplication removes 15-30%
of redundant content not detected by within-source deduplication, since distinct
sources frequently capture the same web documents.

**Algorithm — two passes:**
- Pass 1: computes MinHash signatures (128 permutations) and builds the LSH index
- Pass 2: resolves clusters keeping the doc with the highest quality_score

**Source priority within clusters:** gigaverbo > wikipedia > ulysses >
fineweb2 > madlad400 > hplt2 > common_crawl

**Parameters:** 128 permutations · Jaccard threshold 0.80 · word 5-grams

For volumes >50M documents: `--use-datatrove` with SlurmPipelineExecutor.

### Execution

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

## 13. Script 09 — Validation and Statistics

**File:** `corpus/scripts/09_validate_corpus.py`  
**Status:** ✅ Implemented · run after Script 08

### What it produces

| Output | Content |
|-------|----------|
| `stats/corpus_validation_report.json` | Complete report (JSON) |
| `stats/corpus_summary.md` | Markdown summary for the README (The Pile/Dolma format) |
| `stats/corpus_progress.txt` | Progress vs 1 TB goal |

**Metrics computed:**
- Precise token count via the LLM-jp-3 tokenizer (or 4 chars/token estimate)
- Composition per source: docs, tokens, share%, license, domain
- Quality distributions: score, alpha_ratio, TTR, stopword_ratio (p10/p50/p90)
- Integrity: corrupted shards, schema, row count

### Execution

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

## 14. Execution Flow

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

**Full Tier 1 estimate:** ~410-440B PT-BR tokens (~41-44% of the goal)  
**Tier 2 pending:** CulturaX (~200B) + Jabuticaba (~139B) — HuggingFace access in progress

---

## 15. Monitoring

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

## 16. References

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

*Version 0.2.0 — April 2026 | Manacá Project — LNCC × NII/LLM-jp*  
*Instituto-IA-LNCC · `brunolsm@lncc.br`*

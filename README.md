# Manacá-1B — An Open, Reproducible Brazilian-Portuguese Language Model

> **Manacá-1B** is an open decoder-only language model of **~1.72 billion
> parameters**, trained **from scratch** for **Brazilian Portuguese** with a fully
> containerized, reproducible pipeline. This repository contains everything a
> reader of the paper needs to **reproduce the pretraining and the evaluation** of
> the base model: the Docker environment, the corpus pipeline, the tokenizer
> training, the Megatron-LM pretraining, the Megatron to HuggingFace conversion,
> and the full Portuguese benchmark suite with paired significance tests.
>
> **Scientific cooperation LNCC × NII/LLM-jp** — National Laboratory for Scientific
> Computing (Brazil) × National Institute of Informatics (Japan).

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Model: Manacá-1B](https://img.shields.io/badge/Model-Manac%C3%A1--1B-purple.svg)]()
[![Environment: Docker](https://img.shields.io/badge/Environment-Docker%20(reproducible)-2496ED.svg)](docs/environment/setup-guide-docker-pt.md)
[![Language: PT-BR](https://img.shields.io/badge/Language-PT--BR-009c3b.svg)]()
[![Institution: LNCC](https://img.shields.io/badge/Institution-LNCC-002776.svg)](https://www.lncc.br)

<p align="center">
  <img src="assets/figures/manaca-identity.svg" width="520" alt="Manacá — Tibouchina mutabilis: the three flowering colours as a metaphor for language-model maturation"/>
</p>

*Como o manacá-da-serra, que muda de cor a cada estágio de maturação, o Manacá é a*
*inteligência que aprende, evolui e floresce — em Português do Brasil.*

---

## Índice | Table of Contents

- [Resumo | Overview](#resumo--overview)
- [Resultados | Results](#resultados--results)
- [O modelo | The model](#o-modelo--the-model)
- [O corpus | The corpus](#o-corpus--the-corpus)
- [Dinâmica de treino | Training dynamics](#dinâmica-de-treino--training-dynamics)
- [Como reproduzir | How to reproduce](#como-reproduzir--how-to-reproduce)
- [Hardware e reprodutibilidade | Hardware & reproducibility](#hardware-e-reprodutibilidade--hardware--reproducibility)
- [O nome | The name](#o-nome--the-name)
- [Documentos | Documents](#documentos--documents)
- [Estrutura do repositório | Repository structure](#estrutura-do-repositório--repository-structure)
- [Equipe | Team](#equipe--team)
- [Trabalho futuro | Future work](#trabalho-futuro--future-work)
- [Referências | References](#referências--references)
- [Licença e citação | License & citation](#licença-e-citação--license--citation)

---

## Resumo | Overview

**PT-BR.** O Manacá-1B é um modelo base (pré-treinado, sem ajuste de instrução)
treinado do zero para o português do Brasil. A contribuição é dupla: um **modelo
aberto e reprodutível**, com todo o pipeline containerizado, os logs de treino e
avaliação e os vetores de predição por exemplo liberados; e uma **avaliação
rigorosa e pareada** contra nove modelos abertos em quatro benchmarks de
português, sob um único harness, com erro-padrão e teste de significância em cada
comparação. Documentamos também um achado de fidelidade de tokenizador: a
conversão para o formato HuggingFace descarta silenciosamente o normalizador de
case-folding do SentencePiece, e mostramos a correção. O preprint está em
[`paper/manaca1b.tex`](paper/manaca1b.tex).

**EN.** Manacá-1B is a base (pretrained, non-instruction-tuned) model trained from
scratch for Brazilian Portuguese. The contribution is twofold: an **open,
reproducible model**, with the full containerized pipeline, the raw training and
evaluation logs, and the per-example prediction vectors released; and a
**rigorous, paired evaluation** against nine open baselines on four Portuguese
benchmarks, under a single harness, with a standard error and a significance test
on every comparison. We also document a tokenizer-fidelity finding: converting the
SentencePiece tokenizer to the HuggingFace fast format silently drops the
case-folding normalizer, and we provide the fix. The preprint is in
[`paper/manaca1b.tex`](paper/manaca1b.tex).

---

## Resultados | Results

Acurácia (%) com erro-padrão em quatro benchmarks de português do Brasil, todos os
modelos sob o mesmo harness. CALAME-PT é pontuado por geração da última palavra;
ARC-Challenge-PT (25-shot), HellaSwag-PT (10-shot) e LAMBADA-PT (0-shot) por
log-verossimilhança. O Manacá-1B usa o tokenizador do treino.

| Modelo | Params (B) | CALAME-PT | ARC-Ch-PT | HellaSwag-PT | LAMBADA-PT |
|--------|-----------:|:---------:|:---------:|:------------:|:----------:|
| Tucano-1b1 | 1.10 | 59.08 | 29.66 | **44.23** | 31.50 |
| GlórIA-1b3 | 1.30 | 60.39 | 24.44 | 25.83 | 35.30 |
| mGPT-1b3 | 1.30 | 55.57 | 23.93 | 25.42 | 37.38 |
| **Manacá-1B** | 1.72 | **60.63** | 27.18 | 41.61 | **45.31** |
| Tucano-2b4 | 2.40 | 59.57 | 30.85 | 48.63 | 34.35 |
| Sabiá-7B | 7.00 | **63.23** | **46.67** | **64.55** | **63.67** |

Leitura honesta: o Manacá-1B é o **modelo mais forte abaixo de 7B na predição da
última palavra** (LAMBADA-PT 45.31, acima de Tucano-1b1, Tucano-2b4, GlórIA-1b3 e
mGPT-1b3, com margens pareadas grandes; só o Sabiá-7B, quatro vezes maior, supera).
Ele **empata** com os melhores modelos de 1 a 2 B do português no CALAME-PT e
**supera** claramente os pares de mesmo porte de outras variedades e línguas; em
raciocínio de múltipla escolha (ARC-Challenge-PT) fica na faixa de acaso, como todo
modelo base nessa escala. A tabela completa, os testes pareados de McNemar e a
validação do harness estão no [preprint](paper/manaca1b.tex) e em
[`docs/evaluation/`](docs/evaluation/).

<p align="center">
  <img src="docs/evaluation/benchmarks_paper_en.png" width="640" alt="Manacá-1B em quatro benchmarks de português, acurácia vs. parâmetros (escala log), com IC95%"/>
</p>

---

## O modelo | The model

| Item | Valor |
|------|-------|
| Parâmetros | 1.722.951.680 (~1,72B) |
| Camadas / dim / FFN | 24 / 2048 / 8192 (SwiGLU) |
| Cabeças / grupos KV / head dim | 32 / 8 (GQA) / 64 |
| Posição / norma / bias | RoPE (θ=500000) / RMSNorm / bias em todas as lineares |
| Contexto / precisão | 4096 / bfloat16 |
| Tokenizador | SentencePiece unigram, 64k (padded 64.128), `nmt_nfkc_cf` |
| Framework | Megatron-LM (fork LLM-jp), otimizador distribuído |
| Passos / batch global / tokens | 20.000 / 512 / ~41,9B (~24 tok/param) |
| Referência de arquitetura | LLM-jp-3.1-1.8B |

Detalhes de arquitetura e otimização: [`paper/manaca1b.tex`](paper/manaca1b.tex) e
[`models/README.md`](models/README.md). A conversão Megatron para HuggingFace, que
preserva os biases e desempacota QKV/GQA e o SwiGLU, está em
[`scripts/ckpt_converter/`](scripts/ckpt_converter/).

---

## O corpus | The corpus

O corpus foi dimensionado ao modelo e ao hardware. Para ~1,72B de parâmetros, o
ponto de Chinchilla (Hoffmann et al., 2022) é da ordem de ~34B tokens. O corpus tem
**~20,1B tokens únicos** e o pré-treino faz **~2 épocas** (orçamento de ~41,9B
tokens de atualização, ~24 tokens/parâmetro), próximo do compute-optimal;
repetição de dados até ~4 épocas rende quase o mesmo que dados frescos (Muennighoff
et al., 2023). Todas as fontes são de **licença aberta**.

Composição medida (validação sobre 674 shards, 33,7M documentos; estimativa de ~4
caracteres/token):

| Fonte | Domínio | Tokens | Licença |
|-------|---------|-------:|---------|
| **GigaVerbo** (TucanoBR) — subamostra | Web geral PT-BR | ~14,7 B (73,1%) | Apache-2.0 |
| **Ulysses Tesemõ** (USP) | Jurídico / legislativo | ~4,8 B (23,9%) | Domínio público |
| **Wikipedia-pt** (2023-11) | Enciclopédico | ~0,6 B (3,1%) | CC BY-SA 4.0 |
| **Total** | 33,7M docs | **~20,1 B** | aberta |

Formato: Apache Parquet + Zstandard, shards de 50.000 documentos, schema fixo
`{text, source, id, lang, score}`. Detalhes de cada fonte e do pipeline de limpeza:
[`corpus/README.md`](corpus/README.md) e
[`reports/corpora-survey/`](reports/corpora-survey/).

---

## Dinâmica de treino | Training dynamics

A corrida foi estável do zero: **0 iterações puladas e 0 NaN** em toda a execução,
a loss de treino caiu de 11,41 para 2,48 e a loss de validação chegou a 2,07 nats
(perplexidade 7,96), ainda em queda ao final do cronograma. A norma do gradiente
fica perto de 0,10, com poucos transientes que se recuperam sem pulo. O log bruto
está em [`docs/training/`](docs/training/).

<p align="center">
  <img src="docs/training/training_dynamics_en.png" width="720" alt="Dinâmica de pré-treino do Manacá-1B: loss de treino e validação, norma do gradiente e learning rate ao longo de ~42B tokens"/>
</p>

---

## Como reproduzir | How to reproduce

Pré-requisitos: Docker Engine 24+, Docker Compose v2 e, para as fases de GPU, o
NVIDIA Container Toolkit. Guia completo:
[`docs/environment/setup-guide-docker-pt.md`](docs/environment/setup-guide-docker-pt.md).

```bash
# 1. Clonar e configurar
git clone https://github.com/brunoleomenezes/manaca-1b-base.git
cd manaca-1b-base
cp .env.example .env            # ajuste DATA_DIR (disco com espaço), HF_TOKEN, ...

# 2. Ambiente e verificação (Fase 1 — CPU)
make build-corpus
make verify                     # todos os itens devem retornar OK

# 3. Corpus (containers destacados; sobrevivem ao fechamento do SSH)
make gigaverbo                  # GigaVerbo (subamostrar conforme o alvo de tokens)
make wikipedia                  # Wikipedia-pt
make ulysses-gdrive             # Ulysses Tesemõ
make validate                   # integridade + estatísticas do corpus

# 4. Tokenizador (Fase 2a — CPU)
make tokenizer                  # amostra balanceada -> SentencePiece unigram -> HF

# 5. Preparação dos dados + pré-treino (Fase 2b — GPU)
make build-train
make preprocess-megatron        # Parquet -> Megatron .bin/.idx
make pretrain                   # pré-treino Megatron-LM

# 6. Avaliação em português (Fase 4 — GPU)
make build-eval
./scripts/eval/run_lm_eval_pt.sh   # ARC-Ch-PT, HellaSwag-PT, LAMBADA-PT
./scripts/eval/run_eval.sh         # CALAME-PT (scorer próprio, tokenizador do treino)
```

O `Makefile` documenta todos os atalhos (`make help`). Cada fase é um serviço
Docker isolado; o código do repositório é montado no container (bind mount), de
modo que os scripts são editáveis sem reconstruir a imagem. Os scripts de avaliação
e os testes pareados de significância estão em [`scripts/eval/`](scripts/eval/).

---

## Hardware e reprodutibilidade | Hardware & reproducibility

O Manacá-1B foi treinado em **hardware acessível**, e não em um supercomputador: o
pipeline foi validado, ponta a ponta, em uma máquina com **2 GPUs de 24 GB (NVIDIA
Ampere, sem NVLink)** para o pré-treino do 1,72B com bf16 + FlashAttention, mais
CPU de servidor e RAM ampla para a aquisição/limpeza do corpus.

Princípios adotados:

- **Ambiente congelado:** imagens Docker com dependências pinadas
  (`requirements/*.txt`), sem depender de `module load`/`conda` mantido à mão.
- **Proveniência de execução:** cada corrida grava um registro com o commit do
  repositório, os hiperparâmetros efetivos e as versões do container.
- **Portabilidade HPC:** as mesmas imagens convertem-se para Apptainer/Singularity
  (`.sif`) para produção em cluster, sem divergência de ambiente (ver o guia de
  ambiente).

---

## O nome | The name

| | PT-BR | EN | JA |
|---|---|---|---|
| **Nome** | Manacá | Manacá | マナカ (Manaka) |
| **Origem** | Tupi *manaká* — "dom abundante" | Tupi *manaká* — "generous gift" | 真中 — "true center" |
| **Espécie** | *Tibouchina mutabilis* (Vell.) Cogn. | *Tibouchina mutabilis* | *Tibouchina mutabilis* |
| **Bioma** | Mata Atlântica — endêmica | Atlantic Forest — endemic | 大西洋岸森林 固有種 |

O manacá-da-serra é endêmico da Mata Atlântica e tem flores que mudam de cor,
branco, rosa-lilás e púrpura, coexistindo na mesma árvore, uma metáfora dos
estágios de maturação de um modelo de linguagem. Em japonês, **マナカ** significa
"verdadeiro centro", uma ponte cultural Brasil-Japão. Fundamentação completa:
[`docs/identity/name-proposal-pt.md`](docs/identity/name-proposal-pt.md).

---

## Documentos | Documents

| Documento | Link |
|-----------|------|
| ★ Preprint (arXiv, LaTeX) | [paper/manaca1b.tex](paper/manaca1b.tex) |
| Guia de configuração Docker | [setup-guide-docker-pt.md](docs/environment/setup-guide-docker-pt.md) |
| Relatório de avaliação (10 modelos, testes pareados) | [manaca-1b-base-eval-pt.md](docs/evaluation/manaca-1b-base-eval-pt.md) |
| Manual técnico do corpus | [corpus/README.md](corpus/README.md) |
| Levantamento de corpora PT-BR | [pt](reports/corpora-survey/corpora-survey-pt.md) · [en](reports/corpora-survey/corpora-survey-en.md) |
| Proposta do nome (identidade) | [pt](docs/identity/name-proposal-pt.md) · [en](docs/identity/name-proposal-en.md) |

---

## Estrutura do repositório | Repository structure

```
manaca-1b-base/
├── paper/                        # preprint (LaTeX arXiv) + figuras
├── docker-compose.yml · Makefile · .env.example    # orquestração Docker
├── docker/                       # Dockerfiles (corpus, train, eval) + entrypoint
├── requirements/                 # dependências pinadas por imagem
├── corpus/                       # Fase 1: pipeline do corpus PT-BR (scripts 00..11)
├── tokenizer/                    # Fase 2a: amostra + treino do SentencePiece + conversão HF
├── scripts/
│   ├── docker/run_pretrain.sh    # Fase 2b: pré-treino Megatron-LM
│   ├── ckpt_converter/           # conversão Megatron -> HuggingFace (preserva biases)
│   └── eval/                     # Fase 4: CALAME/ARC/HellaSwag/LAMBADA-PT + testes pareados
├── docs/
│   ├── environment/              # guia de ambiente Docker
│   ├── training/                 # log bruto de pré-treino + figura da dinâmica
│   ├── evaluation/               # resultados, logs por modelo, vetores por exemplo, figuras
│   └── identity/                 # proposta do nome (PT/EN)
├── reports/corpora-survey/       # levantamento de corpora PT-BR
├── models/ · assets/             # ficha do modelo · identidade visual
└── .github/                      # issue templates · CI (url-checker)
```

---

## Equipe | Team

| Nome | Instituição | Papel |
|------|-------------|-------|
| **Bruno Leonardo Santos Menezes** | LNCC / Instituto de IA | Pesquisador principal |
| **Carlos Leonardo Souza Cardoso** | LNCC / Instituto de IA | Pesquisador |
| **Prof. Fábio André Machado Porto** | LNCC / Instituto de IA | Coordenador científico |

**Contato:** `brunolsm@lncc.br` · `cardoso@lncc.br` · `fporto@lncc.br`

Contribuições são bem-vindas, especialmente reproduções independentes e relatos de
execução em outros hardwares. Ver [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Trabalho futuro | Future work

Este repositório entrega o **modelo base** (pré-treino + avaliação). As próximas
etapas, deixadas para trabalho futuro, incluem os benchmarks nativos do Open
Portuguese LLM Leaderboard sob o mesmo protocolo pareado, a extensão do orçamento
de tokens para além do compute-optimal, e o ajuste de instrução do Manacá. Esses
artefatos serão adicionados quando estiverem prontos.

---

## Referências | References

- Kiyomaru, H. et al. (2024). *LLM-jp: A Cross-organizational Project for Fully Open Japanese LLMs*. arXiv:2407.03963.
- Corrêa, N.K. et al. (2024). *Tucano: Advancing Neural Text Generation for Portuguese*. arXiv:2411.07854. *(GigaVerbo)*
- Lopes, R., Magalhães, J., Semedo, D. (2024). *GlórIA: A Generative and Open Large Language Model for Portuguese*. PROPOR. *(CALAME-PT)*
- Hoffmann, J. et al. (2022). *Training Compute-Optimal Large Language Models* (Chinchilla). arXiv:2203.15556.
- Muennighoff, N. et al. (2023). *Scaling Data-Constrained Language Models*. NeurIPS. arXiv:2305.16264.
- [LLM-jp](https://llm-jp.nii.ac.jp/en/home-en/) · [LNCC](https://www.lncc.br) · [NII](https://www.nii.ac.jp/en/)

---

## Licença e citação | License & citation

Licenciado sob **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Um
arquivo [`CITATION.cff`](CITATION.cff) acompanha o repositório.

```
Menezes, B.L.S., Cardoso, C.L.S., & Porto, F.A.M. (2026). Manacá-1B: An Open,
Reproducible Brazilian-Portuguese Language Model and a Tokenizer-Aware, Paired
Evaluation. Preprint. LNCC (Instituto de IA) × NII/LLM-jp.
GitHub: https://github.com/brunoleomenezes/manaca-1b-base · License: CC BY 4.0.
```

---

*Projeto Manacá — LNCC (Instituto de IA) × NII/LLM-jp · Manacá-1B (base)*

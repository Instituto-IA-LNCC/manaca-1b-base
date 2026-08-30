# Modelos Manacá | Manacá Models

**[🇧🇷 Português](#português)** · **[🇬🇧 English](#english)**

## Português

Esta pasta será populada com configurações e logs dos modelos à medida que o
pré-treinamento e a avaliação avançam.

### Manacá-1B (base)

O **Manacá-1B** é um modelo de linguagem decoder-only de ~1,72B de parâmetros,
treinado do zero para o **português do Brasil**. Este repositório entrega o
**pipeline reprodutível** de pré-treino e avaliação do modelo base (ver
[README principal](../README.md) e o [preprint](../paper/manaca1b.tex)).

| Item | Valor |
|------|-------|
| Parâmetros | 1.722.951.680 (~1,72B) |
| Arquitetura | Llama-style, 24 camadas, dim 2048, FFN 8192 (SwiGLU), 32 cabeças, GQA 8 KV, RoPE θ=500000, RMSNorm |
| Contexto / precisão | 4096 / bfloat16 |
| Tokens de atualização | ~41,9B (~24 tokens/parâmetro, ~2 épocas de um corpus curado de ~20,1B) |
| Tokenizador | SentencePiece unigram, 64k (padded 64.128), `nmt_nfkc_cf` |
| Framework | Megatron-LM (fork LLM-jp), otimizador distribuído |

**Referência de arquitetura:** LLM-jp-3.1-1.8B (NII / LLM-jp).

Os pesos do Manacá-1B base serão publicados no Hugging Face. Até lá, este
repositório entrega tudo o que é necessário para reconstruí-lo e reavaliá-lo.

## English

This folder will hold model configurations and logs as pretraining and
evaluation progress.

### Manacá-1B (base)

**Manacá-1B** is a decoder-only language model of ~1.72B parameters, trained from
scratch for **Brazilian Portuguese**. This repository delivers the
**reproducible pipeline** for pretraining and evaluating the base model (see the
[main README](../README.md) and the [preprint](../paper/manaca1b.tex)).

| Item | Value |
|------|-------|
| Parameters | 1,722,951,680 (~1.72B) |
| Architecture | Llama-style, 24 layers, dim 2048, FFN 8192 (SwiGLU), 32 heads, GQA 8 KV, RoPE θ=500000, RMSNorm |
| Context / precision | 4096 / bfloat16 |
| Update tokens | ~41.9B (~24 tokens/parameter, ~2 epochs of a curated corpus of ~20.1B) |
| Tokenizer | SentencePiece unigram, 64k (padded 64,128), `nmt_nfkc_cf` |
| Framework | Megatron-LM (LLM-jp fork), distributed optimizer |

**Architecture reference:** LLM-jp-3.1-1.8B (NII / LLM-jp).

The Manacá-1B base weights will be published on Hugging Face. Until then, this
repository delivers everything needed to rebuild and re-evaluate it.

---

*Projeto Manacá — LNCC (Instituto de IA) × NII/LLM-jp*

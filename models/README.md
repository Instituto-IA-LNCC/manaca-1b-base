# Modelos Manacá | Manacá Models

Esta pasta será populada com configurações e logs dos modelos à medida que o
pré-treinamento e a avaliação avançam.

*This folder will hold model configurations and logs as pretraining and
evaluation progress.*

---

## Manacá-1B (base)

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

---

*Projeto Manacá — LNCC (Instituto de IA) × NII/LLM-jp*

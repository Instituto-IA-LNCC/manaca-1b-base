# Changelog

Todas as mudanças relevantes deste repositório são registradas aqui. O formato
segue, de modo aproximado, o [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] — Manacá-1B base (pré-treino + avaliação)

Primeira versão pública, focada na **reprodutibilidade do modelo base**
(pré-treino e avaliação) descrito no preprint em [`paper/`](paper/).

### Modelo e treino
- Manacá-1B: decoder-only Llama-style de ~1,72B de parâmetros para o português do
  Brasil (24 camadas, dim 2048, FFN 8192 SwiGLU, 32 cabeças, GQA 8 KV, RoPE
  θ=500000, RMSNorm, bias em todas as lineares, embeddings desamarradas).
- Pré-treino com o fork LLM-jp do Megatron-LM: 20.000 passos, batch global 512,
  seq 4096 → ~41,9B tokens de atualização (~24 tok/param, ~2 épocas de um corpus
  curado de ~20,1B). Otimizador distribuído (ZeRO-1), z-loss, bf16, recompute full.
- Log bruto de treino e figura da dinâmica em [`docs/training/`](docs/training/).

### Corpus e tokenizador
- Pipeline de corpus em Docker (aquisição, limpeza por fonte, validação): GigaVerbo
  (~14,7B), Ulysses Tesemõ (~4,8B) e Wikipedia-pt (~0,6B), todos de licença aberta.
- Tokenizador SentencePiece unigram, vocab 64k (padded 64.128), `nmt_nfkc_cf`.

### Conversão e avaliação
- Conversor Megatron → HuggingFace que preserva os biases, desempacota QKV/GQA e o
  SwiGLU e mantém as embeddings desamarradas ([`scripts/ckpt_converter/`](scripts/ckpt_converter/)).
- Suíte de avaliação em português: CALAME-PT (scorer próprio) e ARC-Challenge-PT,
  HellaSwag-PT e LAMBADA-PT via lm-evaluation-harness, com erro-padrão, IC por
  bootstrap e teste pareado de McNemar; validação do harness contra números
  publicados. Resultados, logs por modelo e vetores por exemplo em
  [`docs/evaluation/`](docs/evaluation/).
- Correção de fidelidade do tokenizador HF (injeta `Sequence([NFKC, Lowercase])`),
  que reproduz a tokenização de treino ([`scripts/eval/fix_hf_tokenizer.py`](scripts/eval/fix_hf_tokenizer.py)).

### Infraestrutura
- Ambiente Docker com três imagens (`corpus`, `train`, `eval`) e dependências
  pinadas; `docker-compose.yml` + `Makefile` para orquestração.
- Preprint em LaTeX (arXiv) com as duas figuras em [`paper/`](paper/).

---
license: cc-by-4.0
language:
- pt
library_name: transformers
pipeline_tag: text-generation
tags:
- portuguese
- brazilian-portuguese
- pt-br
- llama
- megatron-lm
- causal-lm
- text-generation
datasets:
- TucanoBR/GigaVerbo
- wikimedia/wikipedia
---

# Manacá-1B (base)

**[🇧🇷 Português](#português)** · **[🇬🇧 English](#english)**

Manacá-1B é um modelo de linguagem decoder-only de ~1,72 bilhão de parâmetros,
treinado do zero para o **português do Brasil**. Modelo **base** (pré-treinado, sem
ajuste de instrução).
Manacá-1B is a ~1.72B-parameter decoder-only language model trained from scratch for
**Brazilian Portuguese**. This is a **base** (pretrained, non-instruction-tuned) model.

- **Código, logs e pipeline reprodutível | Code, logs, and reproducible pipeline:** https://github.com/brunoleomenezes/manaca-1b-base
- **Cooperação | Cooperation:** LNCC (Instituto de IA) × NII/LLM-jp

---

## Português

### Descrição do modelo

Manacá-1B é um transformer decoder-only no estilo Llama-3, treinado do zero para o
português do Brasil com um pipeline totalmente containerizado e reprodutível. É um
**modelo base**: aprendeu a modelar a língua por pré-treino, mas **não** passou por
ajuste de instrução (SFT) nem alinhamento (DPO/RLHF). Ele completa e continua texto;
não segue instruções nem conversa.

| Item | Valor |
|------|-------|
| Parâmetros | 1.722.951.680 (~1,72B) |
| Camadas / dim / FFN | 24 / 2048 / 8192 (SwiGLU) |
| Cabeças / grupos KV / head dim | 32 / 8 (GQA) / 64 |
| Posição / norma / bias | RoPE (θ=500000) / RMSNorm (eps 1e-5) / bias em todas as lineares |
| Embeddings | desamarradas (entrada e saída separadas) |
| Contexto / precisão | 4096 / bfloat16 |
| Vocabulário | 64.000 (padded para 64.128) |
| Framework | Megatron-LM (fork LLM-jp), otimizador distribuído |
| Referência de arquitetura | LLM-jp-3.1-1.8B |

### Tokenizador (leia isto)

O tokenizador é um **SentencePiece unigram** de 64k com a normalização
`nmt_nfkc_cf` (NFKC + *case folding*). **O modelo é lowercase por construção**: o
texto de entrada é normalizado para minúsculas antes da segmentação. Os arquivos de
tokenizador **publicados aqui já incluem a correção** que reproduz exatamente a
tokenização de treino (normalizador `Sequence([NFKC, Lowercase])`). Use sempre o
`AutoTokenizer` deste repositório; um tokenizador sem esse normalizador tokeniza
maiúsculas por *byte-fallback* e degrada os resultados de forma invisível.

### Dados de treino

Corpus curado, aberto, de ~20,1B tokens únicos (pré-treino de ~2 épocas ⇒ ~41,9B
tokens de atualização, ~24 tokens/parâmetro, próximo do compute-optimal de
Chinchilla; repetição até ~4 épocas é quase tão boa quanto dados frescos,
Muennighoff et al., 2023).

| Fonte | Domínio | Tokens | Licença |
|-------|---------|-------:|---------|
| GigaVerbo (TucanoBR) — subamostra | Web geral PT-BR | ~14,7 B (73,1%) | Apache-2.0 |
| Ulysses Tesemõ (USP) | Jurídico / legislativo | ~4,8 B (23,9%) | Domínio público |
| Wikipedia-pt (2023-11) | Enciclopédico | ~0,6 B (3,1%) | CC BY-SA 4.0 |

### Detalhes de treino

20.000 passos, batch global 512, sequência 4096. Adam (β 0,9/0,999, ε 1e-8),
weight decay 0,1, *clip* de gradiente 1,0, learning rate 3e-4 → 3e-5 (cosseno,
warmup de 2.000), z-loss, bf16, *recompute* full, seed 1234. Treinado em 2 GPUs de
24 GB (sem NVLink) com paralelismo de dados (ZeRO-1). A corrida foi estável: 0
iterações puladas e 0 NaN; loss de treino 11,41 → 2,48; loss de validação 2,07 nats.

### Avaliação

Acurácia (%) em quatro benchmarks de português, mesmo harness. CALAME-PT por geração
da última palavra; ARC-Challenge-PT (25-shot), HellaSwag-PT (10-shot) e LAMBADA-PT
(0-shot) por log-verossimilhança.

| Benchmark | Manacá-1B | Melhor par sub-7B | Sabiá-7B |
|-----------|:---------:|:-----------------:|:--------:|
| CALAME-PT | **60,63** | 60,39 (GlórIA-1b3) | 63,23 |
| LAMBADA-PT | **45,31** | 37,38 (mGPT-1b3) | 63,67 |
| HellaSwag-PT | 41,61 | 48,63 (Tucano-2b4) | 64,55 |
| ARC-Ch-PT | 27,18 | 30,85 (Tucano-2b4) | 46,67 |

O Manacá-1B é o modelo mais forte abaixo de 7B na predição da última palavra
(LAMBADA-PT), empata com os melhores modelos de 1 a 2 B no CALAME-PT, e fica na faixa
de acaso em ARC-Challenge-PT (como todo modelo base nessa escala). Detalhes, testes
pareados de McNemar e validação do harness no
[repositório](https://github.com/brunoleomenezes/manaca-1b-base) e no preprint.

### Uso

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "menezesbruno/manaca-1b-base"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto")

# O modelo é lowercase por construção (o tokenizador normaliza a entrada).
prompt = "a capital do brasil é"
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
```

### Uso pretendido e limitações

Pesquisa e uso como **modelo base** para português do Brasil: modelagem de língua,
continuação de texto, e ponto de partida para ajuste fino/instrução. **Não é** um
modelo de instrução nem de chat; não segue comandos sem ajuste posterior. É um
modelo base de ~1,72B: erra fatos, pode gerar conteúdo incorreto ou enviesado
refletindo os dados de treino, e tem raciocínio limitado (próximo do acaso em
múltipla escolha). Avalie antes de qualquer uso sensível.

---

## English

### Model description

Manacá-1B is a Llama-3-style decoder-only transformer trained from scratch for
Brazilian Portuguese with a fully containerized, reproducible pipeline. It is a
**base model**: it learned to model the language through pretraining but has **not**
been instruction-tuned (SFT) or aligned (DPO/RLHF). It completes and continues text;
it does not follow instructions or chat.

| Item | Value |
|------|-------|
| Parameters | 1,722,951,680 (~1.72B) |
| Layers / model dim / FFN | 24 / 2048 / 8192 (SwiGLU) |
| Heads / KV groups / head dim | 32 / 8 (GQA) / 64 |
| Position / norm / bias | RoPE (θ=500000) / RMSNorm (eps 1e-5) / all linear layers |
| Embeddings | untied (separate input and output) |
| Context length / precision | 4096 / bfloat16 |
| Vocabulary | 64,000 (padded to 64,128) |
| Framework | Megatron-LM (LLM-jp fork), distributed optimizer |
| Architecture reference | LLM-jp-3.1-1.8B |

### Tokenizer (read this)

The tokenizer is a 64k **SentencePiece unigram** with `nmt_nfkc_cf` normalization
(NFKC + case folding). **The model is lowercase by construction**: input text is
lowercased before segmentation. The tokenizer files **published here already include
the fix** that reproduces the training tokenization exactly (normalizer
`Sequence([NFKC, Lowercase])`). Always use the `AutoTokenizer` from this repository;
a tokenizer without that normalizer routes capitalized text to byte-fallback and
silently degrades results.

### Training data

A curated, openly licensed corpus of ~20.1B unique tokens (pretraining for ~2 epochs
⇒ ~41.9B update tokens, ~24 tokens/parameter, close to the Chinchilla compute-optimal
point; repeating data up to ~4 epochs is nearly as good as fresh data, Muennighoff et
al., 2023).

| Source | Domain | Tokens | License |
|--------|--------|-------:|---------|
| GigaVerbo (TucanoBR) — subsample | General web PT-BR | ~14.7B (73.1%) | Apache-2.0 |
| Ulysses Tesemõ (USP) | Legal / legislative | ~4.8B (23.9%) | Public domain |
| Wikipedia-pt (2023-11) | Encyclopedic | ~0.6B (3.1%) | CC BY-SA 4.0 |

### Training details

20,000 steps, global batch 512, sequence length 4096. Adam (β 0.9/0.999, ε 1e-8),
weight decay 0.1, gradient clip 1.0, learning rate 3e-4 → 3e-5 (cosine, 2,000-step
warmup), z-loss, bf16, full recompute, seed 1234. Trained on 2 GPUs of 24 GB (no
NVLink) with data parallelism (ZeRO-1). The run was stable: 0 skipped and 0 NaN
steps; training loss 11.41 → 2.48; validation loss 2.07 nats.

### Evaluation

Accuracy (%) on four Portuguese benchmarks, one harness. CALAME-PT by last-word
generation; ARC-Challenge-PT (25-shot), HellaSwag-PT (10-shot), and LAMBADA-PT
(0-shot) by log-likelihood.

| Benchmark | Manacá-1B | Best sub-7B peer | Sabiá-7B |
|-----------|:---------:|:----------------:|:--------:|
| CALAME-PT | **60.63** | 60.39 (GlórIA-1b3) | 63.23 |
| LAMBADA-PT | **45.31** | 37.38 (mGPT-1b3) | 63.67 |
| HellaSwag-PT | 41.61 | 48.63 (Tucano-2b4) | 64.55 |
| ARC-Ch-PT | 27.18 | 30.85 (Tucano-2b4) | 46.67 |

Manacá-1B is the strongest model below 7B on last-word prediction (LAMBADA-PT), ties
the best 1-to-2B models on CALAME-PT, and sits near chance on ARC-Challenge-PT (as
does every base model at this scale). Details, paired McNemar tests, and harness
validation are in the
[repository](https://github.com/brunoleomenezes/manaca-1b-base) and the preprint.

### Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "menezesbruno/manaca-1b-base"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto")

# The model is lowercase by construction (the tokenizer normalizes the input).
prompt = "a capital do brasil é"
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=40, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
```

### Intended use and limitations

Research and use as a **base model** for Brazilian Portuguese: language modeling,
text continuation, and a starting point for fine-tuning/instruction-tuning. It is
**not** an instruction or chat model and will not follow commands without further
tuning. As a ~1.72B base model it makes factual mistakes, may produce incorrect or
biased content reflecting its training data, and has limited reasoning (near chance
on multiple choice). Evaluate before any sensitive use.

---

## Citation

```bibtex
@misc{manaca1b2026,
  title  = {Manac\'a-1B: An Open, Reproducible Brazilian-Portuguese Language Model
            and a Tokenizer-Aware, Paired Evaluation},
  author = {Menezes, Bruno Leonardo Santos and Cardoso, Carlos Leonardo Souza and
            Porto, Fabio Andr\'e Machado},
  year   = {2026},
  note   = {LNCC (AI Institute) $\times$ NII/LLM-jp},
  url    = {https://github.com/brunoleomenezes/manaca-1b-base}
}
```

## Acknowledgments

Developed at the Artificial Intelligence Institute of the National Laboratory for
Scientific Computing (LNCC), Brazil, in cooperation with the National Institute of
Informatics (NII), Japan, and the LLM-jp project, whose open methodology, tooling
(the LLM-jp fork of Megatron-LM), and the LLM-jp-3.1-1.8B recipe this model builds on.

License: **Creative Commons Attribution 4.0 International (CC BY 4.0)**.

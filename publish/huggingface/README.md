# Kit de publicação no Hugging Face | Hugging Face publication kit

**[🇧🇷 Português](#português)** · **[🇬🇧 English](#english)**

Kit para publicar o **Manacá-1B (base)** em
[`menezesbruno/manaca-1b-base`](https://huggingface.co/menezesbruno/manaca-1b-base),
sob **CC BY 4.0**, como ciência aberta. Conteúdo:

- [`MODEL_CARD.md`](MODEL_CARD.md) — o *model card* bilíngue (vira o `README.md` do repo no HF).
- [`upload_to_hf.sh`](upload_to_hf.sh) — monta modelo + tokenizador corrigido + card e faz o upload.

---

## Português

### Pré-requisitos (os pesos NÃO estão neste repositório Git)

Você precisa, na máquina que tem o checkpoint de treino:

1. **Modelo em formato HuggingFace** — saída do
   [`scripts/ckpt_converter/megatron_to_hf.sh`](../../scripts/ckpt_converter/megatron_to_hf.sh)
   (um diretório com `config.json`, `*.safetensors`, arquivos de tokenizador).
2. **Tokenizador corrigido** — saída do
   [`scripts/eval/fix_hf_tokenizer.py`](../../scripts/eval/fix_hf_tokenizer.py)
   (com o normalizador `Sequence([NFKC, Lowercase])`). **Isto é obrigatório**: sem
   ele o tokenizador rápido (fast) tokeniza maiúsculas por *byte-fallback* e degrada
   os resultados de forma invisível.
3. A sua conta no **Hugging Face** (`menezesbruno`) e um **token de escrita**.

### Passo a passo

```bash
# 1. Instale a CLI e faça login (token de ESCRITA)
pip install -U "huggingface_hub[cli]"
huggingface-cli login

# 2. (Se ainda não converteu) gere os artefatos
./scripts/ckpt_converter/megatron_to_hf.sh <ckpt_megatron> <saida_hf> <tokenizer.model>
python scripts/eval/fix_hf_tokenizer.py --src <saida_hf> --spm <tokenizer.model> --out <tok_corrigido>

# 3. Suba tudo (modelo + tokenizador corrigido + model card)
MODEL_DIR=<saida_hf> TOKENIZER_DIR=<tok_corrigido> \
  ./publish/huggingface/upload_to_hf.sh
```

O script cria o repositório (se não existir), junta o tokenizador corrigido ao
modelo, copia o `MODEL_CARD.md` como `README.md` e faz o upload. Ele **verifica** que
o `tokenizer.json` tem o normalizador NFKC+Lowercase antes de subir.

### Depois de publicar

Confirme que qualquer pessoa consegue carregar o modelo:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("menezesbruno/manaca-1b-base")
model = AutoModelForCausalLM.from_pretrained("menezesbruno/manaca-1b-base")
```

Marque o repo do HF com a licença **CC BY 4.0** (já vem no *front-matter* do card) e
confira que os arquivos do tokenizador e o `config.json` foram enviados.

---

## English

### Prerequisites (the weights are NOT in this Git repository)

On the machine that holds the training checkpoint you need:

1. **Model in HuggingFace format** — output of
   [`scripts/ckpt_converter/megatron_to_hf.sh`](../../scripts/ckpt_converter/megatron_to_hf.sh)
   (a directory with `config.json`, `*.safetensors`, tokenizer files).
2. **Fixed tokenizer** — output of
   [`scripts/eval/fix_hf_tokenizer.py`](../../scripts/eval/fix_hf_tokenizer.py)
   (with the `Sequence([NFKC, Lowercase])` normalizer). **This is mandatory**: without
   it the fast tokenizer routes capitalized text to byte-fallback and silently
   degrades results.
3. Your **Hugging Face** account (`menezesbruno`) and a **write token**.

### Step by step

```bash
# 1. Install the CLI and log in (WRITE token)
pip install -U "huggingface_hub[cli]"
huggingface-cli login

# 2. (If not converted yet) produce the artifacts
./scripts/ckpt_converter/megatron_to_hf.sh <megatron_ckpt> <hf_out> <tokenizer.model>
python scripts/eval/fix_hf_tokenizer.py --src <hf_out> --spm <tokenizer.model> --out <fixed_tok>

# 3. Upload everything (model + fixed tokenizer + model card)
MODEL_DIR=<hf_out> TOKENIZER_DIR=<fixed_tok> \
  ./publish/huggingface/upload_to_hf.sh
```

The script creates the repository (if missing), merges the fixed tokenizer into the
model, copies `MODEL_CARD.md` as `README.md`, and uploads. It **verifies** that
`tokenizer.json` carries the NFKC+Lowercase normalizer before uploading.

### After publishing

Confirm anyone can load the model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("menezesbruno/manaca-1b-base")
model = AutoModelForCausalLM.from_pretrained("menezesbruno/manaca-1b-base")
```

Mark the HF repo with the **CC BY 4.0** license (already in the card front-matter) and
check that the tokenizer files and `config.json` were uploaded.

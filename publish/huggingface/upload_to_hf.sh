#!/usr/bin/env bash
# =============================================================================
# Manacá-1B — Publicação no Hugging Face Hub | Publish to the Hugging Face Hub
# =============================================================================
# PT --------------------------------------------------------------------------
# Monta um diretório de publicação com (1) o modelo HF convertido, (2) o
# tokenizador CORRIGIDO (com o normalizador NFKC+Lowercase) e (3) o model card,
# e faz o upload para o repositório de pesos no Hugging Face. NÃO regenera nada:
# usa os artefatos que você já produziu com megatron_to_hf.py e fix_hf_tokenizer.py.
#
# Pré-requisitos:
#   pip install -U "huggingface_hub[cli]"
#   huggingface-cli login          # token de ESCRITA da conta menezesbruno
#
# Uso:
#   MODEL_DIR=/caminho/para/manaca-1b-hf \
#   TOKENIZER_DIR=/caminho/para/tokenizador-corrigido \
#   ./publish/huggingface/upload_to_hf.sh
#
# EN --------------------------------------------------------------------------
# Assembles a publish directory with (1) the converted HF model, (2) the FIXED
# tokenizer (with the NFKC+Lowercase normalizer) and (3) the model card, and
# uploads it to the Hugging Face weights repository. It regenerates nothing:
# it uses the artifacts you already produced with megatron_to_hf.py and
# fix_hf_tokenizer.py.
#
# Prerequisites:
#   pip install -U "huggingface_hub[cli]"
#   huggingface-cli login          # WRITE token for the menezesbruno account
#
# Usage:
#   MODEL_DIR=/path/to/manaca-1b-hf \
#   TOKENIZER_DIR=/path/to/fixed-tokenizer \
#   ./publish/huggingface/upload_to_hf.sh
# =============================================================================
set -euo pipefail

# ── Configuração | Configuration ─────────────────────────────────────────────
HF_REPO="${HF_REPO:-menezesbruno/manaca-1b-base}"
MODEL_DIR="${MODEL_DIR:-/workspace/manaca-1b-hf}"          # saída do megatron_to_hf.py
TOKENIZER_DIR="${TOKENIZER_DIR:-}"                         # saída do fix_hf_tokenizer.py
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
CARD="${CARD:-$SELF_DIR/MODEL_CARD.md}"
STAGE_DIR="${STAGE_DIR:-$(mktemp -d)/manaca-1b-base}"

echo "[hf] repo    : $HF_REPO"
echo "[hf] modelo  | model     : $MODEL_DIR"
echo "[hf] tokenizador | tokenizer: ${TOKENIZER_DIR:-[usando o do MODEL_DIR | using the one in MODEL_DIR]}"
echo "[hf] staging : $STAGE_DIR"

# ── Pré-condições | Preconditions ────────────────────────────────────────────
# Detecta a CLI do Hugging Face | Detect the Hugging Face CLI.
# No huggingface_hub >= 1.0 a CLI é 'hf'; 'huggingface-cli' é legada e pode estar
# quebrada por uma instalação antiga em /usr/local/bin.
# In huggingface_hub >= 1.0 the CLI is 'hf'; 'huggingface-cli' is legacy and may be
# broken by an old install under /usr/local/bin.
HF_CLI="$(command -v hf || command -v huggingface-cli || true)"
[ -n "$HF_CLI" ] || {
  echo "ERRO | ERROR: CLI do Hugging Face não encontrada no PATH | Hugging Face CLI not on PATH."
  echo "  A CLI nova (hf) costuma ficar em ~/.local/bin | The new CLI (hf) is usually in ~/.local/bin. Faça | do:"
  echo "    pip install -U huggingface_hub"
  echo "    export PATH=\"\$HOME/.local/bin:\$PATH\"; hash -r"
  echo "    hf auth login"
  exit 1; }
echo "[hf] cli     : $HF_CLI"
[ -f "$MODEL_DIR/config.json" ] || { echo "ERRO | ERROR: $MODEL_DIR/config.json ausente | missing"; exit 1; }
ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1 || {
  echo "ERRO | ERROR: nenhum .safetensors em | no .safetensors in $MODEL_DIR"; exit 1; }
[ -f "$CARD" ] || { echo "ERRO | ERROR: model card ausente | missing: $CARD"; exit 1; }

# ── Monta o diretório de publicação | Assemble the publish directory ─────────
mkdir -p "$STAGE_DIR"
# 1) modelo (pesos + config + tokenizador do converter) | model (weights + config + converter tokenizer)
cp -a "$MODEL_DIR"/. "$STAGE_DIR"/
# 2) tokenizador corrigido sobrescreve o do converter | fixed tokenizer overrides the converter's
if [ -n "$TOKENIZER_DIR" ]; then
  [ -d "$TOKENIZER_DIR" ] || { echo "ERRO | ERROR: TOKENIZER_DIR não existe | does not exist: $TOKENIZER_DIR"; exit 1; }
  cp -a "$TOKENIZER_DIR"/. "$STAGE_DIR"/
else
  echo "[hf] AVISO | WARNING: sem TOKENIZER_DIR — verifique que o tokenizador do MODEL_DIR já tem"
  echo "[hf]                   o normalizador NFKC+Lowercase (rode fix_hf_tokenizer.py se não tiver)."
fi
# 3) model card -> README.md
cp "$CARD" "$STAGE_DIR/README.md"

# ── Verificação do normalizador (crítico) | Normalizer check (critical) ──────
python3 - "$STAGE_DIR" <<'PY' || echo "[hf] AVISO | WARNING: não foi possível verificar o normalizador | could not verify the normalizer"
import json, sys, pathlib
tj = pathlib.Path(sys.argv[1]) / "tokenizer.json"
if tj.exists():
    norm = json.load(open(tj, encoding="utf-8")).get("normalizer")
    s = json.dumps(norm or {})
    ok = ("Lowercase" in s) and ("NFKC" in s)
    print(f"[hf] normalizer NFKC+Lowercase: {'OK' if ok else 'AUSENTE/MISSING -> rode fix_hf_tokenizer.py'}")
else:
    print("[hf] sem tokenizer.json (fast) — o AutoTokenizer usará o SentencePiece (slow), que já normaliza.")
PY

echo
echo "[hf] conteúdo a publicar | contents to publish:"
ls -lh "$STAGE_DIR"

# ── Upload (cria o repo se não existir) | Upload (creates the repo if missing) ──
# 'hf upload' cria o repositório automaticamente se ele ainda não existir.
# 'hf upload' auto-creates the repository if it does not exist yet.
"$HF_CLI" upload "$HF_REPO" "$STAGE_DIR" . --repo-type model \
  --commit-message "Manacá-1B base: weights, fixed tokenizer, and model card"

echo
echo "[hf] pronto | done: https://huggingface.co/$HF_REPO"

#!/usr/bin/env bash
# =============================================================================
# Manaca-1B - Wrapper: converter checkpoint mcore -> HuggingFace (plano §7.10)
# -----------------------------------------------------------------------------
# Roda megatron_to_hf.py dentro da imagem de treino (que tem torch+transformers),
# LENDO a copia do checkpoint como somente-leitura (:ro) e ESCREVENDO num
# diretorio de saida NOVO. O original e o backup nunca sao tocados.
#
# Uso:
#   ./scripts/ckpt_converter/megatron_to_hf.sh \
#       /workspace/checkpoints/manaca-1b \
#       /workspace/manaca-1b-hf \
#       /caminho/para/manaca-tokenizer.model
#
# Imagem: usa ${TRAIN_IMAGE:-manaca-train:latest}. Se a conversao reclamar de
# transformers antigo (mlp_bias), atualize o transformers na imagem de treino
# (>=4.41) e reconstrua com `make build-train`.
# =============================================================================
set -euo pipefail

LOAD="${1:?uso: $0 <dir_checkpoint_copia> <dir_saida_hf> <tokenizer.model>}"
SAVE="${2:?falta o diretorio de saida HF}"
TOK="${3:?falta o caminho do manaca-tokenizer.model}"
IMAGE="${TRAIN_IMAGE:-manaca-train:latest}"

[ -d "$LOAD" ] || { echo "[ERRO] checkpoint nao encontrado: $LOAD" >&2; exit 1; }
[ -f "$TOK" ]  || { echo "[ERRO] tokenizer .model nao encontrado: $TOK" >&2; exit 1; }

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SAVE"

LOAD_ABS="$(realpath "$LOAD")"
SAVE_ABS="$(realpath "$SAVE")"
TOK_ABS="$(realpath "$TOK")"
TOK_DIR="$(dirname "$TOK_ABS")"
TOK_BASE="$(basename "$TOK_ABS")"

echo "[megatron_to_hf] imagem : $IMAGE"
echo "[megatron_to_hf] le de  : $LOAD_ABS  (montado :ro)"
echo "[megatron_to_hf] tok    : $TOK_ABS   (montado :ro)"
echo "[megatron_to_hf] grava  : $SAVE_ABS"

docker run --rm \
  -v "$LOAD_ABS":/ckpt:ro \
  -v "$TOK_DIR":/tok:ro \
  -v "$SAVE_ABS":/out \
  -v "$SELF_DIR":/conv:ro \
  "$IMAGE" bash -lc "python /conv/megatron_to_hf.py \
      --load-dir /ckpt \
      --save-dir /out \
      --tokenizer-model /tok/$TOK_BASE \
      --validate"

echo "[megatron_to_hf] pronto. Modelo HF em: $SAVE_ABS"
echo "[megatron_to_hf] teste rapido:"
echo "    python -c \"from transformers import AutoModelForCausalLM,AutoTokenizer; \\"
echo "      m=AutoModelForCausalLM.from_pretrained('$SAVE_ABS'); print(m.config)\""

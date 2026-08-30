#!/bin/sh
# =============================================================================
# Manacá — entrypoint comum dos containers
# =============================================================================
# Prepara o volume de trabalho (WORK_DIR) e o cache HuggingFace (HF_HOME) antes
# de executar o comando. Substitui o "source ~/.bashrc && conda activate ..." do
# fluxo HPC: no Docker o ambiente já está pronto na imagem.
set -e

WORK_DIR="${WORK_DIR:-/workspace/manaca-corpus}"
HF_HOME="${HF_HOME:-/workspace/hf-cache}"

# Subdiretórios usados pelos scripts de corpus (Script 00 os espera).
for d in raw processed deduped final stats logs checkpoints; do
    mkdir -p "${WORK_DIR}/${d}" 2>/dev/null || true
done
mkdir -p "${HF_HOME}/datasets" "${HF_HOME}/fasttext" 2>/dev/null || true

# Login opcional no HuggingFace Hub se HF_TOKEN estiver definido (fontes gated).
if [ -n "${HF_TOKEN:-}" ]; then
    export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

# Sem argumentos: abre um shell interativo.
if [ "$#" -eq 0 ]; then
    exec /bin/sh
fi

exec "$@"

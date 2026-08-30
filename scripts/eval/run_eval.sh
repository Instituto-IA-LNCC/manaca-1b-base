#!/usr/bin/env bash
# =============================================================================
# Manaca-1B - Wrapper de avaliacao base (BPB + CALAME-PT) na imagem de treino
# -----------------------------------------------------------------------------
# PT --------------------------------------------------------------------------
# Roda eval_base.py com GPU, montando:
#   - o modelo HF do Manaca em /m  (somente-leitura)
#   - o SPM do Manaca em /tok      (somente-leitura)
#   - um cache HF em /hf           (p/ baixar Tucano/Llama de comparacao)
#   - a pasta do script em /eval   (somente-leitura)
# Tudo que voce passar depois vai direto para o eval_base.py.
#
# Exemplos:
#   # Manaca (tokeniza fiel com o SPM):
#   ./scripts/eval/run_eval.sh --model /m --spm /tok/manaca-tokenizer.model --calame
#
#   # Tucano (aberto, baixa do hub):
#   ./scripts/eval/run_eval.sh --model TucanoBR/Tucano-1b1 --calame
#
#   # Llama-3.2-1B (gated: exige HF_TOKEN com licenca aceita):
#   HF_TOKEN=hf_xxx ./scripts/eval/run_eval.sh --model meta-llama/Llama-3.2-1B --calame
#
#   # comparacao justa de caixa (todos em minusculas):
#   ./scripts/eval/run_eval.sh --model /m --spm /tok/manaca-tokenizer.model --calame --lowercase
#
# Overrides por env: MANACA_HF, SPM, HF_CACHE, TRAIN_IMAGE, HF_TOKEN.
#
# EN --------------------------------------------------------------------------
# Manaca-1B - Base-evaluation wrapper (BPB + CALAME-PT) on the training image
# Runs eval_base.py with GPU, mounting:
#   - Manaca's HF model at /m  (read-only)
#   - Manaca's SPM at /tok     (read-only)
#   - an HF cache at /hf       (to download Tucano/Llama for comparison)
#   - the script folder at /eval (read-only)
# Anything you pass afterward goes straight to eval_base.py.
#
# Examples:
#   # Manaca (tokenizes faithfully with the SPM):
#   ./scripts/eval/run_eval.sh --model /m --spm /tok/manaca-tokenizer.model --calame
#
#   # Tucano (open, downloads from the hub):
#   ./scripts/eval/run_eval.sh --model TucanoBR/Tucano-1b1 --calame
#
#   # Llama-3.2-1B (gated: requires HF_TOKEN with the license accepted):
#   HF_TOKEN=hf_xxx ./scripts/eval/run_eval.sh --model meta-llama/Llama-3.2-1B --calame
#
#   # fair case comparison (everyone in lowercase):
#   ./scripts/eval/run_eval.sh --model /m --spm /tok/manaca-tokenizer.model --calame --lowercase
#
# Env overrides: MANACA_HF, SPM, HF_CACHE, TRAIN_IMAGE, HF_TOKEN.
# =============================================================================
set -euo pipefail

# Imagem DEDICADA de avaliacao (tem datasets/evaluate/vllm baked in). Se ainda nao
# construiu (make build-eval), rode com EVAL_IMAGE=manaca-train:latest para usar a
# imagem de treino (o script instala 'datasets' sob demanda como fallback).
IMAGE="${EVAL_IMAGE:-manaca-eval:latest}"
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
MANACA_HF="${MANACA_HF:-/workspace/manaca-1b-hf}"
SPM="${SPM:-/workspace/tokenizer/manaca-tokenizer.model}"
HF_CACHE="${HF_CACHE:-$HOME/hf_cache_eval}"

mkdir -p "$HF_CACHE"
[ -d "$MANACA_HF" ] || echo "[aviso] MANACA_HF nao existe ($MANACA_HF); ok se for avaliar so modelo do hub."

# Usa GPU por padrao (o treino ja usou --gpus all). Opt-out: NO_GPU=1.
GPU_FLAG=(--gpus all)
[ "${NO_GPU:-0}" = "1" ] && GPU_FLAG=()

# DNS: por padrao usa o DNS INTERNO do container (o do host/instituicao, que
# resolve). Neste host o DNS externo (8.8.8.8) e bloqueado, entao NAO forcamos.
# Se precisar de um DNS especifico: DNS_SERVERS="1.2.3.4 5.6.7.8".
DNS_FLAG=()
if [ -n "${DNS_SERVERS:-}" ]; then
  for d in $DNS_SERVERS; do DNS_FLAG+=(--dns "$d"); done
fi

# IPv6: o DNS resolve pypi/HF em IPv6, mas este host NAO tem rota IPv6, entao o pip
# trava tentando conectar. Desabilita IPv6 no container -> forca IPv4. Off: NO_IP4=1.
NET_FLAG=(--sysctl net.ipv6.conf.all.disable_ipv6=1
          --sysctl net.ipv6.conf.default.disable_ipv6=1)
[ "${NO_IP4:-0}" = "1" ] && NET_FLAG=()

# Log automatico: grava tudo num arquivo com timestamp (reprodutibilidade).
LOG_DIR="${LOG_DIR:-$HOME/manaca-eval-logs}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/eval_$(date +%Y%m%d_%H%M%S).log"
echo "[run_eval] gravando log em: $LOG"

set -o pipefail
docker run --rm -i "${GPU_FLAG[@]}" "${DNS_FLAG[@]}" "${NET_FLAG[@]}" \
  -e HF_HOME=/hf \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e HF_HUB_ENABLE_HF_TRANSFER=0 \
  -v "$MANACA_HF":/m:ro \
  -v "$(dirname "$SPM")":/tok:ro \
  -v "$HF_CACHE":/hf \
  -v "$SELF_DIR":/eval:ro \
  "$IMAGE" python /eval/eval_base.py "$@" 2>&1 | tee "$LOG"
# Mensagem final bilingue (PT + EN) — onde o log foi gravado.
echo "[run_eval] log salvo em: $LOG"
echo "[run_eval] log saved to: $LOG"

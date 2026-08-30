#!/usr/bin/env bash
# =============================================================================
# Manacá-1B — Pré-treino (Fase 2b) em Docker  [substitui o job SLURM do §7.7]
# =============================================================================
# PT --------------------------------------------------------------------------
# Hiperparâmetros FIÉIS ao LLM-jp-3.1-1.8B (arquitetura Llama-3, plano §7.2/§7.3),
# mas AJUSTADOS para 2× GPU de 24 GB SEM NVLink (ex.: 2× RTX A5000) e corpus curado ~20B:
#
#   - Paralelismo: DP=2 (TP=1, PP=1). O LLM-jp usava TP=2 intra-nó com NVLink;
#     aqui as GPUs são ligadas por PCIe (nvidia-smi topo = "NODE"), então tensor
#     parallel seria lento — data parallel + distributed optimizer (ZeRO-1) é melhor.
#   - Batch global de ~2M tokens (fiel ao LLM-jp): micro-batch 1 + acumulação.
#   - Recompute de ativações (gradient checkpointing) p/ caber em 24 GB.
#   - train-iters 20k (~42B tokens de atualização = ~2 épocas do corpus curado de ~20,1B).
#   - LR schedule cosine com warmup (o fork nii-geniac não tem WSD nos choices).
#   - --ffn-hidden-size 8192 (paridade exata com o LLM-jp-3.1-1.8B).
#
# Todos os valores são sobreponíveis por variáveis de ambiente. Exemplos no fim.
#
# Single-node (2 GPUs):
#   make build-train
#   ./scripts/docker/run_pretrain.sh
#
# Multinó (execute em CADA nó, variando NODE_RANK; MASTER_ADDR = IP do nó 0):
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 ./scripts/docker/run_pretrain.sh
#   NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 ./scripts/docker/run_pretrain.sh
#
# EN --------------------------------------------------------------------------
# Manacá-1B — Pre-training (Phase 2b) in Docker  [replaces the SLURM job from §7.7]
# Hyperparameters FAITHFUL to LLM-jp-3.1-1.8B (Llama-3 architecture, plan §7.2/§7.3),
# but ADJUSTED for 2× 24 GB GPUs WITHOUT NVLink (e.g.: 2× RTX A5000) and a curated corpus ~20B:
#
#   - Parallelism: DP=2 (TP=1, PP=1). LLM-jp used TP=2 intra-node with NVLink;
#     here the GPUs are connected by PCIe (nvidia-smi topo = "NODE"), so tensor
#     parallel would be slow — data parallel + distributed optimizer (ZeRO-1) is better.
#   - Global batch of ~2M tokens (faithful to LLM-jp): micro-batch 1 + accumulation.
#   - Activation recompute (gradient checkpointing) to fit in 24 GB.
#   - train-iters 20k (~42B update tokens = ~2 epochs of the ~20.1B curated corpus).
#   - Cosine LR schedule with warmup (the nii-geniac fork has no WSD in its choices).
#   - --ffn-hidden-size 8192 (exact parity with LLM-jp-3.1-1.8B).
#
# All values are overridable by environment variables. Examples at the end.
#
# Single-node (2 GPUs):
#   make build-train
#   ./scripts/docker/run_pretrain.sh
#
# Multi-node (run on EACH node, varying NODE_RANK; MASTER_ADDR = IP of node 0):
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 ./scripts/docker/run_pretrain.sh
#   NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 ./scripts/docker/run_pretrain.sh
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# Carrega .env como DEFAULTS — ambiente/CLI têm precedência sobre o .env.
load_env_defaults() {
    [ -f "$1" ] || return 0
    local line key
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in ''|\#*) continue ;; esac
        key=${line%%=*}; key=${key// /}
        case "$key" in ''|*[!A-Za-z0-9_]*) continue ;; esac
        [ -z "${!key+x}" ] && export "${key}=${line#*=}"
    done < "$1"
    # IMPORTANTE: retorno explícito 0. Sem isto, se a última linha do .env for
    # uma variável já definida no ambiente (ex.: você sobrepôs GPUS_PER_NODE na
    # CLI), o '&&' acima retorna 1, e o 'set -e' mataria o script silenciosamente.
    return 0
}
load_env_defaults .env

IMAGE="${TRAIN_IMAGE:-manaca-train:latest}"
DATA_DIR="$(realpath -m "${DATA_DIR:-./data}")"
HF_CACHE_DIR="$(realpath -m "${HF_CACHE_DIR:-./hf-cache}")"
CKPT_DIR="$(realpath -m "${CKPT_DIR:-./checkpoints}")"

# Topologia — padrão 2 GPUs (hardware acessível). Sobreponha via .env/CLI.
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"

MODEL="${MODEL:-manaca-1b}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"            # imprime a loss a cada N iters
SAVE="${SAVE:-1}"                             # 0 = não salva checkpoint (testes rápidos)
DATA_PATH="${DATA_PATH:-/workspace/manaca-corpus/megatron-data/manaca_text_text_document}"
VOCAB_FILE="${VOCAB_FILE:-/workspace/manaca-corpus/tokenizer/manaca-tokenizer.model}"

# ── Arquitetura (FIEL ao LLM-jp-3.1-1.8B) ────────────────────────────────────
NUM_LAYERS="${NUM_LAYERS:-24}"
HIDDEN="${HIDDEN:-2048}"
FFN_HIDDEN="${FFN_HIDDEN:-8192}"
NUM_HEADS="${NUM_HEADS:-32}"
KV_GROUPS="${KV_GROUPS:-8}"
SEQ_LEN="${SEQ_LEN:-4096}"
ROPE_BASE="${ROPE_BASE:-500000}"

# ── Paralelismo (2× 24 GB sem NVLink → DP) ───────────────────────────────────
TP="${TP:-1}"
PP="${PP:-1}"

# ── Batch (global ~2M tokens, fiel ao LLM-jp) ────────────────────────────────
MICRO_BATCH="${MICRO_BATCH:-1}"
GLOBAL_BATCH="${GLOBAL_BATCH:-512}"    # 512 × 4096 ≈ 2,1M tokens

# ── Cronograma (~42B de atualização ⇒ 20k iters; LLM-jp usa 200B+ em cluster) ─
TRAIN_ITERS="${TRAIN_ITERS:-20000}"
LR="${LR:-3e-4}"
MIN_LR="${MIN_LR:-3e-5}"
WARMUP="${WARMUP:-2000}"
# O scheduler exige lr_warmup_iters < lr_decay_iters (= TRAIN_ITERS aqui). Em
# runs curtos de teste (ex.: TRAIN_ITERS=50) o warmup padrão 2000 quebraria;
# reduzimos automaticamente. No treino real (20000) nada muda.
if [ "${WARMUP}" -ge "${TRAIN_ITERS}" ]; then
    WARMUP=$(( TRAIN_ITERS / 10 )); [ "${WARMUP}" -lt 1 ] && WARMUP=1
    echo "[run_pretrain] WARMUP >= TRAIN_ITERS; reduzido para ${WARMUP} (run curto)."
fi
# Este fork (nii-geniac) só aceita constant|linear|cosine|inverse-square-root
# em --lr-decay-style (NÃO tem WSD). cosine é o schedule padrão e adequado.
LR_DECAY_STYLE="${LR_DECAY_STYLE:-cosine}"

# ── Memória (recompute p/ caber em 24 GB) ────────────────────────────────────
# 'full' é o padrão porque, em 2× A5000 (24 GB), 'selective' estoura por ~200 MB
# no pico da cross-entropy (logits [1,4096,64128] em fp32 ≈ 1 GB). 'full'
# recomputa as ativações no backward e libera vários GB. Com mais VRAM, use
# 'selective' (≈30% mais rápido).
RECOMPUTE="${RECOMPUTE:-full}"                 # full | selective | none

Z_LOSS="${Z_LOSS:-1e-4}"

mkdir -p "${CKPT_DIR}/${MODEL}" "${DATA_DIR}" "${HF_CACHE_DIR}"

# Resume automático: só carrega se houver iter_* + tracker consistente.
LOAD_FLAG=()
if compgen -G "${CKPT_DIR}/${MODEL}/iter_*" >/dev/null 2>&1 \
   && [ -f "${CKPT_DIR}/${MODEL}/latest_checkpointed_iteration.txt" ]; then
    echo "[run_pretrain] Checkpoint encontrado — retomando de ${CKPT_DIR}/${MODEL}"
    LOAD_FLAG=(--load "/workspace/checkpoints/${MODEL}")
fi

# Salvamento (SAVE=0 desabilita — útil p/ smoke tests, pois o fork salva a cada
# iteração nas 10 primeiras, e cada save do estado do optimizer leva ~90s).
SAVE_FLAG=(--save "/workspace/checkpoints/${MODEL}")
if [ "${SAVE}" = "0" ]; then
    SAVE_FLAG=()
    echo "[run_pretrain] SAVE=0 — checkpoints desabilitados (teste)."
fi

# ── Montagem dos argumentos do Megatron ──────────────────────────────────────
MEGATRON_ARGS=(
  --num-layers "${NUM_LAYERS}" --hidden-size "${HIDDEN}"
  --ffn-hidden-size "${FFN_HIDDEN}"
  --num-attention-heads "${NUM_HEADS}"
  --group-query-attention --num-query-groups "${KV_GROUPS}"
  --seq-length "${SEQ_LEN}" --max-position-embeddings "${SEQ_LEN}"
  --micro-batch-size "${MICRO_BATCH}" --global-batch-size "${GLOBAL_BATCH}"
  --lr "${LR}" --min-lr "${MIN_LR}" --lr-warmup-iters "${WARMUP}"
  --weight-decay 0.1 --clip-grad 1.0
  --train-iters "${TRAIN_ITERS}"
  --log-interval "${LOG_INTERVAL}" --eval-interval 1000 --eval-iters 100
  --tensorboard-dir "/workspace/checkpoints/${MODEL}/tensorboard"
  --log-timers-to-tensorboard --log-validation-ppl-to-tensorboard
  --data-path "${DATA_PATH}"
  --tokenizer-model "${VOCAB_FILE}"
  --tokenizer-type SentencePieceTokenizer
  --bf16
  --use-mcore-models
  --tensor-model-parallel-size "${TP}" --pipeline-model-parallel-size "${PP}"
  --use-flash-attn
  --normalization RMSNorm --position-embedding-type rope --rope-theta "${ROPE_BASE}"
  --swiglu --untie-embeddings-and-output-weights
  --use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather
)

# z-loss (estabilidade numérica, PaLM/LLM-jp) — neste fork é flag booleana
# (--use-z-loss), não recebe peso. Ativa se Z_LOSS != 0.
if [ "${Z_LOSS}" != "0" ] && [ -n "${Z_LOSS}" ]; then
    MEGATRON_ARGS+=(--use-z-loss)
fi

# LR schedule. WSD não existe neste fork — se pedido, cai para cosine.
if [ "${LR_DECAY_STYLE}" = "WSD" ]; then
    echo "[run_pretrain] WSD não é suportado por este fork do Megatron; usando cosine."
    LR_DECAY_STYLE="cosine"
fi
MEGATRON_ARGS+=(--lr-decay-style "${LR_DECAY_STYLE}" --lr-decay-iters "${TRAIN_ITERS}")

# Recompute de ativações (memória)
case "${RECOMPUTE}" in
    selective) MEGATRON_ARGS+=(--recompute-activations) ;;
    full)      MEGATRON_ARGS+=(--recompute-granularity full --recompute-method uniform --recompute-num-layers 1) ;;
    none|"")   ;;
esac

# sequence-parallel só faz sentido com TP>1
if [ "${TP}" -gt 1 ]; then
    MEGATRON_ARGS+=(--sequence-parallel)
fi

echo "[run_pretrain] modelo=${MODEL} gpus=${GPUS_PER_NODE} TP=${TP} PP=${PP} DP=$((NNODES*GPUS_PER_NODE/TP/PP))"
echo "[run_pretrain] global_batch=${GLOBAL_BATCH} (~$((GLOBAL_BATCH*SEQ_LEN/1000000))M tokens) micro=${MICRO_BATCH} iters=${TRAIN_ITERS} lr_style=${LR_DECAY_STYLE} recompute=${RECOMPUTE}"

# Log persistente (além do stdout ao vivo no tmux). Fica junto do modelo, no
# disco montado; o autoprune só mexe em iter_*, então o .log é preservado.
LOG_TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${CKPT_DIR}/${MODEL}"
LOG_FILE="${RUN_DIR}/train_${LOG_TS}.log"
PROV_FILE="${RUN_DIR}/provenance_${LOG_TS}.txt"
mkdir -p "${RUN_DIR}"

# ── Provenance: registro científico da execução (estilo LLM-jp) ──────────────
# Grava commit do repo, hardware, hiperparâmetros efetivos, versões e o comando
# completo — tudo no volume de dados, para citar/reproduzir no artigo.
{
  echo "Manacá-1B — Registro de execução (provenance)"
  echo "============================================="
  echo "Timestamp:    ${LOG_TS}"
  echo "Host:         $(hostname)   User: $(id -un)"
  echo "Repo commit:  $(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo '?')"
  echo "Repo branch:  $(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
  echo "Repo dirty:   $(git -C "${REPO_ROOT}" status --porcelain 2>/dev/null | wc -l) arquivo(s) modificado(s) sem commit"
  echo "Imagem:       ${IMAGE}"
  echo "Megatron:     ${MEGATRON_REPO:-?} @ ${MEGATRON_REF:-?}"
  echo ""
  echo "── GPUs / driver ──"
  nvidia-smi -L 2>/dev/null || echo "(nvidia-smi indisponível)"
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
  echo ""
  echo "── Hiperparâmetros efetivos ──"
  echo "MODEL=${MODEL}  SEED=1234"
  echo "NUM_LAYERS=${NUM_LAYERS} HIDDEN=${HIDDEN} FFN_HIDDEN=${FFN_HIDDEN} NUM_HEADS=${NUM_HEADS} KV_GROUPS=${KV_GROUPS}"
  echo "SEQ_LEN=${SEQ_LEN} ROPE_BASE=${ROPE_BASE}"
  echo "NNODES=${NNODES} GPUS_PER_NODE=${GPUS_PER_NODE} TP=${TP} PP=${PP} DP=$((NNODES*GPUS_PER_NODE/TP/PP))"
  echo "MICRO_BATCH=${MICRO_BATCH} GLOBAL_BATCH=${GLOBAL_BATCH} TRAIN_ITERS=${TRAIN_ITERS}"
  echo "LR=${LR} MIN_LR=${MIN_LR} WARMUP=${WARMUP} LR_DECAY_STYLE=${LR_DECAY_STYLE}"
  echo "RECOMPUTE=${RECOMPUTE} Z_LOSS=${Z_LOSS}"
  echo "DATA_PATH=${DATA_PATH}"
  echo "TOKENIZER=${VOCAB_FILE}"
  echo ""
  echo "── Comando Megatron (argumentos completos) ──"
  printf '  %s\n' "${MEGATRON_ARGS[@]}"
} > "${PROV_FILE}"
# Versões dentro do container (torch, commit exato do Megatron) — não-fatal.
{
  echo ""
  echo "── Versões (container) ──"
  docker run --rm "${IMAGE}" sh -c \
    'python -c "import torch;print(\"torch\",torch.__version__)"; printf "megatron-commit "; git -C /opt/Megatron-LM rev-parse HEAD'
} >> "${PROV_FILE}" 2>/dev/null || true

echo "[run_pretrain] log:        ${LOG_FILE}"
echo "[run_pretrain] provenance: ${PROV_FILE}"
echo "[run_pretrain]   acompanhar a loss:  tail -f '${LOG_FILE}' | grep 'lm loss'"

# Sem 'exec': precisamos do pipe p/ o tee (stdout ao vivo + arquivo). O
# 'set -o pipefail' garante que a falha do docker propague o código de saída.
docker run --rm --gpus all --network host --ipc host \
  --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
  -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -e WORK_DIR=/workspace/manaca-corpus \
  -e HF_HOME=/workspace/hf-cache \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -e WANDB_PROJECT="${WANDB_PROJECT:-manaca-pretrain}" \
  -v "${REPO_ROOT}":/workspace/manaca \
  -v "${DATA_DIR}":/workspace/manaca-corpus \
  -v "${HF_CACHE_DIR}":/workspace/hf-cache \
  -v "${CKPT_DIR}":/workspace/checkpoints \
  -w /workspace/manaca \
  "${IMAGE}" \
  torchrun \
    --nnodes "${NNODES}" --node_rank "${NODE_RANK}" \
    --nproc_per_node "${GPUS_PER_NODE}" \
    --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}" \
    /opt/Megatron-LM/pretrain_gpt.py \
    "${MEGATRON_ARGS[@]}" ${SAVE_FLAG[@]+"${SAVE_FLAG[@]}"} ${LOAD_FLAG[@]+"${LOAD_FLAG[@]}"} "$@" \
  2>&1 | tee -a "${LOG_FILE}"

# =============================================================================
# Ajustes finos comuns (sobreponha por env):
#   Ainda com OOM?       SEQ_LEN=2048 ./scripts/docker/run_pretrain.sh (menos ativação)
#   Tem VRAM sobrando?   RECOMPUTE=selective ./scripts/docker/run_pretrain.sh (~30% + rápido)
#   Trocar schedule:     LR_DECAY_STYLE=constant ./scripts/docker/run_pretrain.sh
#   Mais/menos tokens:   TRAIN_ITERS=40000 ./scripts/docker/run_pretrain.sh
#   Testar em 1 GPU:     GPUS_PER_NODE=1 TRAIN_ITERS=50 ./scripts/docker/run_pretrain.sh
# =============================================================================

#!/usr/bin/env bash
# =============================================================================
# Manaca-1B - ARC-Challenge-PT / HellaSwag-PT / LAMBADA-PT via lm-eval-harness
# -----------------------------------------------------------------------------
# PT --------------------------------------------------------------------------
# Roda os tres benchmarks de multipla escolha / ultima palavra no MESMO protocolo
# do Tucano, para todos os modelos, dentro da imagem de avaliacao (que tem lm-eval,
# ver requirements/eval.txt).
#
#   arc_pt        25-shot  acc_norm   (dataset alexandrainst/m_arc, config pt)
#   hellaswag_pt  10-shot  acc_norm   (dataset alexandrainst/m_hellaswag, config pt)
#   lambada_pt     0-shot  acc        (YAML proprio em lm_eval_tasks/, TucanoBR/lambada-pt)
#
# Uso (dentro de tmux):
#   ./scripts/eval/build_lmeval_image.sh  # cria manaca-lmeval (run+commit; DNS ok)
#   ./scripts/eval/run_lm_eval_pt.sh      # roda os 9 modelos do hub + Manaca
# (Em hosts com DNS/IPv6 problematico use o build_lmeval_image.sh, NAO `docker build`: o build nao aceita
#  --sysctl e o DNS quebra com IPv6. docker/Dockerfile.lmeval so serve onde o DNS
#  do build funciona. Alternativa: EVAL_IMAGE=manaca-eval:latest, imagem com vLLM.)
#
# Pre-requisito do Manaca: o lm-eval usa o tokenizador HF do modelo. Se o HF do
# Manaca ainda nao reproduz o nmt_nfkc_cf, o numero sai injustamente baixo. Rode
# antes scripts/eval/fix_hf_tokenizer.py e exporte
#   MANACA_TOKENIZER=/caminho/para/tokenizador-corrigido
#
# Overrides por env: EVAL_IMAGE, MANACA_HF, MANACA_TOKENIZER, HF_CACHE, OUT_DIR,
#   HF_TOKEN (p/ modelos com licenca), ONLY="label1 label2" (subconjunto).
#
# EN --------------------------------------------------------------------------
# Manaca-1B - ARC-Challenge-PT / HellaSwag-PT / LAMBADA-PT via lm-eval-harness
# Runs the three multiple-choice / last-word benchmarks under the SAME protocol
# as Tucano, for all models, inside the evaluation image (which has lm-eval,
# see requirements/eval.txt).
#
#   arc_pt        25-shot  acc_norm   (dataset alexandrainst/m_arc, config pt)
#   hellaswag_pt  10-shot  acc_norm   (dataset alexandrainst/m_hellaswag, config pt)
#   lambada_pt     0-shot  acc        (own YAML in lm_eval_tasks/, TucanoBR/lambada-pt)
#
# Usage (inside tmux):
#   ./scripts/eval/build_lmeval_image.sh  # creates manaca-lmeval (run+commit; DNS ok)
#   ./scripts/eval/run_lm_eval_pt.sh      # runs the 9 hub models + Manaca
# (On hosts with problematic DNS/IPv6 use build_lmeval_image.sh, NOT `docker build`: the build does not accept
#  --sysctl and DNS breaks with IPv6. docker/Dockerfile.lmeval only works where the
#  build's DNS works. Alternative: EVAL_IMAGE=manaca-eval:latest, an image with vLLM.)
#
# Manaca prerequisite: lm-eval uses the model's HF tokenizer. If Manaca's HF still
# does not reproduce nmt_nfkc_cf, its number comes out unfairly low. First run
# scripts/eval/fix_hf_tokenizer.py and export
#   MANACA_TOKENIZER=/path/to/fixed-tokenizer
#
# Env overrides: EVAL_IMAGE, MANACA_HF, MANACA_TOKENIZER, HF_CACHE, OUT_DIR,
#   HF_TOKEN (for licensed models), ONLY="label1 label2" (subset).
# =============================================================================
set -uo pipefail

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE="${EVAL_IMAGE:-manaca-lmeval:latest}"
MANACA_HF="${MANACA_HF:-/workspace/manaca-1b-hf}"
MANACA_TOK="${MANACA_TOKENIZER:-}"
HF_CACHE="${HF_CACHE:-$HOME/hf_cache_eval}"
OUT_DIR="${OUT_DIR:-$HOME/manaca-lmeval-out}"
mkdir -p "$HF_CACHE" "$OUT_DIR"

GPU_FLAG=(--gpus all); [ "${NO_GPU:-0}" = "1" ] && GPU_FLAG=()
# Sem rota IPv6 neste host: forca IPv4 no container (mesmo motivo do run_eval.sh).
NET_FLAG=(--sysctl net.ipv6.conf.all.disable_ipv6=1 --sysctl net.ipv6.conf.default.disable_ipv6=1)
[ "${NO_IP4:-0}" = "1" ] && NET_FLAG=()

# label|hf_id  (Manaca entra a parte, e local)
HUB=(
  "gloria-1b3|NOVA-vision-language/GlorIA-1.3B"
  "mgpt-1b3|ai-forever/mGPT"
  "ttl-160m|nicholasKluge/TeenyTinyLlama-160m"
  "ttl-460m|nicholasKluge/TeenyTinyLlama-460m"
  "tucano-160m|TucanoBR/Tucano-160m"
  "tucano-630m|TucanoBR/Tucano-630m"
  "tucano-1b1|TucanoBR/Tucano-1b1"
  "tucano-2b4|TucanoBR/Tucano-2b4"
  "sabia-7b|maritaca-ai/sabia-7b"
)
# tarefa:few-shot  (HellaSwag = 10 conforme o artigo do Tucano; o codigo deles usa 0)
TAREFAS=( "arc_pt:25" "hellaswag_pt:10" "lambada_pt:0" )

# --log_samples grava o acerto por exemplo (necessario para o McNemar pareado, ver
# scripts/eval/paired_lm_eval.py). Ligado por padrao; desligue com LOG_SAMPLES=0.
SAMPLES_FLAG=""
[ "${LOG_SAMPLES:-1}" = "1" ] && SAMPLES_FLAG="--log_samples"

selecionado() {  # respeita ONLY="a b c" se definido
  [ -z "${ONLY:-}" ] && return 0
  for x in $ONLY; do [ "$x" = "$1" ] && return 0; done
  return 1
}

ja_tem_resultado() {  # $1=label $2=task -> 0 se ja existe um results*.json
  find "$OUT_DIR/$1/$2" -name 'results*.json' 2>/dev/null | grep -q .
}

run_one() {  # $1=label  $2=model_args
  local label="$1" margs="$2"
  for tf in "${TAREFAS[@]}"; do
    local task="${tf%%:*}" shots="${tf##*:}"
    if [ "${FORCE:-0}" != "1" ] && ja_tem_resultado "$label" "$task"; then
      echo "[skip] $label/$task ja tem resultado (use FORCE=1 para refazer)"; continue
    fi
    echo "======== $label : $task (${shots}-shot) ========"
    local log="$OUT_DIR/lmeval_${label}_${task}_$(date +%Y%m%d_%H%M%S).log"
    docker run --rm -i "${GPU_FLAG[@]}" "${NET_FLAG[@]}" \
      -e HF_HOME=/hf -e HF_TOKEN="${HF_TOKEN:-}" \
      -v "$MANACA_HF":/m:ro ${MANACA_TOK:+-v "$MANACA_TOK":/mtok:ro} \
      -v "$SELF_DIR/lm_eval_tasks":/tasks:ro \
      -v "$HF_CACHE":/hf -v "$OUT_DIR":/out \
      "$IMAGE" lm_eval --model hf \
        --model_args "$margs" \
        --tasks "$task" --num_fewshot "$shots" \
        --include_path /tasks --batch_size auto ${SAMPLES_FLAG} \
        --output_path "/out/${label}/${task}" 2>&1 | tee "$log" \
      || echo "[AVISO] $label/$task falhou (segue)."
  done
}

for entry in "${HUB[@]}"; do
  label="${entry%%|*}"; hf="${entry#*|}"
  selecionado "$label" || continue
  run_one "$label" "pretrained=${hf},dtype=bfloat16"
done

if selecionado "manaca"; then
  if [ -n "$MANACA_TOK" ]; then
    run_one "manaca" "pretrained=/m,tokenizer=/mtok,dtype=bfloat16"
  else
    echo "[AVISO] Rodando Manaca com o tokenizador do proprio /m."
    echo "[AVISO] Se ele nao tiver NFKC+Lowercase, o numero do Manaca sai injusto."
    echo "[AVISO] Rode fix_hf_tokenizer.py e exporte MANACA_TOKENIZER antes."
    run_one "manaca" "pretrained=/m,dtype=bfloat16"
  fi
fi

echo
# Mensagem final bilingue (PT + EN) — onde ficaram as saidas e como juntar a tabela.
echo "FIM. Saidas em $OUT_DIR (JSON em <label>/<task>/). Junte a tabela com:"
echo "  python scripts/eval/merge_pt_benchmarks.py --lm-eval-dir $OUT_DIR"
echo "END. Outputs in $OUT_DIR (JSON in <label>/<task>/). Merge the table with:"
echo "  python scripts/eval/merge_pt_benchmarks.py --lm-eval-dir $OUT_DIR"

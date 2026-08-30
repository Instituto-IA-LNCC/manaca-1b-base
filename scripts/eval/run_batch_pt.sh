#!/usr/bin/env bash
# =============================================================================
# Manaca-1B - Lote de avaliacao CALAME-PT com modelos PT/PT-BR de referencia
# -----------------------------------------------------------------------------
# PT --------------------------------------------------------------------------
# Roda o mesmo protocolo (eval_base.py via run_eval.sh) para varios modelos, no
# MESMO harness, salvando para cada um:
#   - um log com timestamp em $HOME/manaca-eval-logs/
#   - o vetor de acertos por exemplo em $HOME/hf_cache_eval/<label>_calame.json
#     (necessario para os testes pareados de McNemar).
#
# Rode dentro de tmux (resiliente a queda de conexao):
#   tmux new -s lote
#   EVAL_IMAGE=manaca-train:latest ./scripts/eval/run_batch_pt.sh
#   (Ctrl-b d para destacar; tmux attach -t lote para voltar)
#
# Modelos que ja temos (nao re-rodados aqui): Manaca-1B, Tucano-1b1, Tucano-2b4.
# Este lote fecha a curva de escala PT-BR e adiciona GlorIA, mGPT e Sabia.
#
# Observacoes:
#   - Sabia-7B tem ~13 GB de download e mais memoria de GPU; se a licenca exigir,
#     exporte HF_TOKEN antes (HF_TOKEN=hf_xxx ./run_batch_pt.sh).
#   - Um modelo que falhar nao interrompe o lote (segue para o proximo).
#
# EN --------------------------------------------------------------------------
# Manaca-1B - CALAME-PT evaluation batch with PT/PT-BR reference models
# Runs the same protocol (eval_base.py via run_eval.sh) for several models, on the
# SAME harness, saving for each one:
#   - a timestamped log in $HOME/manaca-eval-logs/
#   - the per-example correctness vector in $HOME/hf_cache_eval/<label>_calame.json
#     (needed for the paired McNemar tests).
#
# Run inside tmux (resilient to connection drops):
#   tmux new -s lote
#   EVAL_IMAGE=manaca-train:latest ./scripts/eval/run_batch_pt.sh
#   (Ctrl-b d to detach; tmux attach -t lote to return)
#
# Models we already have (not re-run here): Manaca-1B, Tucano-1b1, Tucano-2b4.
# This batch closes the PT-BR scaling curve and adds GlorIA, mGPT and Sabia.
#
# Notes:
#   - Sabia-7B has a ~13 GB download and needs more GPU memory; if the license
#     requires it, export HF_TOKEN first (HF_TOKEN=hf_xxx ./run_batch_pt.sh).
#   - A model that fails does not interrupt the batch (it moves on to the next).
# =============================================================================
set -uo pipefail

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"

# label|hf_id  (o label vira o nome do vetor <label>_calame.json)
MODELOS=(
  "gloria-1b3|NOVA-vision-language/GlorIA-1.3B"
  "mgpt-1b3|ai-forever/mGPT"
  "ttl-160m|nicholasKluge/TeenyTinyLlama-160m"
  "ttl-460m|nicholasKluge/TeenyTinyLlama-460m"
  "tucano-160m|TucanoBR/Tucano-160m"
  "tucano-630m|TucanoBR/Tucano-630m"
  "sabia-7b|maritaca-ai/sabia-7b"
)

ok=(); falhou=()
for entry in "${MODELOS[@]}"; do
  label="${entry%%|*}"; hf="${entry#*|}"
  echo
  echo "########################################################################"
  echo "# $label  ($hf)"
  echo "########################################################################"
  if EVAL_IMAGE="${EVAL_IMAGE:-manaca-train:latest}" \
     "$SELF_DIR/run_eval.sh" --model "$hf" \
        --text /eval/holdout_pt.txt --calame \
        --save-calame "/hf/${label}_calame.json"; then
    ok+=("$label")
  else
    echo "[AVISO] $label falhou; seguindo para o proximo."
    falhou+=("$label")
  fi
done

echo
# Resumo final bilingue (PT + EN) — o resultado do lote que o usuario le.
echo "======================== RESUMO DO LOTE ========================"
echo "OK    : ${ok[*]:-(nenhum)}"
echo "FALHOU: ${falhou[*]:-(nenhum)}"
echo "Vetores em: $HOME/hf_cache_eval/  (copie os *_calame.json para o repo)"
echo "Logs em   : $HOME/manaca-eval-logs/"
echo "======================== BATCH SUMMARY ========================"
echo "OK     : ${ok[*]:-(nenhum)}"
echo "FAILED : ${falhou[*]:-(nenhum)}"
echo "Vectors in: $HOME/hf_cache_eval/  (copy the *_calame.json into the repo)"
echo "Logs in   : $HOME/manaca-eval-logs/"

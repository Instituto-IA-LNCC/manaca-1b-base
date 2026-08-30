#!/usr/bin/env bash
# =============================================================================
# Manacá — Backup de checkpoint Megatron ANTES da conversão para HuggingFace
# =============================================================================
# PT --------------------------------------------------------------------------
# Por quê: a conversão Megatron -> HF (plano §7.10) lê o checkpoint e escreve o
# safetensors em OUTRO diretório. Ainda assim, um bug no conversor, a etapa de
# consolidação TP/PP, ou um engano de caminho podem corromper/apagar o original.
# Este script faz uma cópia IMUTÁVEL (somente-leitura) e VERIFICADA por SHA-256
# do(s) checkpoint(s), para sempre existir uma versão pré-conversão intacta.
#
# O script SÓ LÊ o diretório de origem (nunca escreve nele).
#
# Uso típico (na máquina de treino, após as 20k iterações):
#   ./scripts/backup_checkpoint.sh                 # backup do último checkpoint
#   ./scripts/backup_checkpoint.sh --last 2        # backup dos 2 mais recentes
#   ./scripts/backup_checkpoint.sh --iter 20000    # backup de uma iteração específica
#   ./scripts/backup_checkpoint.sh --tar zst       # + tarball .tar.zst p/ off-site
#   ./scripts/backup_checkpoint.sh --verify        # só re-verifica um backup existente
#
# Espaço: um checkpoint deste 1.8B (pesos bf16 + distributed optimizer fp32) tem
# ~20-30 GB. Aponte BACKUP_DIR para um disco DIFERENTE do CKPT_DIR (proteção real
# contra falha de disco). --no-optim exclui o estado do optimizer (distrib_optim*)
# e reduz ~metade+ do tamanho: suficiente para a conversão HF, mas NÃO permite
# retomar o treino a partir do backup.
#
# EN --------------------------------------------------------------------------
# Manacá — Megatron checkpoint backup BEFORE the conversion to HuggingFace
# Why: the Megatron -> HF conversion (plan §7.10) reads the checkpoint and writes
# the safetensors to ANOTHER directory. Even so, a bug in the converter, the TP/PP
# consolidation step, or a wrong path could corrupt/delete the original.
# This script makes an IMMUTABLE (read-only) copy of the checkpoint(s), VERIFIED by
# SHA-256, so that an intact pre-conversion version always exists.
#
# The script ONLY READS the source directory (never writes to it).
#
# Typical usage (on the training machine, after the 20k iterations):
#   ./scripts/backup_checkpoint.sh                 # backup of the latest checkpoint
#   ./scripts/backup_checkpoint.sh --last 2        # backup of the 2 most recent
#   ./scripts/backup_checkpoint.sh --iter 20000    # backup of a specific iteration
#   ./scripts/backup_checkpoint.sh --tar zst       # + .tar.zst tarball for off-site
#   ./scripts/backup_checkpoint.sh --verify        # only re-verify an existing backup
#
# Space: a checkpoint of this 1.8B (bf16 weights + fp32 distributed optimizer) is
# ~20-30 GB. Point BACKUP_DIR to a DIFFERENT disk from CKPT_DIR (real protection
# against disk failure). --no-optim excludes the optimizer state (distrib_optim*)
# and cuts ~half+ of the size: enough for the HF conversion, but does NOT allow
# resuming training from the backup.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Carrega .env como DEFAULTS — ambiente/CLI têm precedência (igual run_pretrain.sh).
load_env_defaults() {
    [ -f "$1" ] || return 0
    local line key
    while IFS= read -r line || [ -n "$line" ]; do
        case "$line" in ''|\#*) continue ;; esac
        key=${line%%=*}; key=${key// /}
        case "$key" in ''|*[!A-Za-z0-9_]*) continue ;; esac
        [ -z "${!key+x}" ] && export "${key}=${line#*=}"
    done < "$1"
    return 0
}
load_env_defaults .env

MODEL="${MODEL:-manaca-1b}"
CKPT_DIR="$(realpath -m "${CKPT_DIR:-./checkpoints}")"
# Backup, por padrão, ao lado dos checkpoints. RECOMENDADO: outro disco (env).
BACKUP_DIR="$(realpath -m "${BACKUP_DIR:-${CKPT_DIR}-backup}")"

SRC="${CKPT_DIR}/${MODEL}"
DST_ROOT="${BACKUP_DIR}/${MODEL}"

# ── Parsing de argumentos ────────────────────────────────────────────────────
ITER=""; LAST=1; NO_OPTIM=0; TAR=""; FORCE=0; VERIFY_ONLY=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --iter)      ITER="$2"; shift 2 ;;
        --last)      LAST="$2"; shift 2 ;;
        --no-optim)  NO_OPTIM=1; shift ;;
        --tar)       TAR="${2:-none}"; shift 2 ;;   # none|gz|zst
        --tar=*)     TAR="${1#*=}"; shift ;;
        --force)     FORCE=1; shift ;;
        --verify)    VERIFY_ONLY=1; shift ;;
        -h|--help)
            sed -n '2,49p' "$0"; exit 0 ;;
        *) echo "[backup] argumento desconhecido: $1"; exit 2 ;;
    esac
done

log() { echo "[backup] $*"; }
die() { echo "[backup] ERRO: $*" >&2; exit 1; }

command -v sha256sum >/dev/null || die "sha256sum não encontrado (instale coreutils)."
[ -d "$SRC" ] || die "origem não existe: ${SRC}  (rode na máquina de treino; ajuste CKPT_DIR/MODEL no .env)"

# Copiadora: rsync se houver (resumível, mais seguro), senão cp -a.
if command -v rsync >/dev/null; then COPY="rsync"; else COPY="cp"; fi

# ── Modo --verify: só re-checa a integridade de um backup existente ──────────
if [ "$VERIFY_ONLY" = "1" ]; then
    [ -d "$DST_ROOT" ] || die "não há backup em ${DST_ROOT}"
    rc=0
    while IFS= read -r m; do
        d="$(dirname "$m")"
        log "verificando $(basename "$d") ..."
        ( cd "$d" && sha256sum -c --quiet MANIFEST.sha256 ) || rc=1
    done < <(find "$DST_ROOT" -name MANIFEST.sha256 | sort)
    [ "$rc" = "0" ] && log "OK — backup íntegro." || die "backup CORROMPIDO (hashes divergem)."
    exit $rc
fi

# ── Descobrir quais iterações copiar ─────────────────────────────────────────
declare -a ITERS=()
if [ -n "$ITER" ]; then
    d="$(printf '%s/iter_%07d' "$SRC" "$ITER")"
    [ -d "$d" ] || die "iteração não encontrada: ${d}"
    ITERS=("$d")
else
    mapfile -t all < <(find "$SRC" -maxdepth 1 -type d -name 'iter_*' | sort -V)
    [ "${#all[@]}" -gt 0 ] || die "nenhum diretório iter_* em ${SRC}"
    start=$(( ${#all[@]} - LAST )); [ "$start" -lt 0 ] && start=0
    ITERS=("${all[@]:$start}")
fi

log "origem:  ${SRC}"
log "destino: ${DST_ROOT}"
log "copiar:  ${#ITERS[@]} checkpoint(s): $(for i in "${ITERS[@]}"; do basename "$i"; done | paste -sd' ' -)"
[ "$NO_OPTIM" = "1" ] && log "modo --no-optim: excluindo distrib_optim* (serve p/ conversão, NÃO p/ retomar treino)"

# ── Checagem de espaço ───────────────────────────────────────────────────────
mkdir -p "$DST_ROOT"
need=0
for d in "${ITERS[@]}"; do need=$(( need + $(du -sb "$d" | cut -f1) )); done
avail=$(df -PB1 "$BACKUP_DIR" | awk 'NR==2{print $4}')
log "necessário ~$(( need/1024/1024/1024 )) GB · disponível ~$(( avail/1024/1024/1024 )) GB em ${BACKUP_DIR}"
if [ "$need" -gt "$(( avail * 95 / 100 ))" ] && [ "$FORCE" != "1" ]; then
    die "espaço insuficiente (margem <5%). Use outro BACKUP_DIR, --no-optim, ou --force."
fi
# Aviso: mesmo disco não protege contra falha de hardware, só contra a conversão.
if [ "$(stat -c %d "$CKPT_DIR" 2>/dev/null)" = "$(stat -c %d "$BACKUP_DIR" 2>/dev/null)" ]; then
    log "AVISO: BACKUP_DIR está no MESMO disco do CKPT_DIR. Protege contra erro de"
    log "       conversão, mas NÃO contra falha do disco. Ideal: BACKUP_DIR em outro volume."
fi

# ── Copiar + manifesto + verificação + somente-leitura ───────────────────────
EXCLUDES=()
[ "$NO_OPTIM" = "1" ] && EXCLUDES=(distrib_optim)

copy_one() {
    local src="$1" dst="$2"
    if [ -d "$dst" ] && [ "$FORCE" != "1" ]; then
        if [ -f "$dst/MANIFEST.sha256" ]; then
            log "$(basename "$dst") já existe — verificando (use --force p/ refazer)"
            ( cd "$dst" && sha256sum -c --quiet MANIFEST.sha256 ) \
                && { log "  íntegro, pulando."; return 0; } \
                || die "backup existente CORROMPIDO em $dst; investigue antes de --force."
        fi
    fi
    [ -d "$dst" ] && chmod -R u+w "$dst"
    rm -rf "$dst"; mkdir -p "$dst"
    if [ "$COPY" = "rsync" ]; then
        local ex=(); for e in "${EXCLUDES[@]}"; do ex+=(--exclude="${e}*"); done
        rsync -a "${ex[@]}" "$src/" "$dst/"
    else
        cp -a "$src/." "$dst/"
        for e in "${EXCLUDES[@]}"; do find "$dst" -name "${e}*" -delete; done
    fi
    log "  gerando manifesto SHA-256..."
    ( cd "$dst" && find . -type f ! -name MANIFEST.sha256 -print0 \
        | sort -z | xargs -0 sha256sum > MANIFEST.sha256 )
    log "  verificando cópia..."
    ( cd "$dst" && sha256sum -c --quiet MANIFEST.sha256 ) \
        || die "verificação FALHOU em $dst (cópia divergiu da origem)."
    log "  OK ($(du -sh "$dst" | cut -f1))."
}

for d in "${ITERS[@]}"; do
    copy_one "$d" "${DST_ROOT}/$(basename "$d")"
done

# Tracker: torna o backup carregável direto pelo Megatron (--load ${DST_ROOT}).
newest_iter="$(basename "${ITERS[-1]}")"; newest_iter="${newest_iter#iter_}"
echo "$(( 10#$newest_iter ))" > "${DST_ROOT}/latest_checkpointed_iteration.txt"

# Provenance do backup (estilo do repo).
INFO="${DST_ROOT}/BACKUP_INFO.txt"
{
  echo "Manacá — Backup de checkpoint (pré-conversão HF)"
  echo "==============================================="
  echo "Origem:      ${SRC}"
  echo "Destino:     ${DST_ROOT}"
  echo "Host:        $(hostname)   User: $(id -un)"
  echo "Repo commit: $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo '?')"
  echo "Checkpoints: $(for i in "${ITERS[@]}"; do basename "$i"; done | paste -sd' ' -)"
  echo "no-optim:    ${NO_OPTIM}    (1 = sem estado do optimizer)"
  echo "Tamanho:     $(du -sh "$DST_ROOT" | cut -f1)"
  echo "Verificação: sha256sum -c MANIFEST.sha256 em cada iter_* -> OK"
} > "$INFO"

# Somente-leitura: o conversor (ou um rm acidental) não consegue clobberar.
chmod -R a-w "$DST_ROOT" 2>/dev/null || true
log "backup marcado como SOMENTE-LEITURA (chmod a-w)."

# ── Tarball opcional para cópia off-site ─────────────────────────────────────
if [ -n "$TAR" ] && [ "$TAR" != "none" ]; then
    TS="$(date +%Y%m%d_%H%M%S)"
    base="${BACKUP_DIR}/${MODEL}_backup_${TS}"
    case "$TAR" in
        zst) command -v zstd >/dev/null || die "zstd não instalado (use --tar gz)"
             tarball="${base}.tar.zst"
             tar -C "$BACKUP_DIR" -cf - "${MODEL}" | zstd -T0 -3 -o "$tarball" ;;
        gz)  tarball="${base}.tar.gz"; tar -C "$BACKUP_DIR" -czf "$tarball" "${MODEL}" ;;
        *)   die "formato de --tar inválido: '$TAR' (use zst|gz)" ;;
    esac
    sha256sum "$tarball" > "${tarball}.sha256"
    log "tarball: ${tarball} ($(du -sh "$tarball" | cut -f1))  +  .sha256"
fi

# Mensagem final bilingue (PT + EN) — o desfecho e o fluxo que o usuario le.
log "CONCLUÍDO. Backup íntegro em: ${DST_ROOT}"
log "DONE. Backup verified at:     ${DST_ROOT}"
echo
echo "Fluxo seguro de conversão:"
echo "  1) (feito) backup somente-leitura + verificado"
echo "  2) converter LENDO de ${SRC} e ESCREVENDO em um diretório NOVO (nunca por cima)"
echo "  3) validar o HF: AutoModelForCausalLM.from_pretrained(<novo_dir>)"
echo "  4) se algo der errado, o original e este backup continuam intactos"
echo "  Re-checar o backup a qualquer momento:  $0 --verify"
echo
echo "Safe conversion workflow:"
echo "  1) (done) read-only + verified backup"
echo "  2) convert READING from ${SRC} and WRITING to a NEW directory (never in place)"
echo "  3) validate the HF: AutoModelForCausalLM.from_pretrained(<new_dir>)"
echo "  4) if anything goes wrong, the original and this backup remain intact"
echo "  Re-check the backup at any time:  $0 --verify"

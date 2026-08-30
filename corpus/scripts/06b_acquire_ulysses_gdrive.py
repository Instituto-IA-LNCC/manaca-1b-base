#!/usr/bin/env python3
"""
Manacá LLM — Script 06b: Aquisição do Ulysses Tesemõ via Google Drive
=====================================================================
LNCC × NII/LLM-jp | Fase 1 — Corpus PT-BR

================================ PT (Português) ================================

O corpus Ulysses Tesemõ NÃO está no repositório Git (que é só documentação)
nem no HuggingFace — os ~30 GB de dados estão em pastas públicas do Google
Drive (dono: felipe.siqueira@usp.br, USP). Este script substitui o Script 06
(git clone) para essa fonte: baixa via gdown e converte para Parquet,
reutilizando exatamente a limpeza jurídica e o schema do Script 06.

Pipeline:
    gdown (pasta pública) -> extrai arquivos compactados -> varredura ->
    read_file_text -> clean_legal_text -> filtro -> Parquet (shards de 50k)

Idempotente: pula o download se já existir e retoma dos shards já escritos.

DIAGNÓSTICO: o gdown tem limites com pastas MUITO grandes. Este script loga a
estrutura do que foi baixado (contagem, histograma de extensões, tamanho) logo
após o download, para que se possa avaliar se o gdown deu conta ou se é preciso
outra estratégia (ex.: baixar por subpasta, ou rclone).

Uso:
    make ulysses-gdrive
    # ou:
    docker compose run -d --name ulysses corpus \\
        python corpus/scripts/06b_acquire_ulysses_gdrive.py
    docker logs -f ulysses

    # Se já baixou antes e quer só (re)converter:
    docker compose run --rm corpus \\
        python corpus/scripts/06b_acquire_ulysses_gdrive.py --skip-download

================================= EN (English) =================================

Manacá LLM — Script 06b: Ulysses Tesemõ Acquisition via Google Drive

The Ulysses Tesemõ corpus is NOT in the Git repository (which is documentation only)
nor on HuggingFace — the ~30 GB of data are in public Google
Drive folders (owner: felipe.siqueira@usp.br, USP). This script replaces Script 06
(git clone) for this source: it downloads via gdown and converts to Parquet,
reusing exactly the legal cleaning and schema of Script 06.

Pipeline:
    gdown (public folder) -> extracts compressed files -> scan ->
    read_file_text -> clean_legal_text -> filter -> Parquet (50k shards)

Idempotent: skips the download if it already exists and resumes from shards already written.

DIAGNOSTICS: gdown has limits with VERY large folders. This script logs the
structure of what was downloaded (count, extension histogram, size) right
after the download, so one can assess whether gdown coped or whether another
strategy is needed (e.g. downloading per subfolder, or rclone).

Usage:
    make ulysses-gdrive
    # or:
    docker compose run -d --name ulysses corpus \\
        python corpus/scripts/06b_acquire_ulysses_gdrive.py
    docker logs -f ulysses

    # If you already downloaded before and only want to (re)convert:
    docker compose run --rm corpus \\
        python corpus/scripts/06b_acquire_ulysses_gdrive.py --skip-download

Autor | Author: Projeto Manacá — fork manaca-1b (Docker)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

try:
    from loguru import logger
    from tqdm import tqdm
except ImportError:
    print("[ERRO] Rode dentro do container 'corpus' "
          "(docker compose run --rm corpus ...).")
    sys.exit(1)

# ── Reuso das rotinas do Script 06 (mesma limpeza jurídica e schema) ──────────
_SCRIPTS_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "ulysses06", _SCRIPTS_DIR / "06_acquire_ulysses.py"
)
_u06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_u06)

clean_legal_text   = _u06.clean_legal_text
passes_filter      = _u06.passes_filter
quality_score      = _u06.quality_score
write_shard        = _u06.write_shard
read_file_text     = _u06.read_file_text
extract_source_name = _u06.extract_source_name
get_existing_shards = _u06.get_existing_shards
load_manifest      = _u06.load_manifest
save_manifest      = _u06.save_manifest
SHARD_SIZE         = _u06.SHARD_SIZE

# ── Configuração ─────────────────────────────────────────────────────────────
SOURCE_NAME = "ulysses"
LICENSE     = "public-domain"
REFERENCE   = "Nascimento et al. (2023). Ulysses Tesemõ. STIL 2023."

# Pastas públicas do Google Drive (README do ulysses-camara/ulysses-tesemo).
DRIVE_FOLDERS = [
    "https://drive.google.com/drive/folders/1hRugg8mC5R_COB11DI3O1qOBaaXJxdx0",
]

WORK_DIR    = Path(os.environ.get("WORK_DIR", Path.home() / "manaca-corpus"))
OUTPUT_DIR  = WORK_DIR / "raw" / SOURCE_NAME
DOWNLOAD_DIR = OUTPUT_DIR / "download"
LOG_DIR     = WORK_DIR / "logs"

READABLE_EXT = {".txt", ".json", ".jsonl"}
ARCHIVE_EXT  = {".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz"}
TEXT_FIELDS  = ["texto", "text", "conteudo", "content", "body", "corpo"]


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"ulysses_gdrive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")
    logger.add(log_file, level="DEBUG", encoding="utf-8",
               format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}")


# ── Download via gdown ───────────────────────────────────────────────────────

def gdown_folder(url: str, dest: Path) -> None:
    """Baixa uma pasta pública do Google Drive via gdown (CLI)."""
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["gdown", "--folder", url, "-O", str(dest), "--remaining-ok"]
    logger.info(f"gdown: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for line in proc.stdout:
        logger.info(f"[gdown] {line.rstrip()}")
    rc = proc.wait()
    if rc != 0:
        logger.warning(f"gdown terminou com código {rc} (pode ter baixado parcialmente).")


# ── Extração de arquivos compactados ─────────────────────────────────────────

def extract_archives(root: Path) -> int:
    """Extrai zips/tars encontrados (in-place). Retorna quantos extraiu."""
    n = 0
    for f in list(root.rglob("*")):
        if not f.is_file():
            continue
        try:
            if f.suffix.lower() == ".zip":
                with zipfile.ZipFile(f) as z:
                    z.extractall(f.parent / f"{f.stem}_extracted")
                n += 1
            elif f.suffix.lower() in {".tar", ".tgz", ".gz", ".bz2", ".xz"} and \
                    tarfile.is_tarfile(f):
                with tarfile.open(f) as t:
                    t.extractall(f.parent / f"{f.stem}_extracted")
                n += 1
        except Exception as e:
            logger.warning(f"Falha ao extrair {f.name}: {e}")
    if n:
        logger.info(f"Arquivos compactados extraídos: {n}")
    return n


# ── Diagnóstico da estrutura baixada ─────────────────────────────────────────

def survey(root: Path) -> list[Path]:
    """Loga contagem, tamanho e histograma de extensões. Retorna todos os arquivos."""
    files = [f for f in root.rglob("*") if f.is_file()]
    total_bytes = sum(f.stat().st_size for f in files)
    ext_hist = Counter(f.suffix.lower() or "<sem_ext>" for f in files)

    logger.info("=" * 60)
    logger.info(f"ESTRUTURA BAIXADA em {root}")
    logger.info(f"  Arquivos: {len(files):,}  |  Tamanho: {total_bytes/1e9:.2f} GB")
    logger.info("  Extensões (top 15):")
    for ext, c in ext_hist.most_common(15):
        logger.info(f"    {ext:<12} {c:,}")
    logger.info("  Amostra de caminhos:")
    for f in files[:8]:
        logger.info(f"    {f.relative_to(root)}  ({f.stat().st_size/1024:.0f} KB)")
    logger.info("=" * 60)

    readable = [f for f in files if f.suffix.lower() in READABLE_EXT]
    if not readable:
        logger.error(
            "Nenhum arquivo .txt/.json/.jsonl encontrado. Se o histograma acima "
            "mostrar outro formato dominante (ou pouquíssimos arquivos = gdown "
            "capado em pastas grandes), avise para adaptarmos a estratégia."
        )
    return readable


# ── Iterador de documentos ───────────────────────────────────────────────────

def iter_docs(files: list[Path], root: Path) -> Iterator[tuple[str, str, str]]:
    """Gera (source, doc_id, texto_bruto) para cada documento legível."""
    for fp in files:
        ext = fp.suffix.lower()
        src = extract_source_name(fp, root)
        if ext in (".txt", ".json"):
            txt = read_file_text(fp)
            if txt:
                yield src, str(fp.relative_to(root)), txt
        elif ext == ".jsonl":
            try:
                for i, line in enumerate(fp.read_text(encoding="utf-8",
                                                       errors="ignore").splitlines()):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for field in TEXT_FIELDS:
                        if isinstance(obj.get(field), str):
                            yield src, f"{fp.relative_to(root)}:{i}", obj[field]
                            break
            except Exception as e:
                logger.warning(f"Falha ao ler {fp.name}: {e}")


# ── Pipeline principal ───────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Ulysses Tesemõ via Google Drive (gdown)")
    ap.add_argument("--skip-download", action="store_true",
                    help="Não baixar; usar o que já está em download/.")
    ap.add_argument("--url", action="append", default=None,
                    help="URL de pasta do Drive (repetível). Default: pastas do README.")
    args = ap.parse_args()

    setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"Fonte:       {SOURCE_NAME} (Google Drive)")
    logger.info(f"Licença:     {LICENSE}")
    logger.info(f"Referência:  {REFERENCE}")
    logger.info(f"Output:      {OUTPUT_DIR}")
    logger.info("=" * 60)

    # ── 1. Download ──────────────────────────────────────────────────────────
    if not args.skip_download:
        for url in (args.url or DRIVE_FOLDERS):
            gdown_folder(url, DOWNLOAD_DIR)
    else:
        logger.info("--skip-download: usando conteúdo existente em download/.")
    if not DOWNLOAD_DIR.exists():
        logger.error(f"Nada baixado em {DOWNLOAD_DIR}.")
        return 1

    # ── 2. Extrair compactados ───────────────────────────────────────────────
    extract_archives(DOWNLOAD_DIR)

    # ── 3. Diagnóstico + varredura ───────────────────────────────────────────
    readable = survey(DOWNLOAD_DIR)
    if not readable:
        return 1

    # ── 4. Conversão para Parquet (retoma dos shards existentes) ─────────────
    existing = get_existing_shards(OUTPUT_DIR)
    shard_id = (max(existing) + 1) if existing else 0
    if existing:
        logger.info(f"Retomando: {len(existing)} shards já escritos. Próximo: {shard_id:06d}")

    buffer: list[dict] = []
    n_docs = n_kept = 0
    manifest = load_manifest()

    for src, doc_id, raw in tqdm(iter_docs(readable, DOWNLOAD_DIR),
                                 desc="Ulysses", unit="doc"):
        n_docs += 1
        text = clean_legal_text(raw)
        if not passes_filter(text):
            continue
        buffer.append({
            "text": text, "source": SOURCE_NAME, "id": f"{src}/{doc_id}",
            "lang": "por_Latn", "score": quality_score(text),
        })
        n_kept += 1
        if len(buffer) >= SHARD_SIZE:
            info = write_shard(buffer, shard_id, OUTPUT_DIR)
            logger.info(f"Shard {shard_id:04d} | {n_kept:,} docs | {info['size_mb']} MB")
            shard_id += 1
            buffer = []

    if buffer:
        info = write_shard(buffer, shard_id, OUTPUT_DIR)
        logger.info(f"Shard final {shard_id:04d} | {len(buffer):,} docs | {info['size_mb']} MB")

    # ── 5. Manifesto ─────────────────────────────────────────────────────────
    manifest.setdefault("sources", {})[SOURCE_NAME] = {
        "status": "completed",
        "method": "google-drive/gdown",
        "docs_seen": n_docs,
        "docs_kept": n_kept,
        "retention": round(n_kept / max(n_docs, 1), 4),
        "shards": len(get_existing_shards(OUTPUT_DIR)),
        "license": LICENSE,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    save_manifest(manifest)

    logger.info("=" * 60)
    logger.info(f"CONCLUÍDO — {SOURCE_NAME}")
    logger.info(f"  Docs vistos:  {n_docs:,}")
    logger.info(f"  Docs mantidos:{n_kept:,}")
    logger.info(f"  Retenção:     {n_kept/max(n_docs,1)*100:.1f}%")
    logger.info(f"  Shards:       {len(get_existing_shards(OUTPUT_DIR))}")
    logger.info("-" * 60)
    logger.info(f"DONE — {SOURCE_NAME}")
    logger.info(f"  Docs seen:    {n_docs:,}")
    logger.info(f"  Docs kept:    {n_kept:,}")
    logger.info(f"  Retention:    {n_kept/max(n_docs,1)*100:.1f}%")
    logger.info(f"  Shards:       {len(get_existing_shards(OUTPUT_DIR))}")
    logger.info("=" * 60)
    logger.info("Dica: para liberar espaço, apague raw/ulysses/download/ após conferir os shards.")
    logger.info("Tip: to free space, delete raw/ulysses/download/ after checking the shards.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

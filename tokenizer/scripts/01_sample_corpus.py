#!/usr/bin/env python3
"""
Manacá-1B — Fase 2a · Passo 1: Amostragem balanceada para o tokenizador
=======================================================================
Amostra ~N GB de texto do corpus (GigaVerbo + Wikipedia + Ulysses),
balanceado por fonte, para treinar o SentencePiece (Passo 2).

Fiel ao LLM-jp (amostra estratificada por fonte, plano §6.4/§6.5), porém em
UM único modelo — o pipeline multi-idioma do LLM-jp (treinar por idioma e depois
mesclar) existe porque eles misturam vários idiomas; o Manacá-1B é PT quase puro,
então uma amostra única é o correto e mais simples.

Uso (dentro do container 'corpus'):
    make tokenizer-sample
    # ou: docker compose run --rm corpus python tokenizer/scripts/01_sample_corpus.py --gb 5
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

try:
    import pyarrow.parquet as pq
    from loguru import logger
except ImportError:
    print("[ERRO] Rode dentro do container 'corpus' (docker compose run --rm corpus ...).")
    sys.exit(1)

WORK_DIR = Path(os.environ.get("WORK_DIR", Path.home() / "manaca-corpus"))
RAW_DIR  = WORK_DIR / "raw"
OUT_DIR  = WORK_DIR / "tokenizer"

# Fontes do corpus do Manacá-1B (nome da subpasta em raw/).
SOURCES = ["gigaverbo", "wikipedia_ptbr", "ulysses"]


def sample_source(src_dir: Path, budget_bytes: int, out_fh, rng: random.Random) -> int:
    """Lê shards Parquet da fonte e escreve linhas de texto até atingir o budget."""
    written = 0
    shards = sorted(src_dir.glob("*.parquet"))
    if not shards:
        logger.warning(f"Sem shards em {src_dir} — pulando.")
        return 0
    rng.shuffle(shards)
    for shard in shards:
        if written >= budget_bytes:
            break
        try:
            pf = pq.ParquetFile(shard)
        except Exception as e:
            logger.warning(f"Ignorando {shard.name}: {e}")
            continue
        for batch in pf.iter_batches(batch_size=1000, columns=["text"]):
            for value in batch.column("text"):
                text = value.as_py()
                if not text:
                    continue
                # Uma linha por documento (o SentencePiece trata linha a linha).
                line = " ".join(text.split()) + "\n"
                b = line.encode("utf-8")
                out_fh.write(b)
                written += len(b)
                if written >= budget_bytes:
                    break
            if written >= budget_bytes:
                break
    logger.info(f"  {src_dir.name:<16} {written/1e9:.2f} GB")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="Amostragem balanceada p/ tokenizador")
    ap.add_argument("--gb", type=float, default=5.0, help="Tamanho total da amostra (GB). Default: 5.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "sample.txt")
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")

    rng = random.Random(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    available = [s for s in SOURCES if (RAW_DIR / s).exists()]
    if not available:
        logger.error(f"Nenhuma fonte encontrada em {RAW_DIR}. Fontes esperadas: {SOURCES}")
        return 1

    per_source = int(args.gb * 1e9 / len(available))
    logger.info(f"Amostra total: {args.gb} GB | fontes: {available} | ~{per_source/1e9:.2f} GB por fonte")

    total = 0
    with open(args.out, "wb") as fh:
        for src in available:
            total += sample_source(RAW_DIR / src, per_source, fh, rng)

    logger.info("=" * 50)
    logger.info(f"Amostra escrita: {args.out}  ({total/1e9:.2f} GB)")
    logger.info("Proximo: make tokenizer-train")
    return 0


if __name__ == "__main__":
    sys.exit(main())

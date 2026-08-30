#!/usr/bin/env python3
"""
Manacá-1B — Fase 2a · Passo 2: Treino do SentencePiece Unigram
==============================================================

PT ------------------------------------------------------------------------
Treina o tokenizador com as decisões FIÉIS ao LLM-jp (plano §6.3):
  - modelo Unigram (D2a.1)
  - character_coverage = 0.9995 (D2a.2)
  - byte_fallback = True (D2a.3)
  - normalization = nmt_nfkc_cf (D2a.4)
  - split_digits = True (D2a.5)
  - allow_whitespace_only_pieces = True (D2a.6)
  - tokens especiais ChatML/Alpaca-PT (D2a.7)

Ajuste para o Manacá-1B: vocab dimensionado ao corpus PT (default 64k). O LLM-jp
usa ~100k porque é multilíngue (JA+EN+código+...); para PT quase puro, 32-64k é o
tamanho adequado. Sobreponha com --vocab-size se quiser paridade (ex.: 100000).

Uso (container 'corpus'):
    make tokenizer-train
    # ou: docker compose run --rm corpus python tokenizer/scripts/02_train_spm.py --vocab-size 64000

EN ------------------------------------------------------------------------
Manacá-1B — Phase 2a · Step 2: SentencePiece Unigram training
Trains the tokenizer with the decisions FAITHFUL to LLM-jp (plan §6.3):
  - Unigram model (D2a.1)
  - character_coverage = 0.9995 (D2a.2)
  - byte_fallback = True (D2a.3)
  - normalization = nmt_nfkc_cf (D2a.4)
  - split_digits = True (D2a.5)
  - allow_whitespace_only_pieces = True (D2a.6)
  - special ChatML/Alpaca-PT tokens (D2a.7)

Adjustment for Manacá-1B: vocab sized to the PT corpus (default 64k). LLM-jp uses
~100k because it is multilingual (JA+EN+code+...); for almost pure PT, 32-64k is the
right size. Override with --vocab-size if you want parity (e.g.: 100000).

Usage ('corpus' container):
    make tokenizer-train
    # or: docker compose run --rm corpus python tokenizer/scripts/02_train_spm.py --vocab-size 64000
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    import sentencepiece as spm
except ImportError:
    print("[ERRO] sentencepiece ausente — rode dentro do container 'corpus'.")
    sys.exit(1)

WORK_DIR = Path(os.environ.get("WORK_DIR", Path.home() / "manaca-corpus"))
OUT_DIR  = WORK_DIR / "tokenizer"

# Tokens especiais (ChatML + marcadores Alpaca-PT do template de SFT, §8.2).
SPECIAL_TOKENS = [
    "<|im_start|>", "<|im_end|>",
    "<|instrucao|>", "<|resposta|>",
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Treino SentencePiece Unigram (LLM-jp-fiel)")
    ap.add_argument("--input", type=Path, default=OUT_DIR / "sample.txt")
    ap.add_argument("--vocab-size", type=int, default=64000,
                    help="Tamanho do vocabulário. Default 64000 (PT-mono). LLM-jp: ~100000 (multilíngue).")
    ap.add_argument("--model-prefix", type=str, default=str(OUT_DIR / "manaca-tokenizer"))
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--input-sentence-size", type=int, default=10_000_000,
                    help="Máx. de sentenças amostradas no treino (controla RAM). Default 10M.")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"[ERRO] Amostra não encontrada: {args.input}. Rode antes: make tokenizer-sample")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[tokenizer] treinando Unigram vocab={args.vocab_size} threads={args.threads}")
    print(f"[tokenizer] input={args.input} -> {args.model_prefix}.model")

    spm.SentencePieceTrainer.train(
        input=str(args.input),
        model_prefix=args.model_prefix,
        model_type="unigram",
        vocab_size=args.vocab_size,
        character_coverage=0.9995,
        byte_fallback=True,
        normalization_rule_name="nmt_nfkc_cf",
        split_digits=True,
        allow_whitespace_only_pieces=True,
        remove_extra_whitespaces=False,
        # ids de controle (compatível com Llama/HF): unk=0, bos=1, eos=2, pad=3
        unk_id=0, bos_id=1, eos_id=2, pad_id=3,
        unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>", pad_piece="<pad>",
        user_defined_symbols=SPECIAL_TOKENS,
        num_threads=args.threads,
        input_sentence_size=args.input_sentence_size,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True,
    )

    # Resumo final bilingue (PT + EN) — o resultado que o usuario le ao fim.
    print(f"[tokenizer] OK -> {args.model_prefix}.model / .vocab")
    print("[tokenizer] Proximo: make tokenizer-convert")
    print("[tokenizer] Next: make tokenizer-convert")
    return 0


if __name__ == "__main__":
    sys.exit(main())

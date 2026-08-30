#!/usr/bin/env python3
"""
Manacá-1B — Fase 2a · Passo 3: Conversão SentencePiece → HuggingFace
====================================================================
Converte o .model do SentencePiece em um tokenizador HuggingFace rápido
(LlamaTokenizerFast), com os tokens especiais registrados (plano §6.5).

Uso (container 'corpus'):
    make tokenizer-convert
    # ou: docker compose run --rm corpus python tokenizer/scripts/03_convert_to_hf.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

WORK_DIR = Path(os.environ.get("WORK_DIR", Path.home() / "manaca-corpus"))
TOK_DIR  = WORK_DIR / "tokenizer"

SPECIAL_TOKENS = ["<|im_start|>", "<|im_end|>", "<|instrucao|>", "<|resposta|>"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Conversão SPM -> HuggingFace fast tokenizer")
    ap.add_argument("--model", type=Path, default=TOK_DIR / "manaca-tokenizer.model")
    ap.add_argument("--out", type=Path, default=TOK_DIR / "hf")
    args = ap.parse_args()

    if not args.model.exists():
        print(f"[ERRO] Modelo SPM não encontrado: {args.model}. Rode antes: make tokenizer-train")
        return 1

    try:
        from transformers import LlamaTokenizerFast
    except ImportError:
        print("[ERRO] 'transformers' ausente na imagem. Reconstrua: make build-corpus")
        return 1

    print(f"[convert] {args.model} -> {args.out}")
    tok = LlamaTokenizerFast(
        vocab_file=str(args.model),
        bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>",
        legacy=False,
    )
    added = tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    args.out.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(str(args.out))

    # Round-trip de sanidade
    sample = "Abaixo está uma instrução. ### Resposta:\nO manacá floresce em três cores."
    ids = tok.encode(sample)
    dec = tok.decode(ids)
    print(f"[convert] vocab={tok.vocab_size} (+{added} especiais) | round-trip: "
          f"{len(ids)} tokens")
    print(f"[convert] OK -> {args.out}")
    print("[convert] Use como --tokenizer no preprocess_data.py e nas Fases 2b/3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manaca-1B - Corrige o tokenizador HF para reproduzir o nmt_nfkc_cf do treino
============================================================================

PT ------------------------------------------------------------------------
O tokenizador HF gerado a partir do .model NAO minuscula/normaliza como o treino,
entao texto com maiusculas cai em byte-fallback e diverge da tokenizacao de treino.
Isso pune o Manaca em qualquer avaliacao que use o tokenizador HF (lm-eval etc.).

Este script edita o `tokenizer.json` NO NIVEL DO JSON (injeta um normalizador
Sequence([NFKC, Lowercase, <existente>])), o que e confiavel: nao passa pelo
save_pretrained do LlamaTokenizer, que reconverte a partir do SPM e perde o
normalizador. Grava um diretorio de tokenizador PEQUENO (so os arquivos de
tokenizador, classe PreTrainedTokenizerFast) e verifica contra o SPM do treino.
NAO toca no modelo original.

Uso (na imagem manaca-lmeval/manaca-train, que tem transformers + tokenizers + spm):
    python scripts/eval/fix_hf_tokenizer.py \
        --src /m  --spm /tok/manaca-tokenizer.model  --out /hf/manaca-tok-fixed

Depois, no lm-eval: MANACA_HF=/m  MANACA_TOKENIZER=/hf/manaca-tok-fixed (ver
run_lm_eval_pt.sh, que injeta tokenizer=/mtok no model_args).

EN ------------------------------------------------------------------------
Manaca-1B - Fix the HF tokenizer to reproduce the training's nmt_nfkc_cf
The HF tokenizer generated from the .model does NOT lowercase/normalize like the
training, so text with uppercase falls into byte-fallback and diverges from the
training tokenization. This penalizes Manaca in any evaluation that uses the HF
tokenizer (lm-eval etc.).

This script edits `tokenizer.json` AT THE JSON LEVEL (injects a normalizer
Sequence([NFKC, Lowercase, <existing>])), which is reliable: it does not go through
LlamaTokenizer's save_pretrained, which re-converts from the SPM and loses the
normalizer. It writes a SMALL tokenizer directory (only the tokenizer files, class
PreTrainedTokenizerFast) and checks it against the training SPM.
It does NOT touch the original model.

Usage (in the manaca-lmeval/manaca-train image, which has transformers + tokenizers + spm):
    python scripts/eval/fix_hf_tokenizer.py \
        --src /m  --spm /tok/manaca-tokenizer.model  --out /hf/manaca-tok-fixed

Then, in lm-eval: MANACA_HF=/m  MANACA_TOKENIZER=/hf/manaca-tok-fixed (see
run_lm_eval_pt.sh, which injects tokenizer=/mtok into model_args).

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

TESTES = [
    "Qual e a capital do Brasil? A resposta correta e Brasilia.",
    "Maria foi ao mercado e comprou pao, leite e ovos.",
    "Em 1822, Dom Pedro I proclamou a Independencia do Brasil.",
    "A AGUA do ACUDE estava GELADA naquela manha de julho.",
    "Numeros: 1.234,56 e simbolos como no 1o e no 2o caso.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="diretorio HF original do Manaca (so leitura)")
    ap.add_argument("--spm", required=True, help="manaca-tokenizer.model (verificacao)")
    ap.add_argument("--out", required=True, help="diretorio de saida do tokenizador corrigido (pequeno)")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    src_json = os.path.join(a.src, "tokenizer.json")
    if os.path.isfile(src_json):
        with open(src_json, encoding="utf-8") as fh:
            tj = json.load(fh)
    else:
        # Modelo so tem o tokenizador LENTO (SPM). Converte para fast em memoria e
        # extrai o tokenizer.json canonico (a conversao reproduz o SPM, mas sem a
        # normalizacao nmt_nfkc_cf -- que e justamente o que injetamos abaixo).
        print(f"[fix] {src_json} ausente; convertendo o tokenizador lento -> fast em memoria")
        _tok = AutoTokenizer.from_pretrained(a.src, use_fast=True)
        tj = json.loads(_tok.backend_tokenizer.to_str())

    existente = tj.get("normalizer")
    passos = [{"type": "NFKC"}, {"type": "Lowercase"}]
    if existente:
        passos.append(existente)  # preserva o tratamento de espaco/metaspace, se houver
    tj["normalizer"] = {"type": "Sequence", "normalizers": passos}
    print(f"[fix] normalizador antigo: {existente}")
    print(f"[fix] normalizador novo  : Sequence([NFKC, Lowercase{', <existente>' if existente else ''}])")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "tokenizer.json"), "w", encoding="utf-8") as fh:
        json.dump(tj, fh, ensure_ascii=False)

    # tokenizer_config.json: copia do original mas forca PreTrainedTokenizerFast
    cfg = {}
    src_cfg = os.path.join(a.src, "tokenizer_config.json")
    if os.path.isfile(src_cfg):
        with open(src_cfg, encoding="utf-8") as fh:
            cfg = json.load(fh)
    cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    cfg.pop("auto_map", None)
    with open(os.path.join(a.out, "tokenizer_config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)
    for extra in ("special_tokens_map.json",):
        p = os.path.join(a.src, extra)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(a.out, extra))
    print(f"[fix] tokenizador corrigido salvo em: {a.out}")

    # ---- verificacao contra o SentencePiece do treino ----
    import sentencepiece as spm
    from transformers import AutoTokenizer
    sp = sp_proc = spm.SentencePieceProcessor(model_file=a.spm)
    novo = AutoTokenizer.from_pretrained(a.out, use_fast=True)

    print("\n[verificacao] ids (SPM do treino x HF corrigido):")
    iguais = 0
    for t in TESTES:
        a_ids = sp_proc.encode(t, out_type=int)
        b_ids = novo.encode(t, add_special_tokens=False)
        ok = a_ids == b_ids
        iguais += int(ok)
        print(f"  [{'OK ' if ok else 'DIF'}] {t}")
        if not ok:
            print(f"        SPM: {a_ids}")
            print(f"        HF : {b_ids}")
    # Resultado final bilingue (PT + EN) — o veredito que o usuario le.
    print(f"\n[verificacao] {iguais}/{len(TESTES)} textos com ids identicos.")
    print(f"[verification] {iguais}/{len(TESTES)} texts with identical ids.")
    if iguais == len(TESTES):
        print(">>> Tokenizador corrigido reproduz o SPM do treino. Use no lm-eval com "
              f"MANACA_TOKENIZER={a.out}")
        print(">>> Fixed tokenizer reproduces the training SPM. Use it in lm-eval with "
              f"MANACA_TOKENIZER={a.out}")
        return 0
    print(">>> Ainda ha divergencia; me mande a saida acima que eu ajusto o normalizador.")
    print(">>> There is still a divergence; send me the output above and I will adjust the normalizer.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

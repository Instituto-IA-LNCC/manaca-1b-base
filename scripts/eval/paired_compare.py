#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manaca-1B - Comparacao PAREADA de dois modelos no CALAME-PT
===========================================================

PT ------------------------------------------------------------------------
Os dois modelos veem os MESMOS exemplos, entao a forma rigorosa de dizer se um e
melhor que o outro nao e comparar dois intervalos de confianca (que ignoram a
correlacao), e sim testar a DIFERENCA pareada. Este script:

  * le os vetores de acertos por exemplo salvos por eval_base.py (--save-calame),
  * calcula a diferenca de acuracia (A - B) com IC 95% por bootstrap pareado,
  * roda o teste de McNemar (p-valor) sobre os discordantes.

Uso:
    # 1) gere os vetores de acertos de cada modelo (mesmo protocolo!):
    ./run_eval.sh --model /m --spm /tok/manaca-tokenizer.model \
        --text /eval/holdout_pt.txt --calame --save-calame /hf/manaca_calame.json
    ./run_eval.sh --model TucanoBR/Tucano-1b1 \
        --text /eval/holdout_pt.txt --calame --save-calame /hf/tucano_calame.json
    # 2) compare:
    python paired_compare.py /hf/manaca_calame.json /hf/tucano_calame.json

EN ------------------------------------------------------------------------
Manaca-1B - PAIRED comparison of two models on CALAME-PT
The two models see the SAME examples, so the rigorous way to say whether one is
better than the other is not to compare two confidence intervals (which ignore the
correlation), but to test the paired DIFFERENCE. This script:

  * reads the per-example correctness vectors saved by eval_base.py (--save-calame),
  * computes the accuracy difference (A - B) with a 95% CI via paired bootstrap,
  * runs the McNemar test (p-value) over the discordant pairs.

Usage:
    # 1) generate the correctness vectors for each model (same protocol!):
    ./run_eval.sh --model /m --spm /tok/manaca-tokenizer.model \
        --text /eval/holdout_pt.txt --calame --save-calame /hf/manaca_calame.json
    ./run_eval.sh --model TucanoBR/Tucano-1b1 \
        --text /eval/holdout_pt.txt --calame --save-calame /hf/tucano_calame.json
    # 2) compare:
    python paired_compare.py /hf/manaca_calame.json /hf/tucano_calame.json

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import json
import math
import random
import sys


def carregar(caminho: str):
    with open(caminho, encoding="utf-8") as fh:
        d = json.load(fh)
    return d["model"], d["corretos"]


def main() -> int:
    if len(sys.argv) != 3:
        print("uso: python paired_compare.py <A.json> <B.json>")
        return 2
    nome_a, a = carregar(sys.argv[1])
    nome_b, b = carregar(sys.argv[2])
    if len(a) != len(b):
        print(f"[ERRO] tamanhos diferentes: {len(a)} vs {len(b)} (rode o MESMO protocolo).")
        return 1
    n = len(a)
    acc_a = sum(a) / n
    acc_b = sum(b) / n
    dif = acc_a - acc_b

    # bootstrap pareado da diferenca (reamostra os MESMOS indices para os dois)
    rng = random.Random(0)
    difs = []
    for _ in range(5000):
        sa = sb = 0
        for _ in range(n):
            i = rng.randrange(n)
            sa += a[i]; sb += b[i]
        difs.append((sa - sb) / n)
    difs.sort()
    lo, hi = difs[int(0.025 * 5000)], difs[int(0.975 * 5000)]

    # McNemar sobre discordantes: b10 = A acerta e B erra; b01 = A erra e B acerta
    b10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    b01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    # aproximacao normal com correcao de continuidade
    if (b10 + b01) > 0:
        chi = (abs(b10 - b01) - 1) ** 2 / (b10 + b01)
        # p-valor bicaudal via qui-quadrado 1 g.l. = erfc(sqrt(chi/2))
        p = math.erfc(math.sqrt(chi / 2))
    else:
        chi, p = 0.0, 1.0

    print("=" * 64)
    print(f"A = {nome_a}")
    print(f"B = {nome_b}")
    print(f"n = {n} exemplos (pareados)")
    print("-" * 64)
    print(f"acuracia A : {100*acc_a:.2f}%")
    print(f"acuracia B : {100*acc_b:.2f}%")
    print(f"diferenca  : {100*dif:+.2f} pontos   IC95% pareado [{100*lo:+.2f}, {100*hi:+.2f}]")
    print(f"McNemar    : A>B em {b10} casos, B>A em {b01} casos   p = {p:.4f}")
    print("-" * 64)
    signif = "SIM" if (lo > 0 or hi < 0) and p < 0.05 else "NAO"
    # Veredito final bilingue (PT + EN) — a conclusao que o usuario le.
    print(f"diferenca estatisticamente significativa (5%)? {signif}")
    print(f"statistically significant difference (5%)?      {signif}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())

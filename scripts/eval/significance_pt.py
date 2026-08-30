#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manaca-1B - Significancia dos benchmarks (a partir de benchmarks-pt.json)
=========================================================================
Le a tabela consolidada (docs/evaluation/benchmarks-pt.json, com valor e SE por
modelo/benchmark) e compara um modelo de referencia (default Manaca-1B) com os
demais, por benchmark, usando o teste de DUAS PROPORCOES (nao pareado):

    z = (p_ref - p_alvo) / sqrt(SE_ref^2 + SE_alvo^2),  p-valor bicaudal.

E o teste CONSERVADOR (marginal). O pareado (McNemar), como no CALAME (§9 do
relatorio, via paired_compare.py), exige os acertos por exemplo, que o lm-eval so
salva com --log_samples; quando existirem, use paired_compare.py.

Uso:
    python scripts/eval/significance_pt.py                 # Manaca vs todos
    python scripts/eval/significance_pt.py --ref Tucano-1b1

Autor: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import json
import math

BENCHES = [("calame", "CALAME-PT"), ("arc", "ARC-Challenge-PT"),
           ("hellaswag", "HellaSwag-PT"), ("lambada", "LAMBADA-PT")]


def cel(m, chave):
    """Devolve (valor, se) de uma celula, ou (None, None)."""
    v = m.get(chave)
    if isinstance(v, (list, tuple)) and v and v[0] is not None:
        return float(v[0]), (float(v[1]) if len(v) > 1 and v[1] is not None else None)
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", default="docs/evaluation/benchmarks-pt.json")
    ap.add_argument("--ref", default="Manaca-1B", help="modelo de referencia")
    a = ap.parse_args()

    linhas = json.load(open(a.benchmarks, encoding="utf-8"))["linhas"]
    if a.ref not in linhas:
        print(f"[ERRO] '{a.ref}' nao esta em {a.benchmarks}. Modelos: {list(linhas)}")
        return 1

    print(f"Referencia: {a.ref}   (teste de duas proporcoes, NAO pareado)\n")
    for chave, nome in BENCHES:
        va, sa = cel(linhas[a.ref], chave)
        if va is None:
            continue
        print(f"== {nome} ==   {a.ref}: {va:.2f}%")
        for modelo, m in linhas.items():
            if modelo == a.ref:
                continue
            vb, sb = cel(m, chave)
            if vb is None:
                continue
            dif = va - vb
            if sa is None or sb is None:
                print(f"   vs {modelo:12s} {dif:+6.2f}   (sem SE p/ testar)")
                continue
            sed = math.hypot(sa, sb)
            z = dif / sed if sed else 0.0
            p = math.erfc(abs(z) / math.sqrt(2))
            marca = "SIG " if p < 0.05 else "n.s."
            print(f"   vs {modelo:12s} {dif:+6.2f}   z={z:+5.2f}  p={p:6.3f}  [{marca}]")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

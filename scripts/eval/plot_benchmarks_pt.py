#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manaca-1B - Painel dos 4 benchmarks PT (acuracia vs parametros)
===============================================================

PT ------------------------------------------------------------------------
Le docs/evaluation/benchmarks-pt.json (valor e SE por modelo/benchmark) e monta
um painel 2x2 (CALAME-PT, ARC-Challenge-PT, HellaSwag-PT, LAMBADA-PT), cada um com
acuracia vs numero de parametros (escala log), barras de erro de IC95% (±1,96·SE),
o Manaca-1B destacado e a familia PT-BR ligada por tendencia. Gera PNG e PDF.

Nao exige internet nem GPU. Uso:
    python scripts/eval/plot_benchmarks_pt.py
    python scripts/eval/plot_benchmarks_pt.py --benchmarks docs/evaluation/benchmarks-pt.json

EN ------------------------------------------------------------------------
Manaca-1B - Panel of the 4 PT benchmarks (accuracy vs parameters)
Reads docs/evaluation/benchmarks-pt.json (value and SE per model/benchmark) and
builds a 2x2 panel (CALAME-PT, ARC-Challenge-PT, HellaSwag-PT, LAMBADA-PT), each
with accuracy vs number of parameters (log scale), 95% CI error bars (±1.96·SE),
Manaca-1B highlighted and the PT-BR family linked by a trend line. Produces PNG and PDF.

Requires no internet and no GPU. Usage:
    python scripts/eval/plot_benchmarks_pt.py
    python scripts/eval/plot_benchmarks_pt.py --benchmarks docs/evaluation/benchmarks-pt.json

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# nome -> grupo (cor/marcador)
GRUPO = {
    "Manaca-1B": "manaca",
    "Tucano-160m": "ptbr", "Tucano-630m": "ptbr", "Tucano-1b1": "ptbr",
    "Tucano-2b4": "ptbr", "TTL-160m": "ptbr", "TTL-460m": "ptbr", "Sabia-7B": "ptbr",
    "GlorIA-1b3": "ptpt", "mGPT-1b3": "multi",
}
ESTILO = {
    "manaca": ("#d1495b", "*", "Manaca-1B (este trabalho)"),
    "ptbr":   ("#2e6f95", "o", "PT-BR (TTL / Tucano / Sabia)"),
    "ptpt":   ("#e8871e", "s", "PT-PT (GlorIA)"),
    "multi":  ("#8d99ae", "^", "Multilingue (mGPT)"),
}
PAINEIS = [("calame", "CALAME-PT"), ("arc", "ARC-Challenge-PT"),
           ("hellaswag", "HellaSwag-PT"), ("lambada", "LAMBADA-PT")]


def val_se(cell):
    if isinstance(cell, (list, tuple)) and cell and cell[0] is not None:
        return float(cell[0]), (float(cell[1]) if len(cell) > 1 and cell[1] is not None else None)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", default="docs/evaluation/benchmarks-pt.json")
    ap.add_argument("--out", default="docs/evaluation/benchmarks_escala_pt")
    a = ap.parse_args()

    linhas = json.load(open(a.benchmarks, encoding="utf-8"))["linhas"]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (chave, titulo) in zip(axes.flat, PAINEIS):
        # linha de tendencia PT-BR (por tamanho)
        ptbr = []
        for nome, m in linhas.items():
            if GRUPO.get(nome) != "ptbr":
                continue
            v, _ = val_se(m.get(chave))
            if v is not None and m.get("params"):
                ptbr.append((m["params"], v))
        ptbr.sort()
        if len(ptbr) >= 2:
            ax.plot([p for p, _ in ptbr], [v for _, v in ptbr], "-",
                    color=ESTILO["ptbr"][0], alpha=0.30, lw=1.4, zorder=1)

        vistos = set()
        for nome, m in linhas.items():
            par = m.get("params")
            v, se = val_se(m.get(chave))
            if par is None or v is None:
                continue
            g = GRUPO.get(nome, "multi")
            cor, marc, leg = ESTILO[g]
            rot = leg if g not in vistos else None
            vistos.add(g)
            if se is not None:
                ax.errorbar(par, v, yerr=1.96 * se, fmt="none", ecolor=cor,
                            elinewidth=1.0, capsize=2.5, alpha=0.8, zorder=2)
            ax.scatter([par], [v], s=260 if g == "manaca" else 70, marker=marc,
                       color=cor, edgecolor="white", linewidth=0.7, label=rot, zorder=3)
            if g == "manaca":
                ax.annotate("Manaca", (par, v), textcoords="offset points",
                            xytext=(0, 9), ha="center", fontsize=8, fontweight="bold",
                            color=cor)
        if chave == "arc":
            ax.axhline(25, ls=":", color="#999", lw=1)
            ax.annotate("acaso (25%)", (ax.get_xlim()[0], 25), fontsize=7, color="#999",
                        va="bottom")
        ax.set_xscale("log")
        ax.set_title(titulo, fontsize=11)
        ax.set_xlabel("Parametros (B, log)", fontsize=9)
        ax.set_ylabel("Acuracia (%)", fontsize=9)
        ax.grid(True, which="both", alpha=0.22)
        ax.tick_params(labelsize=8)

    axes.flat[0].legend(loc="lower right", fontsize=7.5, framealpha=0.9)
    fig.suptitle("Manaca-1B nos benchmarks PT (barras: IC95%)", fontsize=13)
    fig.text(0.01, 0.005,
             "CALAME (geracao, SPM), ARC/HellaSwag/LAMBADA (log-verossimilhanca, lm-eval). "
             "Mesmo harness; ver docs/evaluation/. Manaca tokenizado com o SPM/tokenizador corrigido.",
             fontsize=6.6, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))

    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=160)
        print("salvo:", f"{a.out}.{ext}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

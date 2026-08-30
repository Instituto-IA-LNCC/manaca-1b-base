#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manaca-1B - Painel de benchmarks para o artigo (ingles, qualidade de publicacao)
================================================================================
Le docs/evaluation/benchmarks-pt.json e monta um painel 2x2 (CALAME-PT,
ARC-Challenge-PT, HellaSwag-PT, LAMBADA-PT), acuracia vs parametros (escala log),
com barras de IC95% (+-1.96*SE), Manaca-1B destacado, tendencia da familia PT-BR e
paleta Okabe-Ito (colorblind-safe, validada). Gera PDF (vetorial) + PNG 300 dpi.

Uso:
    python scripts/eval/plot_benchmarks_paper.py

Autor: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

# Okabe-Ito (CVD-safe), validado por scripts/validate_palette.js
GRUPO = {
    "Manaca-1B": "ours",
    "Tucano-160m": "ptbr", "Tucano-630m": "ptbr", "Tucano-1b1": "ptbr",
    "Tucano-2b4": "ptbr", "TTL-160m": "ptbr", "TTL-460m": "ptbr", "Sabia-7B": "ptbr",
    "GlorIA-1b3": "ptpt", "mGPT-1b3": "multi",
}
STYLE = {  # grupo -> (cor, marcador, rotulo de legenda)
    "ours":  ("#D55E00", "*", "Manacá-1B (ours)"),
    "ptbr":  ("#0072B2", "o", "PT-BR (TTL / Tucano / Sabiá)"),
    "ptpt":  ("#E69F00", "s", "PT-PT (GlórIA)"),
    "multi": ("#009E73", "^", "Multilingual (mGPT)"),
}
DISPLAY = {"Sabia-7B": "Sabiá-7B", "Manaca-1B": "Manacá-1B"}
PANELS = [("calame", "CALAME-PT"), ("arc", "ARC-Challenge-PT"),
          ("hellaswag", "HellaSwag-PT"), ("lambada", "LAMBADA-PT")]
XTICKS = [0.16, 0.3, 0.6, 1, 2, 7]


def val_se(cell):
    if isinstance(cell, (list, tuple)) and cell and cell[0] is not None:
        return float(cell[0]), (float(cell[1]) if len(cell) > 1 and cell[1] is not None else None)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", default="docs/evaluation/benchmarks-pt.json")
    ap.add_argument("--out", default="docs/evaluation/benchmarks_paper_en")
    a = ap.parse_args()

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#444", "axes.linewidth": 0.8,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })

    linhas = json.load(open(a.benchmarks, encoding="utf-8"))["linhas"]
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6.0))

    for ax, (chave, titulo) in zip(axes.flat, PANELS):
        # tendencia PT-BR (por tamanho)
        ptbr = sorted((m["params"], val_se(m.get(chave))[0])
                      for nome, m in linhas.items()
                      if GRUPO.get(nome) == "ptbr" and m.get("params")
                      and val_se(m.get(chave))[0] is not None)
        if len(ptbr) >= 2:
            ax.plot([p for p, _ in ptbr], [v for _, v in ptbr], "-",
                    color=STYLE["ptbr"][0], alpha=0.28, lw=1.4, zorder=1)

        seen = set()
        for nome, m in linhas.items():
            par = m.get("params")
            v, se = val_se(m.get(chave))
            if par is None or v is None:
                continue
            g = GRUPO.get(nome, "multi")
            cor, marc, leg = STYLE[g]
            rotulo = leg if g not in seen else None
            seen.add(g)
            if se is not None:
                ax.errorbar(par, v, yerr=1.96 * se, fmt="none", ecolor=cor,
                            elinewidth=0.9, capsize=2.2, alpha=0.85, zorder=2)
            ax.scatter([par], [v], s=300 if g == "ours" else 60, marker=marc,
                       facecolor=cor, edgecolor="#222", linewidth=0.6,
                       label=rotulo, zorder=4 if g == "ours" else 3)

        # rotulos diretos: Manaca (destaque) e Sabia-7B (ancora superior)
        for nome, dx, dy, peso, cor in [
            ("Manaca-1B", 0, 11, "bold", STYLE["ours"][0]),
            ("Sabia-7B", 0, -13, "normal", "#333"),
        ]:
            m = linhas.get(nome)
            if not m:
                continue
            v, _ = val_se(m.get(chave))
            if v is None:
                continue
            ax.annotate(DISPLAY.get(nome, nome), (m["params"], v),
                        textcoords="offset points", xytext=(dx, dy), ha="center",
                        fontsize=7.2, fontweight=peso, color=cor, zorder=5)

        if chave == "arc":
            ax.axhline(25, ls=(0, (3, 3)), color="#888", lw=0.9, zorder=0)
            ax.text(0.985, 25, "chance", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=6.6, color="#888")

        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(XTICKS))
        ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in XTICKS]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(0.12, 9)
        ax.set_title(titulo, pad=6)
        ax.set_xlabel("Parameters (billions, log scale)")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, axis="y", alpha=0.18, lw=0.6)

    # legenda unica compartilhada embaixo (libera os paineis)
    handles = [Line2D([0], [0], marker=STYLE[g][1], linestyle="none",
                      markerfacecolor=STYLE[g][0], markeredgecolor="#222",
                      markeredgewidth=0.6, markersize=(12 if g == "ours" else 8),
                      label=STYLE[g][2]) for g in ("ours", "ptbr", "ptpt", "multi")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.005), handletextpad=0.35, columnspacing=1.6)
    fig.suptitle("Manacá-1B across Portuguese benchmarks",
                 fontsize=12, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0.045, 1, 0.965), h_pad=1.6, w_pad=1.8)

    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=300, bbox_inches="tight")
        print("saved:", f"{a.out}.{ext}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

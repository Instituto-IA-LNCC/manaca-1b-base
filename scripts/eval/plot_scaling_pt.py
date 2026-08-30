#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manaca-1B - Curva de escala no CALAME-PT (acuracia vs parametros)
=================================================================

PT ------------------------------------------------------------------------
Le os vetores de acertos por exemplo (--save-calame de eval_base.py) presentes em
docs/evaluation/logs/ e monta a curva de acuracia CALAME-PT versus numero de
parametros, com barras de erro de 95% (SE binomial), destacando o Manaca-1B e
ligando a familia PT-BR (TeenyTinyLlama + Tucano) numa linha de tendencia.

Nao exige internet nem GPU: so le os *_calame.json ja gerados. Gera PNG e PDF.

Uso:
    python scripts/eval/plot_scaling_pt.py
    python scripts/eval/plot_scaling_pt.py --logs docs/evaluation/logs --out docs/evaluation

EN ------------------------------------------------------------------------
Manaca-1B - Scaling curve on CALAME-PT (accuracy vs parameters)
Reads the per-example correctness vectors (--save-calame from eval_base.py) present
in docs/evaluation/logs/ and builds the CALAME-PT accuracy curve versus number of
parameters, with 95% error bars (binomial SE), highlighting Manaca-1B and linking
the PT-BR family (TeenyTinyLlama + Tucano) with a trend line.

Requires no internet and no GPU: it only reads the *_calame.json already generated.
Produces PNG and PDF.

Usage:
    python scripts/eval/plot_scaling_pt.py
    python scripts/eval/plot_scaling_pt.py --logs docs/evaluation/logs --out docs/evaluation

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# label do arquivo (<label>_calame.json) -> (parametros_B, nome, grupo)
META = {
    "manaca":      (1.72, "Manaca-1B",   "manaca"),
    "tucano":      (1.10, "Tucano-1b1",  "ptbr"),
    "tucano2b4":   (2.40, "Tucano-2b4",  "ptbr"),
    "tucano-160m": (0.16, "Tucano-160m", "ptbr"),
    "tucano-630m": (0.63, "Tucano-630m", "ptbr"),
    "ttl-160m":    (0.16, "TTL-160m",    "ptbr"),
    "ttl-460m":    (0.46, "TTL-460m",    "ptbr"),
    "gloria-1b3":  (1.30, "GlorIA-1b3",  "ptpt"),
    "mgpt-1b3":    (1.30, "mGPT-1b3",    "multi"),
    "sabia-7b":    (7.00, "Sabia-7B",    "ptbr"),
}

ESTILO = {  # grupo -> (cor, marcador, rotulo de legenda)
    "manaca": ("#d1495b", "*", "Manaca-1B (este trabalho)"),
    "ptbr":   ("#2e6f95", "o", "PT-BR (TeenyTinyLlama + Tucano)"),
    "ptpt":   ("#e8871e", "s", "PT-PT (GlorIA)"),
    "multi":  ("#8d99ae", "^", "Multilingue (mGPT)"),
}


def carregar(caminho):
    with open(caminho, encoding="utf-8") as fh:
        d = json.load(fh)
    v = d["corretos"]
    n = len(v)
    p = sum(v) / n
    se = math.sqrt(p * (1 - p) / n)
    return p, se, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="docs/evaluation/logs")
    ap.add_argument("--out", default="docs/evaluation")
    a = ap.parse_args()

    pontos = []  # (params, acc, se, nome, grupo, label)
    for caminho in sorted(glob.glob(os.path.join(a.logs, "*_calame.json"))):
        label = os.path.basename(caminho)[:-len("_calame.json")]
        if label not in META:
            print(f"[aviso] sem metadados para '{label}', pulando ({caminho})")
            continue
        params, nome, grupo = META[label]
        acc, se, n = carregar(caminho)
        pontos.append((params, acc, se, nome, grupo, label))
        print(f"  {nome:12s} {params:>5.2f}B  {100*acc:5.2f}%  SE {100*se:.2f}%  (n={n})")

    if not pontos:
        print("Nenhum *_calame.json encontrado com metadados. Rode o lote antes.")
        return 1

    fig, ax = plt.subplots(figsize=(8.2, 5.4))

    # linha de tendencia PT-BR (ligando a familia por tamanho), sem o Manaca
    ptbr = sorted([p for p in pontos if p[4] == "ptbr"], key=lambda x: x[0])
    if len(ptbr) >= 2:
        ax.plot([p[0] for p in ptbr], [100 * p[1] for p in ptbr],
                "-", color=ESTILO["ptbr"][0], alpha=0.35, lw=1.6, zorder=1)

    vistos = set()
    for params, acc, se, nome, grupo, label in pontos:
        cor, marc, leg = ESTILO[grupo]
        rotulo = leg if grupo not in vistos else None
        vistos.add(grupo)
        tam = 320 if grupo == "manaca" else 90
        ax.errorbar(params, 100 * acc, yerr=196 * se, fmt="none",
                    ecolor=cor, elinewidth=1.2, capsize=3, alpha=0.8, zorder=2)
        ax.scatter([params], [100 * acc], s=tam, marker=marc, color=cor,
                   edgecolor="white", linewidth=0.8, label=rotulo, zorder=3)
        dy = 0.9 if grupo != "manaca" else 1.3
        ax.annotate(nome, (params, 100 * acc), textcoords="offset points",
                    xytext=(0, 9 if grupo == "manaca" else 7),
                    ha="center", fontsize=8.5,
                    fontweight="bold" if grupo == "manaca" else "normal",
                    color=cor)

    ax.set_xscale("log")
    ax.set_xlabel("Parametros (bilhoes, escala log)")
    ax.set_ylabel("Acuracia CALAME-PT (%)")
    ax.set_title("Manaca-1B na curva de escala do CALAME-PT (PT-BR)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    fig.text(0.01, 0.005,
             "CALAME-PT (NOVA-vision-language), n=2075, mesmo harness (scripts/eval). "
             "Barras: IC95% binomial. Diferencas de ~1 ponto ficam no ruido (ver teste pareado).",
             fontsize=6.8, color="#555")
    fig.tight_layout(rect=(0, 0.02, 1, 1))

    os.makedirs(a.out, exist_ok=True)
    for ext in ("png", "pdf"):
        caminho = os.path.join(a.out, f"calame_escala_pt.{ext}")
        fig.savefig(caminho, dpi=160)
        print("salvo:", caminho)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

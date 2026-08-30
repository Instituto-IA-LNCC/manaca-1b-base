#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manaca-1B - Dinamica de pre-treino (figura de artigo, ingles)
=============================================================

PT ------------------------------------------------------------------------
Le o log cru do Megatron (docs/training/logs/train_20260706_030015.log) e monta um
painel 1x3 com o comportamento do treino: (a) loss de treino e de validacao, (b)
norma do gradiente (escala log, com os transientes precoces), (c) learning rate
(warmup + cosseno). Eixo x em bilhoes de tokens. Paleta Okabe-Ito, PDF + PNG.

Uso:
    python scripts/eval/plot_training_dynamics.py

EN ------------------------------------------------------------------------
Manaca-1B - Pre-training dynamics (paper figure, English)
Reads the raw Megatron log (docs/training/logs/train_20260706_030015.log) and builds
a 1x3 panel with the training behavior: (a) training and validation loss, (b)
gradient norm (log scale, with the early transients), (c) learning rate
(warmup + cosine). X axis in billions of tokens. Okabe-Ito palette, PDF + PNG.

Usage:
    python scripts/eval/plot_training_dynamics.py

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TOK_PER_ITER = 512 * 4096 / 1e9   # bilhoes de tokens por iteracao (batch 512, seq 4096)
C_TRAIN, C_VAL, C_GRAD, C_LR = "#0072B2", "#D55E00", "#333333", "#009E73"

IT = re.compile(r"iteration\s+(\d+)/\s*20000.*?learning rate:\s*([0-9.E+-]+)"
                r".*?lm loss:\s*([0-9.E+-]+).*?grad norm:\s*([0-9.E+naN-]+)"
                r".*?number of skipped iterations:\s*(\d+).*?number of nan iterations:\s*(\d+)")
VAL = re.compile(r"validation loss at iteration\s+(\d+)\s*\|\s*lm loss value:\s*([0-9.E+-]+)")


def parse(path):
    tr, va = [], []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = IT.search(line)
        if m:
            try:
                gn = float(m.group(4))
            except ValueError:
                gn = float("nan")
            tr.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), gn,
                       int(m.group(5)), int(m.group(6))))
        v = VAL.search(line)
        if v:
            va.append((int(v.group(1)), float(v.group(2))))
    return tr, va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="docs/training/logs/train_20260706_030015.log")
    ap.add_argument("--out", default="docs/training/training_dynamics_en")
    a = ap.parse_args()

    tr, va = parse(a.log)
    it = [r[0] for r in tr]
    tok = [i * TOK_PER_ITER for i in it]
    lr = [r[1] for r in tr]
    loss = [r[2] for r in tr]
    grad = [r[3] for r in tr]
    skip = sum(r[4] for r in tr)
    nan = sum(r[5] for r in tr)
    vtok = [i * TOK_PER_ITER for i, _ in va]
    vloss = [l for _, l in va]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#444", "axes.linewidth": 0.8,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    fig, (axL, axG, axR) = plt.subplots(1, 3, figsize=(10.5, 3.2))

    # (a) loss treino + validacao
    axL.plot(tok, loss, "-", color=C_TRAIN, lw=1.1, label="Training loss")
    axL.plot(vtok, vloss, "o-", color=C_VAL, lw=1.2, ms=3.5, mec="white", mew=0.5,
             label="Validation loss")
    axL.set_title("(a) Loss")
    axL.set_xlabel("Training tokens (billions)")
    axL.set_ylabel("LM loss (nats)")
    axL.legend(loc="upper right", frameon=False)
    axL.grid(True, alpha=0.18, lw=0.6)

    # (b) grad norm (log)
    axG.plot(tok, grad, "-", color=C_GRAD, lw=0.9)
    axG.set_yscale("log")
    axG.set_title("(b) Gradient norm")
    axG.set_xlabel("Training tokens (billions)")
    axG.set_ylabel("Global grad norm")
    axG.grid(True, which="both", alpha=0.16, lw=0.6)
    imax = max(range(len(grad)), key=lambda k: grad[k])
    axG.annotate(f"max {grad[imax]:.1f}", (tok[imax], grad[imax]),
                 textcoords="offset points", xytext=(6, 2), fontsize=7, color=C_GRAD)
    axG.text(0.97, 0.05, f"{skip} skipped / {nan} NaN steps", transform=axG.transAxes,
             ha="right", va="bottom", fontsize=7, color="#444",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#ddd", lw=0.6))

    # (c) learning rate
    axR.plot(tok, lr, "-", color=C_LR, lw=1.2)
    axR.set_title("(c) Learning rate")
    axR.set_xlabel("Training tokens (billions)")
    axR.set_ylabel("Learning rate")
    axR.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axR.grid(True, alpha=0.18, lw=0.6)

    fig.tight_layout(w_pad=1.8)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=300, bbox_inches="tight")
        print("saved:", f"{a.out}.{ext}")
    print(f"final: train loss {loss[-1]:.3f}, val loss {vloss[-1]:.3f} "
          f"(ppl {2.718281828 ** vloss[-1]:.2f}), lr {lr[-1]:.2e}, "
          f"grad max {max(grad):.2f}, skipped {skip}, nan {nan}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

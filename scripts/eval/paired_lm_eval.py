#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manaca-1B - McNemar PAREADO para ARC-PT / HellaSwag-PT / LAMBADA-PT
===================================================================
Fecha a estatistica pareada desses tres benchmarks no mesmo padrao do CALAME
(§9 do relatorio). Le os acertos por exemplo que o lm-eval grava com
`--log_samples` (arquivos samples_<task>_*.jsonl), alinha os modelos por `doc_id`
e, para um modelo de referencia (default Manaca-1B) contra cada outro:

  * diferenca de acuracia com IC95% por bootstrap PAREADO (reamostra os mesmos docs),
  * teste de McNemar (p-valor) sobre os discordantes.

Salva um resumo (docs/evaluation/paired-benchmarks-pt.md/.json) e os vetores de
acerto compactos (docs/evaluation/vectors-pt.json), para reproduzir sem os samples
brutos (que sao grandes e ficam fora do git).

Gerar os samples antes (re-run do lote com --log_samples ligado):
    LOG_SAMPLES=1 FORCE=1 MANACA_TOKENIZER=$HOME/hf_cache_eval/manaca-tok-fixed \
        ./scripts/eval/run_lm_eval_pt.sh
Depois:
    python scripts/eval/paired_lm_eval.py --samples-dir $HOME/manaca-lmeval-out

Autor: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random

METRIC = {"arc_pt": "acc_norm", "hellaswag_pt": "acc_norm", "lambada_pt": "acc"}
NOME = {
    "manaca": "Manaca-1B", "gloria-1b3": "GlorIA-1b3", "mgpt-1b3": "mGPT-1b3",
    "ttl-160m": "TTL-160m", "ttl-460m": "TTL-460m", "tucano-160m": "Tucano-160m",
    "tucano-630m": "Tucano-630m", "tucano-1b1": "Tucano-1b1",
    "tucano-2b4": "Tucano-2b4", "sabia-7b": "Sabia-7B",
}
ORDEM_TAREFAS = ["lambada_pt", "hellaswag_pt", "arc_pt"]


def val_metric(s: dict, metric: str):
    for k in (metric, metric + ",none"):
        v = s.get(k)
        if isinstance(v, (int, float)):
            return int(round(float(v)))
    for k, v in s.items():
        if isinstance(v, (int, float)) and k.split(",")[0] == metric:
            return int(round(float(v)))
    return None


def carregar(samples_dir: str, label: str, task: str):
    """{doc_id: 0/1} do samples jsonl mais recente de label/task, ou None."""
    padrao = os.path.join(samples_dir, label, task, "**", f"samples_{task}_*.jsonl")
    fs = sorted(glob.glob(padrao, recursive=True))  # ISO no nome -> ordena
    if not fs:
        return None
    metric = METRIC[task]
    d = {}
    with open(fs[-1], encoding="utf-8") as fh:
        for linha in fh:
            linha = linha.strip()
            if not linha:
                continue
            try:
                s = json.loads(linha)
            except Exception:
                continue
            did = s.get("doc_id")
            v = val_metric(s, metric)
            if did is not None and v is not None:
                d[did] = v
    return d or None


def mcnemar(a, b):
    b10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    b01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    tot = b10 + b01
    chi = (abs(b10 - b01) - 1) ** 2 / tot if tot else 0.0
    p = math.erfc(math.sqrt(chi / 2)) if tot else 1.0
    return b10, b01, p


def boot_pareado(a, b, n=5000):
    rng = random.Random(0)
    N = len(a)
    difs = []
    for _ in range(n):
        sa = sb = 0
        for _ in range(N):
            i = rng.randrange(N)
            sa += a[i]; sb += b[i]
        difs.append((sa - sb) / N)
    difs.sort()
    return difs[int(0.025 * n)], difs[int(0.975 * n)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", default=os.path.expanduser("~/manaca-lmeval-out"))
    ap.add_argument("--ref", default="manaca")
    ap.add_argument("--out", default="docs/evaluation/paired-benchmarks-pt")
    ap.add_argument("--vectors", default="docs/evaluation/vectors-pt.json")
    a = ap.parse_args()

    resumo = {"ref": NOME.get(a.ref, a.ref), "tarefas": {}}
    vetores = {}
    linhas_md = [f"# McNemar pareado - referencia {NOME.get(a.ref, a.ref)}", ""]

    for task in ORDEM_TAREFAS:
        ref = carregar(a.samples_dir, a.ref, task)
        if not ref:
            print(f"== {task}: sem samples do '{a.ref}' (rode com LOG_SAMPLES=1). Pulando.")
            continue
        vetores.setdefault(task, {})[a.ref] = ref
        acc_ref = 100 * sum(ref.values()) / len(ref)
        print(f"== {task} ({METRIC[task]}) ==  {NOME.get(a.ref)} acc={acc_ref:.2f}%  n={len(ref)}")
        linhas_md += [f"## {task} ({METRIC[task]})  -  {NOME.get(a.ref)} {acc_ref:.2f}%",
                      "", "| vs | dif | IC95% pareado | McNemar p | sig |", "|---|---|---|---|---|"]
        comps = []
        for label in NOME:
            if label == a.ref:
                continue
            o = carregar(a.samples_dir, label, task)
            if not o:
                continue
            vetores[task][label] = o
            comuns = sorted(set(ref) & set(o))
            if not comuns:
                continue
            va = [ref[i] for i in comuns]
            vb = [o[i] for i in comuns]
            dif = 100 * (sum(va) - sum(vb)) / len(comuns)
            b10, b01, p = mcnemar(va, vb)
            lo, hi = boot_pareado(va, vb)
            sig = "SIM" if (p < 0.05 and (lo > 0 or hi < 0)) else "nao"
            print(f"   vs {NOME[label]:12s} {dif:+6.2f}  IC95%[{100*lo:+.2f},{100*hi:+.2f}]"
                  f"  McNemar b10={b10} b01={b01} p={p:.4f} [{sig}]")
            linhas_md.append(f"| {NOME[label]} | {dif:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] "
                             f"| {p:.4f} | {sig} |")
            comps.append({"vs": NOME[label], "n": len(comuns), "dif_pontos": round(dif, 2),
                          "ic95_pareado": [round(100 * lo, 2), round(100 * hi, 2)],
                          "mcnemar_b10": b10, "mcnemar_b01": b01, "mcnemar_p": round(p, 4),
                          "significativo_5pct": bool(sig == "SIM")})
        resumo["tarefas"][task] = {"metric": METRIC[task], "acc_ref": round(acc_ref, 2),
                                   "n": len(ref), "comparacoes": comps}
        linhas_md.append("")
        print()

    if not resumo["tarefas"]:
        print("Nenhum samples encontrado. Rode o lote com LOG_SAMPLES=1 antes.")
        return 1

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out + ".md", "w", encoding="utf-8").write("\n".join(linhas_md) + "\n")
    json.dump(resumo, open(a.out + ".json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    # vetores compactos: por tarefa, doc_ids comuns + acertos por modelo
    compacto = {}
    for task, mods in vetores.items():
        docs = sorted(set.intersection(*[set(v) for v in mods.values()]))
        compacto[task] = {"doc_ids": docs,
                          "modelos": {lab: [mods[lab][i] for i in docs] for lab in mods}}
    json.dump(compacto, open(a.vectors, "w", encoding="utf-8"))
    print("salvo:", a.out + ".md", ",", a.out + ".json", "e", a.vectors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

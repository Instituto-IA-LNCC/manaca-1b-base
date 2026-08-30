#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manaca-1B - Junta CALAME-PT + ARC-PT + HellaSwag-PT + LAMBADA-PT numa tabela
============================================================================

PT ------------------------------------------------------------------------
Combina:
  * o CALAME-PT que ja temos (docs/evaluation/results-base.json), e
  * as saidas do lm-evaluation-harness para ARC-Challenge-PT, HellaSwag-PT e
    LAMBADA-PT (JSONs gerados com `lm_eval --output_path <dir>`),
numa unica tabela por modelo (markdown + JSON), destacando o Manaca-1B.

Robusto ao id exato da tarefa: classifica por substring (arc / hellaswag /
lambada) e escolhe a metrica primaria (acc_norm p/ ARC e HellaSwag, acc p/
LAMBADA), com fallback para acc.

Uso:
    python scripts/eval/merge_pt_benchmarks.py --lm-eval-dir <dir_das_saidas>
    # opcional: --out docs/evaluation/benchmarks-pt

EN ------------------------------------------------------------------------
Manaca-1B - Merge CALAME-PT + ARC-PT + HellaSwag-PT + LAMBADA-PT into a table
Combines:
  * the CALAME-PT we already have (docs/evaluation/results-base.json), and
  * the lm-evaluation-harness outputs for ARC-Challenge-PT, HellaSwag-PT and
    LAMBADA-PT (JSONs generated with `lm_eval --output_path <dir>`),
into a single per-model table (markdown + JSON), highlighting Manaca-1B.

Robust to the exact task id: classifies by substring (arc / hellaswag /
lambada) and picks the primary metric (acc_norm for ARC and HellaSwag, acc for
LAMBADA), with a fallback to acc.

Usage:
    python scripts/eval/merge_pt_benchmarks.py --lm-eval-dir <output_dir>
    # optional: --out docs/evaluation/benchmarks-pt

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import glob
import json
import os

# id do modelo (como aparece no HF / no pretrained=) -> (nome, params_b)
CONHECIDOS = {
    "manaca": ("Manaca-1B", 1.72),
    "teenytinyllama-160m": ("TTL-160m", 0.16),
    "teenytinyllama-460m": ("TTL-460m", 0.46),
    "tucano-160m": ("Tucano-160m", 0.16),
    "tucano-630m": ("Tucano-630m", 0.63),
    "tucano-1b1": ("Tucano-1b1", 1.10),
    "tucano-2b4": ("Tucano-2b4", 2.40),
    "gloria-1.3b": ("GlorIA-1b3", 1.30),
    "gloria": ("GlorIA-1b3", 1.30),
    "mgpt": ("mGPT-1b3", 1.30),
    "sabia-7b": ("Sabia-7B", 7.00),
}


def id_para_nome(model_id: str):
    s = model_id.lower()
    for chave, (nome, par) in CONHECIDOS.items():
        if chave in s:
            return nome, par
    # heuristica: se e um caminho local do Manaca
    if "/m" == model_id or "manaca-1b-hf" in s:
        return "Manaca-1B", 1.72
    return model_id, None


def metrica(task_res: dict, prefer: str):
    """Escolhe a metrica primaria e seu stderr. Devolve (nome, valor, stderr)."""
    ordem = [prefer, "acc_norm", "acc"]
    for p in ordem:
        val = se = None
        for k, v in task_res.items():
            if not isinstance(v, (int, float)):
                continue
            base = k.split(",")[0]
            if base == p:
                val = float(v)
            elif base == p + "_stderr":
                se = float(v)
        if val is not None:
            return p, val, se
    return None, None, None


def classificar(task_name: str):
    t = task_name.lower()
    if "arc" in t:
        return "arc", "acc_norm"
    if "hellaswag" in t or "hswag" in t:
        return "hellaswag", "acc_norm"
    if "lambada" in t:
        return "lambada", "acc"
    return None, None


def carregar_lm_eval(dirpath: str):
    """Varre um diretorio de saidas do lm_eval; devolve {model_id: {bench: (val, se)}}.

    Quando ha varios resultados para o mesmo modelo/tarefa (ex.: re-run com FORCE=1),
    o MAIS NOVO vence. Usa o campo `date` gravado pelo proprio lm-eval (timestamp
    interno) em vez da data do arquivo, porque o git zera a mtime no checkout.
    """
    registros = []  # (date, model_id, {bench: (val, se)})
    for caminho in glob.glob(os.path.join(dirpath, "**", "*.json"), recursive=True):
        try:
            d = json.load(open(caminho, encoding="utf-8"))
        except Exception:
            continue
        if "results" not in d or not isinstance(d["results"], dict):
            continue
        mid = d.get("model_name") or ""
        if not mid:
            cfg = d.get("config", {}) or {}
            ma = cfg.get("model_args", "") if isinstance(cfg, dict) else ""
            if isinstance(ma, str) and "pretrained=" in ma:
                mid = ma.split("pretrained=")[1].split(",")[0]
        mid = mid or os.path.basename(os.path.dirname(caminho))
        try:
            dt = float(d.get("date"))
        except (TypeError, ValueError):
            dt = os.path.getmtime(caminho)  # fallback
        benches = {}
        for task, res in d["results"].items():
            bench, prefer = classificar(task)
            if not bench:
                continue
            _, val, se = metrica(res, prefer)
            if val is not None:
                v = 100.0 * val if val <= 1.0 else val
                s = (100.0 * se) if (se is not None and se <= 1.0) else se
                benches[bench] = (v, s)
        if benches:
            registros.append((dt, mid, benches))
    registros.sort(key=lambda r: r[0])  # antigo -> novo (novo sobrescreve)
    dados = {}
    for _, mid, benches in registros:
        dados.setdefault(mid, {}).update(benches)
    return dados


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm-eval-dir", required=True,
                    help="diretorio com as saidas do lm_eval (--output_path)")
    ap.add_argument("--calame", default="docs/evaluation/results-base.json")
    ap.add_argument("--out", default="docs/evaluation/benchmarks-pt")
    a = ap.parse_args()

    # CALAME que ja temos (com SE, quando disponivel)
    base = json.load(open(a.calame, encoding="utf-8"))
    calame = {}
    for r in base.get("results", []):
        nome = r.get("nome") or r.get("model")
        se = r.get("calame_se")
        calame[nome] = (r.get("params_b"),
                        (100.0 * r["calame_acc"], (100.0 * se) if se is not None else None))

    lm = carregar_lm_eval(a.lm_eval_dir)
    linhas = {}  # nome -> dict de metricas
    for nome, (par, cal) in calame.items():
        linhas.setdefault(nome, {"params": par})["calame"] = cal
    for mid, benches in lm.items():
        nome, par = id_para_nome(mid)
        row = linhas.setdefault(nome, {"params": par})
        if row.get("params") is None:
            row["params"] = par
        row.update(benches)

    ordenado = sorted(linhas.items(),
                      key=lambda kv: (kv[1].get("params") is None, kv[1].get("params") or 0))

    def fmt(x):  # params (numero simples)
        return f"{x:.2f}" if isinstance(x, (int, float)) else "-"

    def fmt_se(cell):  # metrica: (valor, se) -> "valor ±se"
        if not isinstance(cell, (list, tuple)):
            return "-"
        v, s = cell[0], (cell[1] if len(cell) > 1 else None)
        if v is None:
            return "-"
        return f"{v:.2f} ±{s:.2f}" if isinstance(s, (int, float)) else f"{v:.2f}"

    linhas_md = ["Valores: acuracia (%) ± erro padrao (SE). IC95% ~ ±1,96·SE.",
                 "",
                 "| Modelo | Params (B) | CALAME-PT | ARC-Ch-PT | HellaSwag-PT | LAMBADA-PT |",
                 "|---|---|---|---|---|---|"]
    for nome, m in ordenado:
        estrela = "**" if "Manaca" in nome else ""
        linhas_md.append(
            f"| {estrela}{nome}{estrela} | {fmt(m.get('params'))} | {fmt_se(m.get('calame'))} | "
            f"{fmt_se(m.get('arc'))} | {fmt_se(m.get('hellaswag'))} | {fmt_se(m.get('lambada'))} |")
    md = "\n".join(linhas_md) + "\n"

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    open(a.out + ".md", "w", encoding="utf-8").write(md)
    json.dump({"linhas": {n: m for n, m in ordenado}}, open(a.out + ".json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(md)
    # Linha final bilingue (PT + EN) — onde a tabela consolidada foi gravada.
    print("salvo:", a.out + ".md", "e", a.out + ".json")
    print("saved:", a.out + ".md", "and", a.out + ".json")
    faltando = [n for n, m in ordenado if any(k not in m for k in ("arc", "hellaswag", "lambada"))]
    if faltando:
        print("[nota] sem todas as tarefas ainda para:", ", ".join(faltando))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

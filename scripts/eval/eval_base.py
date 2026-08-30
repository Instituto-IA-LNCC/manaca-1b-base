#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Manaca-1B - Avaliacao JUSTA de modelo base (BPB + perplexidade + CALAME-PT)
===========================================================================

PT ------------------------------------------------------------------------
Compara modelos base de linguagem no MESMO texto e MESMO protocolo, com metricas
que nao dependem do tokenizador:

  * BPB (bits por byte): NLL total em bits dividido pelos BYTES do texto. E a
    metrica justa entre tokenizadores diferentes (vocab de 32k, 64k, 128k dao
    perplexidades por token diferentes, mas o mesmo texto tem os mesmos bytes).
  * bits/token e perplexidade por token: uteis, mas comparaveis so entre modelos
    com o MESMO tokenizador (registro interno).
  * CALAME-PT: acuracia em prever a ULTIMA palavra de um paragrafo (nativo PT-BR,
    funciona em modelo base, e por acerto -> comparavel).

Roda o MESMO script para cada modelo (Manaca local, Tucano, Llama-3.2) e junta-se
a tabela. Para o Manaca, passe --spm para tokenizar com o SentencePiece (aplica a
normalizacao nmt_nfkc_cf do treino); os demais usam o proprio tokenizador HF.

Uso (imagem manaca-train, com --gpus all):
    python eval_base.py --model /m --spm /tok/manaca-tokenizer.model --calame
    python eval_base.py --model TucanoBR/Tucano-1b1 --calame
    python eval_base.py --model meta-llama/Llama-3.2-1B --calame

Justica de caixa (case): o Manaca e lowercase (tokenizador com case-fold). Para
nivelar, use --lowercase em TODOS os modelos (avalia todo mundo em minusculas) e
reporte tambem sem, com a ressalva. O script imprime os dois lados quando possivel.

EN ------------------------------------------------------------------------
Manaca-1B - FAIR base-model evaluation (BPB + perplexity + CALAME-PT)
Compares base language models on the SAME text and SAME protocol, with metrics
that do not depend on the tokenizer:

  * BPB (bits per byte): total NLL in bits divided by the BYTES of the text. It is
    the fair metric across different tokenizers (vocab of 32k, 64k, 128k give
    different per-token perplexities, but the same text has the same bytes).
  * bits/token and per-token perplexity: useful, but comparable only between models
    with the SAME tokenizer (internal record).
  * CALAME-PT: accuracy at predicting the LAST word of a paragraph (native PT-BR,
    works on a base model, and by exact match -> comparable).

Run the SAME script for each model (local Manaca, Tucano, Llama-3.2) and merge into
the table. For Manaca, pass --spm to tokenize with SentencePiece (applies the
training's nmt_nfkc_cf normalization); the others use their own HF tokenizer.

Usage (manaca-train image, with --gpus all):
    python eval_base.py --model /m --spm /tok/manaca-tokenizer.model --calame
    python eval_base.py --model TucanoBR/Tucano-1b1 --calame
    python eval_base.py --model meta-llama/Llama-3.2-1B --calame

Case fairness: Manaca is lowercase (case-fold tokenizer). To level the field, use
--lowercase on ALL models (evaluate everyone in lowercase) and also report without
it, with the caveat. The script prints both sides when possible.

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch


def carregar_modelo(nome: str, device: str):
    from transformers import AutoModelForCausalLM
    dt = torch.float16 if device == "cuda" else torch.float32
    m = AutoModelForCausalLM.from_pretrained(nome, torch_dtype=dt)
    m.to(device).eval()
    return m


def fazer_encoder(args):
    """Devolve encode(text)->list[int] e um rotulo do tokenizador."""
    if args.spm:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=args.spm)

        def enc(t: str):
            return sp.encode(t)
        return enc, f"SPM({sp.vocab_size()})", sp
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)

    def enc(t: str):
        return tok.encode(t, add_special_tokens=False)
    return enc, f"HF({tok.vocab_size})", tok


@torch.no_grad()
def bpb_de_texto(model, enc, texto: str, device: str, window: int, stride: int):
    """BPB, bits/token e perplexidade por token de um texto, por janela deslizante."""
    ids = enc(texto)
    if len(ids) < 2:
        return None
    ids = torch.tensor([ids], dtype=torch.long)
    n = ids.size(1)
    n_bytes = len(texto.encode("utf-8"))
    total_nll = 0.0
    total_tok = 0
    prev = 0
    for begin in range(0, n, stride):
        end = min(begin + window, n)
        trg = end - prev
        inp = ids[:, begin:end].to(device)
        tgt = inp.clone()
        tgt[:, :-trg] = -100
        out = model(inp, labels=tgt)
        valid = int((tgt[:, 1:] != -100).sum())
        if valid > 0:
            total_nll += out.loss.item() * valid
            total_tok += valid
        prev = end
        if end == n:
            break
    avg = total_nll / total_tok
    return {
        "tokens": total_tok, "bytes": n_bytes,
        "bytes_per_token": n_bytes / total_tok,
        "bits_per_token": avg / math.log(2),
        "ppl_token": math.exp(avg),
        "bpb": (total_nll / math.log(2)) / n_bytes,
    }


@torch.no_grad()
def calame(model, enc, decode, exemplos, device: str, lower: bool, suppress=None):
    """Acuracia da ultima palavra. Devolve a lista de acertos (0/1) por exemplo,
    para calcular IC por bootstrap e permitir teste pareado entre modelos."""
    _pont = ".,;:!?\"'()[]"
    pad_id = getattr(model.config, "eos_token_id", None) or 0
    corretos = []
    for ctx, alvo in exemplos:
        alvo_cmp = alvo.lower().strip().strip(_pont)
        if lower:
            ctx = ctx.lower()
        ids = enc(ctx)
        if not ids:
            continue
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        mask = torch.ones_like(inp)
        # gera tokens suficientes para cobrir a palavra alvo; suppress evita que o
        # modelo emita ids de padding (>= vocab do SPM) que o decoder nao conhece.
        gen = model.generate(inp, attention_mask=mask, max_new_tokens=8,
                             do_sample=False, pad_token_id=pad_id,
                             suppress_tokens=suppress)
        novos = gen[0, inp.size(1):].tolist()
        cont = decode(novos).strip()
        palavra = cont.split()[0] if cont.split() else ""
        corretos.append(1 if palavra.lower().strip(_pont) == alvo_cmp else 0)
    return corretos


# ----------------- incerteza (bootstrap) -----------------
def _boot_proporcao(corretos, boot=2000, seed=0):
    """IC 95% de uma proporcao (acuracia) por bootstrap sobre os exemplos."""
    import random
    rng = random.Random(seed)
    n = len(corretos)
    if n < 2:
        return (None, None)
    amostras = []
    for _ in range(boot):
        s = sum(corretos[rng.randrange(n)] for _ in range(n))
        amostras.append(s / n)
    amostras.sort()
    return amostras[int(0.025 * boot)], amostras[int(0.975 * boot)]


def _boot_razao(bits, bytes_, boot=2000, seed=0):
    """IC 95% de sum(bits)/sum(bytes) por bootstrap sobre segmentos (para BPB)."""
    import random
    rng = random.Random(seed)
    n = len(bits)
    if n < 2:
        return (None, None)
    amostras = []
    for _ in range(boot):
        b = by = 0.0
        for _ in range(n):
            i = rng.randrange(n)
            b += bits[i]; by += bytes_[i]
        amostras.append(b / by)
    amostras.sort()
    return amostras[int(0.025 * boot)], amostras[int(0.975 * boot)]


@torch.no_grad()
def bpb_por_segmentos(model, enc, segmentos, device: str):
    """BPB medindo cada segmento (frase) de forma independente, para ter muitos
    pontos e estimar o IC por bootstrap. Devolve (bpb, lo, hi, n_segmentos)."""
    seg_bits, seg_bytes = [], []
    for seg in segmentos:
        seg = seg.strip()
        if not seg:
            continue
        ids = enc(seg)
        if len(ids) < 2:
            continue
        t = torch.tensor([ids], dtype=torch.long, device=device)
        out = model(t, labels=t)
        nvalid = t.size(1) - 1
        seg_bits.append(out.loss.item() * nvalid / math.log(2))
        seg_bytes.append(len(seg.encode("utf-8")))
    if not seg_bits:
        return None
    bpb = sum(seg_bits) / sum(seg_bytes)
    lo, hi = _boot_razao(seg_bits, seg_bytes)
    return bpb, lo, hi, len(seg_bits)


def carregar_calame(path: str | None, lower: bool):
    """Devolve lista de (contexto, ultima_palavra)."""
    linhas = []
    if path:
        import json
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                d = json.loads(ln)
                s = d.get("sentence") or d.get("text") or ""
                lw = d.get("last_word") or ""
                linhas.append((s, lw))
    else:
        # SEM pip e SEM a lib 'datasets': baixa o JSONL do CALAME direto do
        # HuggingFace com huggingface_hub (ja na imagem). Neste host o pypi e
        # intermitente, mas o HF resolve; e o HF cacheia em HF_HOME.
        import json
        from huggingface_hub import hf_hub_download
        repo = "NOVA-vision-language/calame-pt"
        subset = os.environ.get("CALAME_SUBSET", "all")
        fname = {"all": "calamept_all.jsonl",
                 "generated": "calamept_gen_only.jsonl",
                 "handwritten": "calamept_handwritten_only.jsonl"}.get(subset, "calamept_all.jsonl")
        print(f"[eval] CALAME: baixando {fname} do HF (sem pip)...")
        local = hf_hub_download(repo, fname, repo_type="dataset")
        primeiro = None
        with open(local, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                d = json.loads(ln)
                if primeiro is None:
                    primeiro = d
                s = d.get("sentence") or d.get("text") or d.get("context") or ""
                lw = d.get("last_word") or d.get("target") or d.get("word") or ""
                if s and lw:
                    linhas.append((s, lw))
        if not linhas and primeiro is not None:
            raise RuntimeError(f"CALAME: campos inesperados; chaves do 1o exemplo: {list(primeiro.keys())}")
    # No CALAME-PT o campo 'sentence' JA e o contexto (nao inclui a ultima
    # palavra); 'last_word' e o alvo. Entao contexto = sentence, sem cortar.
    # (So removemos a palavra se, por acaso, a frase terminar com ela.)
    out = []
    for s, lw in linhas:
        s = s.rstrip()
        lw = (lw or "").strip()
        if not (s and lw):
            continue
        ctx = s[: -len(lw)].rstrip() if s.endswith(lw) else s
        out.append((ctx, lw))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Avaliacao base: BPB + CALAME-PT")
    ap.add_argument("--model", required=True, help="Caminho local ou nome no HF Hub.")
    ap.add_argument("--spm", default=None, help="(Manaca) .model do SentencePiece p/ tokenizar fiel ao treino.")
    ap.add_argument("--text", default=None, help="Arquivo de texto p/ BPB (UTF-8).")
    ap.add_argument("--calame", action="store_true", help="Rodar CALAME-PT (baixa o dataset).")
    ap.add_argument("--calame-path", default=None, help="JSONL local do CALAME (offline).")
    ap.add_argument("--lowercase", action="store_true", help="Minuscula o texto p/ comparacao justa de caixa.")
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=1024)
    ap.add_argument("--max-calame", type=int, default=0, help="Limita nº de exemplos CALAME (0 = todos).")
    ap.add_argument("--save-calame", default=None, help="Salva os acertos por exemplo (JSON) p/ teste pareado.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] device={device}  modelo={args.model}")
    model = carregar_modelo(args.model, device)
    enc, tok_label, tokobj = fazer_encoder(args)
    print(f"[eval] tokenizador={tok_label}  janela={args.window} stride={args.stride} lowercase={args.lowercase}")

    # decoder para o CALAME (usa o mesmo tokenizador). Para o SPM, filtra ids de
    # padding (>= vocab do SPM) e prepara a lista de tokens a suprimir na geracao.
    suppress = None
    if args.spm:
        _vsz = tokobj.vocab_size()
        _mvsz = int(getattr(model.config, "vocab_size", _vsz))
        if _mvsz > _vsz:
            suppress = list(range(_vsz, _mvsz))
        decode = lambda ids: tokobj.decode([i for i in ids if 0 <= i < _vsz])  # noqa: E731
    else:
        decode = lambda ids: tokobj.decode(ids, skip_special_tokens=True)  # noqa: E731

    resultado = {"model": args.model, "tokenizer": tok_label}

    # --- texto para BPB: --text, senao as frases do CALAME ---
    texto = None
    calame_ex = None
    if args.calame or args.calame_path:
        calame_ex = carregar_calame(args.calame_path, args.lowercase)
        if args.max_calame:
            calame_ex = calame_ex[: args.max_calame]
        print(f"[eval] CALAME-PT: {len(calame_ex)} exemplos")

    if args.text:
        with open(args.text, encoding="utf-8") as fh:
            texto = fh.read()
    elif calame_ex:
        texto = "\n".join(f"{c} {w}" for c, w in calame_ex)

    if texto is not None:
        if args.lowercase:
            texto = texto.lower()
        r = bpb_de_texto(model, enc, texto, device, args.window, args.stride)
        if r:
            resultado.update(r)
            print("[eval] ---- modelagem de linguagem (texto reservado) ----")
            print(f"[eval]   BPB (bits/byte)   : {r['bpb']:.4f}   (menor e melhor)")
            print(f"[eval]   perplexidade/token: {r['ppl_token']:.3f}")
            print(f"[eval]   bits/token        : {r['bits_per_token']:.4f}")
            print(f"[eval]   bytes/token       : {r['bytes_per_token']:.4f}   ({r['tokens']} tokens, {r['bytes']} bytes)")

    if calame_ex:
        # BPB com IC sobre as frases do CALAME (muitos segmentos -> IC estreito)
        segs = [c.lower() if args.lowercase else c for c, _ in calame_ex]
        bs = bpb_por_segmentos(model, enc, segs, device)
        if bs:
            bpb_s, lo, hi, nseg = bs
            resultado["bpb_calame"] = bpb_s
            resultado["bpb_calame_ci95"] = [lo, hi]
            resultado["bpb_calame_nseg"] = nseg
            print("[eval] ---- BPB sobre CALAME (com IC) ----")
            print(f"[eval]   BPB              : {bpb_s:.4f}   IC95% [{lo:.4f}, {hi:.4f}]   ({nseg} frases)")

        corretos = calame(model, enc, decode, calame_ex, device, args.lowercase, suppress)
        n = len(corretos)
        ac = sum(corretos)
        acc = ac / n if n else 0.0
        se = math.sqrt(acc * (1 - acc) / n) if n else 0.0
        lo, hi = _boot_proporcao(corretos)
        resultado["calame_acc"] = acc
        resultado["calame_se"] = se
        resultado["calame_ci95"] = [lo, hi]
        resultado["calame_n"] = n
        print("[eval] ---- CALAME-PT (com IC) ----")
        print(f"[eval]   acuracia: {ac}/{n} = {100*acc:.2f}%   "
              f"SE {100*se:.2f}%   IC95% [{100*lo:.2f}%, {100*hi:.2f}%]")
        # salva o vetor de acertos p/ teste pareado entre modelos
        if args.save_calame:
            with open(args.save_calame, "w", encoding="utf-8") as fh:
                import json as _json
                _json.dump({"model": args.model, "lowercase": args.lowercase,
                            "corretos": corretos}, fh)
            print(f"[eval]   acertos por exemplo salvos em: {args.save_calame}")

    # Resumo final bilingue (PT + EN) — a saida que o usuario le ao fim da execucao.
    _resumo = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in resultado.items()}
    print("[eval] RESUMO:", _resumo)
    print("[eval] SUMMARY:", _resumo)
    return 0


if __name__ == "__main__":
    sys.exit(main())

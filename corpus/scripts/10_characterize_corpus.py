#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
corpus/scripts/10_characterize_corpus.py
=========================================
Projeto Manacá LLM — LNCC × NII/LLM-jp
Fase 1: Caracterização Estatística Completa do Corpus PT-BR

Coleta as seguintes métricas por fonte e para o total, baseado na
literatura de alto impacto (ACL/EMNLP/NeurIPS Datasets & Benchmarks):

BLOCO 1 — Volume e Tokenização
  - Bytes UTF-8 brutos
  - Palavras (whitespace)
  - Tokens com 3 tokenizadores: GPT-2 BPE (50k), Llama-3 (128k), Gemma-2 (256k)
  - Fertility (tokens/palavra) e bytes/token por tokenizador
  - STRR — Single Token Retention Rate (Nayeem et al., arXiv:2510.09947)

BLOCO 2 — Distribuição de Comprimento de Documento
  - Percentis [1,5,10,25,50,75,90,95,99] de chars e tokens-Gemma
  - Histograma de comprimento (20 bins)

BLOCO 3 — Sinais de Qualidade Gopher
  13 sinais da Tabela A1 de Rae et al. (arXiv:2112.11446, 2021):
  - alpha_ratio, mean_word_length, symbol_word_ratio
  - frac_lines_end_punct, frac_no_alpha_lines
  - dup_line_frac, dup_para_frac
  - dup_line_char_frac, dup_para_char_frac
  - top2_char_gram_frac, top3_char_gram_frac, top4_char_gram_frac
  - stop_word_ratio (PT-BR stoplist)

BLOCO 4 — Perplexidade KenLM
  - Modelo 5-gram Kneser–Ney treinado em Wikipedia PT-BR
  - Perplexidade média, mediana, p10, p90 por documento (Wenzek et al., 2019)

BLOCO 5 — Diversidade Lexical
  - MATTR (Moving Average TTR, window=500, Covington & McFall 2010)
  - MTLD (McCarthy & Jarvis 2010)
  - Yule's K (Yule 1944)
  - Herdan's C = log(V) / log(N)
  Calculado em amostra aleatória de 100k docs por fonte

BLOCO 6 — Cobertura Temporal
  - Histograma de timestamps por fonte (campo 'date' quando disponível)
  - Distribuição por ano

BLOCO 7 — Distribuição de Domínios (fontes web)
  - Top 100 TLDs/domínios (campo 'url' quando disponível)
  - Proporção .br, .com.br, .gov.br, .edu.br

BLOCO 8 — Sobreposição Inter-Fonte (MinHash)
  - Jaccard pairwise entre fontes via MinHash 13-grams (256 hashes)
  - Antes e depois de deduplicação
  (amostra 50k docs/fonte — estima o overlap, não contagem exata)

BLOCO 9 — Detecção de PII
  - Emails (RFC 5322 regex)
  - CPFs (regex + verificação de dígitos)
  - Telefones brasileiros (DD + 8/9 dígitos)
  - IPs IPv4

BLOCO 10 — Contaminação de Benchmark (n-grams)
  - 13-grams contra benchmarks PT-BR declarados
  - ENEM, BLUEX, OAB, ASSIN2, FAQUAD, HateBR, PoEta-v2
  (requer arquivos de benchmark em BENCHMARK_DIR)

OUTPUT
  $WORK_DIR/stats/corpus_characterization.json  — métricas completas
  $WORK_DIR/stats/corpus_characterization.md    — relatório legível
  $WORK_DIR/stats/per_source/                   — métricas por fonte

REFERÊNCIAS-CHAVE
  Rae et al. (2021). Scaling Language Models (Gopher). arXiv:2112.11446
  Wenzek et al. (2020). CCNet. LREC 2020. arXiv:1911.00359
  Lee et al. (2022). Deduplicating Training Data. ACL 2022. arXiv:2107.06499
  Petrov et al. (2023). Tokenizers Introduce Unfairness. NeurIPS. arXiv:2305.15425
  Nayeem et al. (2025). Beyond Fertility (STRR). arXiv:2510.09947
  McCarthy & Jarvis (2010). MTLD. Behavior Research Methods 42:381–392
  Covington & McFall (2010). MATTR. JQL 17:94–100

Autores: Bruno L. S. Menezes, LNCC Instituto de IA
         Co-authored-by: Fabio Porto <fporto@lncc.br>
"""

# ════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import pyarrow.parquet as pq
from loguru import logger
from tqdm import tqdm

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO
# ════════════════════════════════════════════════════════════════════════════
WORK_DIR      = Path(os.environ.get("WORK_DIR", "/workspace/manaca-corpus"))
STATS_DIR     = WORK_DIR / "stats"
PER_SOURCE_DIR = STATS_DIR / "per_source"
BENCHMARK_DIR = WORK_DIR / "benchmarks"          # colocar arquivos .txt de benchmark aqui
KENLM_MODEL   = WORK_DIR / "kenlm" / "wikipedia_ptbr_5gram.arpa.bin"

# Número de documentos amostrados para métricas custosas (léxico, KenLM, PII)
SAMPLE_SIZE_LEXICAL   = 100_000  # por fonte
SAMPLE_SIZE_MINHASH   = 50_000   # por fonte (para matriz inter-fonte)
SAMPLE_SIZE_KENLM     = 50_000   # por fonte
SAMPLE_SIZE_PII       = 100_000  # por fonte

# Fontes do Tier 1
SOURCES = {
    "gigaverbo":     WORK_DIR / "raw" / "gigaverbo",
    "madlad400":     WORK_DIR / "raw" / "madlad400",
    "fineweb2":      WORK_DIR / "raw" / "fineweb2",
    "hplt2":         WORK_DIR / "raw" / "hplt2",
    "wikipedia_ptbr": WORK_DIR / "raw" / "wikipedia_ptbr",
    "ulysses":       WORK_DIR / "raw" / "ulysses",
}

# Tokenizadores (carregados sob demanda)
TOKENIZER_IDS = {
    "gpt2":   "gpt2",
    "llama3": "NousResearch/Meta-Llama-3-8B",
    "gemma2": "unsloth/gemma-2-2b",
}

# Stopwords PT-BR (Gopher rule adaptada)
STOPWORDS_PTBR = frozenset({
    "o", "a", "os", "as", "um", "uma", "uns", "umas",
    "de", "do", "da", "dos", "das", "no", "na", "nos", "nas",
    "ao", "aos", "à", "às", "pelo", "pela", "pelos", "pelas",
    "em", "com", "por", "para", "que", "se", "não", "mais",
    "como", "mas", "foi", "ele", "ela", "seu", "sua",
    "ou", "quando", "muito", "também", "já", "eu", "é", "são",
    "este", "esta", "isso", "aquele", "aquela", "neste", "nessa",
})

# ════════════════════════════════════════════════════════════════════════════
# SETUP
# ════════════════════════════════════════════════════════════════════════════
STATS_DIR.mkdir(parents=True, exist_ok=True)
PER_SOURCE_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, format="{time:HH:mm:ss} | {level:<7} | {message}", level="INFO")
logger.add(STATS_DIR / "characterization.log", level="DEBUG", rotation="100 MB")


# ════════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS GERAIS
# ════════════════════════════════════════════════════════════════════════════

def iter_parquet_texts(
    source_dir: Path,
    columns: List[str] = None,
    max_docs: Optional[int] = None,
    shuffle_seed: Optional[int] = 42,
) -> Generator[Dict, None, None]:
    """
    Itera sobre todos os shards Parquet de uma fonte.
    Se max_docs for definido, retorna uma amostra aleatória.
    """
    if columns is None:
        columns = ["text", "source", "lang", "score"]
    shards = sorted(source_dir.glob("shard_*.parquet"))
    if not shards:
        logger.warning(f"Nenhum shard em {source_dir}")
        return

    # Coleta metadados para amostragem
    total_rows = 0
    shard_sizes = []
    for s in shards:
        pf = pq.ParquetFile(s)
        n = pf.metadata.num_rows
        shard_sizes.append(n)
        total_rows += n

    if max_docs is None or max_docs >= total_rows:
        # Iterar tudo
        for s in shards:
            table = pq.read_table(s, columns=columns)
            for i in range(len(table)):
                yield {c: table[c][i].as_py() for c in columns if c in table.schema.names}
        return

    # Amostragem proporcional por shard
    rng = random.Random(shuffle_seed)
    for s, n in zip(shards, shard_sizes):
        k = max(1, round(max_docs * n / total_rows))
        indices = sorted(rng.sample(range(n), min(k, n)))
        table = pq.read_table(s, columns=columns)
        for i in indices:
            yield {c: table[c][i].as_py() for c in columns if c in table.schema.names}


def percentiles(values: List[float], qs: List[float] = None) -> Dict[str, float]:
    """Calcula percentis de uma lista de valores."""
    if not values:
        return {}
    if qs is None:
        qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    arr = np.array(values, dtype=np.float64)
    return {f"p{q}": float(np.percentile(arr, q)) for q in qs}


def histogram(values: List[float], n_bins: int = 20) -> Dict:
    """Histograma simplificado."""
    if not values:
        return {"bins": [], "counts": []}
    arr = np.array(values, dtype=np.float64)
    counts, edges = np.histogram(arr, bins=n_bins)
    return {
        "bins": [round(float(e), 4) for e in edges],
        "counts": [int(c) for c in counts],
    }


# ════════════════════════════════════════════════════════════════════════════
# BLOCO 1 — TOKENIZAÇÃO
# ════════════════════════════════════════════════════════════════════════════

_tokenizers: Dict[str, Any] = {}

def load_tokenizer(name: str) -> Optional[Any]:
    """Carrega tokenizador HuggingFace com cache."""
    if name in _tokenizers:
        return _tokenizers[name]
    try:
        from transformers import AutoTokenizer
        tok_id = TOKENIZER_IDS[name]
        logger.info(f"Carregando tokenizador: {tok_id}")
        tok = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
        _tokenizers[name] = tok
        return tok
    except Exception as e:
        logger.warning(f"Tokenizador {name} indisponivel: {e}")
        return None


def count_tokens(tokenizer: Any, text: str) -> int:
    """Conta tokens sem adicionar tokens especiais."""
    if tokenizer is None:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False, truncation=False))


def strr(tokenizer: Any, text: str) -> float:
    """
    Single Token Retention Rate — proporção de palavras 'whitespace'
    representadas como token único. Nayeem et al. (2025), arXiv:2510.09947.
    """
    if tokenizer is None:
        return 0.0
    words = text.split()
    if not words:
        return 0.0
    single = 0
    for w in words[:200]:  # amostra por eficiência
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            single += 1
    return single / min(len(words), 200)


def compute_tokenization_metrics(
    source_name: str,
    source_dir: Path,
    sample_size: int = 50_000,
) -> Dict:
    """
    Bloco 1: Métricas de volume e tokenização.
    Percorre TODOS os shards para bytes/palavras; usa amostra para tokens.
    """
    logger.info(f"[Tokenização] {source_name}")

    # Passagem 1: bytes e palavras — todos os docs
    total_docs  = 0
    total_bytes = 0
    total_words = 0
    shards = sorted(source_dir.glob("shard_*.parquet"))

    for shard in tqdm(shards, desc=f"  bytes/words {source_name}", leave=False):
        table = pq.read_table(shard, columns=["text"])
        for text in table["text"].to_pylist():
            if not isinstance(text, str):
                continue
            total_docs  += 1
            total_bytes += len(text.encode("utf-8"))
            total_words += len(text.split())

    result: Dict[str, Any] = {
        "total_docs":  total_docs,
        "total_bytes": total_bytes,
        "total_words": total_words,
        "bytes_per_word": round(total_bytes / max(total_words, 1), 4),
    }

    # Passagem 2: tokens — amostra
    tokenizer_results = {}
    for tok_name in TOKENIZER_IDS:
        tok = load_tokenizer(tok_name)
        if tok is None:
            continue
        token_counts  = []
        fertility_vals = []
        strr_vals     = []
        for doc in tqdm(
            iter_parquet_texts(source_dir, columns=["text"], max_docs=sample_size),
            desc=f"  tokens-{tok_name} {source_name}", total=sample_size, leave=False
        ):
            text = doc.get("text", "")
            if not isinstance(text, str) or len(text) < 10:
                continue
            n_tok  = count_tokens(tok, text[:8192])   # limitar por memória
            n_word = max(len(text[:8192].split()), 1)
            token_counts.append(n_tok)
            fertility_vals.append(n_tok / n_word)
            strr_vals.append(strr(tok, text[:2000]))

        if token_counts:
            n_sampled = len(token_counts)
            # Extrapolar para total_docs
            mean_tok = float(np.mean(token_counts))
            tokenizer_results[tok_name] = {
                "sample_size":          n_sampled,
                "mean_tokens_per_doc":  round(mean_tok, 2),
                "total_tokens_est":     int(mean_tok * total_docs),
                "fertility_mean":       round(float(np.mean(fertility_vals)), 4),
                "fertility_p50":        round(float(np.percentile(fertility_vals, 50)), 4),
                "fertility_p95":        round(float(np.percentile(fertility_vals, 95)), 4),
                "bytes_per_token":      round(total_bytes / max(mean_tok * total_docs, 1), 4),
                "strr_mean":            round(float(np.mean(strr_vals)), 4),
            }
        del token_counts, fertility_vals, strr_vals
        gc.collect()

    result["tokenizers"] = tokenizer_results
    return result


# ════════════════════════════════════════════════════════════════════════════
# BLOCO 2 — DISTRIBUIÇÃO DE COMPRIMENTO
# ════════════════════════════════════════════════════════════════════════════

def compute_length_distribution(
    source_name: str,
    source_dir: Path,
    sample_size: int = 200_000,
) -> Dict:
    """Bloco 2: Distribuição de comprimento de documento."""
    logger.info(f"[Comprimento] {source_name}")
    char_lens  = []
    word_lens  = []

    for doc in tqdm(
        iter_parquet_texts(source_dir, columns=["text"], max_docs=sample_size),
        desc=f"  comprimento {source_name}", total=sample_size, leave=False
    ):
        text = doc.get("text", "")
        if not isinstance(text, str):
            continue
        char_lens.append(len(text))
        word_lens.append(len(text.split()))

    return {
        "sample_size":    len(char_lens),
        "chars":          percentiles(char_lens) | {"mean": round(float(np.mean(char_lens)), 2)},
        "words":          percentiles(word_lens) | {"mean": round(float(np.mean(word_lens)), 2)},
        "chars_histogram": histogram(char_lens),
        "words_histogram": histogram(word_lens),
    }


# ════════════════════════════════════════════════════════════════════════════
# BLOCO 3 — SINAIS DE QUALIDADE GOPHER
# ════════════════════════════════════════════════════════════════════════════

_TERMINAL_PUNCT = frozenset(".!?;:")

def gopher_signals(text: str) -> Dict[str, float]:
    """
    Computa os 13 sinais de qualidade Gopher (Rae et al. 2021, Tabela A1).
    Retorna floats normalizados [0,1] ou contagens.
    """
    if not text:
        return {}

    words  = text.split()
    n_word = max(len(words), 1)
    chars  = list(text)
    n_char = max(len(chars), 1)
    lines  = text.splitlines()
    n_line = max(len(lines), 1)
    paras  = [p.strip() for p in text.split("\n\n") if p.strip()]
    n_para = max(len(paras), 1)

    # 1. Proporção de caracteres alfabéticos
    n_alpha = sum(1 for c in chars if c.isalpha())
    alpha_ratio = n_alpha / n_char

    # 2. Comprimento médio de palavra
    mean_word_len = sum(len(w) for w in words) / n_word

    # 3. Razão símbolo-palavra (Gopher: caracteres não-alfanuméricos exceto espaço)
    n_symbol = sum(1 for c in chars if not c.isalnum() and not c.isspace())
    symbol_word_ratio = n_symbol / n_word

    # 4. Fração de linhas terminando em pontuação terminal
    n_end_punct = sum(1 for l in lines if l.rstrip() and l.rstrip()[-1] in _TERMINAL_PUNCT)
    frac_lines_end_punct = n_end_punct / n_line

    # 5. Fração de linhas sem caractere alfabético
    n_no_alpha = sum(1 for l in lines if not any(c.isalpha() for c in l))
    frac_no_alpha_lines = n_no_alpha / n_line

    # 6–9. Duplicatas de linha e parágrafo
    line_ctr  = Counter(l.strip() for l in lines if l.strip())
    para_ctr  = Counter(p for p in paras)

    dup_line_count = sum(v for v in line_ctr.values() if v > 1)
    dup_line_frac  = dup_line_count / n_line

    dup_para_count = sum(v for v in para_ctr.values() if v > 1)
    dup_para_frac  = dup_para_count / n_para

    dup_line_chars = sum(len(l) * (v - 1)
                         for l, v in line_ctr.items() if v > 1)
    dup_line_char_frac = dup_line_chars / n_char

    dup_para_chars = sum(len(p) * (v - 1)
                         for p, v in para_ctr.items() if v > 1)
    dup_para_char_frac = dup_para_chars / n_char

    # 10–12. Top n-gram char fractions (2-gram, 3-gram, 4-gram de palavras)
    def top_ngram_char_frac(ws: List[str], n: int) -> float:
        if len(ws) < n:
            return 0.0
        ngrams = Counter(tuple(ws[i:i+n]) for i in range(len(ws)-n+1))
        if not ngrams:
            return 0.0
        top_val = ngrams.most_common(1)[0][1]
        top_chars = sum(len(w) for w in ngrams.most_common(1)[0][0]) * top_val
        return top_chars / n_char

    top2 = top_ngram_char_frac(words, 2)
    top3 = top_ngram_char_frac(words, 3)
    top4 = top_ngram_char_frac(words, 4)

    # 13. Stopword ratio PT-BR
    n_stop = sum(1 for w in words if w.lower() in STOPWORDS_PTBR)
    stop_word_ratio = n_stop / n_word

    return {
        "alpha_ratio":          round(alpha_ratio, 4),
        "mean_word_length":     round(mean_word_len, 4),
        "symbol_word_ratio":    round(symbol_word_ratio, 4),
        "frac_lines_end_punct": round(frac_lines_end_punct, 4),
        "frac_no_alpha_lines":  round(frac_no_alpha_lines, 4),
        "dup_line_frac":        round(dup_line_frac, 4),
        "dup_para_frac":        round(dup_para_frac, 4),
        "dup_line_char_frac":   round(dup_line_char_frac, 4),
        "dup_para_char_frac":   round(dup_para_char_frac, 4),
        "top2_char_gram_frac":  round(top2, 4),
        "top3_char_gram_frac":  round(top3, 4),
        "top4_char_gram_frac":  round(top4, 4),
        "stop_word_ratio":      round(stop_word_ratio, 4),
    }


def compute_gopher_metrics(
    source_name: str,
    source_dir: Path,
    sample_size: int = 100_000,
) -> Dict:
    """Bloco 3: Agrega sinais Gopher sobre amostra."""
    logger.info(f"[Gopher] {source_name}")

    accum: Dict[str, List[float]] = defaultdict(list)
    n = 0

    for doc in tqdm(
        iter_parquet_texts(source_dir, columns=["text"], max_docs=sample_size),
        desc=f"  gopher {source_name}", total=sample_size, leave=False
    ):
        text = doc.get("text", "")
        if not isinstance(text, str) or len(text) < 50:
            continue
        sigs = gopher_signals(text)
        for k, v in sigs.items():
            accum[k].append(v)
        n += 1

    result: Dict = {"sample_size": n}
    for key, vals in accum.items():
        result[key] = {
            "mean":   round(float(np.mean(vals)), 5),
            "median": round(float(np.median(vals)), 5),
            "std":    round(float(np.std(vals)), 5),
            "p10":    round(float(np.percentile(vals, 10)), 5),
            "p90":    round(float(np.percentile(vals, 90)), 5),
        }
    return result


# ════════════════════════════════════════════════════════════════════════════
# BLOCO 4 — PERPLEXIDADE KENLM
# ════════════════════════════════════════════════════════════════════════════

def compute_kenlm_perplexity(
    source_name: str,
    source_dir: Path,
    sample_size: int = SAMPLE_SIZE_KENLM,
) -> Dict:
    """
    Bloco 4: Perplexidade KenLM 5-gram (Wenzek et al., 2020).
    Requer modelo em KENLM_MODEL.
    """
    logger.info(f"[KenLM] {source_name}")
    if not KENLM_MODEL.exists():
        msg = (f"Modelo KenLM não encontrado em {KENLM_MODEL}. "
               "Execute build_kenlm_model() após rodar o Script 05.")
        logger.warning(msg)
        return {"error": msg}

    try:
        import kenlm
        model = kenlm.Model(str(KENLM_MODEL))
    except Exception as e:
        return {"error": str(e)}

    perps = []
    for doc in tqdm(
        iter_parquet_texts(source_dir, columns=["text"], max_docs=sample_size),
        desc=f"  kenlm {source_name}", total=sample_size, leave=False
    ):
        text = doc.get("text", "")
        if not isinstance(text, str) or len(text.split()) < 10:
            continue
        try:
            p = model.perplexity(text[:2000])
            if math.isfinite(p):
                perps.append(p)
        except Exception:
            pass

    if not perps:
        return {"sample_size": 0, "error": "nenhum doc valido"}

    return {
        "sample_size": len(perps),
        "mean":   round(float(np.mean(perps)), 2),
        "median": round(float(np.median(perps)), 2),
        "std":    round(float(np.std(perps)), 2),
        "p10":    round(float(np.percentile(perps, 10)), 2),
        "p25":    round(float(np.percentile(perps, 25)), 2),
        "p75":    round(float(np.percentile(perps, 75)), 2),
        "p90":    round(float(np.percentile(perps, 90)), 2),
        "histogram": histogram(np.log10(np.array(perps) + 1).tolist()),
    }


# ════════════════════════════════════════════════════════════════════════════
# BLOCO 5 — DIVERSIDADE LEXICAL
# ════════════════════════════════════════════════════════════════════════════

def _mattr(tokens: List[str], window: int = 500) -> float:
    """
    Moving Average Type-Token Ratio.
    Covington & McFall (2010). JQL 17:94-100.
    """
    n = len(tokens)
    if n < window:
        return len(set(tokens)) / max(n, 1)
    ttrs = []
    for i in range(n - window + 1):
        window_tokens = tokens[i: i + window]
        ttrs.append(len(set(window_tokens)) / window)
    return float(np.mean(ttrs)) if ttrs else 0.0


def _mtld(tokens: List[str], threshold: float = 0.72) -> float:
    """
    Measure of Textual Lexical Diversity (bidirectional).
    McCarthy & Jarvis (2010). Behavior Research Methods 42:381–392.
    """
    def _one_direction(toks: List[str]) -> float:
        if len(toks) < 10:
            return 0.0
        factors = 0.0
        types   = set()
        n_toks  = 0
        for t in toks:
            types.add(t)
            n_toks += 1
            if n_toks == 0:
                continue
            ttr = len(types) / n_toks
            if ttr <= threshold:
                factors += 1.0
                types   = set()
                n_toks  = 0
        if n_toks > 0 and len(types) / max(n_toks, 1) > threshold:
            factors += n_toks / max(len(toks), 1)
        return len(toks) / max(factors, 1.0)

    fwd = _one_direction(tokens)
    rev = _one_direction(tokens[::-1])
    return (fwd + rev) / 2.0


def _yule_k(tokens: List[str]) -> float:
    """Yule's K (1944). Mede repetitividade."""
    freq = Counter(tokens)
    M1   = len(tokens)
    M2   = sum(v * v for v in freq.values())
    if M1 == 0:
        return 0.0
    return 10_000 * (M2 - M1) / max(M1 * M1, 1)


def _herdan_c(tokens: List[str]) -> float:
    """Herdan's C = log(V) / log(N)."""
    N = len(tokens)
    V = len(set(tokens))
    if N < 2 or V < 2:
        return 0.0
    return math.log(V) / math.log(N)


def compute_lexical_diversity(
    source_name: str,
    source_dir: Path,
    sample_size: int = SAMPLE_SIZE_LEXICAL,
) -> Dict:
    """Bloco 5: Diversidade lexical sobre amostra."""
    logger.info(f"[Léxico] {source_name}")

    mattr_vals = []
    mtld_vals  = []
    yule_vals  = []
    herdan_vals = []

    for doc in tqdm(
        iter_parquet_texts(source_dir, columns=["text"], max_docs=sample_size),
        desc=f"  léxico {source_name}", total=sample_size, leave=False
    ):
        text = doc.get("text", "")
        if not isinstance(text, str) or len(text.split()) < 50:
            continue
        # Tokenização simples: lowercase, apenas palavras
        tokens = re.findall(r"\b[a-záâãàäéêèëíîìïóôõòöúûùüçñ]{2,}\b", text.lower())
        if len(tokens) < 50:
            continue

        mattr_vals.append(_mattr(tokens))
        mtld_vals.append(_mtld(tokens))
        yule_vals.append(_yule_k(tokens))
        herdan_vals.append(_herdan_c(tokens))

    def agg(vals: List[float], name: str) -> Dict:
        if not vals:
            return {"n": 0}
        return {
            "n":      len(vals),
            "mean":   round(float(np.mean(vals)), 5),
            "median": round(float(np.median(vals)), 5),
            "std":    round(float(np.std(vals)), 5),
            "p10":    round(float(np.percentile(vals, 10)), 5),
            "p90":    round(float(np.percentile(vals, 90)), 5),
        }

    return {
        "sample_size": len(mattr_vals),
        "mattr_window500":  agg(mattr_vals,  "MATTR"),
        "mtld_bidirectional": agg(mtld_vals, "MTLD"),
        "yule_k":           agg(yule_vals,   "Yule-K"),
        "herdan_c":         agg(herdan_vals, "Herdan-C"),
    }


# ════════════════════════════════════════════════════════════════════════════
# BLOCO 6 — COBERTURA TEMPORAL
# ════════════════════════════════════════════════════════════════════════════

def compute_temporal_coverage(
    source_name: str,
    source_dir: Path,
) -> Dict:
    """Bloco 6: Distribuição temporal por ano."""
    logger.info(f"[Temporal] {source_name}")

    year_counter: Counter = Counter()
    n_with_date = 0
    n_total     = 0

    for shard in sorted(source_dir.glob("shard_*.parquet")):
        schema = pq.read_schema(shard)
        if "date" not in schema.names:
            # Fonte sem campo de data
            break
        table = pq.read_table(shard, columns=["date"])
        for val in table["date"].to_pylist():
            n_total += 1
            if val is None:
                continue
            try:
                year = str(val)[:4]
                if year.isdigit() and 1990 <= int(year) <= 2030:
                    year_counter[year] += 1
                    n_with_date += 1
            except Exception:
                pass

    return {
        "total_docs":         n_total,
        "docs_with_date":     n_with_date,
        "coverage_pct":       round(100 * n_with_date / max(n_total, 1), 2),
        "by_year":            dict(sorted(year_counter.items())),
    }


# ════════════════════════════════════════════════════════════════════════════
# BLOCO 7 — DISTRIBUIÇÃO DE DOMÍNIOS
# ════════════════════════════════════════════════════════════════════════════

def extract_tld(url: str) -> str:
    """Extrai TLD de URL."""
    try:
        netloc = urlparse(url).netloc.lower()
        if not netloc:
            return "unknown"
        parts = netloc.split(".")
        if len(parts) >= 3 and parts[-1] in {"br", "uk", "au", "jp", "de", "fr", "pt"}:
            return ".".join(parts[-2:])
        return parts[-1] if parts else "unknown"
    except Exception:
        return "unknown"


def compute_domain_distribution(
    source_name: str,
    source_dir: Path,
    top_n: int = 100,
) -> Dict:
    """Bloco 7: Distribuição de domínios/TLDs (fontes web)."""
    logger.info(f"[Domínios] {source_name}")

    tld_counter:    Counter = Counter()
    domain_counter: Counter = Counter()
    n_with_url = 0
    n_total    = 0

    for shard in sorted(source_dir.glob("shard_*.parquet")):
        schema = pq.read_schema(shard)
        if "url" not in schema.names:
            break
        table = pq.read_table(shard, columns=["url"])
        for url in table["url"].to_pylist():
            n_total += 1
            if not url:
                continue
            try:
                netloc = urlparse(url).netloc.lower()
                domain_counter[netloc] += 1
                tld_counter[extract_tld(url)] += 1
                n_with_url += 1
            except Exception:
                pass

    br_count = sum(v for k, v in tld_counter.items() if k in {"br", "com.br", "gov.br", "edu.br", "org.br", "net.br"})

    return {
        "total_docs":     n_total,
        "docs_with_url":  n_with_url,
        "br_docs":        br_count,
        "br_pct":         round(100 * br_count / max(n_with_url, 1), 2),
        "top_tlds":       dict(tld_counter.most_common(top_n)),
        "top_domains":    dict(domain_counter.most_common(top_n)),
    }


# ════════════════════════════════════════════════════════════════════════════
# BLOCO 8 — MINHASH INTER-FONTE
# ════════════════════════════════════════════════════════════════════════════

def _minhash_signature(text: str, n_hashes: int = 128, n_gram: int = 13) -> np.ndarray:
    """
    Assinatura MinHash para um documento.
    13-grams de palavras, 128 hashes. Lee et al. (2022), arXiv:2107.06499.
    """
    words  = text.split()
    ngrams = {" ".join(words[i:i+n_gram]) for i in range(max(1, len(words)-n_gram+1))}
    sig    = np.full(n_hashes, np.iinfo(np.uint64).max, dtype=np.uint64)
    for ng in ngrams:
        h = int(hashlib.sha256(ng.encode()).hexdigest(), 16) & 0xFFFFFFFFFFFFFFFF
        for i in range(n_hashes):
            v = ((h * (i + 1)) ^ (i * 2654435761)) & 0xFFFFFFFFFFFFFFFF
            if v < sig[i]:
                sig[i] = v
    return sig


def compute_minhash_overlap(
    sample_size: int = SAMPLE_SIZE_MINHASH,
) -> Dict:
    """
    Bloco 8: Matriz de Jaccard estimada inter-fontes via MinHash 13-grams.
    """
    logger.info("[MinHash] Calculando sobreposição inter-fontes")
    source_sigs: Dict[str, List[np.ndarray]] = {}

    for src_name, src_dir in SOURCES.items():
        if not src_dir.exists():
            continue
        sigs = []
        for doc in tqdm(
            iter_parquet_texts(src_dir, columns=["text"], max_docs=sample_size),
            desc=f"  minhash {src_name}", total=sample_size, leave=False
        ):
            text = doc.get("text", "")
            if isinstance(text, str) and len(text.split()) >= 13:
                sigs.append(_minhash_signature(text))
        source_sigs[src_name] = sigs
        logger.info(f"  {src_name}: {len(sigs)} assinaturas")

    # Calcular Jaccard pairwise
    names = list(source_sigs.keys())
    matrix: Dict[str, Dict[str, float]] = {n: {} for n in names}

    for i in range(len(names)):
        for j in range(i, len(names)):
            n1, n2 = names[i], names[j]
            s1, s2 = source_sigs[n1], source_sigs[n2]
            if not s1 or not s2:
                matrix[n1][n2] = matrix[n2][n1] = 0.0
                continue

            # Amostrar min(5k, len) de cada
            k  = min(5_000, len(s1), len(s2))
            arr1 = np.array(random.sample(s1, k))
            arr2 = np.array(random.sample(s2, k))

            # Jaccard estimado: E[min_k == min_k] por hash position
            # Como são de fontes diferentes, usar média de matches dos mins
            if i == j:
                jacc = 1.0
            else:
                matches = np.mean(arr1.min(axis=0) == arr2.min(axis=0))
                jacc = float(matches)

            matrix[n1][n2] = round(jacc, 6)
            matrix[n2][n1] = round(jacc, 6)

    return {
        "method": "MinHash 13-grams, 128 hashes",
        "sample_per_source": sample_size,
        "jaccard_matrix": matrix,
    }


# ════════════════════════════════════════════════════════════════════════════
# BLOCO 9 — DETECÇÃO DE PII
# ════════════════════════════════════════════════════════════════════════════

# Padrões compilados
_RE_EMAIL    = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_RE_CPF_RAW  = re.compile(r"\b(\d{3}[.\-]?\d{3}[.\-]?\d{3}[.\-]?\d{2})\b")
_RE_PHONE_BR = re.compile(r"\(?\b([1-9][1-9])\)?\s*[\s\-]?\b(9\d{4}|\d{4})[\s\-]?\d{4}\b")
_RE_IP       = re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")


def _valid_cpf(cpf: str) -> bool:
    """Valida CPF pelos dígitos verificadores."""
    d = re.sub(r"\D", "", cpf)
    if len(d) != 11 or len(set(d)) == 1:
        return False
    for i in range(9, 11):
        s = sum(int(d[j]) * (i + 1 - j) for j in range(i))
        chk = (s * 10 % 11) % 10
        if chk != int(d[i]):
            return False
    return True


def pii_in_text(text: str) -> Dict[str, int]:
    """Conta instâncias de PII em um texto."""
    cpf_matches = [m for m in _RE_CPF_RAW.finditer(text) if _valid_cpf(m.group())]
    return {
        "emails":   len(_RE_EMAIL.findall(text)),
        "cpfs":     len(cpf_matches),
        "phones":   len(_RE_PHONE_BR.findall(text)),
        "ips":      len(_RE_IP.findall(text)),
    }


def compute_pii_metrics(
    source_name: str,
    source_dir: Path,
    sample_size: int = SAMPLE_SIZE_PII,
) -> Dict:
    """Bloco 9: Prevalência de PII."""
    logger.info(f"[PII] {source_name}")

    totals: Dict[str, int] = defaultdict(int)
    docs_with_any = 0
    n = 0

    for doc in tqdm(
        iter_parquet_texts(source_dir, columns=["text"], max_docs=sample_size),
        desc=f"  PII {source_name}", total=sample_size, leave=False
    ):
        text = doc.get("text", "")
        if not isinstance(text, str):
            continue
        counts = pii_in_text(text[:5000])
        n += 1
        for k, v in counts.items():
            totals[k] += v
        if any(counts.values()):
            docs_with_any += 1

    return {
        "sample_size":    n,
        "docs_with_pii":  docs_with_any,
        "docs_pii_pct":   round(100 * docs_with_any / max(n, 1), 3),
        "total_emails":   totals["emails"],
        "total_cpfs":     totals["cpfs"],
        "total_phones":   totals["phones"],
        "total_ips":      totals["ips"],
        "emails_per_1k":  round(1000 * totals["emails"] / max(n, 1), 3),
        "cpfs_per_1k":    round(1000 * totals["cpfs"] / max(n, 1), 3),
    }


# ════════════════════════════════════════════════════════════════════════════
# BLOCO 10 — CONTAMINAÇÃO DE BENCHMARK
# ════════════════════════════════════════════════════════════════════════════

def _ngram_set(text: str, n: int = 13) -> set:
    """Conjunto de n-grams de palavras de um texto."""
    words = text.lower().split()
    return {" ".join(words[i:i+n]) for i in range(max(1, len(words)-n+1))}


def compute_benchmark_contamination(
    source_name: str,
    source_dir: Path,
    n: int = 13,
    sample_size: int = 50_000,
) -> Dict:
    """
    Bloco 10: Overlap n-gram (13-gram) contra benchmarks PT-BR.
    Requer arquivos .txt em BENCHMARK_DIR.
    (Singh et al. 2024, ConTAM; Brown et al. 2020, GPT-3)
    """
    logger.info(f"[Contaminação] {source_name}")

    if not BENCHMARK_DIR.exists():
        return {"error": f"BENCHMARK_DIR não existe: {BENCHMARK_DIR}"}

    benchmark_files = list(BENCHMARK_DIR.glob("*.txt"))
    if not benchmark_files:
        return {"error": f"Nenhum arquivo .txt em {BENCHMARK_DIR}"}

    # Carregar n-grams de todos os benchmarks
    bench_ngrams: Dict[str, set] = {}
    for bf in benchmark_files:
        try:
            text = bf.read_text(encoding="utf-8", errors="ignore")
            bench_ngrams[bf.stem] = _ngram_set(text, n)
            logger.debug(f"  benchmark {bf.stem}: {len(bench_ngrams[bf.stem])} {n}-grams")
        except Exception as e:
            logger.warning(f"  Erro ao ler {bf}: {e}")

    if not bench_ngrams:
        return {"error": "nenhum benchmark carregado com sucesso"}

    # Verificar overlap por documento
    overlap_counts: Dict[str, int] = defaultdict(int)
    docs_with_any = 0
    n_docs = 0

    for doc in tqdm(
        iter_parquet_texts(source_dir, columns=["text"], max_docs=sample_size),
        desc=f"  contam {source_name}", total=sample_size, leave=False
    ):
        text = doc.get("text", "")
        if not isinstance(text, str) or len(text.split()) < n:
            continue
        doc_ngrams = _ngram_set(text, n)
        n_docs += 1
        found = False
        for bname, bset in bench_ngrams.items():
            hits = len(doc_ngrams & bset)
            if hits > 0:
                overlap_counts[bname] += 1
                found = True
        if found:
            docs_with_any += 1

    return {
        "n_gram":         n,
        "sample_size":    n_docs,
        "docs_with_any_overlap": docs_with_any,
        "contamination_rate_pct": round(100 * docs_with_any / max(n_docs, 1), 4),
        "per_benchmark":  {
            k: {
                "docs_with_overlap": v,
                "overlap_rate_pct":  round(100 * v / max(n_docs, 1), 4),
            }
            for k, v in sorted(overlap_counts.items())
        },
    }


# ════════════════════════════════════════════════════════════════════════════
# SCORE COMPOSTO E RELATÓRIO MARKDOWN
# ════════════════════════════════════════════════════════════════════════════

def generate_markdown_report(stats: Dict) -> str:
    """Gera relatório Markdown completo a partir do JSON de métricas."""
    lines = []
    lines.append("# Projeto Manacá LLM — Caracterização do Corpus Tier 1 PT-BR\n")
    lines.append(f"**Gerado em:** {datetime.now(timezone.utc).isoformat()}\n")
    lines.append("**Metodologia:** Rae et al. (2021), Wenzek et al. (2020), "
                 "McCarthy & Jarvis (2010), Covington & McFall (2010), "
                 "Petrov et al. (2023), Nayeem et al. (2025), Lee et al. (2022).\n")

    # Sumário geral
    lines.append("---\n## 1. Sumário Geral\n")
    total = stats.get("total", {})
    tok   = total.get("tokenization", {})
    lines.append("| Métrica | Valor |")
    lines.append("|---------|-------|")
    lines.append(f"| Total de documentos | {tok.get('total_docs', 0):,} |")
    lines.append(f"| Tamanho (bytes) | {tok.get('total_bytes', 0) / 1e9:.2f} GB |")
    lines.append(f"| Palavras (whitespace) | {tok.get('total_words', 0) / 1e9:.2f} B |")
    for tok_name in ["gpt2", "llama3", "gemma2"]:
        t = tok.get("tokenizers", {}).get(tok_name, {})
        if t:
            est = t.get("total_tokens_est", 0)
            fert = t.get("fertility_mean", 0)
            lines.append(f"| Tokens ({tok_name}) est. | {est / 1e9:.1f}B "
                         f"(fertility={fert:.3f}) |")
    lines.append("")

    # Por fonte
    lines.append("---\n## 2. Volume por Fonte\n")
    lines.append("| Fonte | Docs | GB | Tokens-Gemma2 est. | Fertility-Gemma2 |")
    lines.append("|-------|------|----|-------------------|-----------------|")
    for src_name in SOURCES:
        sd = stats.get("per_source", {}).get(src_name, {})
        tt = sd.get("tokenization", {})
        n_docs = tt.get("total_docs", 0)
        gb     = tt.get("total_bytes", 0) / 1e9
        g2     = tt.get("tokenizers", {}).get("gemma2", {})
        toks   = g2.get("total_tokens_est", 0)
        fert   = g2.get("fertility_mean", "-")
        lines.append(f"| {src_name} | {n_docs:,} | {gb:.1f} | "
                     f"{toks/1e9:.1f}B | {fert} |")
    lines.append("")

    # Qualidade Gopher resumida
    lines.append("---\n## 3. Sinais de Qualidade Gopher (médias por fonte)\n")
    gopher_keys = ["alpha_ratio", "mean_word_length", "symbol_word_ratio",
                   "stop_word_ratio", "dup_line_frac", "dup_para_frac"]
    header = "| Fonte | " + " | ".join(gopher_keys) + " |"
    sep    = "|-------|" + "|".join(["-------"] * len(gopher_keys)) + "|"
    lines.append(header)
    lines.append(sep)
    for src_name in SOURCES:
        sd = stats.get("per_source", {}).get(src_name, {})
        gd = sd.get("gopher", {})
        row = f"| {src_name} |"
        for k in gopher_keys:
            v = gd.get(k, {}).get("mean", "-")
            row += f" {v} |"
        lines.append(row)
    lines.append("")

    # Diversidade lexical
    lines.append("---\n## 4. Diversidade Lexical (médias por fonte)\n")
    lines.append("| Fonte | MATTR (w=500) | MTLD | Yule's K | Herdan's C |")
    lines.append("|-------|--------------|------|----------|------------|")
    for src_name in SOURCES:
        sd  = stats.get("per_source", {}).get(src_name, {})
        lex = sd.get("lexical_diversity", {})
        mattr  = lex.get("mattr_window500",       {}).get("mean", "-")
        mtld   = lex.get("mtld_bidirectional",    {}).get("mean", "-")
        yule   = lex.get("yule_k",                {}).get("mean", "-")
        herdan = lex.get("herdan_c",              {}).get("mean", "-")
        lines.append(f"| {src_name} | {mattr} | {mtld} | {yule} | {herdan} |")
    lines.append("")

    # PII
    lines.append("---\n## 5. PII por Fonte\n")
    lines.append("| Fonte | Docs com PII (%) | Emails/1k | CPFs/1k |")
    lines.append("|-------|-----------------|-----------|---------|")
    for src_name in SOURCES:
        sd  = stats.get("per_source", {}).get(src_name, {})
        pii = sd.get("pii", {})
        pct    = pii.get("docs_pii_pct", "-")
        emails = pii.get("emails_per_1k", "-")
        cpfs   = pii.get("cpfs_per_1k", "-")
        lines.append(f"| {src_name} | {pct} | {emails} | {cpfs} |")
    lines.append("")

    # Contaminação
    lines.append("---\n## 6. Contaminação de Benchmarks\n")
    lines.append("| Fonte | Taxa geral (%) | Benchmarks afetados |")
    lines.append("|-------|----------------|---------------------|")
    for src_name in SOURCES:
        sd  = stats.get("per_source", {}).get(src_name, {})
        cont = sd.get("contamination", {})
        rate   = cont.get("contamination_rate_pct", "-")
        benches = [k for k, v in cont.get("per_benchmark", {}).items()
                   if v.get("overlap_rate_pct", 0) > 0]
        lines.append(f"| {src_name} | {rate} | {', '.join(benches) or 'nenhum'} |")
    lines.append("")

    # Referências
    lines.append("---\n## 7. Referências Metodológicas\n")
    refs = [
        "Rae et al. (2021). Scaling Language Models: Gopher. arXiv:2112.11446",
        "Wenzek et al. (2020). CCNet. LREC 2020. arXiv:1911.00359",
        "Lee et al. (2022). Deduplicating Training Data. ACL 2022. arXiv:2107.06499",
        "McCarthy & Jarvis (2010). MTLD. Behavior Research Methods 42:381–392",
        "Covington & McFall (2010). MATTR. JQL 17:94–100",
        "Petrov et al. (2023). Tokenizers Introduce Unfairness. NeurIPS. arXiv:2305.15425",
        "Nayeem et al. (2025). Beyond Fertility (STRR). arXiv:2510.09947",
        "Singh et al. (2024). ConTAM. arXiv:2411.03923",
        "Soldaini et al. (2024). Dolma. ACL 2024. arXiv:2402.00159",
        "Penedo et al. (2024). FineWeb. NeurIPS D&B. arXiv:2406.17557",
        "Correa et al. (2024). Tucano. arXiv:2411.07854",
    ]
    for r in refs:
        lines.append(f"- {r}")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ════════════════════════════════════════════════════════════════════════════

def run_characterization(skip_existing: bool = True) -> Dict:
    """
    Pipeline completo de caracterização.
    Executa todos os blocos por fonte e agrega totais.
    """
    logger.info("=" * 70)
    logger.info("Manacá LLM — Caracterização do Corpus Tier 1")
    logger.info("=" * 70)

    all_stats: Dict[str, Any] = {
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "work_dir":       str(WORK_DIR),
        "sources_scanned": list(SOURCES.keys()),
        "per_source":     {},
    }

    # ── Por fonte ────────────────────────────────────────────────────────
    for src_name, src_dir in SOURCES.items():
        if not src_dir.exists():
            logger.warning(f"Diretório não existe: {src_dir} — pulando {src_name}")
            continue

        checkpoint = PER_SOURCE_DIR / f"{src_name}.json"
        if skip_existing and checkpoint.exists():
            logger.info(f"[checkpoint] {src_name} — carregando cache")
            with open(checkpoint) as f:
                all_stats["per_source"][src_name] = json.load(f)
            continue

        logger.info(f"\n{'─'*60}")
        logger.info(f"Processando fonte: {src_name}")
        logger.info(f"{'─'*60}")
        src_stats: Dict[str, Any] = {"source": src_name, "dir": str(src_dir)}

        t0 = time.time()

        src_stats["tokenization"]        = compute_tokenization_metrics(src_name, src_dir)
        src_stats["length_distribution"] = compute_length_distribution(src_name, src_dir)
        src_stats["gopher"]              = compute_gopher_metrics(src_name, src_dir)
        src_stats["kenlm_perplexity"]    = compute_kenlm_perplexity(src_name, src_dir)
        src_stats["lexical_diversity"]   = compute_lexical_diversity(src_name, src_dir)
        src_stats["temporal_coverage"]   = compute_temporal_coverage(src_name, src_dir)
        src_stats["domain_distribution"] = compute_domain_distribution(src_name, src_dir)
        src_stats["pii"]                 = compute_pii_metrics(src_name, src_dir)
        src_stats["contamination"]       = compute_benchmark_contamination(src_name, src_dir)

        elapsed = time.time() - t0
        src_stats["processing_time_s"] = round(elapsed, 1)
        logger.info(f"  {src_name} concluído em {elapsed:.0f}s")

        # Salvar checkpoint
        with open(checkpoint, "w") as f:
            json.dump(src_stats, f, indent=2, ensure_ascii=False)

        all_stats["per_source"][src_name] = src_stats

    # ── MinHash inter-fontes (uma única passagem global) ─────────────────
    minhash_ckpt = STATS_DIR / "minhash_matrix.json"
    if skip_existing and minhash_ckpt.exists():
        logger.info("[checkpoint] MinHash — carregando cache")
        with open(minhash_ckpt) as f:
            all_stats["minhash_inter_source"] = json.load(f)
    else:
        all_stats["minhash_inter_source"] = compute_minhash_overlap()
        with open(minhash_ckpt, "w") as f:
            json.dump(all_stats["minhash_inter_source"], f, indent=2)

    # ── Totais agregados ─────────────────────────────────────────────────
    logger.info("\nAgregando totais...")
    total_docs  = sum(s.get("tokenization", {}).get("total_docs", 0)
                      for s in all_stats["per_source"].values())
    total_bytes = sum(s.get("tokenization", {}).get("total_bytes", 0)
                      for s in all_stats["per_source"].values())
    total_words = sum(s.get("tokenization", {}).get("total_words", 0)
                      for s in all_stats["per_source"].values())

    tok_totals: Dict[str, Dict] = {}
    for tok_name in TOKENIZER_IDS:
        est_tokens = sum(
            s.get("tokenization", {}).get("tokenizers", {}).get(tok_name, {}).get("total_tokens_est", 0)
            for s in all_stats["per_source"].values()
        )
        tok_totals[tok_name] = {
            "total_tokens_est": est_tokens,
            "total_tokens_est_billions": round(est_tokens / 1e9, 2),
        }

    all_stats["total"] = {
        "tokenization": {
            "total_docs":    total_docs,
            "total_bytes":   total_bytes,
            "total_bytes_gb": round(total_bytes / 1e9, 2),
            "total_words":   total_words,
            "tokenizers":    tok_totals,
        }
    }

    # ── Salvar JSON e Markdown ───────────────────────────────────────────
    json_path = STATS_DIR / "corpus_characterization.json"
    md_path   = STATS_DIR / "corpus_characterization.md"

    with open(json_path, "w") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    logger.info(f"JSON salvo: {json_path}")

    md_content = generate_markdown_report(all_stats)
    with open(md_path, "w") as f:
        f.write(md_content)
    logger.info(f"Markdown salvo: {md_path}")

    # ── Sumário no terminal ──────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("CARACTERIZAÇÃO CONCLUÍDA")
    logger.info("=" * 70)
    logger.info(f"  Total documentos:  {total_docs:,}")
    logger.info(f"  Total bytes:       {total_bytes/1e9:.2f} GB")
    logger.info(f"  Total palavras:    {total_words/1e9:.2f}B")
    for tok_name, tv in tok_totals.items():
        logger.info(f"  Tokens ({tok_name}):  {tv['total_tokens_est_billions']}B")
    logger.info(f"\n  JSON:     {json_path}")
    logger.info(f"  Markdown: {md_path}")
    logger.info("=" * 70)

    return all_stats


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Manacá LLM — Caracterização Estatística do Corpus Tier 1 PT-BR"
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Ignorar checkpoints e reprocessar tudo do zero"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Processar apenas uma fonte (ex: --source gigaverbo)"
    )
    args = parser.parse_args()

    if args.source:
        # Modo single-source para debugging
        if args.source not in SOURCES:
            logger.error(f"Fonte desconhecida: {args.source}. Disponíveis: {list(SOURCES.keys())}")
            sys.exit(1)
        src_name = args.source
        src_dir  = SOURCES[src_name]
        logger.info(f"Processando apenas: {src_name}")
        result = {
            "tokenization":        compute_tokenization_metrics(src_name, src_dir),
            "length_distribution": compute_length_distribution(src_name, src_dir),
            "gopher":              compute_gopher_metrics(src_name, src_dir),
            "kenlm_perplexity":    compute_kenlm_perplexity(src_name, src_dir),
            "lexical_diversity":   compute_lexical_diversity(src_name, src_dir),
            "pii":                 compute_pii_metrics(src_name, src_dir),
        }
        out = PER_SOURCE_DIR / f"{src_name}_debug.json"
        with open(out, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Resultado salvo em {out}")
    else:
        run_characterization(skip_existing=not args.no_cache)

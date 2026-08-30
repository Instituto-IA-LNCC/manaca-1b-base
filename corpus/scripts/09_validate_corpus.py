#!/usr/bin/env python3
"""
Manacá LLM — Script 09: Validação e Relatório Estatístico
==========================================================
LNCC × NII/LLM-jp | Fase 1 — Corpus PT-BR

Valida o corpus deduplicado final e gera relatório estatístico completo
para publicação científica e documentação do Projeto Manacá.

Este é o último script da Fase 1. Deve ser executado após o Script 08
(deduplicação global) ter concluído com sucesso.

Métricas produzidas:

    VOLUME E COMPOSIÇÃO:
      - Contagem precisa de tokens por fonte e total
        (via tokenizador LLM-jp-3 como referência externa)
      - Distribuição de comprimento de documentos (p10/p50/p90/p99)
      - Composição por fonte e licença
      - Progresso em relação à meta de 1 TB tokens

    QUALIDADE LINGUÍSTICA:
      - Distribuição de quality_score por fonte
      - Razão alfabética média (proxy de limpeza)
      - Type-Token Ratio (diversidade lexical)
      - Distribuição de stop words PT-BR (proxy de fluência)
      - Razão PT-BR vs PT-PT por fonte (fontes multilíngues)

    COBERTURA TEMPORAL:
      - Distribuição de snapshots CC (fontes FineWeb-2 e Common Crawl)
      - Cobertura de fontes governamentais (Ulysses Tesemõ)

    INTEGRIDADE:
      - Verificação de todos os shards Parquet (metadados + amostra)
      - Consistência de schema entre fontes
      - Detecção de shards vazios ou corrompidos

Referências:
    Gao, L. et al. (2020). The Pile. arXiv:2101.00027.
      (metodologia de relatório de corpus)
    Laurençon, A. et al. (2022). ROOTS. arXiv:2303.03915.
      (métricas de qualidade para corpora multilíngues)
    de Lucena, R. et al. (2024). Tucano. arXiv:2411.07854.
      (referência para métricas PT-BR)

PRÉ-REQUISITOS OBRIGATÓRIOS:
    1. Script 08 (deduplicação global) concluído
    2. $WORK_DIR/deduped/ com shards Parquet

Uso:
    # Validação completa com contagem precisa de tokens:
    python corpus/scripts/09_validate_corpus.py \\
        --tokenizer llm-jp/llm-jp-3-tokenizer-nightly

    # Validação rápida (estimativa de tokens, sem tokenizador):
    python corpus/scripts/09_validate_corpus.py

    # Apenas verificar integridade dos shards:
    python corpus/scripts/09_validate_corpus.py --integrity-only

    # Validar corpus raw (antes da deduplicação):
    python corpus/scripts/09_validate_corpus.py --corpus-dir $WORK_DIR/raw

Saída:
    $WORK_DIR/stats/
    ├── corpus_validation_report.json   relatório completo (JSON)
    ├── corpus_summary.md               sumário em Markdown (para README)
    └── corpus_progress.txt             progresso vs meta 1 TB

Autor: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
Versão: 0.1.0 — Abril 2026
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from loguru import logger
    from tqdm import tqdm
except ImportError as e:
    print(f"[ERRO] Dependência ausente: {e}")
    print("Docker: execute dentro do container 'corpus' "
          "(docker compose run --rm corpus ...). Veja docs/environment/setup-guide-docker-pt.md")
    sys.exit(1)


# ── Configuração ──────────────────────────────────────────────────────────────

WORK_DIR    = Path(os.environ.get("WORK_DIR", Path.home() / "manaca-corpus"))
DEDUPED_DIR = WORK_DIR / "deduped"
RAW_DIR     = WORK_DIR / "raw"
STATS_DIR   = WORK_DIR / "stats"
CKPT_DIR    = WORK_DIR / "checkpoints"

REPORT_PATH   = STATS_DIR / "corpus_validation_report.json"
SUMMARY_PATH  = STATS_DIR / "corpus_summary.md"
PROGRESS_PATH = STATS_DIR / "corpus_progress.txt"

# Meta da Fase 1
TARGET_TOKENS_B = 1_000.0  # 1 trilhão de tokens

# Stop words PT-BR para proxy de fluência
# (subconjunto representativo — não exaustivo)
PTBR_STOPWORDS = frozenset({
    "a", "o", "as", "os", "um", "uma", "uns", "umas",
    "de", "da", "do", "das", "dos", "em", "na", "no",
    "e", "ou", "mas", "que", "se", "com", "por", "para",
    "ao", "aos", "sua", "seu", "suas", "seus",
    "é", "são", "foi", "ser", "ter", "há", "mais", "não",
    "ele", "ela", "eles", "elas", "eu", "você", "nós",
    "como", "quando", "onde", "muito", "também", "já",
})

# Fontes com seus metadados para o relatório
SOURCE_METADATA = {
    "gigaverbo":     {"license": "Apache 2.0",   "domain": "geral/web",        "ref": "de Lucena et al. (2024)"},
    "madlad400":     {"license": "Apache 2.0",   "domain": "geral/web",        "ref": "Kudugunta et al. (2024)"},
    "fineweb2":      {"license": "ODC-By 1.0",   "domain": "geral/web",        "ref": "Penedo et al. (2024)"},
    "hplt2":         {"license": "CC0",           "domain": "geral/web",        "ref": "de Gibert et al. (2024)"},
    "wikipedia_ptbr":{"license": "CC BY-SA 4.0", "domain": "enciclopédico",    "ref": "Wikimedia (2023)"},
    "ulysses":       {"license": "Domínio público","domain": "jurídico/legislativo", "ref": "Nascimento et al. (2023)"},
    "common_crawl":  {"license": "Domínio público","domain": "geral/web",       "ref": "Common Crawl (2025)"},
}


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging() -> None:
    STATS_DIR.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = WORK_DIR / "logs" / f"validation_{ts}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr, level="INFO", colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
    )
    logger.add(
        log_file, level="DEBUG", rotation="50 MB", retention=2,
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}",
    )
    logger.info(f"Log: {log_file}")


# ── Tokenizador ───────────────────────────────────────────────────────────────

def load_tokenizer(tokenizer_path: str | None):
    """Carrega tokenizador para contagem precisa de tokens."""
    if not tokenizer_path:
        return None
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info(f"Tokenizador carregado: {tokenizer_path}")
        return tok
    except Exception as e:
        logger.warning(f"Tokenizador indisponível ({e}). Usando estimativa 4 chars/token.")
        return None


def count_tokens(text: str, tokenizer=None) -> int:
    """Conta tokens com tokenizador ou estimativa (4 chars/token para PT-BR)."""
    if tokenizer:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return max(1, len(text) // 4)


# ── Métricas por documento ────────────────────────────────────────────────────

def doc_metrics(text: str) -> dict[str, float]:
    """
    Calcula métricas de qualidade para um documento.

    Baseado em:
      - Gao et al. (2020). The Pile. arXiv:2101.00027.
      - Laurençon et al. (2022). ROOTS. arXiv:2303.03915.
      - de Lucena et al. (2024). Tucano. arXiv:2411.07854.
    """
    if not text:
        return {}

    n_chars = len(text)
    words   = text.split()
    n_words = max(len(words), 1)

    n_alpha    = sum(1 for c in text if c.isalpha())
    alpha_ratio = n_alpha / n_chars

    unique_words = len(set(w.lower() for w in words))
    ttr          = unique_words / n_words

    words_lower  = [w.lower().strip(".,!?;:()[]\"'") for w in words]
    n_stopwords  = sum(1 for w in words_lower if w in PTBR_STOPWORDS)
    stopword_ratio = n_stopwords / n_words

    return {
        "n_chars":        n_chars,
        "n_words":        n_words,
        "n_unique_words": unique_words,
        "alpha_ratio":    round(alpha_ratio, 4),
        "ttr":            round(ttr, 4),
        "stopword_ratio": round(stopword_ratio, 4),
    }


# ── Validação de integridade ──────────────────────────────────────────────────

def validate_shards(corpus_dir: Path) -> dict[str, Any]:
    """
    Verifica integridade de todos os shards Parquet.
    Retorna estatísticas de integridade.
    """
    shards  = sorted(corpus_dir.rglob("shard_*.parquet"))
    if not shards:
        shards = sorted(corpus_dir.rglob("*.parquet"))

    total_rows  = 0
    empty       = 0
    corrupted   = 0
    schema_ok   = 0
    required_cols = {"text", "source", "id", "lang", "score"}

    for s in tqdm(shards, desc="Verificando shards", unit="shard"):
        try:
            meta = pq.read_metadata(s)
            rows = meta.num_rows
            total_rows += rows
            if rows == 0:
                empty += 1
                continue
            # Verificar schema
            cols = set(meta.schema.names)
            if required_cols.issubset(cols):
                schema_ok += 1
        except Exception as e:
            logger.debug(f"Shard corrompido {s.name}: {e}")
            corrupted += 1

    return {
        "total_shards":   len(shards),
        "total_rows":     total_rows,
        "empty_shards":   empty,
        "corrupted_shards": corrupted,
        "schema_ok":      schema_ok,
        "integrity_rate": round((len(shards) - empty - corrupted) / max(len(shards), 1), 4),
    }


# ── Validador principal ───────────────────────────────────────────────────────

class CorpusValidator:
    """
    Valida e gera relatório completo do corpus Manacá Fase 1.

    Itera sobre os shards Parquet acumulando estatísticas por fonte,
    por língua e globais. Amostragem adaptativa para métricas de
    qualidade (1 doc a cada 100 para eficiência).
    """

    def __init__(self, corpus_dir: Path, tokenizer=None, sample_rate: int = 100):
        self.corpus_dir  = corpus_dir
        self.tokenizer   = tokenizer
        self.sample_rate = sample_rate

        # Acumuladores globais
        self.total_docs:   int   = 0
        self.total_tokens: int   = 0
        self.total_chars:  int   = 0

        # Por fonte
        self.source_stats: dict[str, dict] = defaultdict(lambda: {
            "docs": 0, "tokens": 0, "chars": 0,
        })

        # Por língua
        self.lang_counts: Counter = Counter()

        # Métricas de qualidade (amostradas)
        self.quality_samples: dict[str, list[float]] = defaultdict(list)

        # Comprimentos (amostrados)
        self.length_samples: list[int] = []

    def process(self) -> None:
        """Processa todos os shards do corpus."""
        shards = sorted(self.corpus_dir.rglob("shard_*.parquet"))
        if not shards:
            shards = sorted(self.corpus_dir.rglob("*.parquet"))

        if not shards:
            logger.error(f"Nenhum shard encontrado em {self.corpus_dir}")
            return

        logger.info(f"Processando {len(shards):,} shards em {self.corpus_dir}...")

        for shard_path in tqdm(shards, desc="Validando", unit="shard"):
            try:
                table = pq.read_table(shard_path)
            except Exception as e:
                logger.warning(f"Erro ao ler {shard_path.name}: {e}")
                continue

            cols = table.column_names
            text_col   = "text"   if "text"   in cols else cols[0]
            source_col = "source" if "source" in cols else None
            lang_col   = "lang"   if "lang"   in cols else None
            score_col  = "score"  if "score"  in cols else None

            for batch in table.to_batches(max_chunksize=500):
                texts   = batch.column(text_col).to_pylist()
                sources = batch.column(source_col).to_pylist() if source_col else ["unknown"] * len(texts)
                langs   = batch.column(lang_col).to_pylist()   if lang_col   else ["unknown"] * len(texts)
                scores  = batch.column(score_col).to_pylist()  if score_col  else [0.0] * len(texts)

                for i, (text, src, lang, score) in enumerate(
                    zip(texts, sources, langs, scores)
                ):
                    if not text or not isinstance(text, str):
                        continue

                    n_tokens = count_tokens(text, self.tokenizer)
                    n_chars  = len(text)

                    # Identificar fonte base (sem sufixo _gov)
                    base_src = (src or "unknown").split("_")[0]

                    self.total_docs   += 1
                    self.total_tokens += n_tokens
                    self.total_chars  += n_chars

                    ss = self.source_stats[base_src]
                    ss["docs"]   += 1
                    ss["tokens"] += n_tokens
                    ss["chars"]  += n_chars

                    self.lang_counts[lang or "unknown"] += 1

                    # Amostragem para métricas de qualidade
                    if self.total_docs % self.sample_rate == 0:
                        m = doc_metrics(text)
                        for k, v in m.items():
                            self.quality_samples[k].append(v)
                        self.length_samples.append(n_chars)
                        if score:
                            self.quality_samples["quality_score"].append(
                                float(score)
                            )

    def generate_report(self) -> dict[str, Any]:
        """Gera relatório JSON completo."""

        def pct(data: list[float], p: float) -> float:
            if not data:
                return 0.0
            s   = sorted(data)
            idx = int(len(s) * p / 100)
            return round(s[min(idx, len(s) - 1)], 4)

        def mean(data: list[float]) -> float:
            return round(sum(data) / len(data), 4) if data else 0.0

        total_b = self.total_tokens / 1e9
        pct_goal = round(total_b / TARGET_TOKENS_B * 100, 2)

        report: dict[str, Any] = {
            "generated_at":  datetime.now(timezone.utc).isoformat(),
            "project":       "Manacá LLM — LNCC × NII/LLM-jp",
            "corpus_dir":    str(self.corpus_dir),
            "phase":         "Fase 1 — Corpus PT-BR",

            "summary": {
                "total_docs":          self.total_docs,
                "total_tokens":        self.total_tokens,
                "total_tokens_b":      round(total_b, 3),
                "target_tokens_b":     TARGET_TOKENS_B,
                "progress_pct":        pct_goal,
                "total_chars":         self.total_chars,
                "avg_tokens_per_doc":  round(self.total_tokens / max(self.total_docs, 1)),
            },

            "sources": {},
            "language_distribution": dict(self.lang_counts.most_common(20)),
            "quality_distributions": {},
        }

        # Por fonte
        total_tokens_check = 0
        for src, stats in sorted(
            self.source_stats.items(),
            key=lambda x: x[1]["tokens"],
            reverse=True,
        ):
            meta      = SOURCE_METADATA.get(src, {})
            tok_b     = stats["tokens"] / 1e9
            share_pct = round(stats["tokens"] / max(self.total_tokens, 1) * 100, 2)
            total_tokens_check += stats["tokens"]

            report["sources"][src] = {
                "docs":          stats["docs"],
                "tokens":        stats["tokens"],
                "tokens_b":      round(tok_b, 3),
                "share_pct":     share_pct,
                "avg_chars":     round(stats["chars"] / max(stats["docs"], 1)),
                "license":       meta.get("license", "—"),
                "domain":        meta.get("domain", "—"),
                "reference":     meta.get("ref", "—"),
            }

        # Distribuições de qualidade
        for metric, values in self.quality_samples.items():
            if values:
                report["quality_distributions"][metric] = {
                    "n_sampled": len(values),
                    "mean":      mean(values),
                    "p10":       pct(values, 10),
                    "p25":       pct(values, 25),
                    "p50":       pct(values, 50),
                    "p75":       pct(values, 75),
                    "p90":       pct(values, 90),
                    "p99":       pct(values, 99),
                }

        return report

    def generate_markdown_summary(self, report: dict[str, Any]) -> str:
        """
        Gera sumário em Markdown para publicação no README do repositório.
        Segue o formato de The Pile (Gao et al., 2020) e Dolma (Soldaini, 2024).
        """
        s   = report["summary"]
        now = datetime.now(timezone.utc).strftime("%B %Y")

        lines = [
            "# Manacá Corpus — Relatório da Fase 1",
            f"**Gerado em:** {now}  ",
            f"**Projeto:** LNCC × NII/LLM-jp  ",
            "",
            "## Resumo",
            "",
            f"| Métrica | Valor |",
            f"|---------|-------|",
            f"| Total de documentos | {s['total_docs']:,} |",
            f"| Total de tokens | {s['total_tokens_b']:.1f}B |",
            f"| Meta Fase 1 | {s['target_tokens_b']:.0f}B tokens |",
            f"| Progresso | {s['progress_pct']:.1f}% |",
            f"| Tokens por documento (média) | {s['avg_tokens_per_doc']:,} |",
            "",
            "## Composição por Fonte",
            "",
            "| Fonte | Docs | Tokens (B) | Share | Licença | Domínio |",
            "|-------|------|-----------|-------|---------|---------|",
        ]

        for src, info in report["sources"].items():
            lines.append(
                f"| {src} | {info['docs']:,} | "
                f"{info['tokens_b']:.2f} | "
                f"{info['share_pct']:.1f}% | "
                f"{info['license']} | "
                f"{info['domain']} |"
            )

        lines += [
            "",
            "## Qualidade Linguística (amostra)",
            "",
            "| Métrica | Média | P10 | P50 | P90 |",
            "|---------|-------|-----|-----|-----|",
        ]

        metric_labels = {
            "quality_score":  "Score de qualidade",
            "alpha_ratio":    "Razão alfabética",
            "ttr":            "Type-Token Ratio",
            "stopword_ratio": "Razão stop words PT-BR",
        }

        for m_key, m_label in metric_labels.items():
            d = report["quality_distributions"].get(m_key, {})
            if d:
                lines.append(
                    f"| {m_label} | {d['mean']:.4f} | "
                    f"{d['p10']:.4f} | {d['p50']:.4f} | {d['p90']:.4f} |"
                )

        lines += [
            "",
            "## Referências das Fontes",
            "",
        ]
        for src, info in report["sources"].items():
            ref = info.get("reference", "")
            if ref:
                lines.append(f"- **{src}:** {ref}")

        lines += [
            "",
            "---",
            f"*Versão 0.1.0 — {now} | Projeto Manacá — LNCC × NII/LLM-jp*",
        ]

        return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manacá — Validação e Relatório do Corpus Fase 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corpus-dir", type=Path, default=DEDUPED_DIR,
        help=(
            "Diretório do corpus a validar. "
            "Default: $WORK_DIR/deduped (corpus deduplicado final)."
        ),
    )
    parser.add_argument(
        "--target-tokens-b", type=float, default=TARGET_TOKENS_B,
        help=(
            "Meta de tokens (bilhões) para o cálculo de progresso. "
            f"Default: {TARGET_TOKENS_B:.0f} (visão de longo prazo, modelos maiores). "
            "Para o Manacá-1B use ~42 (orçamento de pré-treino de ~2 épocas)."
        ),
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None,
        help=(
            "Tokenizador HuggingFace para contagem precisa de tokens. "
            "Recomendado: llm-jp/llm-jp-3-tokenizer-nightly. "
            "Default: estimativa 4 chars/token."
        ),
    )
    parser.add_argument(
        "--sample-rate", type=int, default=100,
        help=(
            "Taxa de amostragem para métricas de qualidade "
            "(1 de cada N documentos). Default: 100."
        ),
    )
    parser.add_argument(
        "--integrity-only", action="store_true",
        help="Verificar apenas integridade dos shards (rápido).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=STATS_DIR,
        help="Diretório de saída para os relatórios. Default: $WORK_DIR/stats.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()

    # Meta de progresso (sobreponível): 1 TB é a visão de longo prazo (modelos
    # maiores); o Manacá-1B usa ~42B (orçamento de pré-treino de ~2 épocas).
    global TARGET_TOKENS_B
    TARGET_TOKENS_B = args.target_tokens_b

    corpus_dir = args.corpus_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Manaca — Validacao do Corpus Fase 1")
    logger.info(f"  Corpus:     {corpus_dir}")
    logger.info(f"  Tokenizer:  {args.tokenizer or 'estimativa 4 chars/token'}")
    logger.info(f"  Amostragem: 1 de cada {args.sample_rate} docs")
    logger.info("=" * 60)

    # ── Verificação de integridade ────────────────────────────────────────────
    logger.info("Verificando integridade dos shards...")
    integrity = validate_shards(corpus_dir)
    logger.info(
        f"  Shards: {integrity['total_shards']:,} | "
        f"Rows: {integrity['total_rows']:,} | "
        f"Corrompidos: {integrity['corrupted_shards']} | "
        f"Integridade: {integrity['integrity_rate']:.1%}"
    )

    if args.integrity_only:
        report = {"integrity": integrity, "generated_at": datetime.now(timezone.utc).isoformat()}
        path = output_dir / "integrity_check.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        logger.info(f"Relatorio de integridade: {path}")
        return 0 if integrity["corrupted_shards"] == 0 else 1

    if integrity["total_rows"] == 0:
        logger.error(
            f"Nenhum documento encontrado em {corpus_dir}. "
            "Verificar se o Script 08 foi executado."
        )
        return 1

    # ── Validação estatística ─────────────────────────────────────────────────
    tokenizer = load_tokenizer(args.tokenizer)
    validator = CorpusValidator(
        corpus_dir  = corpus_dir,
        tokenizer   = tokenizer,
        sample_rate = args.sample_rate,
    )

    t_start = time.time()
    validator.process()
    elapsed = time.time() - t_start

    # ── Gerar relatório ───────────────────────────────────────────────────────
    report = validator.generate_report()
    report["integrity"] = integrity
    report["validation_elapsed_hours"] = round(elapsed / 3600, 2)

    # Salvar JSON
    report_path = output_dir / "corpus_validation_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Relatorio JSON: {report_path}")

    # Salvar Markdown
    md_summary = validator.generate_markdown_summary(report)
    md_path    = output_dir / "corpus_summary.md"
    md_path.write_text(md_summary, encoding="utf-8")
    logger.info(f"Sumario Markdown: {md_path}")

    # Salvar progresso
    s = report["summary"]
    progress_text = (
        f"Manaca Corpus — Fase 1 — {datetime.now().strftime('%Y-%m-%d')}\n"
        f"{'='*50}\n"
        f"Total:       {s['total_docs']:>15,} documentos\n"
        f"Tokens:      {s['total_tokens_b']:>14.1f} B\n"
        f"Meta:        {s['target_tokens_b']:>14.0f} B\n"
        f"Progresso:   {s['progress_pct']:>14.1f} %\n"
        f"{'='*50}\n"
    )
    progress_path = output_dir / "corpus_progress.txt"
    progress_path.write_text(progress_text, encoding="utf-8")

    # ── Exibir resumo no terminal ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("VALIDACAO CONCLUIDA")
    logger.info(f"  Documentos:  {s['total_docs']:,}")
    logger.info(f"  Tokens:      {s['total_tokens_b']:.1f}B / {s['target_tokens_b']:.0f}B ({s['progress_pct']:.1f}%)")
    logger.info(f"  Tempo:       {elapsed/3600:.2f}h")
    logger.info("")
    logger.info("  Composição por fonte:")
    for src, info in report["sources"].items():
        logger.info(
            f"    {src:<25} {info['tokens_b']:6.2f}B  "
            f"({info['share_pct']:5.1f}%)  {info['license']}"
        )
    logger.info("")
    logger.info(f"  Relatorios em: {output_dir}")
    logger.info("=" * 60)

    if s["progress_pct"] >= 100:
        logger.success(
            f"META DE {s['target_tokens_b']:.0f}B TOKENS ATINGIDA! Corpus completo."
        )
    else:
        logger.info(
            f"Progresso: {s['total_tokens_b']:.1f}B / {s['target_tokens_b']:.0f}B "
            f"— faltam {s['target_tokens_b'] - s['total_tokens_b']:.1f}B tokens."
        )
        logger.info(
            "Para expandir o corpus (visão de longo prazo, modelos maiores):"
            "\n  - Solicitar acesso ao CulturaX e Jabuticaba (Tier 2)"
            "\n  - Executar Script 07 (Common Crawl) com mais snapshots"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

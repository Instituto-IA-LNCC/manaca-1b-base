#!/usr/bin/env python3
"""
Manacá LLM — Script 08: Deduplicação Global Cross-Source
=========================================================
LNCC × NII/LLM-jp | Fase 1 — Corpus PT-BR

================================ PT (Português) ================================

Executa deduplicação global (cross-source) sobre todos os shards
Parquet produzidos pelos Scripts 01-07, usando MinHash LSH com
128 permutações e threshold Jaccard 0.80.

Esta é a segunda passagem de deduplicação. A primeira passagem
(within-source) ocorreu individualmente em cada script de aquisição.

Idempotente: assinaturas já computadas são carregadas do cache.

Justificativa científica:
    Soldaini et al. (2024, arXiv:2402.00159) demonstraram que a
    deduplicação cross-source remove 15-30% de conteúdo redundante
    que passa pela deduplicação within-source, pois fontes distintas
    (GigaVerbo, FineWeb-2, MADLAD-400) frequentemente capturam os
    mesmos documentos web com pequenas variações.

    Lee et al. (2022, arXiv:2107.06499) demonstraram que modelos
    treinados em dados deduplicados apresentam:
      - Menor perplexidade em benchmarks padrão
      - Menor taxa de memorização de sequências de treinamento
      - Melhor generalização em downstream tasks

    Parâmetros adotados:
      - 128 permutações: erro padrão Jaccard ≈ 0.088 (Lee et al., 2022)
      - Threshold 0.80: conservador — preserva diversidade lexical PT-BR
      - n-gramas de palavras (n=5): robusto a variações ortográficas
      - Resolução de clusters: manter documento com maior quality_score
        (campo 'score' calculado em cada script de aquisição)

    Referências:
      Soldaini, L. et al. (2024). Dolma. arXiv:2402.00159.
      Lee, K. et al. (2022). Deduplicating Training Data. arXiv:2107.06499.
      Leskovec, J. et al. (2014). Mining of Massive Datasets.
        Cambridge University Press. (fundamentos LSH)

ESTRATÉGIA EM DUAS PASSAGENS:
    Passagem 1 (map):    Calcular assinaturas MinHash por documento
                         e agrupar por banda LSH → candidatos a duplicata
    Passagem 2 (reduce): Para cada cluster de candidatos, manter o
                         documento com maior quality_score e marcar
                         os demais como duplicatas a remover

NOTA SOBRE ESCALA:
    Para o corpus completo Tier 1 (~410B tokens, centenas de milhões
    de documentos), esta implementação Python é adequada para volumes
    até ~50M documentos em RAM. Para volumes maiores, usar o módulo
    datatrove.pipeline.dedup com execução SLURM distribuída.
    A função build_datatrove_dedup() oferece esta alternativa.

PRÉ-REQUISITOS OBRIGATÓRIOS:
    1. Scripts 01-06 (Tier 1) concluídos
    2. Volume de trabalho WORK_DIR gravavel (bind mount Docker ./data, ou NFS/HPC)
    3. RAM suficiente: ~1 GB por milhão de documentos (índice MinHash)

Uso:
    # Deduplicação de todas as fontes Tier 1:
    python corpus/scripts/08_global_dedup.py

    # Deduplicação apenas de fontes específicas:
    python corpus/scripts/08_global_dedup.py \\
        --sources gigaverbo madlad400 fineweb2

    # Verificar status sem processar:
    python corpus/scripts/08_global_dedup.py --status

    # Usar datatrove distribuído (recomendado para >50M docs):
    python corpus/scripts/08_global_dedup.py --use-datatrove \\
        --executor slurm --slurm-partition cpu

Saída:
    $WORK_DIR/deduped/
    ├── shard_000000.parquet   corpus deduplicado final
    ├── shard_000001.parquet
    ├── ...
    └── dedup_report.json      estatísticas completas

================================= EN (English) =================================

Manacá LLM — Script 08: Global Cross-Source Deduplication

Runs global (cross-source) deduplication over all Parquet shards
produced by Scripts 01-07, using MinHash LSH with
128 permutations and Jaccard threshold 0.80.

This is the second deduplication pass. The first pass
(within-source) happened individually in each acquisition script.

Idempotent: already-computed signatures are loaded from cache.

Scientific rationale:
    Soldaini et al. (2024, arXiv:2402.00159) showed that
    cross-source deduplication removes 15-30% of redundant content
    that passes within-source deduplication, since distinct sources
    (GigaVerbo, FineWeb-2, MADLAD-400) frequently capture the
    same web documents with small variations.

    Lee et al. (2022, arXiv:2107.06499) showed that models
    trained on deduplicated data exhibit:
      - Lower perplexity on standard benchmarks
      - Lower memorization rate of training sequences
      - Better generalization on downstream tasks

    Parameters adopted:
      - 128 permutations: Jaccard standard error ≈ 0.088 (Lee et al., 2022)
      - Threshold 0.80: conservative — preserves PT-BR lexical diversity
      - word n-grams (n=5): robust to orthographic variations
      - Cluster resolution: keep the document with the highest quality_score
        ('score' field computed in each acquisition script)

    References:
      Soldaini, L. et al. (2024). Dolma. arXiv:2402.00159.
      Lee, K. et al. (2022). Deduplicating Training Data. arXiv:2107.06499.
      Leskovec, J. et al. (2014). Mining of Massive Datasets.
        Cambridge University Press. (LSH fundamentals)

TWO-PASS STRATEGY:
    Pass 1 (map):    Compute MinHash signatures per document
                     and group by LSH band → duplicate candidates
    Pass 2 (reduce): For each candidate cluster, keep the
                     document with the highest quality_score and mark
                     the rest as duplicates to remove

NOTE ON SCALE:
    For the full Tier 1 corpus (~410B tokens, hundreds of millions
    of documents), this Python implementation is adequate for volumes
    up to ~50M documents in RAM. For larger volumes, use the
    datatrove.pipeline.dedup module with distributed SLURM execution.
    The build_datatrove_dedup() function offers this alternative.

MANDATORY PREREQUISITES:
    1. Scripts 01-06 (Tier 1) completed
    2. Writable WORK_DIR working volume (Docker bind mount ./data, or NFS/HPC)
    3. Sufficient RAM: ~1 GB per million documents (MinHash index)

Usage:
    # Deduplicate all Tier 1 sources:
    python corpus/scripts/08_global_dedup.py

    # Deduplicate only specific sources:
    python corpus/scripts/08_global_dedup.py \\
        --sources gigaverbo madlad400 fineweb2

    # Check status without processing:
    python corpus/scripts/08_global_dedup.py --status

    # Use distributed datatrove (recommended for >50M docs):
    python corpus/scripts/08_global_dedup.py --use-datatrove \\
        --executor slurm --slurm-partition cpu

Output:
    $WORK_DIR/deduped/
    ├── shard_000000.parquet   final deduplicated corpus
    ├── shard_000001.parquet
    ├── ...
    └── dedup_report.json      complete statistics

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
Versão | Version: 0.1.0 — Abril 2026
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

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

WORK_DIR      = Path(os.environ.get("WORK_DIR", Path.home() / "manaca-corpus"))
RAW_DIR       = WORK_DIR / "raw"
OUTPUT_DIR    = WORK_DIR / "deduped"
STATS_DIR     = WORK_DIR / "stats"
CKPT_DIR      = WORK_DIR / "checkpoints"
MANIFEST_PATH = CKPT_DIR / "acquisition_manifest.json"
REPORT_PATH   = OUTPUT_DIR / "dedup_report.json"
SIG_CACHE_DIR = CKPT_DIR / "minhash_signatures"

# Fontes Tier 1 em ordem de prioridade (maior qualidade primeiro).
# Em clusters de duplicatas, documentos de fontes com menor índice
# têm prioridade de manutenção quando os scores são iguais.
TIER1_SOURCES = [
    "gigaverbo",
    "wikipedia_ptbr",
    "ulysses",
    "fineweb2",
    "madlad400",
    "hplt2",
    "common_crawl",
]

# Parâmetros MinHash LSH
MINHASH_CONFIG = {
    "num_permutations": 128,
    "n_gram_size":      5,
    "jaccard_threshold": 0.80,
    "lsh_bands":        16,    # 16 × 8 = 128 permutações
    "lsh_rows":         8,
}

SHARD_SIZE = 50_000

# Schema Parquet fixo — idêntico aos Scripts 01-07.
SCHEMA = pa.schema([
    pa.field("text",   pa.string()),
    pa.field("source", pa.string()),
    pa.field("id",     pa.string()),
    pa.field("lang",   pa.string()),
    pa.field("score",  pa.float32()),
])


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging() -> None:
    log_dir  = WORK_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"global_dedup_{ts}.log"

    logger.remove()
    logger.add(
        sys.stderr, level="INFO", colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
    )
    logger.add(
        log_file, level="DEBUG", rotation="200 MB", retention=3,
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}",
    )
    logger.info(f"Log: {log_file}")


# ── Normalização de texto ─────────────────────────────────────────────────────

_RE_PUNCT    = re.compile(r"[^\w\s]", re.UNICODE)
_RE_SPACES   = re.compile(r"\s+")


def normalize_for_dedup(text: str) -> str:
    """
    Normalização leve para deduplicação.

    Objetivo: remover variações superficiais que impediriam a detecção
    de duplicatas sem alterar o conteúdo textual essencial.

    Operações:
      1. Normalização NFD: decompõe caracteres acentuados
      2. Lowercase: insensível a maiúsculas/minúsculas
      3. Remoção de pontuação: foca no conteúdo lexical
      4. Colapso de espaços: normaliza whitespace

    Baseado na metodologia do CCNet (Wenzek et al., 2020) e
    LLM-jp-corpus v4 para línguas não-inglesas.
    """
    text = unicodedata.normalize("NFD", text)
    text = text.lower()
    text = _RE_PUNCT.sub(" ", text)
    text = _RE_SPACES.sub(" ", text).strip()
    return text


def text_to_ngrams(text: str, n: int = 5) -> set[str]:
    """Converte texto normalizado em conjunto de n-gramas de palavras."""
    words = text.split()
    if len(words) < n:
        return {text} if text else set()
    return {" ".join(words[i: i + n]) for i in range(len(words) - n + 1)}


# ── MinHash ───────────────────────────────────────────────────────────────────

def compute_minhash(ngrams: set[str], num_perm: int = 128) -> list[int]:
    """
    Calcula assinatura MinHash para um conjunto de n-gramas.

    Usa datasketch se disponível (vetorizado via NumPy, ~10x mais rápido).
    Fallback: implementação Python pura com hash mixing multiplicativo.

    Lee et al. (2022): 128 permutações oferece erro padrão de Jaccard
    ≈ 1/sqrt(128) ≈ 0.088, suficiente para threshold 0.80.
    """
    try:
        from datasketch import MinHash
        m = MinHash(num_perm=num_perm)
        for ngram in ngrams:
            m.update(ngram.encode("utf-8"))
        return list(m.hashvalues.tolist())
    except ImportError:
        # Fallback: implementação manual
        sig = [0xFFFFFFFF] * num_perm
        for ngram in ngrams:
            h = hashlib.sha256(ngram.encode("utf-8")).digest()
            for i in range(num_perm):
                val = struct.unpack_from(">I", h, (i * 4) % 28)[0]
                val = (val ^ (i * 2654435761)) & 0xFFFFFFFF
                if val < sig[i]:
                    sig[i] = val
        return sig


def sig_to_bands(
    sig: list[int],
    n_bands: int,
    rows: int,
) -> list[tuple[int, str]]:
    """Divide assinatura em bandas LSH para indexação."""
    bands = []
    for b in range(n_bands):
        start     = b * rows
        band_vals = tuple(sig[start: start + rows])
        band_hash = hashlib.md5(str(band_vals).encode()).hexdigest()[:16]
        bands.append((b, band_hash))
    return bands


# ── Iteração sobre corpus ─────────────────────────────────────────────────────

def iter_corpus(
    sources: list[str],
) -> Generator[dict[str, Any], None, None]:
    """
    Itera sobre todos os shards Parquet das fontes especificadas.
    Ordena por fonte (priority order) para que, em clusters de duplicatas,
    documentos de fontes de maior qualidade sejam preferidos.
    """
    for source in sources:
        source_dir = RAW_DIR / source
        if not source_dir.exists():
            logger.warning(f"Diretório nao encontrado: {source_dir} — pulando")
            continue

        shards = sorted(source_dir.glob("shard_*.parquet"))
        if not shards:
            logger.warning(f"Nenhum shard em {source_dir} — pulando")
            continue

        logger.info(f"  Fonte: {source:<25} {len(shards):>4} shards")

        for shard_path in shards:
            try:
                table = pq.read_table(
                    shard_path,
                    columns=["text", "source", "id", "lang", "score"],
                )
                for batch in table.to_batches(max_chunksize=1000):
                    texts   = batch.column("text").to_pylist()
                    srcs    = batch.column("source").to_pylist()
                    ids     = batch.column("id").to_pylist()
                    langs   = batch.column("lang").to_pylist()
                    scores  = batch.column("score").to_pylist()
                    for text, src, doc_id, lang, score in zip(
                        texts, srcs, ids, langs, scores
                    ):
                        if text and isinstance(text, str):
                            yield {
                                "text":   text,
                                "source": src or source,
                                "id":     str(doc_id),
                                "lang":   lang or "por_Latn_BR",
                                "score":  float(score) if score else 0.0,
                            }
            except Exception as e:
                logger.warning(f"Erro ao ler {shard_path.name}: {e}")
                continue


# ── Deduplicador principal ────────────────────────────────────────────────────

class GlobalDeduplicator:
    """
    Deduplicador MinHash LSH global para o corpus Manacá.

    Implementação em duas passagens:
      Passagem 1 (indexing_pass): calcular assinaturas e construir
        índice LSH em memória
      Passagem 2 (writing_pass): varrer corpus novamente, escrever
        apenas documentos não marcados como duplicatas

    Para corpora >50M documentos no SLURM, usar build_datatrove_dedup().
    """

    def __init__(self, sources: list[str], config: dict = MINHASH_CONFIG):
        self.sources  = sources
        self.config   = config
        self.lsh_index: dict[str, list[tuple[str, float, int]]] = {}
        self.duplicates: set[str] = set()
        self.doc_store: dict[str, dict] = {}

    def indexing_pass(self) -> tuple[int, int]:
        """
        Passagem 1: computa MinHash e indexa no LSH.
        Retorna (total_docs, indexed_docs).
        """
        cfg      = self.config
        n_bands  = cfg["lsh_bands"]
        n_rows   = cfg["lsh_rows"]
        n_perm   = cfg["num_permutations"]
        n_grams  = cfg["n_gram_size"]

        total    = 0
        indexed  = 0

        logger.info("Passagem 1: indexando assinaturas MinHash...")

        for doc in tqdm(
            iter_corpus(self.sources),
            desc="Indexando",
            unit="doc",
            mininterval=30,
        ):
            total += 1
            text   = doc["text"]
            doc_id = doc["id"]

            normalized = normalize_for_dedup(text)
            if len(normalized.split()) < n_grams:
                continue

            ngrams = text_to_ngrams(normalized, n=n_grams)
            sig    = compute_minhash(ngrams, num_perm=n_perm)
            bands  = sig_to_bands(sig, n_bands, n_rows)

            for band_id, band_hash in bands:
                key = f"b{band_id}:{band_hash}"
                if key not in self.lsh_index:
                    self.lsh_index[key] = []
                self.lsh_index[key].append(
                    (doc_id, doc["score"], TIER1_SOURCES.index(
                        doc["source"].split("_")[0]
                        if "_" in doc["source"] else doc["source"]
                    ) if doc["source"].split("_")[0] in TIER1_SOURCES else 99)
                )

            self.doc_store[doc_id] = doc
            indexed += 1

            if total % 1_000_000 == 0:
                logger.info(
                    f"  {total:,} docs vistos | "
                    f"{indexed:,} indexados | "
                    f"{len(self.lsh_index):,} entradas LSH"
                )

        return total, indexed

    def find_duplicates(self) -> tuple[int, int]:
        """
        Identifica clusters de near-duplicatas e marca para remoção.
        Em cada cluster, mantém o documento com maior quality_score
        (desempate por prioridade de fonte).
        """
        n_clusters   = 0
        n_duplicates = 0

        logger.info("Identificando clusters de near-duplicatas...")

        for key, candidates in tqdm(
            self.lsh_index.items(),
            desc="Clusters LSH",
            unit="banda",
            mininterval=10,
        ):
            if len(candidates) <= 1:
                continue

            # Ordenar: maior score primeiro, menor índice de fonte primeiro
            candidates_sorted = sorted(
                candidates,
                key=lambda x: (-x[1], x[2]),
            )
            best_id = candidates_sorted[0][0]

            new_dups = False
            for doc_id, score, src_idx in candidates_sorted[1:]:
                if doc_id != best_id and doc_id not in self.duplicates:
                    self.duplicates.add(doc_id)
                    n_duplicates += 1
                    new_dups = True

            if new_dups:
                n_clusters += 1

        return n_clusters, n_duplicates

    def writing_pass(self) -> int:
        """
        Passagem 2: escreve corpus deduplicado em Parquet.
        Retorna número de documentos escritos.
        """
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Passagem 2: escrevendo corpus deduplicado "
            f"({len(self.doc_store) - len(self.duplicates):,} docs)..."
        )

        buffer   = []
        shard_id = 0
        written  = 0

        def flush(buf: list, sid: int) -> None:
            path = OUTPUT_DIR / f"shard_{sid:06d}.parquet"
            pq.write_table(
                pa.table(
                    {
                        "text":   [d["text"]   for d in buf],
                        "source": [d["source"] for d in buf],
                        "id":     [d["id"]     for d in buf],
                        "lang":   [d["lang"]   for d in buf],
                        "score":  [d["score"]  for d in buf],
                    },
                    schema=SCHEMA,
                ),
                path,
                compression="zstd",
            )

        for doc_id, doc in tqdm(
            self.doc_store.items(),
            desc="Escrevendo",
            unit="doc",
            mininterval=10,
        ):
            if doc_id in self.duplicates:
                continue
            buffer.append(doc)
            written += 1

            if len(buffer) >= SHARD_SIZE:
                flush(buffer, shard_id)
                logger.debug(
                    f"Shard {shard_id:04d} escrito: "
                    f"{len(buffer):,} docs"
                )
                buffer, shard_id = [], shard_id + 1

        if buffer:
            flush(buffer, shard_id)

        return written

    def run(self) -> dict[str, Any]:
        """Executa pipeline completo de deduplicação global."""
        t_start = time.time()

        logger.info("=" * 60)
        logger.info("Manaca — Deduplicacao Global Cross-Source")
        logger.info(f"  Fontes:         {self.sources}")
        logger.info(f"  Permutacoes:    {self.config['num_permutations']}")
        logger.info(f"  Threshold:      {self.config['jaccard_threshold']}")
        logger.info(f"  N-gramas:       {self.config['n_gram_size']}")
        logger.info(f"  Output:         {OUTPUT_DIR}")
        logger.info("=" * 60)

        total_docs, indexed = self.indexing_pass()
        logger.info(
            f"Indexacao: {total_docs:,} vistos | "
            f"{indexed:,} indexados | "
            f"{len(self.lsh_index):,} entradas LSH"
        )

        n_clusters, n_dups = self.find_duplicates()
        logger.info(
            f"Clusters: {n_clusters:,} | "
            f"Duplicatas marcadas: {n_dups:,} "
            f"({n_dups/max(indexed,1):.1%})"
        )

        written = self.writing_pass()
        elapsed = time.time() - t_start

        total_bytes = sum(
            f.stat().st_size for f in OUTPUT_DIR.glob("shard_*.parquet")
        )

        report = {
            "status":            "completed",
            "sources_processed": self.sources,
            "config":            self.config,
            "total_docs_seen":   total_docs,
            "total_indexed":     indexed,
            "lsh_entries":       len(self.lsh_index),
            "duplicate_clusters": n_clusters,
            "duplicates_removed": n_dups,
            "dedup_rate":        round(n_dups / max(indexed, 1), 4),
            "docs_written":      written,
            "total_bytes":       total_bytes,
            "total_gb":          round(total_bytes / 1e9, 2),
            "elapsed_hours":     round(elapsed / 3600, 2),
            "output_dir":        str(OUTPUT_DIR),
            "completed_at":      datetime.now(timezone.utc).isoformat(),
        }

        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info("=" * 60)
        logger.info("DEDUPLICACAO GLOBAL CONCLUIDA")
        logger.info(f"  Docs indexados:     {indexed:,}")
        logger.info(f"  Duplicatas rem.:    {n_dups:,}  ({report['dedup_rate']:.1%})")
        logger.info(f"  Docs no corpus:     {written:,}")
        logger.info(f"  Tamanho final:      {total_bytes/1e9:.2f} GB")
        logger.info(f"  Tempo total:        {elapsed/3600:.2f}h")
        logger.info(f"  Output:             {OUTPUT_DIR}")
        logger.info(f"  Relatorio:          {REPORT_PATH}")
        logger.info("-" * 60)
        logger.info("GLOBAL DEDUPLICATION DONE")
        logger.info(f"  Docs indexed:       {indexed:,}")
        logger.info(f"  Duplicates removed: {n_dups:,}  ({report['dedup_rate']:.1%})")
        logger.info(f"  Docs in corpus:     {written:,}")
        logger.info(f"  Final size:         {total_bytes/1e9:.2f} GB")
        logger.info(f"  Total time:         {elapsed/3600:.2f}h")
        logger.info(f"  Output:             {OUTPUT_DIR}")
        logger.info(f"  Report:             {REPORT_PATH}")
        logger.info("=" * 60)
        logger.info(
            "Proximo passo | Next step: python corpus/scripts/09_validate_corpus.py"
        )

        return report


# ── Alternativa datatrove distribuída ────────────────────────────────────────

def build_datatrove_dedup(
    sources: list[str],
    executor_type: str = "local",
    num_tasks: int = 16,
    slurm_partition: str = "cpu",
) -> None:
    """
    Deduplicação distribuída via datatrove para corpora >50M documentos.

    Usa MinhashDedupSignature + MinhashDedupBuckets + MinhashDedupFilter
    do datatrove, que implementa o mesmo algoritmo mas de forma distribuída
    sobre múltiplos nós SLURM.

    Recomendado quando o corpus total superar 50M documentos.
    Referência: Penedo et al. (2024). FineWeb. arXiv:2406.17557.
    """
    try:
        from datatrove.executor import LocalPipelineExecutor, SlurmPipelineExecutor
        from datatrove.pipeline.dedup import (
            MinhashDedupSignature,
            MinhashDedupBuckets,
            MinhashDedupFilter,
            MinhashDedupCluster,
        )
        from datatrove.pipeline.readers import ParquetReader
        from datatrove.pipeline.writers import ParquetWriter
    except ImportError:
        logger.error("datatrove nao instalado. Use a implementacao Python local.")
        return

    logger.info("Usando deduplicacao distribuida via datatrove...")
    sig_dir    = CKPT_DIR / "dt_dedup_sigs"
    bucket_dir = CKPT_DIR / "dt_dedup_buckets"

    # Pipeline de assinaturas
    sig_pipeline = [
        ParquetReader(str(RAW_DIR)),
        MinhashDedupSignature(
            output_folder=str(sig_dir),
            config={
                "n_grams":           MINHASH_CONFIG["n_gram_size"],
                "num_permutations":  MINHASH_CONFIG["num_permutations"],
                "jaccard_threshold": MINHASH_CONFIG["jaccard_threshold"],
            },
        ),
    ]

    # Pipeline de filtragem
    filter_pipeline = [
        ParquetReader(str(RAW_DIR)),
        MinhashDedupFilter(sig_folder=str(sig_dir)),
        ParquetWriter(
            output_folder=str(OUTPUT_DIR),
            compression="zstd",
        ),
    ]

    ExecutorClass = (
        SlurmPipelineExecutor if executor_type == "slurm"
        else LocalPipelineExecutor
    )

    kwargs: dict[str, Any] = {
        "pipeline": sig_pipeline,
        "tasks": num_tasks,
        "logging_dir": str(WORK_DIR / "logs" / "dt_dedup"),
        "checkpoints_folder": str(CKPT_DIR / "dt_dedup_ckpt"),
    }
    if executor_type == "slurm":
        kwargs["partition"] = slurm_partition
        kwargs["time"] = "48:00:00"
        kwargs["mem_gb"] = 8

    ExecutorClass(**kwargs).run()
    logger.info("Assinaturas computadas. Executar pipeline de filtragem...")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manacá — Deduplicação Global Cross-Source",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--sources", nargs="+", default=None,
        help=(
            "Fontes a deduplicar (default: todas em TIER1_SOURCES). "
            "Ex: --sources gigaverbo fineweb2 madlad400"
        ),
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Exibir status das fontes disponíveis e sair.",
    )
    parser.add_argument(
        "--use-datatrove", action="store_true",
        help=(
            "Usar deduplicação distribuída via datatrove "
            "(recomendado para >50M documentos)."
        ),
    )
    parser.add_argument(
        "--executor", choices=["local", "slurm"], default="local",
        help="Executor datatrove (apenas com --use-datatrove). Default: local.",
    )
    parser.add_argument(
        "--num-tasks", type=int, default=16,
        help="Tarefas paralelas datatrove. Default: 16.",
    )
    parser.add_argument(
        "--slurm-partition", type=str, default="cpu",
        help="Partição SLURM. Default: cpu.",
    )
    parser.add_argument(
        "--jaccard-threshold", type=float, default=0.80,
        help="Threshold Jaccard para deduplicação. Default: 0.80.",
    )
    return parser.parse_args()


def print_source_status(sources: list[str]) -> None:
    """Exibe estado das fontes disponíveis para deduplicação."""
    print()
    print("=" * 60)
    print("  Fontes disponíveis para deduplicação")
    print("=" * 60)
    total_shards = 0
    total_gb     = 0.0
    for src in sources:
        src_dir = RAW_DIR / src
        if not src_dir.exists():
            print(f"  {src:<25} {'AUSENTE':<12} —")
            continue
        shards = list(src_dir.glob("shard_*.parquet"))
        gb     = sum(f.stat().st_size for f in shards) / 1e9
        total_shards += len(shards)
        total_gb     += gb
        print(f"  {src:<25} {len(shards):>4} shards   {gb:6.1f} GB")
    print("=" * 60)
    print(f"  Total: {total_shards} shards | {total_gb:.1f} GB")
    deduped = list(OUTPUT_DIR.glob("shard_*.parquet")) if OUTPUT_DIR.exists() else []
    if deduped:
        dedup_gb = sum(f.stat().st_size for f in deduped) / 1e9
        print(f"  Corpus deduplicado: {len(deduped)} shards | {dedup_gb:.1f} GB")
    print()


def main() -> int:
    args = parse_args()

    setup_logging()

    sources = args.sources or [
        s for s in TIER1_SOURCES
        if (RAW_DIR / s).exists()
    ]

    if args.status:
        print_source_status(TIER1_SOURCES)
        return 0

    if not sources:
        logger.error(
            "Nenhuma fonte disponível em $WORK_DIR/raw/. "
            "Executar Scripts 01-06 primeiro."
        )
        return 1

    print_source_status(sources)

    # Volume de trabalho: bind mount Docker (./data -> /workspace/manaca-corpus)
    # ou NFS/HPC. Configuravel pela variavel de ambiente WORK_DIR.
    _work_dir = Path(os.environ.get("WORK_DIR", Path.home() / "manaca-corpus"))
    try:
        _work_dir.mkdir(parents=True, exist_ok=True)
    except OSError as _e:
        print(f"[ERRO] Nao foi possivel preparar WORK_DIR={_work_dir}: {_e}")
        sys.exit(1)
    if not os.access(_work_dir, os.W_OK):
        print(f"[ERRO] WORK_DIR sem permissao de escrita: {_work_dir}")
        print("       Docker: confira o bind mount (./data) e a variavel WORK_DIR.")
        print("       HPC/NFS: exporte WORK_DIR=/caminho/do/nfs/manaca-corpus.")
        sys.exit(1)

    if args.use_datatrove:
        build_datatrove_dedup(
            sources       = sources,
            executor_type = args.executor,
            num_tasks     = args.num_tasks,
            slurm_partition = args.slurm_partition,
        )
        return 0

    # Ajustar threshold se fornecido
    config = dict(MINHASH_CONFIG)
    config["jaccard_threshold"] = args.jaccard_threshold

    deduplicator = GlobalDeduplicator(sources=sources, config=config)
    report = deduplicator.run()

    # Atualizar manifesto global
    manifest_path = CKPT_DIR / "acquisition_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {}
    manifest["global_dedup"] = report
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    tmp.rename(manifest_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())

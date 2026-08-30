#!/usr/bin/env python3
"""
Manacá LLM — Script 07: Pipeline Common Crawl via datatrove
============================================================
LNCC × NII/LLM-jp | Fase 1 — Corpus PT-BR

Processa snapshots do Common Crawl usando o framework datatrove,
replicando o pipeline FineWeb para PT-BR com snapshots recentes
(2024–2025) não cobertos pelo FineWeb-2 (Script 03).

Idempotente: o datatrove mantém checkpoints por tarefa automaticamente.

Justificativa científica:
    O FineWeb-2 (Script 03) cobre snapshots CC até dezembro de 2024.
    Este script processa snapshots de 2025 usando exatamente o mesmo
    pipeline (Gopher + C4 + FineWeb filters + MinHash deduplication),
    garantindo que o corpus Manacá inclua dados web recentes.

    O framework datatrove (Penedo et al., 2024) foi desenvolvido
    especificamente para este propósito — processar o Common Crawl em
    escala com pipeline de qualidade reprodutível. É o mesmo framework
    usado para construir o FineWeb e o FineWeb-2.

    Pipeline completo aplicado (diferente dos Scripts 01-06 que usam
    apenas filtros de sanidade sobre dados já curados):

      WARCReader → LanguageFilter (GlotLID) → GopherQualityFilter →
      GopherRepetitionFilter → C4QualityFilter → FineWebQualityFilter →
      MinhashDedupSignature → ParquetWriter

    Referências:
      Penedo, G. et al. (2024). The FineWeb Datasets. arXiv:2406.17557.
      Rae, J. et al. (2021). Scaling Language Models. arXiv:2112.11446.
      Raffel, C. et al. (2020). T5/C4. arXiv:1910.10683.
      Lee, K. et al. (2022). Deduplicating Training Data. arXiv:2107.06499.

DIFERENÇA FUNDAMENTAL em relação a todos os Scripts anteriores:
    Este script usa o framework datatrove com executores próprios
    (LocalPipelineExecutor ou SlurmPipelineExecutor), não o loop
    Python customizado dos Scripts 01-06.

    O datatrove gerencia internamente:
      - Paralelização por tarefas (shards CC)
      - Checkpointing automático por tarefa
      - Logging e métricas de pipeline
      - Escrita em Parquet particionado

REQUISITO DE RECURSOS:
    - Mínimo: 8 CPUs, 32 GB RAM (execução local lenta)
    - Recomendado: SLURM com 4+ nós, 64 CPUs/nó (produção)
    - Armazenamento: ~500 GB por snapshot processado

PRÉ-REQUISITOS OBRIGATÓRIOS:
    1. Scripts 01-06 (Tier 1) concluídos
    2. Volume de trabalho WORK_DIR gravavel (bind mount Docker ./data, ou NFS/HPC)
    3. datatrove[processing] instalado (pip install datatrove[processing])
    4. SLURM disponível (recomendado) ou múltiplos CPUs locais

Uso:
    # Listar snapshots configurados:
    python corpus/scripts/07_cc_pipeline.py --list-snapshots

    # Teste local com 1 snapshot, 4 tarefas (lento mas sem SLURM):
    python corpus/scripts/07_cc_pipeline.py \\
        --snapshot CC-MAIN-2025-08 \\
        --num-tasks 4 \\
        --executor local

    # Produção via SLURM (após acesso ao cluster):
    python corpus/scripts/07_cc_pipeline.py \\
        --snapshot CC-MAIN-2025-08 \\
        --num-tasks 64 \\
        --executor slurm \\
        --slurm-partition cpu

    # Todos os snapshots configurados via SLURM:
    python corpus/scripts/07_cc_pipeline.py \\
        --executor slurm \\
        --slurm-partition cpu

    # Monitorar progresso:
    tail -f $WORK_DIR/logs/cc_pipeline/CC-MAIN-2025-08/*.log

Saída:
    $WORK_DIR/raw/common_crawl/<snapshot>/
    ├── filtered/
    │   ├── 000000.parquet
    │   ├── 000001.parquet
    │   └── ...
    └── dedup_signatures/
        └── (assinaturas MinHash para dedup global)

Autor: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
Versão: 0.1.0 — Abril 2026
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Verificação de dependências ───────────────────────────────────────────────
try:
    from loguru import logger
except ImportError:
    print("[ERRO] loguru ausente. Rode dentro do container 'corpus' (docker compose run --rm corpus ...).")
    sys.exit(1)


# ── Configuração ──────────────────────────────────────────────────────────────

SOURCE_NAME = "common_crawl"

# Snapshots do Common Crawl a processar.
# Seleção: snapshots de 2025 não cobertos pelo FineWeb-2 (cutoff: dez/2024)
# + 1 snapshot de 2024 para validação cruzada com FineWeb-2.
# Fonte: https://commoncrawl.org/the-data/get-started/
CC_SNAPSHOTS = [
    "CC-MAIN-2024-51",  # Dezembro 2024 — overlap com FineWeb-2 (validação)
    "CC-MAIN-2025-08",  # Fevereiro 2025 — primeiro snapshot 2025
    "CC-MAIN-2025-18",  # Maio 2025
]

# Configuração de linguagem para GlotLID
# por_Latn: português em escrita latina (PT-BR + PT-PT)
# Threshold: 0.65 — conservador, consistente com Scripts 02-04
LANG_CONFIG = "por_Latn"
LANG_THRESHOLD = 0.65

WORK_DIR      = Path(os.environ.get("WORK_DIR", Path.home() / "manaca-corpus"))
OUTPUT_BASE   = WORK_DIR / "raw" / SOURCE_NAME
LOG_BASE      = WORK_DIR / "logs" / "cc_pipeline"
CKPT_BASE     = WORK_DIR / "checkpoints" / "cc_pipeline"
MANIFEST_PATH = WORK_DIR / "checkpoints" / "acquisition_manifest.json"


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(snapshot: str) -> None:
    log_dir = LOG_BASE / snapshot
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
    )
    logger.add(
        log_file,
        level="DEBUG",
        rotation="200 MB",
        retention=5,
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<7} | {message}",
    )
    logger.info(f"Log: {log_file}")


# ── Manifesto ─────────────────────────────────────────────────────────────────

def load_manifest() -> dict[str, Any]:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {
        "version":    "0.1.0",
        "project":    "Manaca LLM — LNCC x NII/LLM-jp",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources":    {},
    }


def save_manifest(manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = MANIFEST_PATH.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.rename(MANIFEST_PATH)


# ── Verificação de dependências datatrove ─────────────────────────────────────

def check_datatrove_deps() -> bool:
    """
    Verifica se as dependências do datatrove estão disponíveis.

    O datatrove[processing] inclui dependências pesadas (trafilatura,
    fasttext, inscriptis) que não são necessárias nos Scripts 01-06.
    """
    required = [
        "datatrove.pipeline.readers",
        "datatrove.pipeline.filters",
        "datatrove.pipeline.dedup",
        "datatrove.pipeline.writers",
        "datatrove.executor",
    ]
    missing = []
    for module in required:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)

    if missing:
        logger.error("Modulos datatrove ausentes:")
        for m in missing:
            logger.error(f"  {m}")
        logger.error(
            "Instalar: pip install 'datatrove[processing]'"
        )
        return False
    return True


# ── Construção do pipeline datatrove ─────────────────────────────────────────

def build_pipeline(
    snapshot: str,
    output_dir: Path,
    ckpt_dir: Path,
    log_dir: Path,
    num_tasks: int,
) -> tuple[list, Path, Path]:
    """
    Constrói o pipeline datatrove para um snapshot do Common Crawl.

    Pipeline idêntico ao FineWeb (Penedo et al., 2024):
      1. WARCReader      — lê arquivos WARC do CC via HTTPS
      2. LanguageFilter  — GlotLID: manter somente por_Latn
      3. GopherQuality   — filtros heurísticos (Rae et al., 2021)
      4. GopherRepetition — filtros de repetição n-gram
      5. C4Quality       — filtros adicionais (Raffel et al., 2020)
      6. FineWebQuality  — filtros de alta precisão (Penedo et al., 2024)
      7. MinhashDedup    — assinaturas MinHash within-snapshot
      8. ParquetWriter   — saída em Parquet + Zstd

    Nota sobre acesso ao CC:
      Os WARCs são acessados via HTTPS (data.commoncrawl.org) sem custo.
      O código HTTP 403 visto anteriormente era apenas no endpoint raiz
      do S3, não nos dados. O datatrove gerencia o acesso automaticamente.

    Retorna: (pipeline, output_filtered_dir, dedup_dir)
    """
    from datatrove.pipeline.readers import WarcReader
    from datatrove.pipeline.filters import (
        GopherQualityFilter,
        GopherRepetitionFilter,
        C4QualityFilter,
        LanguageFilter,
        FineWebQualityFilter,
    )
    from datatrove.pipeline.dedup import MinhashDedupSignature
    from datatrove.pipeline.writers import ParquetWriter

    output_filtered = output_dir / "filtered"
    dedup_dir       = output_dir / "dedup_signatures"
    output_filtered.mkdir(parents=True, exist_ok=True)
    dedup_dir.mkdir(parents=True, exist_ok=True)

    # ── Estágio 1: WARCReader ─────────────────────────────────────────────────
    # Lê WARCs diretamente do Common Crawl via HTTPS.
    # O padrão glob seleciona todos os segmentos do snapshot.
    warc_reader = WarcReader(
        f"s3://commoncrawl/crawl-data/{snapshot}/segments/",
        glob_pattern="*/warc/*.warc.gz",
        default_metadata={"snapshot": snapshot, "source": SOURCE_NAME},
    )

    # ── Estágio 2: Language Filter (GlotLID) ──────────────────────────────────
    # GlotLID: Kargaran et al. (2023). arXiv:2303.12463.
    # por_Latn distingue PT (Latino) de outras escritas.
    # Threshold 0.65: consistente com Scripts 02-04 para uniformidade.
    language_filter = LanguageFilter(
        language=LANG_CONFIG,
        language_threshold=LANG_THRESHOLD,
    )

    # ── Estágio 3: Gopher Quality Filter ─────────────────────────────────────
    # Rae et al. (2021). Scaling Language Models: Gopher. arXiv:2112.11446.
    # Parâmetros calibrados para PT-BR (stop words diferentes do inglês).
    gopher_quality = GopherQualityFilter(
        min_doc_words=50,
        max_doc_words=100_000,
        min_avg_word_length=3,
        max_avg_word_length=10,
        max_symbol_word_ratio=0.1,
        max_bullet_lines_ratio=0.9,
        max_ellipsis_lines_ratio=0.3,
        language="pt",
    )

    # ── Estágio 4: Gopher Repetition Filter ──────────────────────────────────
    # Remove documentos com alto nível de repetição de n-gramas,
    # indicativo de spam, listas geradas automaticamente ou boilerplate.
    gopher_repetition = GopherRepetitionFilter(
        language="pt",
    )

    # ── Estágio 5: C4 Quality Filter ─────────────────────────────────────────
    # Raffel et al. (2020). T5. arXiv:1910.10683.
    # Filtros adicionais: pontuação terminal, número mínimo de sentenças.
    c4_quality = C4QualityFilter(
        filter_no_terminal_punct=True,
        min_num_sentences=3,
        language="pt",
    )

    # ── Estágio 6: FineWeb Quality Filter ────────────────────────────────────
    # Penedo et al. (2024). FineWeb. arXiv:2406.17557.
    # Filtros de alta precisão desenvolvidos especificamente pelo HuggingFace
    # para o FineWeb — inclui detecção de stop words mínimas.
    fineweb_quality = FineWebQualityFilter(
        min_stop_words=2,
    )

    # ── Estágio 7: MinHash Deduplication (within-snapshot) ───────────────────
    # Lee et al. (2022). Deduplicating Training Data. arXiv:2107.06499.
    # Assinaturas computadas aqui; clustering feito no Script 08.
    # 128 permutações: erro padrão Jaccard ≈ 1/sqrt(128) ≈ 0.088.
    minhash_dedup = MinhashDedupSignature(
        output_folder=str(dedup_dir),
        config={
            "n_grams":           5,
            "num_permutations":  128,
            "jaccard_threshold": 0.80,
        },
    )

    # ── Estágio 8: ParquetWriter ──────────────────────────────────────────────
    # Saída em Parquet + Zstd, compatível com o schema dos Scripts 01-06.
    # max_file_size=1GB: shards grandes para eficiência no Script 08.
    parquet_writer = ParquetWriter(
        output_folder=str(output_filtered),
        output_filename="${rank:06d}.parquet",
        compression="zstd",
        max_file_size=1 << 30,  # 1 GB por arquivo
    )

    pipeline = [
        warc_reader,
        language_filter,
        gopher_quality,
        gopher_repetition,
        c4_quality,
        fineweb_quality,
        minhash_dedup,
        parquet_writer,
    ]

    return pipeline, output_filtered, dedup_dir


# ── Executor local ────────────────────────────────────────────────────────────

def run_local(
    snapshot: str,
    num_tasks: int,
    num_workers: int | None,
) -> dict[str, Any]:
    """
    Executa o pipeline localmente usando LocalPipelineExecutor.

    Adequado para:
      - Teste e validação do pipeline com pequeno número de tarefas
      - Ambientes sem SLURM
      - Desenvolvimento e debugging

    Limitações:
      - Significativamente mais lento que execução SLURM
      - Limitado pela RAM disponível no nó ha4 (94 GiB)
      - Recomendado: num_tasks <= 8 para execução local no ha4

    Para produção completa (todos os snapshots), usar run_slurm().
    """
    from datatrove.executor import LocalPipelineExecutor

    output_dir = OUTPUT_BASE / snapshot
    ckpt_dir   = CKPT_BASE  / snapshot
    log_dir    = LOG_BASE   / snapshot

    for d in [output_dir, ckpt_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f"Executor: LOCAL | Tarefas: {num_tasks} | Workers: {num_workers or num_tasks}")

    pipeline, output_filtered, dedup_dir = build_pipeline(
        snapshot, output_dir, ckpt_dir, log_dir, num_tasks
    )

    t_start = time.time()

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=num_tasks,
        workers=num_workers or min(num_tasks, os.cpu_count() or 4),
        logging_dir=str(log_dir),
        checkpoints_folder=str(ckpt_dir),
    )

    logger.info(f"Iniciando pipeline local para {snapshot}...")
    executor.run()

    elapsed = time.time() - t_start

    # Calcular tamanho do output
    total_bytes = sum(
        f.stat().st_size
        for f in output_filtered.rglob("*.parquet")
        if f.is_file()
    )

    stats = {
        "snapshot":       snapshot,
        "executor":       "local",
        "num_tasks":      num_tasks,
        "elapsed_hours":  round(elapsed / 3600, 2),
        "output_dir":     str(output_filtered),
        "total_bytes":    total_bytes,
        "total_gb":       round(total_bytes / 1e9, 2),
        "completed_at":   datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        f"Pipeline local concluido: {snapshot} | "
        f"{total_bytes/1e9:.2f} GB | "
        f"{elapsed/3600:.1f}h"
    )

    return stats


# ── Executor SLURM ────────────────────────────────────────────────────────────

def run_slurm(
    snapshot: str,
    num_tasks: int,
    slurm_partition: str,
    slurm_time: str,
    slurm_mem_gb: int,
    slurm_cpus: int,
) -> dict[str, Any]:
    """
    Submete o pipeline ao SLURM usando SlurmPipelineExecutor.

    Este é o modo de execução recomendado para produção no cluster DEXL/SDumont.
    O datatrove cria automaticamente os job scripts SLURM e os submete,
    gerenciando dependências entre tarefas internamente.

    Parâmetros calibrados para o ha4/DEXL:
      - partition: verificar com 'sinfo' no nó de submissão SLURM
      - time: 72h por snapshot (conservador)
      - mem_gb_per_task: 4 GB por tarefa (sufficient para WARC processing)

    O SlurmPipelineExecutor monitora os jobs e pode ser re-executado
    para retomar tarefas que falharam (idempotência via checkpoints).
    """
    from datatrove.executor import SlurmPipelineExecutor

    output_dir = OUTPUT_BASE / snapshot
    ckpt_dir   = CKPT_BASE  / snapshot
    log_dir    = LOG_BASE   / snapshot

    for d in [output_dir, ckpt_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Executor: SLURM | Partition: {slurm_partition} | "
        f"Tarefas: {num_tasks} | Time: {slurm_time}"
    )

    pipeline, output_filtered, dedup_dir = build_pipeline(
        snapshot, output_dir, ckpt_dir, log_dir, num_tasks
    )

    mem_per_task = max(1, slurm_mem_gb // num_tasks)

    executor = SlurmPipelineExecutor(
        pipeline=pipeline,
        tasks=num_tasks,
        time=slurm_time,
        partition=slurm_partition,
        cpus_per_task=slurm_cpus,
        mem_gb=mem_per_task,
        job_name=f"manaca-cc-{snapshot[-7:]}",
        logs_folder=str(log_dir),
        checkpoints_folder=str(ckpt_dir),
    )

    logger.info(f"Submetendo jobs SLURM para {snapshot}...")
    executor.run()

    stats = {
        "snapshot":       snapshot,
        "executor":       "slurm",
        "num_tasks":      num_tasks,
        "slurm_partition": slurm_partition,
        "slurm_time":     slurm_time,
        "output_dir":     str(output_filtered),
        "submitted_at":   datetime.now(timezone.utc).isoformat(),
        "status":         "submitted",
    }

    logger.info(
        f"Jobs SLURM submetidos para {snapshot}. "
        f"Monitorar: squeue -u $USER"
    )

    return stats


# ── Pipeline principal ────────────────────────────────────────────────────────

def process_snapshot(
    snapshot: str,
    executor_type: str,
    num_tasks: int,
    num_workers: int | None,
    slurm_partition: str,
    slurm_time: str,
    slurm_mem_gb: int,
    slurm_cpus: int,
) -> dict[str, Any]:
    """
    Processa um snapshot do Common Crawl com o pipeline FineWeb PT-BR.

    Verifica pré-requisitos, configura diretórios e despacha para o
    executor apropriado (local ou SLURM).
    """

    # Volume de trabalho: bind mount Docker (./data -> /workspace/manaca-corpus)
    # ou NFS/HPC. Usa o mesmo diretorio de OUTPUT_BASE (respeita --work-dir),
    # nao apenas a env var — OUTPUT_BASE = <work_dir>/raw/<source>.
    _work_dir = OUTPUT_BASE.parent.parent
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

    # ── Verificar dependências datatrove ──────────────────────────────────────
    if not check_datatrove_deps():
        sys.exit(1)

    # ── Verificar se já foi processado ───────────────────────────────────────
    manifest = load_manifest()
    existing = manifest.get("sources", {}).get(
        f"{SOURCE_NAME}_{snapshot}", {}
    )
    if existing.get("status") == "completed":
        logger.info(f"Snapshot {snapshot} ja processado. Pulando.")
        return existing

    setup_logging(snapshot)

    logger.info("=" * 60)
    logger.info(f"Snapshot:      {snapshot}")
    logger.info(f"Executor:      {executor_type}")
    logger.info(f"Tarefas:       {num_tasks}")
    logger.info(f"Linguagem:     {LANG_CONFIG} (threshold={LANG_THRESHOLD})")
    logger.info(f"Output:        {OUTPUT_BASE / snapshot}")
    logger.info("=" * 60)
    logger.info("Pipeline: WARC → GlotLID → Gopher → C4 → FineWeb → MinHash → Parquet")
    logger.info(
        "Referencia: Penedo et al. (2024). FineWeb. arXiv:2406.17557."
    )

    # ── Despachar para executor ───────────────────────────────────────────────
    if executor_type == "local":
        stats = run_local(snapshot, num_tasks, num_workers)
    elif executor_type == "slurm":
        stats = run_slurm(
            snapshot, num_tasks, slurm_partition,
            slurm_time, slurm_mem_gb, slurm_cpus,
        )
    else:
        logger.error(f"Executor desconhecido: {executor_type}")
        sys.exit(1)

    # ── Atualizar manifesto ───────────────────────────────────────────────────
    manifest["sources"][f"{SOURCE_NAME}_{snapshot}"] = stats
    save_manifest(manifest)

    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manacá — Pipeline Common Crawl via datatrove",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--snapshot", type=str, default=None,
        help=(
            "Snapshot CC a processar (ex: CC-MAIN-2025-08). "
            "Default: todos os snapshots configurados em CC_SNAPSHOTS."
        ),
    )
    parser.add_argument(
        "--list-snapshots", action="store_true",
        help="Listar snapshots configurados e sair.",
    )
    parser.add_argument(
        "--executor", type=str, choices=["local", "slurm"], default="local",
        help="Tipo de executor: 'local' ou 'slurm'. Default: local.",
    )
    parser.add_argument(
        "--num-tasks", type=int, default=8,
        help=(
            "Número de tarefas paralelas. "
            "Local: <= 8 recomendado para ha4. "
            "SLURM: 64-256 para produção. "
            "Default: 8."
        ),
    )
    parser.add_argument(
        "--num-workers", type=int, default=None,
        help=(
            "Workers locais (apenas executor=local). "
            "Default: min(num_tasks, nproc)."
        ),
    )
    parser.add_argument(
        "--slurm-partition", type=str, default="cpu",
        help="Partição SLURM (verificar com 'sinfo'). Default: cpu.",
    )
    parser.add_argument(
        "--slurm-time", type=str, default="72:00:00",
        help="Tempo limite SLURM por job (HH:MM:SS). Default: 72:00:00.",
    )
    parser.add_argument(
        "--slurm-mem-gb", type=int, default=256,
        help="Memória total por nó SLURM (GB). Default: 256.",
    )
    parser.add_argument(
        "--slurm-cpus", type=int, default=1,
        help="CPUs por tarefa SLURM. Default: 1.",
    )
    parser.add_argument(
        "--work-dir", type=Path,
        default=Path(os.environ.get("WORK_DIR", Path.home() / "manaca-corpus")),
        help="Diretório de trabalho. Default: $WORK_DIR.",
    )
    return parser.parse_args()


# ── Sumário de status ─────────────────────────────────────────────────────────

def print_status_summary() -> None:
    """Exibe o estado atual de todos os snapshots configurados."""
    manifest = load_manifest()
    sources  = manifest.get("sources", {})

    print()
    print("=" * 60)
    print("  Status dos snapshots Common Crawl")
    print("=" * 60)

    total_gb = 0.0
    for snap in CC_SNAPSHOTS:
        key    = f"{SOURCE_NAME}_{snap}"
        status = sources.get(key, {})
        state  = status.get("status", "nao iniciado")
        gb     = status.get("total_gb", 0.0)
        total_gb += gb
        print(f"  {snap:<25} {state:<14} {gb:6.1f} GB")

    print("=" * 60)
    print(f"  Total processado: {total_gb:.1f} GB")
    print()


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    # Atualizar WORK_DIR global se fornecido via CLI
    global OUTPUT_BASE, LOG_BASE, CKPT_BASE, MANIFEST_PATH
    OUTPUT_BASE   = args.work_dir / "raw" / SOURCE_NAME
    LOG_BASE      = args.work_dir / "logs" / "cc_pipeline"
    CKPT_BASE     = args.work_dir / "checkpoints" / "cc_pipeline"
    MANIFEST_PATH = args.work_dir / "checkpoints" / "acquisition_manifest.json"

    if args.list_snapshots:
        print("\nSnapshots configurados para o projeto Manacá:")
        for i, snap in enumerate(CC_SNAPSHOTS, 1):
            print(f"  {i}. {snap}")
        print(
            f"\nFineWeb-2 (Script 03) ja cobre snapshots ate dez/2024."
            f"\nEste script processa snapshots 2025 (novos dados)."
        )
        print_status_summary()
        return 0

    # Selecionar snapshots a processar
    snapshots = [args.snapshot] if args.snapshot else CC_SNAPSHOTS

    # Validar snapshots
    for snap in snapshots:
        if not snap.startswith("CC-MAIN-"):
            logger.error(
                f"Formato de snapshot invalido: '{snap}'. "
                "Esperado: CC-MAIN-YYYY-WW (ex: CC-MAIN-2025-08)"
            )
            return 1

    logger.info(f"Snapshots a processar: {snapshots}")
    logger.info(f"Executor: {args.executor} | Tarefas: {args.num_tasks}")

    # Processar cada snapshot
    all_stats = []
    for snapshot in snapshots:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Processando snapshot: {snapshot}")
        logger.info(f"{'='*60}")

        try:
            stats = process_snapshot(
                snapshot       = snapshot,
                executor_type  = args.executor,
                num_tasks      = args.num_tasks,
                num_workers    = args.num_workers,
                slurm_partition = args.slurm_partition,
                slurm_time     = args.slurm_time,
                slurm_mem_gb   = args.slurm_mem_gb,
                slurm_cpus     = args.slurm_cpus,
            )
            all_stats.append(stats)
        except Exception as e:
            logger.error(f"Falha ao processar {snapshot}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Sumário final
    print_status_summary()

    if args.executor == "local":
        logger.info(
            "Proximo passo (local): "
            "python corpus/scripts/08_global_dedup.py"
        )
    else:
        logger.info(
            "Jobs SLURM submetidos. Monitorar com: squeue -u $USER"
        )
        logger.info(
            "Apos conclusao dos jobs: "
            "python corpus/scripts/08_global_dedup.py"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

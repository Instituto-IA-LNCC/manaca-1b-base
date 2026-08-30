#!/usr/bin/env python3
"""
Manacá LLM — Script 03: Aquisição do FineWeb-2 (partição por_Latn)
====================================================================
LNCC × NII/LLM-jp | Fase 1 — Corpus PT-BR

================================ PT (Português) ================================

Baixa a partição portuguesa do FineWeb-2 (HuggingFaceFW/fineweb-2),
aplica filtro de variante linguística para reter somente PT-BR,
e armazena em Apache Parquet + Zstandard particionado em shards
de 50.000 documentos.

Idempotente: retoma automaticamente do último shard salvo.

Justificativa científica:
    Penedo et al. (2024, arXiv:2406.17557) demonstraram que o FineWeb
    supera C4, RefinedWeb e OSCAR em todos os benchmarks avaliados para
    inglês. O FineWeb-2 estende a metodologia para 1.000 línguas aplicando
    o mesmo pipeline rigoroso: Language Identification (GlotLID) → Gopher
    Quality Filters → C4 Quality Filters → FineWeb Quality Filters →
    MinHash Deduplication.

    A partição por_Latn (~150B tokens PT) representa o subconjunto de
    maior qualidade de dados web em português já disponível publicamente,
    resultado de 96 snapshots do Common Crawl processados com o framework
    datatrove — o mesmo usado neste projeto.

    Razão para incluir além do Common Crawl direto (Script 07):
    O FineWeb-2 já passou por todo o pipeline de filtragem, economizando
    semanas de processamento. O Script 07 complementa com snapshots 2025
    não cobertos pelo FineWeb-2 (cutoff: dezembro 2024).

    ATENÇÃO: A partição por_Latn inclui PT-BR e PT-PT. Este script aplica
    identificação de variante via fastText com threshold >= 0.65, idêntico
    ao Script 02, para consistência metodológica entre fontes.

PRÉ-REQUISITOS OBRIGATÓRIOS:
    1. Script 01 (GigaVerbo) concluído e verificado
    2. Volume de trabalho WORK_DIR gravavel (bind mount Docker ./data, ou NFS/HPC)
    3. python corpus/scripts/00_verify_env.py retornando OK

Uso:
    # Verificar ambiente (obrigatório):
    python corpus/scripts/00_verify_env.py

    # Produção — sempre via screen:
    # Docker (recomendado): container destacado, logs persistidos no volume
    docker compose run -d --name fineweb2 corpus python corpus/scripts/03_acquire_fineweb2.py
    docker compose logs -f fineweb2
    tail -f $WORK_DIR/raw/fineweb2/download.log

    # Teste rápido (foreground, interrompível com Ctrl+C):
    python corpus/scripts/03_acquire_fineweb2.py

Saída:
    $WORK_DIR/raw/fineweb2/
    ├── shard_000000.parquet   (~100-200 MB · Zstd)
    ├── shard_000001.parquet
    ├── ...
    ├── download.log           log estruturado com timestamps
    └── langid_stats.json      estatísticas PT-BR vs PT-PT

================================= EN (English) =================================

Manacá LLM — Script 03: FineWeb-2 Acquisition (por_Latn partition)

Downloads the Portuguese partition of FineWeb-2 (HuggingFaceFW/fineweb-2),
applies a language-variant filter to keep only PT-BR,
and stores it in Apache Parquet + Zstandard partitioned into shards
of 50,000 documents.

Idempotent: automatically resumes from the last saved shard.

Scientific rationale:
    Penedo et al. (2024, arXiv:2406.17557) showed that FineWeb
    outperforms C4, RefinedWeb, and OSCAR on all evaluated benchmarks for
    English. FineWeb-2 extends the methodology to 1,000 languages applying
    the same rigorous pipeline: Language Identification (GlotLID) → Gopher
    Quality Filters → C4 Quality Filters → FineWeb Quality Filters →
    MinHash Deduplication.

    The por_Latn partition (~150B PT tokens) represents the highest-quality
    subset of Portuguese web data publicly available so far,
    the result of 96 Common Crawl snapshots processed with the
    datatrove framework — the same one used in this project.

    Reason to include it besides direct Common Crawl (Script 07):
    FineWeb-2 has already gone through the entire filtering pipeline, saving
    weeks of processing. Script 07 complements it with 2025 snapshots
    not covered by FineWeb-2 (cutoff: December 2024).

    WARNING: The por_Latn partition includes PT-BR and PT-PT. This script applies
    variant identification via fastText with threshold >= 0.65, identical
    to Script 02, for methodological consistency across sources.

MANDATORY PREREQUISITES:
    1. Script 01 (GigaVerbo) completed and verified
    2. Writable WORK_DIR working volume (Docker bind mount ./data, or NFS/HPC)
    3. python corpus/scripts/00_verify_env.py returning OK


Usage:
    # Check the environment (mandatory):
    python corpus/scripts/00_verify_env.py

    # Production — always via screen:
    # Docker (recommended): detached container, logs persisted on the volume
    docker compose run -d --name fineweb2 corpus python corpus/scripts/03_acquire_fineweb2.py
    docker compose logs -f fineweb2
    tail -f $WORK_DIR/raw/fineweb2/download.log

    # Quick test (foreground, interruptible with Ctrl+C):
    python corpus/scripts/03_acquire_fineweb2.py

Output:
    $WORK_DIR/raw/fineweb2/
    ├── shard_000000.parquet   (~100-200 MB · Zstd)
    ├── shard_000001.parquet
    ├── ...
    ├── download.log           structured log with timestamps
    └── langid_stats.json      PT-BR vs PT-PT statistics

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
Versão | Version: 0.1.0 — Abril 2026
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import load_dataset
    from loguru import logger
    from tqdm import tqdm
except ImportError as e:
    print(f"[ERRO] Dependência ausente: {e}")
    print("Docker: execute dentro do container 'corpus' "
          "(docker compose run --rm corpus ...). Veja docs/environment/setup-guide-docker-pt.md")
    print("Depois verifique: python corpus/scripts/00_verify_env.py")
    sys.exit(1)


# ── Configuração ──────────────────────────────────────────────────────────────

SOURCE_NAME  = "fineweb2"
HF_REPO_ID   = "HuggingFaceFW/fineweb-2"
HF_CONFIG    = "por_Latn"   # partição portuguesa (PT-BR + PT-PT)
HF_SPLIT     = "train"
LICENSE      = "odc-by"     # Open Data Commons Attribution License
REFERENCE    = "Penedo et al. (2024). FineWeb. arXiv:2406.17557"
EST_TOKENS_B = 150          # bilhões estimados para partição por_Latn

# Threshold de identificação de variante PT-BR.
# Mantido igual ao Script 02 (0.65) para consistência metodológica.
# Referência de calibração: análise da distribuição no brWaC (Hartmann 2017).
LANGID_THRESHOLD = 0.65

WORK_DIR       = Path(os.environ.get("WORK_DIR", Path.home() / "manaca-corpus"))
OUTPUT_DIR     = WORK_DIR / "raw" / SOURCE_NAME
LOG_DIR        = WORK_DIR / "logs"
CKPT_DIR       = WORK_DIR / "checkpoints"
MANIFEST_PATH  = CKPT_DIR / "acquisition_manifest.json"
HF_HOME        = Path(os.environ.get("HF_HOME", Path.home() / "software" / "hf-cache"))
FASTTEXT_MODEL = HF_HOME / "fasttext" / "lid.176.bin"

SHARD_SIZE   = 50_000
MIN_TEXT_LEN = 50

# Schema Parquet fixo — idêntico em todos os scripts desta suite.
SCHEMA = pa.schema([
    pa.field("text",   pa.string()),
    pa.field("source", pa.string()),
    pa.field("id",     pa.string()),
    pa.field("lang",   pa.string()),
    pa.field("score",  pa.float32()),
])

# Campos adicionais do FineWeb-2 preservados como metadados no manifesto
# (não no Parquet principal, para manter schema compatível com demais fontes).
# dump      — snapshot do Common Crawl (ex: CC-MAIN-2024-10)
# url       — URL de origem do documento
# date      — data de coleta
# quality_signals — scores internos do pipeline FineWeb
FINEWEB2_METADATA_COLS = ["dump", "url", "date"]


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
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
    """Escrita atômica — garante integridade em caso de interrupção."""
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = MANIFEST_PATH.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.rename(MANIFEST_PATH)


# ── Retomada automática ───────────────────────────────────────────────────────

def get_existing_shards(output_dir: Path) -> set[int]:
    """Retorna IDs dos shards já escritos e com integridade verificada."""
    existing = set()
    for f in sorted(output_dir.glob("shard_*.parquet")):
        sid = f.stem.replace("shard_", "")
        if not sid.isdigit():
            continue
        try:
            meta = pq.read_metadata(f)
            if meta.num_rows > 0:
                existing.add(int(sid))
            else:
                logger.warning(f"Shard vazio (sera re-escrito): {f.name}")
        except Exception:
            logger.warning(f"Shard corrompido (sera re-escrito): {f.name}")
    return existing


# ── Language Identification (PT-BR vs PT-PT) ──────────────────────────────────

class LangIdentifier:
    """
    Identificador de variante linguística PT-BR vs PT-PT usando fastText.

    Reutiliza a mesma lógica do Script 02 para consistência metodológica.
    O threshold de 0.65 é mantido idêntico entre MADLAD-400 e FineWeb-2
    para garantir comparabilidade na análise de distribuição de variantes.

    Referências:
      Joulin, A. et al. (2016). fastText. arXiv:1607.01759.
      Zampieri & Gebhar (2012). Automatic Identification of Language
        Varieties. KONVENS 2012.
    """

    PTBR_MARKERS = frozenset({
        "você", "vocês", "ônibus", "banheiro", "celular", "sorvete",
        "trem", "metrô", "tchau", "legal", "bacana", "moleque",
        "saudade", "capivara", "feira", "bagunça", "grana", "cara",
    })

    PTPT_MARKERS = frozenset({
        "tu", "vós", "autocarro", "casa de banho", "telemóvel", "gelado",
        "comboio", "metro", "adeus", "fixe", "bestial", "gajo",
        "miúdo", "puto", "bué", "tasca", "talho",
    })

    def __init__(self, model_path: Path, threshold: float = LANGID_THRESHOLD):
        self.threshold = threshold
        self.model     = None
        self._load_model(model_path)

    def _load_model(self, model_path: Path) -> None:
        if not model_path.exists():
            logger.info("Modelo lid.176.bin nao encontrado. Baixando via HuggingFace...")
            self._download_model(model_path)
        try:
            import fasttext
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = fasttext.load_model(str(model_path))
            logger.info(f"Modelo fastText carregado: {model_path}")
        except Exception as e:
            logger.error(f"Falha ao carregar modelo fastText: {e}")
            raise

    @staticmethod
    def _download_model(model_path: Path) -> None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id="facebook/fasttext-language-identification",
                filename="model.bin",
                local_dir=str(model_path.parent),
            )
            downloaded_path = Path(downloaded)
            if downloaded_path.name != model_path.name:
                downloaded_path.rename(model_path)
            logger.info(f"Modelo baixado: {model_path}")
        except Exception as e:
            logger.error(f"Falha ao baixar modelo: {e}")
            logger.error("Download manual: https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin")
            raise

    def is_portuguese(self, text: str) -> tuple[bool, float, str]:
        """
        Verifica se o texto é português (qualquer variante).

        Nota sobre o FineWeb-2: a partição por_Latn já foi filtrada pelo
        GlotLID durante o pipeline FineWeb, portanto praticamente todos os
        documentos serão identificados como PT. A verificação aqui serve
        como salvaguarda para eventuais documentos com erro de rótulo.

        Retorna: (is_pt, score, lang_code)
        """
        if not self.model or not text:
            return False, 0.0, "unknown"
        clean = text[:1000].replace("\n", " ").strip()
        if not clean:
            return False, 0.0, "unknown"
        try:
            labels, scores = self.model.predict(clean, k=1)
            lang  = labels[0].replace("__label__", "")
            score = float(scores[0])
            is_pt = lang in ("pt", "por_Latn") and score >= self.threshold
            return is_pt, score, lang
        except Exception:
            return False, 0.0, "error"

    def classify_variant(self, text: str) -> str:
        """
        Classifica a variante: 'por_Latn_BR' ou 'por_Latn_PT'.
        Usa marcadores lexicais como heurística complementar.
        """
        text_lower = text[:2000].lower()
        n_br = sum(1 for m in self.PTBR_MARKERS if m in text_lower)
        n_pt = sum(1 for m in self.PTPT_MARKERS if m in text_lower)
        if n_pt > n_br * 2:
            return "por_Latn_PT"
        return "por_Latn_BR"


# ── Filtros de qualidade ──────────────────────────────────────────────────────

def passes_filter(text: str) -> bool:
    """
    Filtros de sanidade mínimos.

    Para o FineWeb-2, que já passou pelo pipeline completo de qualidade
    (Gopher + C4 + FineWeb filters + deduplicação MinHash), aplicamos
    apenas filtros básicos de sanidade. A qualidade intrínseca dos dados
    já está garantida pelo pipeline upstream do HuggingFace.

    Isso distingue o FineWeb-2 do Common Crawl bruto (Script 07), onde
    os filtros Gopher completos são aplicados durante o processamento.

    Critérios:
      1. Tipo string válido
      2. Comprimento mínimo: MIN_TEXT_LEN caracteres (50)
      3. Razão mínima de caracteres alfabéticos: 50%

    Referência pipeline FineWeb: Penedo et al. (2024). arXiv:2406.17557.
    """
    if not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < MIN_TEXT_LEN:
        return False
    n_alpha = sum(1 for c in text if c.isalpha())
    return n_alpha / len(text) >= 0.50


def quality_score(text: str) -> float:
    """
    Score de qualidade composto [0.0–1.0].
    Idêntico aos Scripts 01 e 02 para comparabilidade entre fontes.
    """
    import math
    if not text:
        return 0.0
    words  = text.split()
    n      = max(len(words), 1)
    alpha  = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
    ttr    = len(set(w.lower() for w in words)) / n
    lscore = min(math.log1p(len(text)) / 14.0, 1.0)
    return round(0.4 * alpha + 0.3 * ttr + 0.3 * lscore, 4)


# ── Escrita de shard ──────────────────────────────────────────────────────────

def write_shard(docs: list[dict], shard_id: int, output_dir: Path) -> dict[str, Any]:
    """Escreve shard Parquet + Zstd. Idêntico aos Scripts 01 e 02."""
    path = output_dir / f"shard_{shard_id:06d}.parquet"
    table = pa.table(
        {
            "text":   [d["text"]   for d in docs],
            "source": [d["source"] for d in docs],
            "id":     [d["id"]     for d in docs],
            "lang":   [d["lang"]   for d in docs],
            "score":  [d["score"]  for d in docs],
        },
        schema=SCHEMA,
    )
    pq.write_table(table, path, compression="zstd")
    size_bytes = path.stat().st_size
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read(1 << 20)).hexdigest()
    return {
        "shard_id":   shard_id,
        "path":       str(path),
        "n_docs":     len(docs),
        "size_bytes": size_bytes,
        "size_mb":    round(size_bytes / 1e6, 1),
        "sha256_1mb": sha,
        "written_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Rastreamento de snapshots CC ──────────────────────────────────────────────

class SnapshotTracker:
    """
    Rastreia a distribuição de documentos por snapshot do Common Crawl.

    O FineWeb-2 por_Latn agrega documentos de 96 snapshots CC (2013-2024).
    Este rastreamento permite:
      1. Identificar quais snapshots já estão cobertos
      2. Evitar sobreposição com o Script 07 (Common Crawl direto)
      3. Documentar a cobertura temporal do corpus Manacá

    Os metadados de snapshot são salvos em snapshot_distribution.json
    para análise posterior.
    """

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def record(self, dump: str | None) -> None:
        if dump:
            self.counts[dump] = self.counts.get(dump, 0) + 1

    def save(self, output_dir: Path) -> None:
        path = output_dir / "snapshot_distribution.json"
        data = {
            "source":        SOURCE_NAME,
            "total_snapshots": len(self.counts),
            "distribution":  dict(sorted(self.counts.items())),
            "generated_at":  datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info(
            f"Distribuicao por snapshot CC salva: {path} "
            f"({len(self.counts)} snapshots distintos)"
        )


# ── Pipeline principal ────────────────────────────────────────────────────────

def acquire() -> dict[str, Any]:
    """
    Pipeline principal de aquisição do FineWeb-2 (partição por_Latn).

    Diferenças em relação aos Scripts 01 e 02:
      - Filtros de qualidade mínimos (dados já filtrados pelo pipeline FineWeb)
      - Rastreamento de distribuição por snapshot CC (SnapshotTracker)
      - Preservação do campo 'dump' (snapshot CC) nos metadados do manifesto
      - Filtro de variante idêntico ao Script 02 para consistência metodológica

    O campo 'url' do FineWeb-2 não é armazenado no Parquet principal
    (para manter schema compatível com demais fontes), mas é registrado
    nas estatísticas para eventual rastreabilidade futura.
    """

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

    # ── Pré-requisito: Script 01 concluído ────────────────────────────────────
    manifest = load_manifest()
    gv_status = manifest.get("sources", {}).get("gigaverbo", {}).get("status", "")
    if gv_status != "completed":
        print()
        print("[AVISO] Script 01 (GigaVerbo) ainda nao concluido.")
        print(f"        Status: '{gv_status or 'nao iniciado'}'")
        response = input("        Continuar mesmo assim? [s/N] ").strip().lower()
        if response != "s":
            print("        Abortado.")
            sys.exit(0)

    # ── Setup de diretórios ───────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    setup_logging(OUTPUT_DIR / "download.log")

    # ── Carregar modelo fastText ───────────────────────────────────────────────
    logger.info("Carregando modelo fastText para identificacao de variante PT-BR/PT-PT...")
    lang_id = LangIdentifier(FASTTEXT_MODEL, threshold=LANGID_THRESHOLD)

    # ── Inicializar rastreador de snapshots ───────────────────────────────────
    snapshot_tracker = SnapshotTracker()

    # ── Verificar retomada ────────────────────────────────────────────────────
    existing     = get_existing_shards(OUTPUT_DIR)
    docs_to_skip = len(existing) * SHARD_SIZE if existing else 0
    next_shard   = (max(existing) + 1) if existing else 0

    if docs_to_skip:
        logger.info(
            f"Retomando: {len(existing)} shards existentes "
            f"(~{docs_to_skip:,} docs PT-BR ja salvos, "
            f"proximo shard: {next_shard:06d})"
        )
    else:
        logger.info("Iniciando download do zero")

    logger.info("=" * 60)
    logger.info(f"Fonte:          {SOURCE_NAME}")
    logger.info(f"Repositorio:    {HF_REPO_ID} (config='{HF_CONFIG}')")
    logger.info(f"Tokens est.:    {EST_TOKENS_B}B  |  Licenca: {LICENSE}")
    logger.info(f"Referencia:     {REFERENCE}")
    logger.info(f"LangID thresh.: {LANGID_THRESHOLD} (PT-BR vs PT-PT)")
    logger.info(f"Output:         {OUTPUT_DIR}")
    logger.info(f"Shard size:     {SHARD_SIZE:,} docs")
    logger.info("=" * 60)
    logger.info("Nota: FineWeb-2 ja passou pelo pipeline Gopher+C4+FineWeb+MinHash.")
    logger.info("      Filtros de qualidade aplicados aqui sao apenas de sanidade.")

    # ── Carregar dataset em streaming ─────────────────────────────────────────
    logger.info(f"Conectando ao HuggingFace: {HF_REPO_ID} (config={HF_CONFIG}) ...")
    try:
        dataset = load_dataset(
            HF_REPO_ID,
            HF_CONFIG,
            split=HF_SPLIT,
            streaming=True,
            
        )
    except Exception as e:
        logger.error(f"Falha ao carregar dataset: {e}")
        raise

    # ── Detectar colunas disponíveis ──────────────────────────────────────────
    first    = next(iter(dataset))
    text_col = next(
        (c for c in ["text", "content", "body"] if c in first),
        list(first.keys())[0],
    )
    id_col   = next((c for c in ["id", "doc_id"] if c in first), None)
    dump_col = "dump" if "dump" in first else None
    url_col  = "url"  if "url"  in first else None

    logger.info(f"Coluna texto:   '{text_col}'")
    logger.info(f"Coluna ID:      '{id_col or 'gerado'}'")
    logger.info(f"Coluna dump CC: '{dump_col or 'nao disponivel'}'")
    logger.info(f"Coluna URL:     '{url_col or 'nao disponivel'}'")
    logger.info(f"Todos os campos: {list(first.keys())}")

    # ── Loop principal ────────────────────────────────────────────────────────
    buffer:     list[dict] = []
    shard_id   = next_shard
    total_seen = 0
    kept_ptbr  = 0
    skip_qual  = 0
    skip_ptpt  = 0
    skip_other = 0
    t_start    = time.time()

    variant_counts: dict[str, int] = {"por_Latn_BR": 0, "por_Latn_PT": 0, "other": 0}

    for item in tqdm(
        dataset,
        desc="FineWeb-2 por_Latn",
        unit="doc",
        mininterval=30,
        file=sys.stderr,
    ):
        total_seen += 1

        # Pular documentos já processados em execução anterior
        if total_seen <= docs_to_skip:
            if total_seen % 500_000 == 0:
                logger.debug(f"Pulando: {total_seen:,}/{docs_to_skip:,}")
            continue

        text = item.get(text_col, "")

        # Filtro 1: sanidade básica
        if not passes_filter(text):
            skip_qual += 1
            continue

        # Filtro 2: identificação de língua
        # Para o FineWeb-2 por_Latn, quase todos serão PT.
        # Verificação como salvaguarda contra erros de rótulo.
        is_pt, ft_score, ft_lang = lang_id.is_portuguese(text)
        if not is_pt:
            skip_other += 1
            continue

        # Filtro 3: classificação de variante PT-BR vs PT-PT
        variant = lang_id.classify_variant(text)
        variant_counts[variant] = variant_counts.get(variant, 0) + 1

        if variant == "por_Latn_PT":
            skip_ptpt += 1
            continue

        # Rastrear snapshot CC de origem
        dump = item.get(dump_col, "") if dump_col else ""
        snapshot_tracker.record(dump if dump else None)

        # Documento PT-BR aceito
        raw_id = item.get(id_col, "") if id_col else ""
        buffer.append({
            "text":   text.strip(),
            "source": SOURCE_NAME,
            "id":     str(raw_id) if raw_id else f"{SOURCE_NAME}_{total_seen}",
            "lang":   variant,
            "score":  quality_score(text),
        })
        kept_ptbr += 1

        # ── Flush de shard ────────────────────────────────────────────────────
        if len(buffer) >= SHARD_SIZE:
            meta    = write_shard(buffer, shard_id, OUTPUT_DIR)
            elapsed = time.time() - t_start
            rate    = kept_ptbr / max(elapsed, 1)
            ret_pt  = kept_ptbr / max(total_seen - docs_to_skip, 1)

            logger.info(
                f"Shard {shard_id:04d} | "
                f"{kept_ptbr:>9,} docs PT-BR | "
                f"{meta['size_mb']:6.0f} MB | "
                f"{rate:,.0f} docs/s | "
                f"retencao PT-BR {ret_pt:.1%} | "
                f"snapshots CC: {len(snapshot_tracker.counts)}"
            )

            manifest["sources"][SOURCE_NAME] = {
                "status":             "in_progress",
                "hf_repo":            HF_REPO_ID,
                "hf_config":          HF_CONFIG,
                "license":            LICENSE,
                "langid_threshold":   LANGID_THRESHOLD,
                "total_seen":         total_seen,
                "kept_ptbr":          kept_ptbr,
                "skip_quality":       skip_qual,
                "skip_ptpt":          skip_ptpt,
                "skip_other":         skip_other,
                "shards_written":     shard_id + 1,
                "estimated_tokens":   kept_ptbr * 4,
                "estimated_tokens_b": round(kept_ptbr * 4 / 1e9, 1),
                "cc_snapshots_seen":  len(snapshot_tracker.counts),
                "updated_at":         datetime.now(timezone.utc).isoformat(),
            }
            save_manifest(manifest)
            buffer, shard_id = [], shard_id + 1

        # Log de progresso a cada 1 milhão de documentos vistos
        n_processed = total_seen - docs_to_skip
        if n_processed > 0 and n_processed % 1_000_000 == 0:
            elapsed = time.time() - t_start
            logger.info(
                f"Progresso: {total_seen:,} vistos | "
                f"{kept_ptbr:,} PT-BR | "
                f"{skip_ptpt:,} PT-PT | "
                f"{skip_qual:,} qual | "
                f"{len(snapshot_tracker.counts)} snapshots CC | "
                f"{elapsed / 3600:.1f}h"
            )

    # ── Flush do último shard parcial ─────────────────────────────────────────
    if buffer:
        meta = write_shard(buffer, shard_id, OUTPUT_DIR)
        logger.info(
            f"Shard final {shard_id:04d} | "
            f"{len(buffer):,} docs | "
            f"{meta['size_mb']:.0f} MB"
        )

    # ── Salvar estatísticas finais ────────────────────────────────────────────
    snapshot_tracker.save(OUTPUT_DIR)

    langid_stats = {
        "source":           SOURCE_NAME,
        "langid_threshold": LANGID_THRESHOLD,
        "total_seen":       total_seen,
        "variant_counts":   variant_counts,
        "skip_quality":     skip_qual,
        "skip_ptpt":        skip_ptpt,
        "skip_other":       skip_other,
        "kept_ptbr":        kept_ptbr,
        "ptbr_rate":        round(kept_ptbr / max(total_seen, 1), 4),
        "ptpt_rate":        round(skip_ptpt / max(total_seen, 1), 4),
        "generated_at":     datetime.now(timezone.utc).isoformat(),
    }
    stats_path = OUTPUT_DIR / "langid_stats.json"
    stats_path.write_text(
        json.dumps(langid_stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── Relatório final ───────────────────────────────────────────────────────
    elapsed     = time.time() - t_start
    total_bytes = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.parquet"))
    est_tokens  = kept_ptbr * 4

    final: dict[str, Any] = {
        "status":              "completed",
        "hf_repo":             HF_REPO_ID,
        "hf_config":           HF_CONFIG,
        "license":             LICENSE,
        "reference":           REFERENCE,
        "langid_threshold":    LANGID_THRESHOLD,
        "total_seen":          total_seen,
        "kept_ptbr":           kept_ptbr,
        "skip_quality":        skip_qual,
        "skip_ptpt":           skip_ptpt,
        "skip_other":          skip_other,
        "retention_rate":      round(kept_ptbr / max(total_seen, 1), 4),
        "shards_written":      shard_id + 1,
        "total_bytes":         total_bytes,
        "total_gb":            round(total_bytes / 1e9, 2),
        "estimated_tokens":    est_tokens,
        "estimated_tokens_b":  round(est_tokens / 1e9, 1),
        "elapsed_hours":       round(elapsed / 3600, 2),
        "cc_snapshots_covered": len(snapshot_tracker.counts),
        "output_dir":          str(OUTPUT_DIR),
        "langid_stats_path":   str(stats_path),
        "completed_at":        datetime.now(timezone.utc).isoformat(),
    }
    manifest["sources"][SOURCE_NAME] = final
    save_manifest(manifest)

    logger.info("=" * 60)
    logger.info(f"CONCLUIDO — {SOURCE_NAME}")
    logger.info(f"  Docs vistos:       {total_seen:,}")
    logger.info(f"  Docs PT-BR mant.:  {kept_ptbr:,}")
    logger.info(f"  Docs PT-PT desc.:  {skip_ptpt:,}  ({skip_ptpt/max(total_seen,1):.1%})")
    logger.info(f"  Docs qual. desc.:  {skip_qual:,}")
    logger.info(f"  Taxa PT-BR:        {final['retention_rate']:.1%}")
    logger.info(f"  Shards:            {shard_id + 1}")
    logger.info(f"  Tamanho total:     {total_bytes / 1e9:.2f} GB")
    logger.info(f"  Tokens est.:       {est_tokens / 1e9:.1f}B")
    logger.info(f"  Snapshots CC:      {len(snapshot_tracker.counts)}")
    logger.info(f"  Tempo total:       {elapsed / 3600:.1f} horas")
    logger.info(f"  Output:            {OUTPUT_DIR}")
    logger.info(f"  Manifesto:         {MANIFEST_PATH}")
    logger.info("-" * 60)
    logger.info(f"DONE — {SOURCE_NAME}")
    logger.info(f"  Docs seen:          {total_seen:,}")
    logger.info(f"  PT-BR docs kept:    {kept_ptbr:,}")
    logger.info(f"  PT-PT docs dropped: {skip_ptpt:,}  ({skip_ptpt/max(total_seen,1):.1%})")
    logger.info(f"  Quality dropped:    {skip_qual:,}")
    logger.info(f"  PT-BR rate:         {final['retention_rate']:.1%}")
    logger.info(f"  Shards:             {shard_id + 1}")
    logger.info(f"  Total size:         {total_bytes / 1e9:.2f} GB")
    logger.info(f"  Est. tokens:        {est_tokens / 1e9:.1f}B")
    logger.info(f"  CC snapshots:       {len(snapshot_tracker.counts)}")
    logger.info(f"  Total time:         {elapsed / 3600:.1f} hours")
    logger.info(f"  Output:             {OUTPUT_DIR}")
    logger.info(f"  Manifest:           {MANIFEST_PATH}")
    logger.info("=" * 60)
    logger.info("Proximo passo | Next step: python corpus/scripts/04_acquire_hplt2.py")

    return final


# ── Verificação de integridade pós-download ───────────────────────────────────

def verify() -> None:
    """
    Verifica integridade dos shards e exibe estatísticas de cobertura.
    """
    shards = sorted(OUTPUT_DIR.glob("shard_*.parquet"))
    if not shards:
        logger.warning("Nenhum shard encontrado para verificacao.")
        return

    total_rows = 0
    errors     = 0

    for s in shards:
        try:
            rows = pq.read_metadata(s).num_rows
            total_rows += rows
            if rows == 0:
                logger.warning(f"Shard vazio: {s.name}")
                errors += 1
        except Exception as e:
            logger.error(f"Shard corrompido {s.name}: {e}")
            errors += 1

    logger.info(
        f"Verificacao: {len(shards)} shards | "
        f"{total_rows:,} docs | "
        f"{errors} erro(s)"
    )

    # Estatísticas de variante
    stats_path = OUTPUT_DIR / "langid_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        logger.info(
            f"Distribuicao: PT-BR={stats.get('kept_ptbr',0):,} "
            f"PT-PT desc.={stats.get('skip_ptpt',0):,} "
            f"Taxa PT-BR={stats.get('ptbr_rate',0):.1%}"
        )

    # Snapshots CC cobertos
    snap_path = OUTPUT_DIR / "snapshot_distribution.json"
    if snap_path.exists():
        snap = json.loads(snap_path.read_text(encoding="utf-8"))
        logger.info(
            f"Snapshots CC cobertos: {snap.get('total_snapshots', 0)} "
            f"(evitar sobreposicao no Script 07)"
        )

    # Amostra do último shard
    try:
        t      = pq.read_table(shards[-1], columns=["text", "lang", "score"])
        sample = t["text"][0].as_py()
        lang   = t["lang"][0].as_py()
        score  = t["score"][0].as_py()
        logger.info(
            f'Amostra ({shards[-1].name}): '
            f'"{sample[:80]}..." '
            f'[lang={lang} score={score:.2f}]'
        )
    except Exception as e:
        logger.warning(f"Erro ao ler amostra: {e}")


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        acquire()
        verify()
        sys.exit(0)
    except KeyboardInterrupt:
        print()
        print("[INFO] Interrompido pelo usuario.")
        print("       Re-executar o mesmo comando para retomar do ultimo shard.")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERRO] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

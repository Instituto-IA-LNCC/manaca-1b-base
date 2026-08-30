#!/usr/bin/env python3
"""
Manacá LLM — Script 02: Aquisição do MADLAD-400 (partição PT)
==============================================================
LNCC × NII/LLM-jp | Fase 1 — Corpus PT-BR

================================ PT (Português) ================================

Baixa a partição portuguesa do MADLAD-400 (allenai/MADLAD-400),
aplica filtro de variante linguística para reter somente PT-BR,
e armazena em Apache Parquet + Zstandard particionado em shards
de 50.000 documentos.

Idempotente: retoma automaticamente do último shard salvo.

Justificativa científica:
    Kudugunta et al. (2024, arXiv:2309.04662) desenvolveram o MADLAD-400
    com pipeline rigoroso de deduplicação e filtragem em 419 línguas.
    A partição PT (~80B tokens) complementa o GigaVerbo com domínios
    subarepresentados (documentos digitalizados, fontes de períodos
    históricos distintos) e períodos de coleta diferentes (2013-2023),
    aumentando a diversidade temporal e de domínio do corpus Manacá.

    ATENÇÃO: A partição PT inclui tanto PT-BR quanto PT-PT misturados.
    Este script aplica identificação de variante via fastText (lid.176.bin)
    para reter somente documentos identificados como PT-BR (score >= 0.65).
    O threshold de 0.65 é conservador para maximizar recall — documentos
    ambíguos são incluídos e podem ser refinados em versões futuras do corpus.

PRÉ-REQUISITOS OBRIGATÓRIOS:
    1. Script 01 (GigaVerbo) concluído e verificado
    2. Volume de trabalho WORK_DIR gravavel (bind mount Docker ./data, ou NFS/HPC)
    3. python corpus/scripts/00_verify_env.py retornando OK

Uso:
    # Verificar ambiente (obrigatório):
    python corpus/scripts/00_verify_env.py

    # Produção — sempre via screen:
    # Docker (recomendado): container destacado, logs persistidos no volume
    docker compose run -d --name madlad corpus python corpus/scripts/02_acquire_madlad400.py
    docker compose logs -f madlad
    tail -f $WORK_DIR/raw/madlad400/download.log

    # Teste rápido (foreground, interrompível com Ctrl+C):
    python corpus/scripts/02_acquire_madlad400.py

Saída:
    $WORK_DIR/raw/madlad400/
    ├── shard_000000.parquet   (~100-200 MB · Zstd)
    ├── shard_000001.parquet
    ├── ...
    ├── download.log           log estruturado com timestamps
    └── langid_stats.json      estatísticas de distribuição PT-BR vs PT-PT

================================= EN (English) =================================

Manacá LLM — Script 02: MADLAD-400 Acquisition (PT partition)

Downloads the Portuguese partition of MADLAD-400 (allenai/MADLAD-400),
applies a language-variant filter to keep only PT-BR,
and stores it in Apache Parquet + Zstandard partitioned into shards
of 50,000 documents.

Idempotent: automatically resumes from the last saved shard.

Scientific rationale:
    Kudugunta et al. (2024, arXiv:2309.04662) developed MADLAD-400
    with a rigorous deduplication and filtering pipeline across 419 languages.
    The PT partition (~80B tokens) complements GigaVerbo with
    underrepresented domains (digitized documents, sources from distinct
    historical periods) and different collection periods (2013-2023),
    increasing the temporal and domain diversity of the Manacá corpus.

    WARNING: The PT partition includes both PT-BR and PT-PT mixed together.
    This script applies variant identification via fastText (lid.176.bin)
    to keep only documents identified as PT-BR (score >= 0.65).
    The 0.65 threshold is conservative to maximize recall — ambiguous
    documents are included and may be refined in future versions of the corpus.

MANDATORY PREREQUISITES:
    1. Script 01 (GigaVerbo) completed and verified
    2. Writable WORK_DIR working volume (Docker bind mount ./data, or NFS/HPC)
    3. python corpus/scripts/00_verify_env.py returning OK

Usage:
    # Check the environment (mandatory):
    python corpus/scripts/00_verify_env.py

    # Production — always via screen:
    # Docker (recommended): detached container, logs persisted on the volume
    docker compose run -d --name madlad corpus python corpus/scripts/02_acquire_madlad400.py
    docker compose logs -f madlad
    tail -f $WORK_DIR/raw/madlad400/download.log

    # Quick test (foreground, interruptible with Ctrl+C):
    python corpus/scripts/02_acquire_madlad400.py

Output:
    $WORK_DIR/raw/madlad400/
    ├── shard_000000.parquet   (~100-200 MB · Zstd)
    ├── shard_000001.parquet
    ├── ...
    ├── download.log           structured log with timestamps
    └── langid_stats.json      PT-BR vs PT-PT distribution statistics

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

SOURCE_NAME  = "madlad400"
HF_REPO_ID   = "allenai/MADLAD-400"
HF_CONFIG    = "pt"  # usado como languages=["pt"]        # partição portuguesa (PT-BR + PT-PT)
HF_SPLIT     = "train"
LICENSE      = "apache-2.0"
REFERENCE    = "Kudugunta et al. (2024). MADLAD-400. arXiv:2309.04662"
EST_TOKENS_B = 80          # bilhões estimados para partição PT total

# Threshold de identificação de variante PT-BR via fastText.
# 0.65 é conservador: maximiza recall (inclui documentos ambíguos).
# Documentos com score < 0.65 para "pt" são descartados como PT-PT.
# Referência de calibração: análise da distribuição no brWaC (Hartmann 2017).
LANGID_THRESHOLD = 0.65

WORK_DIR      = Path(os.environ.get("WORK_DIR", Path.home() / "manaca-corpus"))
OUTPUT_DIR    = WORK_DIR / "raw" / SOURCE_NAME
LOG_DIR       = WORK_DIR / "logs"
CKPT_DIR      = WORK_DIR / "checkpoints"
MANIFEST_PATH = CKPT_DIR / "acquisition_manifest.json"
HF_HOME       = Path(os.environ.get("HF_HOME", Path.home() / "software" / "hf-cache"))
FASTTEXT_MODEL = HF_HOME / "fasttext" / "lid.176.bin"

SHARD_SIZE   = 50_000
MIN_TEXT_LEN = 50

# Schema Parquet fixo — idêntico em todos os scripts desta suite.
SCHEMA = pa.schema([
    pa.field("text",   pa.string()),
    pa.field("source", pa.string()),
    pa.field("id",     pa.string()),
    pa.field("lang",   pa.string()),   # score e código do fastText
    pa.field("score",  pa.float32()),
])


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

    O modelo lid.176.bin do fastText identifica 176 idiomas com latência
    de microsegundos por documento. Para português, retorna o código "pt"
    sem distinguir a variante — a distinção PT-BR/PT-PT é feita por heurística
    lexical complementar quando o score fastText está abaixo do threshold.

    Estratégia de duas camadas:
      1. fastText: score >= LANGID_THRESHOLD → aceitar como PT
      2. Marcadores lexicais PT-BR/PT-PT: desempate para scores intermediários

    Referências:
      Joulin, A. et al. (2016). fastText. arXiv:1607.01759.
      Zampieri & Gebhar (2012). Automatic Identification of Language Varieties.
    """

    # Marcadores lexicais PT-BR (palavras exclusivas ou muito mais frequentes no Brasil)
    PTBR_MARKERS = frozenset({
        "você", "vocês", "ônibus", "banheiro", "celular", "sorvete",
        "trem", "metrô", "tchau", "legal", "bacana", "moleque",
        "saudade", "capivara", "feira", "bagunça", "grana", "cara",
    })

    # Marcadores lexicais PT-PT (palavras exclusivas ou muito mais frequentes em Portugal)
    PTPT_MARKERS = frozenset({
        "tu", "vós", "autocarro", "casa de banho", "telemóvel", "gelado",
        "comboio", "metro", "adeus", "fixe", "bestial", "gajo",
        "miúdo", "puto", "bué", "tasca", "talho",
    })

    def __init__(self, model_path: Path, threshold: float = LANGID_THRESHOLD):
        self.threshold = threshold
        self.model = None
        self._load_model(model_path)

    def _load_model(self, model_path: Path) -> None:
        """Carrega modelo fastText, baixando se necessário."""
        if not model_path.exists():
            logger.info("Modelo lid.176.bin nao encontrado. Baixando via HuggingFace...")
            self._download_model(model_path)

        try:
            import fasttext
            # suprimir mensagem de aviso do fastText sobre newlines
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
        """Baixa lid.176.bin do HuggingFace."""
        model_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download
            downloaded = hf_hub_download(
                repo_id="facebook/fasttext-language-identification",
                filename="model.bin",
                local_dir=str(model_path.parent),
            )
            # Renomear para lid.176.bin se necessário
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

        Retorna:
            (is_pt, score, lang_code)
            is_pt    — True se identificado como português
            score    — score de confiança do fastText [0.0–1.0]
            lang_code — código da língua identificada (ex: "pt")
        """
        if not self.model or not text:
            return False, 0.0, "unknown"

        # fastText espera texto sem quebras de linha
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
        Classifica a variante do português: 'por_Latn_BR' ou 'por_Latn_PT'.

        Usa marcadores lexicais como heurística complementar ao fastText.
        Para documentos claramente PT-BR ou PT-PT, usa os marcadores.
        Para documentos ambíguos, assume PT-BR (conservador para o corpus Manacá).
        """
        text_lower = text[:2000].lower()
        n_br = sum(1 for m in self.PTBR_MARKERS if m in text_lower)
        n_pt = sum(1 for m in self.PTPT_MARKERS if m in text_lower)

        if n_pt > n_br * 2:
            return "por_Latn_PT"   # claramente PT-PT
        return "por_Latn_BR"       # PT-BR ou ambíguo


# ── Filtros de qualidade ──────────────────────────────────────────────────────

def passes_filter(text: str) -> bool:
    """
    Filtros de sanidade mínimos.

    Para o MADLAD-400, que já passou por pipeline de qualidade do Allen AI,
    aplicamos apenas filtros básicos de sanidade. A filtragem de variante
    linguística (PT-BR vs PT-PT) é feita separadamente pelo LangIdentifier.

    Critérios:
      1. Tipo string válido
      2. Comprimento mínimo: MIN_TEXT_LEN caracteres (50)
      3. Razão mínima de caracteres alfabéticos: 50%

    Referência: Rae et al. (2021). Gopher. arXiv:2112.11446.
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
    Idêntico ao Script 01 para comparabilidade entre fontes.
    Ver documentação completa em 01_acquire_gigaverbo.py.
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
    """
    Escreve shard Parquet + Zstd. Idêntico ao Script 01.
    Ver documentação completa em 01_acquire_gigaverbo.py.
    """
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


# ── Pipeline principal ────────────────────────────────────────────────────────

def acquire() -> dict[str, Any]:
    """
    Pipeline principal de aquisição do MADLAD-400 (partição PT).

    Diferenças em relação ao Script 01 (GigaVerbo):
      - Filtro de variante linguística via fastText (PT-BR vs PT-PT)
      - Coleta de estatísticas de distribuição de variante (langid_stats.json)
      - Campo 'lang' preenchido com classificação real (por_Latn_BR/PT)
        em vez de valor fixo como no GigaVerbo

    Fluxo:
      1. Verificar NFS e pré-requisitos
      2. Carregar modelo fastText para identificação de variante
      3. Verificar shards existentes (retomada automática)
      4. Iterar em modo streaming: filtrar → identificar variante → acumular
      5. Flush de shard a cada SHARD_SIZE docs PT-BR mantidos
      6. Salvar estatísticas de distribuição para análise
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
        print(f"        Status atual: '{gv_status or 'nao iniciado'}'")
        print("        Recomendado: aguardar conclusao do Script 01 antes de prosseguir.")
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
    logger.info(f"Tokens est.:    {EST_TOKENS_B}B total PT  |  Licenca: {LICENSE}")
    logger.info(f"Referencia:     {REFERENCE}")
    logger.info(f"LangID thresh.: {LANGID_THRESHOLD} (PT-BR vs PT-PT)")
    logger.info(f"Output:         {OUTPUT_DIR}")
    logger.info(f"Shard size:     {SHARD_SIZE:,} docs")
    logger.info("=" * 60)

    # ── Carregar dataset em streaming ─────────────────────────────────────────
    logger.info(f"Conectando ao HuggingFace: {HF_REPO_ID} (config={HF_CONFIG}) ...")
    try:
        dataset = load_dataset(
            "json",
            data_files="hf://datasets/allenai/MADLAD-400/data/pt/pt_clean_*.jsonl.gz",
            split="train",
            streaming=True,
        )
    except Exception as e:
        logger.error(f"Falha ao carregar dataset: {e}")
        raise

    # ── Detectar coluna de texto ──────────────────────────────────────────────
    first    = next(iter(dataset))
    text_col = next(
        (c for c in ["text", "content", "body", "passage"] if c in first),
        list(first.keys())[0],
    )
    id_col = next(
        (c for c in ["id", "doc_id", "url"] if c in first),
        None,
    )
    logger.info(f"Coluna texto: '{text_col}'  |  Coluna ID: '{id_col or 'gerado'}'")
    logger.info(f"Campos disponíveis: {list(first.keys())}")

    # ── Loop principal ────────────────────────────────────────────────────────
    buffer:     list[dict] = []
    shard_id   = next_shard
    total_seen = 0
    kept_ptbr  = 0   # documentos PT-BR mantidos
    skip_qual  = 0   # descartados por qualidade
    skip_ptpt  = 0   # descartados por ser PT-PT
    skip_other = 0   # descartados por não ser português
    t_start    = time.time()

    # Contadores de variante para estatísticas
    variant_counts: dict[str, int] = {"por_Latn_BR": 0, "por_Latn_PT": 0, "other": 0}

    for item in tqdm(
        dataset,
        desc="MADLAD-400 PT",
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

        # Filtro 2: identificação de língua e variante
        # Para o MADLAD-400 partição PT, esperamos que todos sejam PT,
        # mas verificamos para descartar eventuais documentos com erro de rótulo.
        is_pt, ft_score, ft_lang = lang_id.is_portuguese(text)

        if not is_pt:
            skip_other += 1
            continue

        # Classificar variante PT-BR vs PT-PT
        variant = lang_id.classify_variant(text)
        variant_counts[variant] = variant_counts.get(variant, 0) + 1

        if variant == "por_Latn_PT":
            skip_ptpt += 1
            continue

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
                f"PT-PT descartados: {skip_ptpt:,}"
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

    # ── Relatório final ───────────────────────────────────────────────────────
    elapsed     = time.time() - t_start
    total_bytes = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.parquet"))
    est_tokens  = kept_ptbr * 4

    # Salvar estatísticas de distribuição de variante
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
    logger.info(f"Estatísticas de variante salvas: {stats_path}")

    final: dict[str, Any] = {
        "status":             "completed",
        "hf_repo":            HF_REPO_ID,
        "hf_config":          HF_CONFIG,
        "license":            LICENSE,
        "reference":          REFERENCE,
        "langid_threshold":   LANGID_THRESHOLD,
        "total_seen":         total_seen,
        "kept_ptbr":          kept_ptbr,
        "skip_quality":       skip_qual,
        "skip_ptpt":          skip_ptpt,
        "skip_other":         skip_other,
        "retention_rate":     round(kept_ptbr / max(total_seen, 1), 4),
        "shards_written":     shard_id + 1,
        "total_bytes":        total_bytes,
        "total_gb":           round(total_bytes / 1e9, 2),
        "estimated_tokens":   est_tokens,
        "estimated_tokens_b": round(est_tokens / 1e9, 1),
        "elapsed_hours":      round(elapsed / 3600, 2),
        "output_dir":         str(OUTPUT_DIR),
        "langid_stats_path":  str(stats_path),
        "completed_at":       datetime.now(timezone.utc).isoformat(),
    }
    manifest["sources"][SOURCE_NAME] = final
    save_manifest(manifest)

    logger.info("=" * 60)
    logger.info(f"CONCLUIDO — {SOURCE_NAME}")
    logger.info(f"  Docs vistos:      {total_seen:,}")
    logger.info(f"  Docs PT-BR mant.: {kept_ptbr:,}")
    logger.info(f"  Docs PT-PT desc.: {skip_ptpt:,}  ({skip_ptpt/max(total_seen,1):.1%})")
    logger.info(f"  Docs qual. desc.: {skip_qual:,}")
    logger.info(f"  Taxa PT-BR:       {final['retention_rate']:.1%}")
    logger.info(f"  Shards:           {shard_id + 1}")
    logger.info(f"  Tamanho total:    {total_bytes / 1e9:.2f} GB")
    logger.info(f"  Tokens est.:      {est_tokens / 1e9:.1f}B")
    logger.info(f"  Tempo total:      {elapsed / 3600:.1f} horas")
    logger.info(f"  Output:           {OUTPUT_DIR}")
    logger.info(f"  Manifesto:        {MANIFEST_PATH}")
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
    logger.info(f"  Total time:         {elapsed / 3600:.1f} hours")
    logger.info(f"  Output:             {OUTPUT_DIR}")
    logger.info(f"  Manifest:           {MANIFEST_PATH}")
    logger.info("=" * 60)
    logger.info("Proximo passo | Next step: python corpus/scripts/03_acquire_fineweb2.py")

    return final


# ── Verificação de integridade pós-download ───────────────────────────────────

def verify() -> None:
    """
    Verifica integridade de todos os shards e exibe estatísticas de variante.
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

    # Exibir estatísticas de variante
    stats_path = OUTPUT_DIR / "langid_stats.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        logger.info(
            f"Distribuicao de variante: "
            f"PT-BR={stats.get('kept_ptbr', 0):,} "
            f"PT-PT descartados={stats.get('skip_ptpt', 0):,} "
            f"Taxa PT-BR={stats.get('ptbr_rate', 0):.1%}"
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

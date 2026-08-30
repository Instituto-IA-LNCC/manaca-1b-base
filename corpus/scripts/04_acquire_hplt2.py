#!/usr/bin/env python3
"""
Manacá LLM — Script 04: Aquisição do HPLT 2.0 (partição PT)
=============================================================
LNCC × NII/LLM-jp | Fase 1 — Corpus PT-BR

Baixa a partição portuguesa do HPLT 2.0 (HPLT/HPLT2.0_cleaned),
aplica filtro de variante linguística para reter somente PT-BR,
e armazena em Apache Parquet + Zstandard particionado em shards
de 50.000 documentos.

Idempotente: retoma automaticamente do último shard salvo.

Justificativa científica:
    de Gibert et al. (2024, arXiv:2403.05010) desenvolveram o HPLT 2.0
    (High Performance Language Technologies) como parte do consórcio
    europeu homônimo. A versão 2.0 passou por pipeline rigoroso de
    deduplicação e filtragem de qualidade, resultando em dados de alta
    densidade lexical.

    Três razões específicas para incluir o HPLT 2.0 além das demais fontes:

    1. COBERTURA TEMPORAL DISTINTA: O HPLT 2.0 inclui snapshots do Common
       Crawl de 2013 a 2023 com metodologia de filtragem diferente do
       FineWeb-2, capturando padrões linguísticos de períodos históricos
       que podem estar subarepresentados nas demais fontes.

    2. LICENÇA CC0 (DOMÍNIO PÚBLICO): É a licença mais permissiva possível,
       sem restrições de uso comercial ou de redistribuição. Relevante para
       a publicação aberta do corpus Manacá sob licença maximamente aberta.

    3. DIVERSIDADE LEXICAL: Análises internas do consórcio HPLT mostram que
       o pipeline de filtragem do HPLT 2.0 preserva maior diversidade de
       domínios de texto em comparação ao C4, resultando em vocabulário mais
       rico para línguas morfologicamente complexas como o português.

    ATENÇÃO: A partição PT inclui PT-BR e PT-PT. Este script aplica o mesmo
    pipeline de identificação de variante dos Scripts 02 e 03 (fastText +
    marcadores lexicais, threshold >= 0.65) para consistência metodológica.

PRÉ-REQUISITOS OBRIGATÓRIOS:
    1. Script 01 (GigaVerbo) concluído e verificado
    2. Volume de trabalho WORK_DIR gravavel (bind mount Docker ./data, ou NFS/HPC)
    3. python corpus/scripts/00_verify_env.py retornando OK

Uso:
    # Verificar ambiente (obrigatório):
    python corpus/scripts/00_verify_env.py

    # Produção — sempre via screen:
    # Docker (recomendado): container destacado, logs persistidos no volume
    docker compose run -d --name hplt2 corpus python corpus/scripts/04_acquire_hplt2.py
    docker compose logs -f hplt2
    tail -f $WORK_DIR/raw/hplt2/download.log

    # Teste rápido (foreground, interrompível com Ctrl+C):
    python corpus/scripts/04_acquire_hplt2.py

Saída:
    $WORK_DIR/raw/hplt2/
    ├── shard_000000.parquet   (~100-200 MB · Zstd)
    ├── shard_000001.parquet
    ├── ...
    ├── download.log           log estruturado com timestamps
    └── langid_stats.json      estatísticas PT-BR vs PT-PT

Autor: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
Versão: 0.1.0 — Abril 2026
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

SOURCE_NAME  = "hplt2"
HF_REPO_ID   = "HPLT/HPLT2.0_cleaned"
HF_CONFIG    = "por_Latn"         # partição portuguesa (PT-BR + PT-PT)
HF_SPLIT     = "train"
LICENSE      = "cc0"        # Creative Commons Zero — domínio público
REFERENCE    = "de Gibert et al. (2024). HPLT 2.0. arXiv:2403.05010"
EST_TOKENS_B = 60           # bilhões estimados para partição PT total

# Threshold de identificação de variante PT-BR.
# Mantido em 0.65 para consistência metodológica com Scripts 02 e 03.
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

# Campos específicos do HPLT 2.0 que serão registrados nos metadados
# (não no Parquet principal, para manter schema compatível entre fontes).
# collection — coleção de origem (ex: CC-MAIN-2013-20)
# url        — URL de origem do documento
# seg_langs  — línguas detectadas pelo pipeline HPLT
HPLT2_METADATA_COLS = ["collection", "url", "seg_langs"]


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

    Implementação idêntica aos Scripts 02 e 03 para consistência metodológica.
    O HPLT 2.0 já inclui o campo 'seg_langs' com detecção de idioma do próprio
    pipeline HPLT, mas optamos por reprocessar com fastText para garantir
    uniformidade com as demais fontes do corpus Manacá.

    Referências:
      Joulin, A. et al. (2016). fastText. arXiv:1607.01759.
      Zampieri & Gebhar (2012). Automatic Identification of Language Varieties.
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
            logger.error(
                "Download manual: "
                "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
            )
            raise

    def is_portuguese(self, text: str) -> tuple[bool, float, str]:
        """
        Verifica se o texto é português (qualquer variante).
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
        Classifica variante: 'por_Latn_BR' ou 'por_Latn_PT'.
        Usa marcadores lexicais como heurística complementar ao fastText.
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
    Filtros de sanidade mínimos para o HPLT 2.0.

    O HPLT 2.0 já passou por pipeline de qualidade do consórcio HPLT,
    portanto aplicamos apenas filtros básicos de sanidade, analogamente
    ao FineWeb-2 (Script 03).

    Critérios:
      1. Tipo string válido
      2. Comprimento mínimo: MIN_TEXT_LEN caracteres (50)
      3. Razão mínima de caracteres alfabéticos: 50%

    Referência pipeline HPLT: de Gibert et al. (2024). arXiv:2403.05010.
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
    Idêntico aos Scripts 01, 02 e 03 para comparabilidade entre fontes.
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
    """Escreve shard Parquet + Zstd. Idêntico aos Scripts 01, 02 e 03."""
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


# ── Rastreamento de coleções HPLT ─────────────────────────────────────────────

class CollectionTracker:
    """
    Rastreia a distribuição de documentos por coleção HPLT.

    O HPLT 2.0 organiza os dados em coleções correspondentes a snapshots
    do Common Crawl (ex: CC-MAIN-2013-20). Este rastreamento:
      1. Documenta a cobertura temporal do corpus Manacá
      2. Permite identificar sobreposição com FineWeb-2 (Script 03)
         e Common Crawl direto (Script 07)
      3. Fornece dado científico para o artigo da Fase 1

    Salvo em collection_distribution.json ao final da execução.
    """

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def record(self, collection: str | None) -> None:
        if collection:
            key = str(collection).strip()
            if key:
                self.counts[key] = self.counts.get(key, 0) + 1

    def save(self, output_dir: Path) -> None:
        path = output_dir / "collection_distribution.json"
        data = {
            "source":              SOURCE_NAME,
            "total_collections":   len(self.counts),
            "distribution":        dict(sorted(self.counts.items())),
            "generated_at":        datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            f"Distribuicao por colecao HPLT salva: {path} "
            f"({len(self.counts)} colecoes distintas)"
        )


# ── Pipeline principal ────────────────────────────────────────────────────────

def acquire() -> dict[str, Any]:
    """
    Pipeline principal de aquisição do HPLT 2.0 (partição PT).

    Diferenças em relação aos Scripts anteriores:
      - CollectionTracker: rastreia distribuição por coleção HPLT
        (análogo ao SnapshotTracker do Script 03)
      - Aproveitamento do campo 'seg_langs' do HPLT 2.0 como contexto
        adicional no log (sem alterar o schema Parquet)
      - Licença CC0 (mais permissiva que Apache 2.0 e ODC-By) registrada
        explicitamente no manifesto para rastreabilidade de licença por fonte

    Fluxo:
      1. Verificar NFS e pré-requisitos
      2. Carregar modelo fastText
      3. Verificar shards existentes (retomada)
      4. Iterar em streaming: filtrar → identificar variante → acumular
      5. Flush de shard a cada SHARD_SIZE docs PT-BR
      6. Salvar distribuição de coleções e estatísticas de variante
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

    # ── Inicializar rastreador de coleções ────────────────────────────────────
    coll_tracker = CollectionTracker()

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
    logger.info("Nota: HPLT 2.0 ja passou pelo pipeline de qualidade HPLT.")
    logger.info("      LangID reprocessado com fastText para uniformidade com demais fontes.")

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
    coll_col = "collection" if "collection" in first else None
    url_col  = "url"        if "url"        in first else None
    lang_col = "seg_langs"  if "seg_langs"  in first else None

    logger.info(f"Coluna texto:      '{text_col}'")
    logger.info(f"Coluna ID:         '{id_col or 'gerado'}'")
    logger.info(f"Coluna colecao:    '{coll_col or 'nao disponivel'}'")
    logger.info(f"Coluna seg_langs:  '{lang_col or 'nao disponivel'}'")
    logger.info(f"Todos os campos:   {list(first.keys())}")

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
        desc="HPLT 2.0 PT",
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

        # Filtro 2: identificação de língua via fastText
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

        # Rastrear coleção HPLT de origem
        collection = item.get(coll_col, "") if coll_col else ""
        coll_tracker.record(collection if collection else None)

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
                f"colecoes HPLT: {len(coll_tracker.counts)}"
            )

            manifest["sources"][SOURCE_NAME] = {
                "status":              "in_progress",
                "hf_repo":             HF_REPO_ID,
                "hf_config":           HF_CONFIG,
                "license":             LICENSE,
                "langid_threshold":    LANGID_THRESHOLD,
                "total_seen":          total_seen,
                "kept_ptbr":           kept_ptbr,
                "skip_quality":        skip_qual,
                "skip_ptpt":           skip_ptpt,
                "skip_other":          skip_other,
                "shards_written":      shard_id + 1,
                "estimated_tokens":    kept_ptbr * 4,
                "estimated_tokens_b":  round(kept_ptbr * 4 / 1e9, 1),
                "hplt_collections_seen": len(coll_tracker.counts),
                "updated_at":          datetime.now(timezone.utc).isoformat(),
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
                f"{len(coll_tracker.counts)} colecoes | "
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
    coll_tracker.save(OUTPUT_DIR)

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
        "status":                "completed",
        "hf_repo":               HF_REPO_ID,
        "hf_config":             HF_CONFIG,
        "license":               LICENSE,
        "reference":             REFERENCE,
        "langid_threshold":      LANGID_THRESHOLD,
        "total_seen":            total_seen,
        "kept_ptbr":             kept_ptbr,
        "skip_quality":          skip_qual,
        "skip_ptpt":             skip_ptpt,
        "skip_other":            skip_other,
        "retention_rate":        round(kept_ptbr / max(total_seen, 1), 4),
        "shards_written":        shard_id + 1,
        "total_bytes":           total_bytes,
        "total_gb":              round(total_bytes / 1e9, 2),
        "estimated_tokens":      est_tokens,
        "estimated_tokens_b":    round(est_tokens / 1e9, 1),
        "elapsed_hours":         round(elapsed / 3600, 2),
        "hplt_collections_covered": len(coll_tracker.counts),
        "output_dir":            str(OUTPUT_DIR),
        "langid_stats_path":     str(stats_path),
        "completed_at":          datetime.now(timezone.utc).isoformat(),
    }
    manifest["sources"][SOURCE_NAME] = final
    save_manifest(manifest)

    logger.info("=" * 60)
    logger.info(f"CONCLUIDO — {SOURCE_NAME}")
    logger.info(f"  Docs vistos:        {total_seen:,}")
    logger.info(f"  Docs PT-BR mant.:   {kept_ptbr:,}")
    logger.info(f"  Docs PT-PT desc.:   {skip_ptpt:,}  ({skip_ptpt/max(total_seen,1):.1%})")
    logger.info(f"  Docs qual. desc.:   {skip_qual:,}")
    logger.info(f"  Taxa PT-BR:         {final['retention_rate']:.1%}")
    logger.info(f"  Shards:             {shard_id + 1}")
    logger.info(f"  Tamanho total:      {total_bytes / 1e9:.2f} GB")
    logger.info(f"  Tokens est.:        {est_tokens / 1e9:.1f}B")
    logger.info(f"  Colecoes HPLT:      {len(coll_tracker.counts)}")
    logger.info(f"  Tempo total:        {elapsed / 3600:.1f} horas")
    logger.info(f"  Output:             {OUTPUT_DIR}")
    logger.info(f"  Manifesto:          {MANIFEST_PATH}")
    logger.info("=" * 60)
    logger.info("Proximo passo: python corpus/scripts/05_acquire_wikipedia.py")

    return final


# ── Verificação de integridade pós-download ───────────────────────────────────

def verify() -> None:
    """Verifica integridade dos shards e exibe estatísticas de cobertura."""
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

    # Coleções HPLT cobertas
    coll_path = OUTPUT_DIR / "collection_distribution.json"
    if coll_path.exists():
        coll = json.loads(coll_path.read_text(encoding="utf-8"))
        logger.info(
            f"Colecoes HPLT cobertas: {coll.get('total_collections', 0)}"
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

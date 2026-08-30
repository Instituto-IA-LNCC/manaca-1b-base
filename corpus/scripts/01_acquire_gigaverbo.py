#!/usr/bin/env python3
"""
Manacá LLM — Script 01: Aquisição do GigaVerbo
===============================================
LNCC × NII/LLM-jp | Fase 1 — Corpus PT-BR

Baixa TucanoBR/GigaVerbo em Apache Parquet + Zstandard,
particionado em shards de 50.000 documentos.
Idempotente: retoma automaticamente do último shard salvo.

Justificativa científica:
    De Lucena et al. (2024, arXiv:2411.07854) demonstram que o GigaVerbo
    como base de pré-treinamento supera corpora multilíngues maiores (mC4,
    CC-100) em benchmarks PT-BR — qualidade e especificidade superam volume
    bruto. 200B tokens · Apache 2.0 · sem autenticação.

PRÉ-REQUISITO OBRIGATÓRIO:
    1. Volume de trabalho WORK_DIR gravavel (bind mount Docker ./data, ou NFS/HPC)
    2. python corpus/scripts/00_verify_env.py retornando OK

Uso:
    # Verificar ambiente (obrigatório):
    python corpus/scripts/00_verify_env.py

    # Produção — sempre via screen:
    # Docker (recomendado): container destacado, logs persistidos no volume
    docker compose run -d --name gigaverbo corpus python corpus/scripts/01_acquire_gigaverbo.py
    docker compose logs -f gigaverbo
    tail -f $WORK_DIR/raw/gigaverbo/download.log

    # Teste rápido (foreground, interrompível com Ctrl+C):
    python corpus/scripts/01_acquire_gigaverbo.py

Saída:
    $WORK_DIR/raw/gigaverbo/
    ├── shard_000000.parquet   (~100-200 MB · Zstd)
    ├── shard_000001.parquet
    ├── ...
    ├── download.log           log estruturado com timestamps
    └── _manifest.json         metadados por shard (SHA-256, contagens)

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

SOURCE_NAME  = "gigaverbo"
HF_REPO_ID   = "TucanoBR/GigaVerbo"
HF_SPLIT     = "train"
LICENSE      = "apache-2.0"
REFERENCE    = "de Lucena et al. (2024). Tucano. arXiv:2411.07854"
EST_TOKENS_B = 200  # bilhões (estimativa)

WORK_DIR      = Path(os.environ.get("WORK_DIR", Path.home() / "manaca-corpus"))
OUTPUT_DIR    = WORK_DIR / "raw" / SOURCE_NAME
LOG_DIR       = WORK_DIR / "logs"
CKPT_DIR      = WORK_DIR / "checkpoints"
MANIFEST_PATH = CKPT_DIR / "acquisition_manifest.json"

SHARD_SIZE   = 50_000  # documentos por arquivo Parquet
MIN_TEXT_LEN = 50      # caracteres mínimos para aceitar um documento

# Schema Parquet fixo — idêntico em todos os scripts desta suite.
# Compatível com HuggingFace datasets.load_from_disk() e datatrove.
# Campos:
#   text   — texto limpo do documento
#   source — identificador da fonte (ex: "gigaverbo")
#   id     — id único do documento dentro da fonte
#   lang   — código de idioma detectado (GlotLID/fastText)
#   score  — score de qualidade [0.0–1.0] para desempate na deduplicação
SCHEMA = pa.schema([
    pa.field("text",   pa.string()),
    pa.field("source", pa.string()),
    pa.field("id",     pa.string()),
    pa.field("lang",   pa.string()),
    pa.field("score",  pa.float32()),
])


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_file: Path) -> None:
    """
    Configura loguru com dois destinos:
      - stderr: nível INFO, colorido, formato conciso para monitoramento
      - arquivo: nível DEBUG, rotação 200 MB, timestamps completos para auditoria
    """
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
    """Carrega manifesto existente ou cria novo."""
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
    """
    Escrita atômica via arquivo temporário.
    Garante que o manifesto nunca fica corrompido em caso de interrupção.
    """
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    tmp = MANIFEST_PATH.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.rename(MANIFEST_PATH)


# ── Retomada automática ───────────────────────────────────────────────────────

def get_existing_shards(output_dir: Path) -> set[int]:
    """
    Retorna IDs dos shards Parquet já escritos e com integridade verificada.
    Shards corrompidos (sem rows ou ilegíveis) são ignorados e serão re-escritos.
    Esta função é o mecanismo central de idempotência do script.
    """
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


# ── Filtros de qualidade ──────────────────────────────────────────────────────

def passes_filter(text: str) -> bool:
    """
    Filtros de sanidade mínimos para o GigaVerbo.

    O GigaVerbo já é um corpus curado pela equipe TucanoBR, portanto
    aplicamos apenas filtros básicos de sanidade — não os filtros Gopher
    completos que serão aplicados a dados web brutos no Script 07.

    Critérios aplicados:
      1. Tipo string válido (descartar None, int, listas)
      2. Comprimento mínimo: MIN_TEXT_LEN caracteres (50)
         Remove fragmentos muito curtos sem valor de treinamento.
      3. Razão mínima de caracteres alfabéticos: 50%
         Remove documentos dominados por código, símbolos ou números.

    Referência sobre filtros heurísticos:
      Rae et al. (2021). Scaling Language Models: Gopher. arXiv:2112.11446.
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
    Score de qualidade composto [0.0–1.0] para uso na deduplicação global.

    Componentes (pesos):
      - alpha_ratio  (0.4): proporção de caracteres alfabéticos
        Documentos com mais texto real pontuam mais alto.
      - ttr          (0.3): type-token ratio (diversidade lexical)
        Documentos com vocabulário mais variado pontuam mais alto.
      - length_score (0.3): comprimento log-normalizado
        Documentos mais longos (até certo limite) pontuam mais alto.

    Uso: na deduplicação global (Script 08), em um cluster de near-duplicatas
    identificadas pelo MinHash LSH, o documento com maior score é mantido e
    os demais são descartados. Não é um filtro — todos os documentos que
    passam em passes_filter() são incluídos independentemente do score.
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
    Escreve um shard Parquet com compressão Zstandard e retorna seus metadados.

    Formato Parquet + Zstd escolhido por:
      - Leitura colunar: processar só o campo 'text' sem ler os demais campos
      - Compressão Zstd: ~3-5x melhor que texto bruto com decompressão rápida
      - Compatibilidade: HuggingFace datasets, Apache Spark, DuckDB, Polars
      - Padrão da indústria: FineWeb (Penedo 2024) e Dolma (Soldaini 2024)

    O SHA-256 dos primeiros 1 MB serve para verificação rápida de integridade
    sem precisar ler o arquivo completo.
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
    Pipeline principal de aquisição do GigaVerbo.

    Fluxo de execução:
      1. Verificar NFS disponível (pré-requisito obrigatório)
      2. Verificar shards existentes → calcular docs a pular (retomada)
      3. Abrir dataset em modo streaming (sem carregar 780 GB na RAM)
      4. Para cada documento: filtrar → calcular score → acumular no buffer
      5. A cada SHARD_SIZE docs: gravar Parquet → atualizar manifesto (checkpoint)
      6. Ao final: flush do buffer parcial → relatório completo → verificação

    Sobre o modo streaming do HuggingFace datasets:
      streaming=True itera sobre os shards remotos incrementalmente via HTTP,
      mantendo apenas um batch na memória por vez. Essencial para datasets de
      centenas de GB em máquinas com RAM limitada.
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

    # ── Setup de diretórios ───────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    setup_logging(OUTPUT_DIR / "download.log")
    manifest = load_manifest()

    # ── Verificar retomada ────────────────────────────────────────────────────
    existing     = get_existing_shards(OUTPUT_DIR)
    docs_to_skip = len(existing) * SHARD_SIZE if existing else 0
    next_shard   = (max(existing) + 1) if existing else 0

    if docs_to_skip:
        logger.info(
            f"Retomando: {len(existing)} shards existentes "
            f"(~{docs_to_skip:,} docs ja salvos, "
            f"proximo shard: {next_shard:06d})"
        )
    else:
        logger.info("Iniciando download do zero")

    logger.info("=" * 60)
    logger.info(f"Fonte:        {SOURCE_NAME}")
    logger.info(f"Repositorio:  {HF_REPO_ID}")
    logger.info(f"Tokens est.:  {EST_TOKENS_B}B  |  Licenca: {LICENSE}")
    logger.info(f"Referencia:   {REFERENCE}")
    logger.info(f"Output:       {OUTPUT_DIR}")
    logger.info(f"Shard size:   {SHARD_SIZE:,} docs")
    logger.info(f"Manifesto:    {MANIFEST_PATH}")
    logger.info("=" * 60)

    # ── Carregar dataset em streaming ─────────────────────────────────────────
    logger.info(f"Conectando ao HuggingFace: {HF_REPO_ID} ...")
    try:
        dataset = load_dataset(
            HF_REPO_ID,
            split=HF_SPLIT,
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error(f"Falha ao carregar dataset: {e}")
        raise

    # ── Detectar coluna de texto automaticamente ──────────────────────────────
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

    # ── Loop principal ────────────────────────────────────────────────────────
    buffer:     list[dict] = []
    shard_id   = next_shard
    total_seen = 0
    kept       = 0
    skipped    = 0
    t_start    = time.time()

    for item in tqdm(
        dataset,
        desc="GigaVerbo",
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
        if not passes_filter(text):
            skipped += 1
            continue

        raw_id = item.get(id_col, "") if id_col else ""
        buffer.append({
            "text":   text.strip(),
            "source": SOURCE_NAME,
            "id":     str(raw_id) if raw_id else f"{SOURCE_NAME}_{total_seen}",
            "lang":   "por_Latn",  # GigaVerbo e PT-BR — LangID desnecessario
            "score":  quality_score(text),
        })
        kept += 1

        # ── Flush de shard ────────────────────────────────────────────────────
        if len(buffer) >= SHARD_SIZE:
            meta    = write_shard(buffer, shard_id, OUTPUT_DIR)
            elapsed = time.time() - t_start
            rate    = kept / max(elapsed, 1)
            ret     = kept / max(total_seen - docs_to_skip, 1)

            logger.info(
                f"Shard {shard_id:04d} | "
                f"{kept:>9,} docs | "
                f"{meta['size_mb']:6.0f} MB | "
                f"{rate:,.0f} docs/s | "
                f"retencao {ret:.1%}"
            )

            # Checkpoint a cada shard — permite retomada segura em caso de falha
            manifest["sources"][SOURCE_NAME] = {
                "status":             "in_progress",
                "hf_repo":            HF_REPO_ID,
                "license":            LICENSE,
                "total_seen":         total_seen,
                "total_kept":         kept,
                "total_skip":         skipped,
                "shards_written":     shard_id + 1,
                "estimated_tokens":   kept * 4,
                "estimated_tokens_b": round(kept * 4 / 1e9, 1),
                "updated_at":         datetime.now(timezone.utc).isoformat(),
            }
            save_manifest(manifest)

            buffer, shard_id = [], shard_id + 1

        # Log de progresso a cada 1 milhão de documentos
        n_processed = total_seen - docs_to_skip
        if n_processed > 0 and n_processed % 1_000_000 == 0:
            elapsed = time.time() - t_start
            logger.info(
                f"Progresso: {total_seen:,} vistos | "
                f"{kept:,} mantidos | "
                f"{skipped:,} descartados | "
                f"{elapsed / 3600:.1f}h decorridas"
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
    est_tokens  = kept * 4  # estimativa: ~4 chars/token para PT-BR

    final: dict[str, Any] = {
        "status":             "completed",
        "hf_repo":            HF_REPO_ID,
        "license":            LICENSE,
        "reference":          REFERENCE,
        "total_seen":         total_seen,
        "total_kept":         kept,
        "total_skipped":      skipped,
        "retention_rate":     round(kept / max(total_seen, 1), 4),
        "shards_written":     shard_id + 1,
        "total_bytes":        total_bytes,
        "total_gb":           round(total_bytes / 1e9, 2),
        "estimated_tokens":   est_tokens,
        "estimated_tokens_b": round(est_tokens / 1e9, 1),
        "elapsed_hours":      round(elapsed / 3600, 2),
        "output_dir":         str(OUTPUT_DIR),
        "completed_at":       datetime.now(timezone.utc).isoformat(),
    }
    manifest["sources"][SOURCE_NAME] = final
    save_manifest(manifest)

    logger.info("=" * 60)
    logger.info(f"CONCLUIDO — {SOURCE_NAME}")
    logger.info(f"  Docs mantidos:  {kept:,}")
    logger.info(f"  Docs ignorados: {skipped:,}")
    logger.info(f"  Taxa retencao:  {final['retention_rate']:.1%}")
    logger.info(f"  Shards:         {shard_id + 1}")
    logger.info(f"  Tamanho total:  {total_bytes / 1e9:.2f} GB")
    logger.info(f"  Tokens est.:    {est_tokens / 1e9:.1f}B")
    logger.info(f"  Tempo total:    {elapsed / 3600:.1f} horas")
    logger.info(f"  Output:         {OUTPUT_DIR}")
    logger.info(f"  Manifesto:      {MANIFEST_PATH}")
    logger.info("=" * 60)
    logger.info("Proximo passo: python corpus/scripts/02_acquire_madlad400.py")

    return final


# ── Verificação de integridade pós-download ───────────────────────────────────

def verify() -> None:
    """
    Verifica integridade de todos os shards escritos.

    Lê apenas os metadados Parquet (sem descomprimir os dados) para checar
    contagem de rows. Em seguida, lê uma amostra do último shard para
    confirmar que o conteúdo é legível e bem formado.
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

    # Amostra do último shard
    try:
        t      = pq.read_table(shards[-1], columns=["text", "source", "score"])
        sample = t["text"][0].as_py()
        score  = t["score"][0].as_py()
        logger.info(
            f'Amostra ({shards[-1].name}): '
            f'"{sample[:80]}..." '
            f'[score={score:.2f}]'
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

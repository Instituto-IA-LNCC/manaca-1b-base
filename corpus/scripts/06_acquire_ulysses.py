#!/usr/bin/env python3
"""
Manacá LLM — Script 06: Aquisição do Ulysses Tesemõ
=====================================================
LNCC × NII/LLM-jp | Fase 1 — Corpus PT-BR

================================ PT (Português) ================================

Baixa e processa o corpus Ulysses Tesemõ (github.com/ulysses-camara/
ulysses-tesemo), o maior corpus jurídico-legislativo em PT-BR, contendo
3,5 milhões de arquivos (30,7 GiB) provenientes de 159 fontes
governamentais brasileiras.

Idempotente: retoma automaticamente do último shard salvo.

Justificativa científica:
    O Ulysses Tesemõ é o principal produto do Projeto Ulysses (Câmara dos
    Deputados / UFRGS / UnB / UFAL), desenvolvido especificamente para
    suprir a lacuna de dados jurídico-legislativos em português brasileiro.

    Três razões estratégicas para incluir esta fonte no corpus Manacá:

    1. COBERTURA DE DOMÍNIO ESPECIALIZADO:
       Modelos de linguagem geral têm desempenho sistematicamente inferior
       em linguagem jurídica por causa do vocabulário técnico, estrutura
       sintática complexa e referências normativas específicas. A inclusão
       do Tesemõ habilita o Manacá para aplicações em direito, regulação,
       compliance e análise de políticas públicas — domínios críticos para
       soberania tecnológica nacional (Niklaus et al., 2022).

    2. DIVERSIDADE DE FONTES GOVERNAMENTAIS:
       159 fontes distintas cobrindo Câmara Federal, Senado, STF, STJ,
       TCU, ministérios e agências reguladoras. Esta diversidade de
       registros linguísticos (projetos de lei, acórdãos, portarias,
       decretos, emendas constitucionais) é qualitativamente diferente
       de qualquer outra fonte do corpus.

    3. LICENÇA PÚBLICA E RASTREABILIDADE:
       Documentos governamentais brasileiros são de domínio público por
       mandato constitucional (Art. 216, CF/1988). O Ulysses Tesemõ
       provê rastreabilidade completa de origem por documento.

    Referências:
      Nascimento, F. et al. (2023). Ulysses: Uma Suíte de Ferramentas de
        NLP para o Parlamento Brasileiro. STIL 2023.
      Niklaus, J. et al. (2022). A Survey of Corpora for Germanic Low-
        Resource Languages. LREC 2022. (referência metodológica para
        corpora jurídicos especializados)

DIFERENÇA FUNDAMENTAL em relação aos Scripts 01-05:
    Esta fonte não é um dataset HuggingFace — é um repositório GitHub
    com arquivos de texto organizados por fonte governamental. O pipeline
    é completamente diferente: clonagem Git → varredura de arquivos →
    limpeza de texto jurídico → conversão para Parquet.

PRÉ-REQUISITOS OBRIGATÓRIOS:
    1. Volume de trabalho WORK_DIR gravavel (bind mount Docker ./data, ou NFS/HPC)
    2. git instalado e acessível no PATH
    3. python corpus/scripts/00_verify_env.py retornando OK

Uso:
    # Verificar ambiente (obrigatório):
    python corpus/scripts/00_verify_env.py

    # Produção — via screen (repositório grande, clone demora):
    # Docker (recomendado): container destacado, logs persistidos no volume
    docker compose run -d --name ulysses corpus python corpus/scripts/06_acquire_ulysses.py
    docker compose logs -f ulysses
    tail -f $WORK_DIR/raw/ulysses/download.log

    # Execução direta (foreground):
    python corpus/scripts/06_acquire_ulysses.py

Saída:
    $WORK_DIR/raw/ulysses/
    ├── repo/                  clone do repositório GitHub
    ├── shard_000000.parquet   (~100-200 MB · Zstd)
    ├── ...
    ├── download.log
    ├── source_distribution.json   contagem de docs por fonte governamental
    └── domain_stats.json          estatísticas de qualidade jurídica

================================= EN (English) =================================

Manacá LLM — Script 06: Ulysses Tesemõ Acquisition

Downloads and processes the Ulysses Tesemõ corpus (github.com/ulysses-camara/
ulysses-tesemo), the largest legal-legislative corpus in PT-BR, containing
3.5 million files (30.7 GiB) from 159 Brazilian government
sources.

Idempotent: automatically resumes from the last saved shard.

Scientific rationale:
    Ulysses Tesemõ is the main product of the Ulysses Project (Câmara dos
    Deputados / UFRGS / UnB / UFAL), developed specifically to
    fill the gap of legal-legislative data in Brazilian Portuguese.

    Three strategic reasons to include this source in the Manacá corpus:

    1. SPECIALIZED-DOMAIN COVERAGE:
       General language models perform systematically worse
       on legal language because of the technical vocabulary, complex
       syntactic structure, and specific normative references. Including
       Tesemõ enables Manacá for applications in law, regulation,
       compliance, and public-policy analysis — domains critical to
       national technological sovereignty (Niklaus et al., 2022).

    2. DIVERSITY OF GOVERNMENT SOURCES:
       159 distinct sources covering the Federal Chamber, Senate, STF, STJ,
       TCU, ministries, and regulatory agencies. This diversity of
       linguistic registers (bills, rulings, ordinances,
       decrees, constitutional amendments) is qualitatively different
       from any other source in the corpus.

    3. PUBLIC LICENSE AND TRACEABILITY:
       Brazilian government documents are public domain by
       constitutional mandate (Art. 216, CF/1988). Ulysses Tesemõ
       provides complete provenance traceability per document.

    References:
      Nascimento, F. et al. (2023). Ulysses: Uma Suíte de Ferramentas de
        NLP para o Parlamento Brasileiro. STIL 2023.
      Niklaus, J. et al. (2022). A Survey of Corpora for Germanic Low-
        Resource Languages. LREC 2022. (methodological reference for
        specialized legal corpora)

FUNDAMENTAL DIFFERENCE relative to Scripts 01-05:
    This source is not a HuggingFace dataset — it is a GitHub repository
    with text files organized by government source. The pipeline
    is completely different: Git cloning → file scan →
    legal-text cleaning → conversion to Parquet.

MANDATORY PREREQUISITES:
    1. Writable WORK_DIR working volume (Docker bind mount ./data, or NFS/HPC)
    2. git installed and accessible on PATH
    3. python corpus/scripts/00_verify_env.py returning OK

Usage:
    # Check the environment (mandatory):
    python corpus/scripts/00_verify_env.py

    # Production — via screen (large repository, clone takes time):
    # Docker (recommended): detached container, logs persisted on the volume
    docker compose run -d --name ulysses corpus python corpus/scripts/06_acquire_ulysses.py
    docker compose logs -f ulysses
    tail -f $WORK_DIR/raw/ulysses/download.log

    # Direct execution (foreground):
    python corpus/scripts/06_acquire_ulysses.py

Output:
    $WORK_DIR/raw/ulysses/
    ├── repo/                  clone of the GitHub repository
    ├── shard_000000.parquet   (~100-200 MB · Zstd)
    ├── ...
    ├── download.log
    ├── source_distribution.json   doc count per government source
    └── domain_stats.json          legal-quality statistics

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
Versão | Version: 0.1.0 — Abril 2026
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
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
    print("Depois verifique: python corpus/scripts/00_verify_env.py")
    sys.exit(1)


# ── Configuração ──────────────────────────────────────────────────────────────

SOURCE_NAME   = "ulysses"
GITHUB_REPO   = "https://github.com/ulysses-camara/ulysses-tesemo.git"
GITHUB_BRANCH = "main"
LICENSE       = "public-domain"  # documentos governamentais brasileiros
REFERENCE     = "Nascimento et al. (2023). Ulysses. STIL 2023."
EST_TOKENS_B  = 10               # ~10 bilhões de tokens estimados

WORK_DIR      = Path(os.environ.get("WORK_DIR", Path.home() / "manaca-corpus"))
OUTPUT_DIR    = WORK_DIR / "raw" / SOURCE_NAME
REPO_DIR      = OUTPUT_DIR / "repo"
LOG_DIR       = WORK_DIR / "logs"
CKPT_DIR      = WORK_DIR / "checkpoints"
MANIFEST_PATH = CKPT_DIR / "acquisition_manifest.json"

SHARD_SIZE    = 50_000
MIN_TEXT_LEN  = 100   # textos jurídicos muito curtos raramente têm valor

# Extensões de arquivo aceitas no repositório Ulysses Tesemõ
ACCEPTED_EXTENSIONS = {".txt", ".json"}

# Schema Parquet fixo — idêntico em todos os scripts desta suite.
SCHEMA = pa.schema([
    pa.field("text",   pa.string()),
    pa.field("source", pa.string()),
    pa.field("id",     pa.string()),
    pa.field("lang",   pa.string()),
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
        rotation="100 MB",
        retention=3,
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


# ── Clonagem / atualização do repositório ────────────────────────────────────

def clone_or_update_repo() -> bool:
    """
    Clona o repositório Ulysses Tesemõ ou atualiza se já existir.

    O repositório (~30 GiB) é clonado com --depth=1 para evitar baixar
    o histórico completo do Git, reduzindo o tempo de download de horas
    para minutos e o espaço em disco necessário.

    Retorna True se o repositório está pronto para uso.
    """
    # Verificar git disponível
    if not _git_available():
        logger.error("git nao encontrado no PATH.")
        logger.error("Instalar: sudo apt-get install git")
        return False

    if REPO_DIR.exists() and (REPO_DIR / ".git").exists():
        logger.info(f"Repositorio ja existe em {REPO_DIR}. Atualizando...")
        try:
            result = subprocess.run(
                ["git", "-C", str(REPO_DIR), "pull", "--quiet", "--ff-only"],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                logger.info("Repositorio atualizado com sucesso.")
            else:
                logger.warning(
                    f"git pull retornou codigo {result.returncode}: "
                    f"{result.stderr[:200]}"
                )
                logger.warning("Prosseguindo com versao existente do repositorio.")
        except subprocess.TimeoutExpired:
            logger.warning("git pull timeout. Prosseguindo com versao existente.")
        return True

    logger.info(f"Clonando repositorio: {GITHUB_REPO}")
    logger.info(f"Destino: {REPO_DIR}")
    logger.info("--depth=1: apenas commit mais recente (economiza ~GB de historico)")
    logger.info("Isso pode levar 10-30 minutos dependendo da banda...")

    REPO_DIR.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            [
                "git", "clone",
                "--depth=1",
                "--branch", GITHUB_BRANCH,
                "--single-branch",
                GITHUB_REPO,
                str(REPO_DIR),
            ],
            capture_output=True,
            text=True,
            timeout=7200,  # 2 horas — repositório grande
        )
        if result.returncode != 0:
            logger.error(f"git clone falhou (code {result.returncode}):")
            logger.error(result.stderr[:500])
            return False

        logger.info(f"Clone concluido: {REPO_DIR}")
        return True

    except subprocess.TimeoutExpired:
        logger.error("git clone timeout (>2h). Verificar conectividade com GitHub.")
        return False
    except Exception as e:
        logger.error(f"Erro durante git clone: {e}")
        return False


def _git_available() -> bool:
    try:
        subprocess.run(
            ["git", "--version"],
            capture_output=True,
            timeout=10,
        )
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ── Limpeza de texto jurídico ─────────────────────────────────────────────────

# Padrões de limpeza específicos para texto jurídico-legislativo brasileiro
_RE_EMENTA    = re.compile(r"^EMENTA\s*[:.]?\s*", re.MULTILINE | re.IGNORECASE)
_RE_PROCESSO  = re.compile(
    r"(PROCESSO\s+N[Oº°]?\.?\s*[\d\.\-/]+|"
    r"AUTOS\s+N[Oº°]?\.?\s*[\d\.\-/]+)",
    re.IGNORECASE,
)
_RE_ASSINATURA = re.compile(
    r"^(Brasília|São Paulo|Rio de Janeiro|Salvador),\s+\d+\s+de\s+\w+\s+de\s+\d{4}\.",
    re.MULTILINE | re.IGNORECASE,
)
_RE_CABECALHO_LEGAL = re.compile(
    r"^(REPÚBLICA FEDERATIVA DO BRASIL|"
    r"PODER (LEGISLATIVO|JUDICIÁRIO|EXECUTIVO)|"
    r"CÂMARA DOS DEPUTADOS|"
    r"SENADO FEDERAL|"
    r"SUPREMO TRIBUNAL FEDERAL)",
    re.MULTILINE | re.IGNORECASE,
)
_RE_MULTIPLAS_LINHAS = re.compile(r"\n{3,}")
_RE_ESPACOS_MULTIPLOS = re.compile(r"[ \t]{2,}")
_RE_PAGINA = re.compile(r"^\s*-?\s*\d+\s*-?\s*$", re.MULTILINE)


def clean_legal_text(text: str) -> str:
    """
    Limpeza especializada para texto jurídico-legislativo brasileiro.

    Mantém a estrutura semântica do documento (artigos, parágrafos,
    incisos) que é informativa para o modelo de linguagem, mas remove:
      - Números de página isolados
      - Espaços múltiplos e tabulações excessivas
      - Linhas em branco excessivas (>2 consecutivas)

    NÃO removemos:
      - Números de processo (informativos para o modelo jurídico)
      - Cabeçalhos institucionais (contexto importante)
      - Ementas (resumo do documento — alta densidade informacional)
      - Datas e assinaturas (metadados temporais relevantes)

    Esta abordagem conservadora preserva mais contexto jurídico que seria
    descartado por filtros genéricos, reconhecendo que a especificidade
    do domínio é a razão de ser desta fonte no corpus Manacá.

    Referência de pré-processamento jurídico:
      Zheng, L. et al. (2021). When Does Pretraining Help?
        Assessing Self-Supervised Learning for Law. ICAIL 2021.
    """
    if not text:
        return ""

    # Remover números de página isolados
    text = _RE_PAGINA.sub("", text)
    # Normalizar espaços múltiplos (preservar quebras de linha)
    text = _RE_ESPACOS_MULTIPLOS.sub(" ", text)
    # Normalizar múltiplas linhas em branco
    text = _RE_MULTIPLAS_LINHAS.sub("\n\n", text)

    return text.strip()


def extract_source_name(file_path: Path, repo_dir: Path) -> str:
    """
    Extrai o nome da fonte governamental a partir do caminho do arquivo.

    O Tesemõ organiza os documentos em subdiretórios por fonte:
      repo/camara/   → Câmara dos Deputados
      repo/senado/   → Senado Federal
      repo/stf/      → Supremo Tribunal Federal
      repo/tcu/      → Tribunal de Contas da União
      etc.

    Retorna o identificador da fonte para rastreabilidade.
    """
    try:
        rel = file_path.relative_to(repo_dir)
        parts = rel.parts
        if len(parts) >= 1:
            return parts[0].lower()
    except ValueError:
        pass
    return "unknown"


# ── Filtros de qualidade ──────────────────────────────────────────────────────

def passes_filter(text: str) -> bool:
    """
    Filtros de qualidade para texto jurídico.

    Critérios calibrados para linguagem jurídica:
      1. Tipo string válido
      2. Comprimento mínimo: MIN_TEXT_LEN caracteres (100)
         Textos jurídicos muito curtos são fragmentos sem contexto.
      3. Razão mínima de caracteres alfabéticos: 40%
         Limiar menor que demais fontes (30%) porque textos jurídicos
         contêm legitimamente muitos números (artigos, parágrafos, datas,
         valores monetários, números de processo).
    """
    if not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < MIN_TEXT_LEN:
        return False
    n_alpha = sum(1 for c in text if c.isalpha())
    # Threshold mais baixo (40%) para acomodar textos jurídicos com números
    return n_alpha / max(len(text), 1) >= 0.40


def quality_score(text: str) -> float:
    """
    Score de qualidade composto [0.0–1.0].
    Idêntico aos Scripts anteriores para comparabilidade entre fontes.

    Nota: textos jurídicos tendem a ter TTR menor que texto web
    (vocabulário jurídico repetitivo) mas maior comprimento, resultando
    em scores típicos de 0.55–0.75.
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
    """Escreve shard Parquet + Zstd. Idêntico aos Scripts anteriores."""
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


# ── Leitura de arquivo do repositório ────────────────────────────────────────

def read_file_text(file_path: Path) -> str | None:
    """
    Lê o conteúdo textual de um arquivo do repositório Tesemõ.

    O Tesemõ contém arquivos .txt (texto plano) e .json (metadados + texto).
    Para .json, extrai o campo de texto relevante.
    Tenta múltiplos encodings para compatibilidade com documentos antigos.
    """
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    if file_path.suffix == ".json":
        for enc in encodings:
            try:
                data = json.loads(file_path.read_text(encoding=enc))
                # Tentar campos de texto comuns no Tesemõ
                for field in ["texto", "text", "conteudo", "content", "body"]:
                    if field in data and isinstance(data[field], str):
                        return data[field]
                # Se não encontrou campo específico, serializar o JSON inteiro
                # (pode conter texto distribuído em múltiplos campos)
                return " ".join(
                    str(v) for v in data.values()
                    if isinstance(v, str) and len(v) > 20
                )
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
        return None

    elif file_path.suffix == ".txt":
        for enc in encodings:
            try:
                return file_path.read_text(encoding=enc)
            except UnicodeDecodeError:
                continue
        return None

    return None


# ── Varredura do repositório ──────────────────────────────────────────────────

def enumerate_repo_files(repo_dir: Path) -> list[Path]:
    """
    Enumera todos os arquivos de texto no repositório Tesemõ.

    Exclui:
      - Arquivos de metadados do Git (.git/)
      - Arquivos de configuração do repositório (README, LICENSE, etc.)
      - Diretórios vazios
      - Arquivos com extensões não aceitas

    Ordena por caminho para processamento determinístico
    (facilita retomada e reprodutibilidade).
    """
    exclude_dirs = {".git", "__pycache__", ".github", "docs", "scripts"}
    exclude_files = {
        "README.md", "README.txt", "LICENSE", "LICENSE.txt",
        ".gitignore", ".gitattributes", "requirements.txt",
    }

    files = []
    for f in sorted(repo_dir.rglob("*")):
        if not f.is_file():
            continue
        # Pular diretórios excluídos
        if any(part in exclude_dirs for part in f.parts):
            continue
        # Pular arquivos excluídos
        if f.name in exclude_files:
            continue
        # Pular extensões não aceitas
        if f.suffix.lower() not in ACCEPTED_EXTENSIONS:
            continue
        files.append(f)

    return files


# ── Pipeline principal ────────────────────────────────────────────────────────

def acquire() -> dict[str, Any]:
    """
    Pipeline principal de aquisição do Ulysses Tesemõ.

    Este script é estruturalmente diferente dos Scripts 01-05 porque
    a fonte é um repositório GitHub, não um dataset HuggingFace.
    O pipeline é: Git clone → varredura de arquivos → limpeza → Parquet.

    Fluxo:
      1. Verificar NFS e pré-requisitos
      2. Clonar ou atualizar o repositório GitHub
      3. Enumerar todos os arquivos de texto
      4. Verificar arquivos já processados (retomada por arquivo)
      5. Para cada arquivo: ler → limpar → filtrar → acumular no buffer
      6. Flush de shard a cada SHARD_SIZE documentos
      7. Salvar distribuição por fonte governamental

    Retomada: a retomada é baseada em shards (como nos demais scripts),
    não em arquivos individuais. Em caso de interrupção, os shards
    completos são preservados e o processamento retoma a partir do
    arquivo correspondente ao início do próximo shard.
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

    logger.info("=" * 60)
    logger.info(f"Fonte:         {SOURCE_NAME}")
    logger.info(f"Repositorio:   {GITHUB_REPO}")
    logger.info(f"Tokens est.:   {EST_TOKENS_B}B  |  Licenca: {LICENSE}")
    logger.info(f"Referencia:    {REFERENCE}")
    logger.info(f"Output:        {OUTPUT_DIR}")
    logger.info("=" * 60)
    logger.info("DIFERENCA: fonte e repositorio GitHub, nao HuggingFace.")
    logger.info("Pipeline: Git clone → varredura → limpeza juridica → Parquet.")

    # ── Clonar / atualizar repositório ───────────────────────────────────────
    if not clone_or_update_repo():
        logger.error("Falha ao preparar o repositorio. Abortando.")
        sys.exit(1)

    # ── Enumerar arquivos ─────────────────────────────────────────────────────
    logger.info("Enumerando arquivos de texto no repositorio...")
    all_files = enumerate_repo_files(REPO_DIR)
    logger.info(f"  {len(all_files):,} arquivos encontrados")

    if not all_files:
        logger.error(
            f"Nenhum arquivo .txt ou .json encontrado em {REPO_DIR}. "
            "Verificar estrutura do repositorio."
        )
        sys.exit(1)

    # ── Verificar retomada ────────────────────────────────────────────────────
    existing     = get_existing_shards(OUTPUT_DIR)
    docs_to_skip = len(existing) * SHARD_SIZE if existing else 0
    next_shard   = (max(existing) + 1) if existing else 0

    if docs_to_skip:
        logger.info(
            f"Retomando: {len(existing)} shards existentes "
            f"(~{docs_to_skip:,} docs ja salvos)"
        )
    else:
        logger.info("Processando do zero")

    # ── Loop principal ────────────────────────────────────────────────────────
    buffer:       list[dict] = []
    shard_id      = next_shard
    total_seen    = 0   # arquivos processados
    kept          = 0   # documentos mantidos
    skip_qual     = 0   # descartados por qualidade
    skip_read     = 0   # não foi possível ler
    t_start       = time.time()

    # Distribuição por fonte governamental
    source_counts: dict[str, int] = {}

    for file_path in tqdm(
        all_files,
        desc="Ulysses Tesemo",
        unit="arquivo",
        mininterval=10,
        file=sys.stderr,
    ):
        total_seen += 1

        # Pular arquivos já processados em execução anterior
        # (aproximação por número sequencial de documentos)
        if kept < docs_to_skip and total_seen <= docs_to_skip:
            continue

        # Ler conteúdo do arquivo
        raw_text = read_file_text(file_path)
        if raw_text is None:
            skip_read += 1
            logger.debug(f"Nao foi possivel ler: {file_path.name}")
            continue

        # Limpar texto jurídico
        text = clean_legal_text(raw_text)

        # Filtro de qualidade
        if not passes_filter(text):
            skip_qual += 1
            continue

        # Identificar fonte governamental
        gov_source = extract_source_name(file_path, REPO_DIR)
        source_counts[gov_source] = source_counts.get(gov_source, 0) + 1

        # Gerar ID único baseado no caminho relativo do arquivo
        try:
            rel_path = str(file_path.relative_to(REPO_DIR))
        except ValueError:
            rel_path = file_path.name
        doc_id = hashlib.md5(rel_path.encode()).hexdigest()[:16]

        score = quality_score(text)

        buffer.append({
            "text":   text,
            "source": f"{SOURCE_NAME}_{gov_source}",
            "id":     doc_id,
            "lang":   "por_Latn_BR",   # documentos governamentais brasileiros
            "score":  score,
        })
        kept += 1

        # ── Flush de shard ────────────────────────────────────────────────────
        if len(buffer) >= SHARD_SIZE:
            meta    = write_shard(buffer, shard_id, OUTPUT_DIR)
            elapsed = time.time() - t_start
            rate    = kept / max(elapsed, 1)

            logger.info(
                f"Shard {shard_id:04d} | "
                f"{kept:>7,} docs | "
                f"{meta['size_mb']:5.0f} MB | "
                f"{rate:,.0f} docs/s | "
                f"{len(source_counts)} fontes gov."
            )

            manifest["sources"][SOURCE_NAME] = {
                "status":             "in_progress",
                "github_repo":        GITHUB_REPO,
                "license":            LICENSE,
                "total_files_seen":   total_seen,
                "total_kept":         kept,
                "skip_quality":       skip_qual,
                "skip_read_error":    skip_read,
                "shards_written":     shard_id + 1,
                "estimated_tokens":   kept * 250,  # ~250 tokens/doc jurídico médio
                "estimated_tokens_b": round(kept * 250 / 1e9, 2),
                "gov_sources_seen":   len(source_counts),
                "updated_at":         datetime.now(timezone.utc).isoformat(),
            }
            save_manifest(manifest)
            buffer, shard_id = [], shard_id + 1

        # Log de progresso a cada 100k arquivos
        if total_seen % 100_000 == 0:
            elapsed = time.time() - t_start
            logger.info(
                f"Progresso: {total_seen:,} arquivos | "
                f"{kept:,} docs mantidos | "
                f"{skip_qual:,} desc. qualidade | "
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

    # ── Salvar distribuição por fonte governamental ───────────────────────────
    source_dist = {
        "source":          SOURCE_NAME,
        "total_docs":      kept,
        "gov_sources":     len(source_counts),
        "distribution":    dict(sorted(
            source_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )),
        "generated_at":    datetime.now(timezone.utc).isoformat(),
    }
    dist_path = OUTPUT_DIR / "source_distribution.json"
    dist_path.write_text(
        json.dumps(source_dist, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        f"Distribuicao por fonte governamental: {dist_path} "
        f"({len(source_counts)} fontes)"
    )

    # Exibir top 10 fontes
    top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("  Top 10 fontes governamentais:")
    for src, count in top_sources:
        logger.info(f"    {src:<30} {count:>8,} docs")

    # ── Relatório final ───────────────────────────────────────────────────────
    elapsed     = time.time() - t_start
    total_bytes = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.parquet"))
    est_tokens  = kept * 250  # estimativa específica para textos jurídicos

    final: dict[str, Any] = {
        "status":               "completed",
        "github_repo":          GITHUB_REPO,
        "license":              LICENSE,
        "reference":            REFERENCE,
        "total_files_seen":     total_seen,
        "total_kept":           kept,
        "skip_quality":         skip_qual,
        "skip_read_error":      skip_read,
        "retention_rate":       round(kept / max(total_seen, 1), 4),
        "shards_written":       shard_id + 1,
        "total_bytes":          total_bytes,
        "total_gb":             round(total_bytes / 1e9, 3),
        "estimated_tokens":     est_tokens,
        "estimated_tokens_b":   round(est_tokens / 1e9, 2),
        "elapsed_hours":        round(elapsed / 3600, 2),
        "gov_sources_covered":  len(source_counts),
        "output_dir":           str(OUTPUT_DIR),
        "source_dist_path":     str(dist_path),
        "completed_at":         datetime.now(timezone.utc).isoformat(),
    }
    manifest["sources"][SOURCE_NAME] = final
    save_manifest(manifest)

    logger.info("=" * 60)
    logger.info(f"CONCLUIDO — {SOURCE_NAME}")
    logger.info(f"  Arquivos processados: {total_seen:,}")
    logger.info(f"  Docs mantidos:        {kept:,}")
    logger.info(f"  Desc. qualidade:      {skip_qual:,}")
    logger.info(f"  Erros de leitura:     {skip_read:,}")
    logger.info(f"  Taxa retencao:        {final['retention_rate']:.1%}")
    logger.info(f"  Shards:               {shard_id + 1}")
    logger.info(f"  Tamanho total:        {total_bytes / 1e9:.3f} GB")
    logger.info(f"  Tokens est.:          {est_tokens / 1e9:.2f}B")
    logger.info(f"  Fontes gov.:          {len(source_counts)}")
    logger.info(f"  Tempo total:          {elapsed / 3600:.2f} horas")
    logger.info(f"  Output:               {OUTPUT_DIR}")
    logger.info(f"  Manifesto:            {MANIFEST_PATH}")
    logger.info("-" * 60)
    logger.info(f"DONE — {SOURCE_NAME}")
    logger.info(f"  Files processed:      {total_seen:,}")
    logger.info(f"  Docs kept:            {kept:,}")
    logger.info(f"  Quality dropped:      {skip_qual:,}")
    logger.info(f"  Read errors:          {skip_read:,}")
    logger.info(f"  Retention rate:       {final['retention_rate']:.1%}")
    logger.info(f"  Shards:               {shard_id + 1}")
    logger.info(f"  Total size:           {total_bytes / 1e9:.3f} GB")
    logger.info(f"  Est. tokens:          {est_tokens / 1e9:.2f}B")
    logger.info(f"  Gov. sources:         {len(source_counts)}")
    logger.info(f"  Total time:           {elapsed / 3600:.2f} hours")
    logger.info(f"  Output:               {OUTPUT_DIR}")
    logger.info(f"  Manifest:             {MANIFEST_PATH}")
    logger.info("=" * 60)
    logger.info(
        "Tier 1 concluido! Proximo: python corpus/scripts/07_cc_pipeline.py "
        "(aguardar SLURM/recursos computacionais adicionais)"
    )
    logger.info(
        "Tier 1 complete! Next: python corpus/scripts/07_cc_pipeline.py "
        "(wait for SLURM/additional compute resources)"
    )
    logger.info(
        "Alternativa imediata: python corpus/scripts/08_global_dedup.py "
        "(deduplicar as fontes Tier 1 ja disponiveis)"
    )
    logger.info(
        "Immediate alternative: python corpus/scripts/08_global_dedup.py "
        "(deduplicate the Tier 1 sources already available)"
    )

    return final


# ── Verificação de integridade pós-download ───────────────────────────────────

def verify() -> None:
    """Verifica integridade dos shards e exibe distribuição por fonte."""
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

    # Distribuição por fonte governamental
    dist_path = OUTPUT_DIR / "source_distribution.json"
    if dist_path.exists():
        dist = json.loads(dist_path.read_text(encoding="utf-8"))
        logger.info(
            f"Fontes governamentais cobertas: {dist.get('gov_sources', 0)}"
        )
        top = list(dist.get("distribution", {}).items())[:5]
        for src, count in top:
            logger.info(f"  {src:<30} {count:>8,} docs")

    # Amostra do último shard
    try:
        t      = pq.read_table(shards[-1], columns=["text", "source", "score"])
        sample = t["text"][0].as_py()
        src    = t["source"][0].as_py()
        score  = t["score"][0].as_py()
        logger.info(
            f'Amostra ({shards[-1].name}) [{src}]: '
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

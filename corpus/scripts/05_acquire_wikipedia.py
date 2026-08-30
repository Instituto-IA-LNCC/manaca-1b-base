#!/usr/bin/env python3
"""
Manacá LLM — Script 05: Aquisição da Wikipedia PT-BR
=====================================================
LNCC × NII/LLM-jp | Fase 1 — Corpus PT-BR

================================ PT (Português) ================================

Baixa a Wikipedia em Português do Brasil (wikimedia/wikipedia, config 20231101.pt),
processa e armazena em Apache Parquet + Zstandard particionado em shards
de 50.000 documentos.

Idempotente: retoma automaticamente do último shard salvo.

Justificativa científica — PAPEL DUPLO desta fonte:

    PAPEL 1 — CORPUS DE TREINAMENTO:
    A Wikipedia PT-BR é texto enciclopédico factual, bem estruturado e
    revisado por humanos — qualidade rara em dados web. O LLM-jp inclui
    a Wikipedia japonesa em todos os seus modelos como âncora de qualidade,
    e o mesmo princípio se aplica ao Manacá. Apesar do volume pequeno
    (~1B tokens), o impacto na qualidade do modelo é desproporcional ao
    tamanho: modelos treinados em Wikipedia tendem a ter melhor desempenho
    em tarefas de conhecimento factual e raciocínio (Gao et al., 2020).

    PAPEL 2 — BOOTSTRAP DO MODELO KENLM:
    A Wikipedia PT-BR é usada para treinar o modelo de linguagem KenLM
    (5-gramas) que serve de filtro de perplexidade nos Scripts 02-07.
    Documentos com perplexidade muito alta em relação ao modelo Wikipedia
    são indicadores de spam, código, ou texto não-português. Esta técnica
    foi introduzida por Wenzek et al. (2020) no CCNet e é adotada pelo
    FineWeb-2 e pelo LLM-jp-corpus v4.

    NÃO aplicamos filtros de variante linguística (PT-BR vs PT-PT):
    A Wikipedia em português é predominantemente PT-BR, e os artigos
    são identificáveis pela grafia brasileira. Aplicar o filtro fastText
    introduziria falsos negativos desnecessários em dados já de alta
    qualidade.

Referências:
    Gao, L. et al. (2020). The Pile: An 800GB Dataset of Diverse Text
        for Language Modeling. arXiv:2101.00027.
    Wenzek, G. et al. (2020). CCNet. arXiv:2011.00180.
    Ikuya Yamada et al. (2020). Wikipedia2Vec. arXiv:1812.06280.

PRÉ-REQUISITOS OBRIGATÓRIOS:
    1. Volume de trabalho WORK_DIR gravavel (bind mount Docker ./data, ou NFS/HPC)
    2. python corpus/scripts/00_verify_env.py retornando OK

Uso:
    # Verificar ambiente (obrigatório):
    python corpus/scripts/00_verify_env.py

    # Produção — via screen:
    # Docker (recomendado): container destacado, logs persistidos no volume
    docker compose run -d --name wikipedia corpus python corpus/scripts/05_acquire_wikipedia.py
    docker compose logs -f wikipedia
    tail -f $WORK_DIR/raw/wikipedia_ptbr/download.log

    # Execução direta (download rápido ~1-2h, aceitável em foreground):
    python corpus/scripts/05_acquire_wikipedia.py

Saída:
    $WORK_DIR/raw/wikipedia_ptbr/
    ├── shard_000000.parquet   (único ou poucos shards — volume pequeno)
    ├── ...
    ├── download.log
    └── wikipedia_stats.json   estatísticas de qualidade do corpus

    $WORK_DIR/kenlm/
    └── wikipedia_ptbr_5gram.arpa   modelo KenLM (se kenlm instalado)

================================= EN (English) =================================

Manacá LLM — Script 05: Wikipedia PT-BR Acquisition

Downloads Brazilian Portuguese Wikipedia (wikimedia/wikipedia, config 20231101.pt),
processes and stores it in Apache Parquet + Zstandard partitioned into shards
of 50,000 documents.

Idempotent: automatically resumes from the last saved shard.

Scientific rationale — DUAL ROLE of this source:

    ROLE 1 — TRAINING CORPUS:
    Wikipedia PT-BR is factual encyclopedic text, well structured and
    human-reviewed — a rare quality in web data. LLM-jp includes
    Japanese Wikipedia in all of its models as a quality anchor,
    and the same principle applies to Manacá. Despite the small volume
    (~1B tokens), the impact on model quality is disproportionate to
    the size: models trained on Wikipedia tend to perform better
    on factual-knowledge and reasoning tasks (Gao et al., 2020).

    ROLE 2 — KENLM MODEL BOOTSTRAP:
    Wikipedia PT-BR is used to train the KenLM language model
    (5-gram) that serves as a perplexity filter in Scripts 02-07.
    Documents with very high perplexity relative to the Wikipedia model
    are indicators of spam, code, or non-Portuguese text. This technique
    was introduced by Wenzek et al. (2020) in CCNet and is adopted by
    FineWeb-2 and by LLM-jp-corpus v4.

    We do NOT apply language-variant filters (PT-BR vs PT-PT):
    Portuguese Wikipedia is predominantly PT-BR, and the articles
    are identifiable by Brazilian spelling. Applying the fastText filter
    would introduce unnecessary false negatives on data that is already high
    quality.

References:
    Gao, L. et al. (2020). The Pile: An 800GB Dataset of Diverse Text
        for Language Modeling. arXiv:2101.00027.
    Wenzek, G. et al. (2020). CCNet. arXiv:2011.00180.
    Ikuya Yamada et al. (2020). Wikipedia2Vec. arXiv:1812.06280.

MANDATORY PREREQUISITES:
    1. Writable WORK_DIR working volume (Docker bind mount ./data, or NFS/HPC)
    2. python corpus/scripts/00_verify_env.py returning OK

Usage:
    # Check the environment (mandatory):
    python corpus/scripts/00_verify_env.py

    # Production — via screen:
    # Docker (recommended): detached container, logs persisted on the volume
    docker compose run -d --name wikipedia corpus python corpus/scripts/05_acquire_wikipedia.py
    docker compose logs -f wikipedia
    tail -f $WORK_DIR/raw/wikipedia_ptbr/download.log

    # Direct execution (fast download ~1-2h, acceptable in foreground):
    python corpus/scripts/05_acquire_wikipedia.py

Output:
    $WORK_DIR/raw/wikipedia_ptbr/
    ├── shard_000000.parquet   (single or few shards — small volume)
    ├── ...
    ├── download.log
    └── wikipedia_stats.json   corpus quality statistics

    $WORK_DIR/kenlm/
    └── wikipedia_ptbr_5gram.arpa   KenLM model (if kenlm installed)

Autor | Author: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
Versão | Version: 0.1.0 — Abril 2026
"""

from __future__ import annotations

import hashlib
import json
import os
import re
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

SOURCE_NAME  = "wikipedia_ptbr"
HF_REPO_ID   = "wikimedia/wikipedia"
HF_CONFIG    = "20231101.pt"   # snapshot novembro 2023, português
HF_SPLIT     = "train"
LICENSE      = "cc-by-sa-4.0"
REFERENCE    = "Wikimedia Foundation (2023). Wikipedia PT-BR. Nov 2023 dump."
EST_TOKENS_B = 1              # ~1 bilhão de tokens

WORK_DIR      = Path(os.environ.get("WORK_DIR", Path.home() / "manaca-corpus"))
OUTPUT_DIR    = WORK_DIR / "raw" / SOURCE_NAME
KENLM_DIR     = WORK_DIR / "kenlm"
LOG_DIR       = WORK_DIR / "logs"
CKPT_DIR      = WORK_DIR / "checkpoints"
MANIFEST_PATH = CKPT_DIR / "acquisition_manifest.json"

SHARD_SIZE   = 50_000
MIN_TEXT_LEN = 100   # Wikipedia: mínimo maior — artigos muito curtos têm pouco valor

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


# ── Limpeza de texto Wikipedia ────────────────────────────────────────────────

# Padrões de limpeza específicos para Wikipedia
# A Wikipedia do HuggingFace já vem com texto extraído (sem wikitext),
# mas pode conter artefatos residuais de templates e marcações.
_RE_SECTION_HEADER = re.compile(r"^={1,6}.+={1,6}\s*$", re.MULTILINE)
_RE_MULTIPLE_NEWLINES = re.compile(r"\n{3,}")
_RE_TEMPLATE_RESIDUAL = re.compile(r"\{\{[^}]*\}\}")
_RE_REFERENCE_MARKS = re.compile(r"\[\d+\]")


def clean_wikipedia_text(text: str) -> str:
    """
    Limpeza leve de artefatos residuais do texto Wikipedia extraído.

    O dataset wikimedia/wikipedia já passou por extração de texto via
    WikiExtractor, mas pode conter:
      - Cabeçalhos de seção com marcação == Título ==
      - Templates residuais {{...}}
      - Marcas de referência [1], [2]
      - Múltiplas linhas em branco consecutivas

    Mantemos os cabeçalhos de seção como texto simples (removendo apenas
    os delimitadores ==) para preservar a estrutura do artigo, que é
    informativa para o modelo de linguagem.

    Referência de pré-processamento Wikipedia:
      Devlin et al. (2019). BERT. arXiv:1810.04805. (Apêndice A)
    """
    if not text:
        return ""

    # Remover delimitadores de cabeçalho, manter texto
    text = _RE_SECTION_HEADER.sub(
        lambda m: m.group().strip("= \n"), text
    )
    # Remover templates residuais
    text = _RE_TEMPLATE_RESIDUAL.sub("", text)
    # Remover marcas de referência numéricas
    text = _RE_REFERENCE_MARKS.sub("", text)
    # Normalizar espaçamento
    text = _RE_MULTIPLE_NEWLINES.sub("\n\n", text)

    return text.strip()


# ── Filtros de qualidade ──────────────────────────────────────────────────────

def passes_filter(text: str) -> bool:
    """
    Filtros de qualidade para artigos Wikipedia.

    Critérios mais restritivos que as demais fontes:
      1. Tipo string válido
      2. Comprimento mínimo: MIN_TEXT_LEN caracteres (100)
         Artigos muito curtos (stubs, redirects) têm pouco valor de treinamento.
      3. Razão mínima de caracteres alfabéticos: 60%
         Wikipedia pode conter tabelas numéricas — filtramos documentos
         dominados por números e símbolos.
      4. Mínimo de 3 sentenças (heurística: pelo menos 3 pontos finais)
         Elimina artigos de redirecionamento e stubs de uma linha.

    O threshold mínimo de tamanho (100 chars) é maior que nas demais fontes
    (50 chars) porque artigos Wikipedia muito curtos são invariavelmente
    stubs ou redirects sem valor de treinamento.
    """
    if not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < MIN_TEXT_LEN:
        return False
    n_alpha = sum(1 for c in text if c.isalpha())
    if n_alpha / len(text) < 0.60:
        return False
    # Mínimo de 3 sentenças
    n_sentences = text.count(".") + text.count("!") + text.count("?")
    return n_sentences >= 3


def quality_score(text: str) -> float:
    """
    Score de qualidade composto [0.0–1.0].

    Para a Wikipedia, esperamos scores altos (0.7–0.9) dado que:
      - Alta razão de caracteres alfabéticos (texto enciclopédico)
      - Alto type-token ratio (vocabulário variado por artigo)
      - Comprimento tipicamente longo (artigos completos)

    Scores baixos podem indicar artigos com muitas tabelas numéricas
    ou listas de dados — ainda úteis, mas de qualidade diferente.
    Idêntico aos Scripts anteriores para comparabilidade entre fontes.
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


# ── Bootstrap do modelo KenLM ─────────────────────────────────────────────────

def build_kenlm_model(text_corpus: list[str]) -> None:
    """
    Treina modelo KenLM 5-gramas na Wikipedia PT-BR para filtragem
    de perplexidade nos Scripts de Common Crawl (Script 07).

    KenLM (Heafield et al., 2013) é o estimador de n-gramas mais eficiente
    disponível. Um modelo treinado na Wikipedia PT-BR serve como referência
    de "português de qualidade": documentos com perplexidade muito alta
    em relação a este modelo são indicadores de texto degradado, spam,
    código ou língua diferente.

    Esta técnica foi introduzida no CCNet (Wenzek et al., 2020) e é adotada
    pelo FineWeb-2, LLM-jp-corpus v4 e Dolma.

    Threshold de perplexidade sugerido (calibrar no Script 07):
      - Manter: perplexidade < 1500 (conservador, alta recall)
      - Manter: perplexidade < 800  (agressivo, alta precision)
      Recomendação inicial: 1500 (ajustar após análise da distribuição)

    Referências:
      Heafield, K. et al. (2013). Scalable Modified Kneser-Ney Language
        Model Estimation. ACL 2013.
      Wenzek, G. et al. (2020). CCNet. arXiv:2011.00180.

    Nota: Se kenlm não estiver instalado, esta etapa é ignorada com aviso.
    Para instalar: pip install https://github.com/kpu/kenlm/archive/master.zip
    """
    try:
        import kenlm  # noqa: F401
    except ImportError:
        logger.warning(
            "kenlm nao instalado — modelo de perplexidade nao sera construido."
        )
        logger.warning(
            "Para instalar: "
            "pip install https://github.com/kpu/kenlm/archive/master.zip"
        )
        logger.warning(
            "O modelo sera construido manualmente apos instalacao do kenlm."
        )
        return

    KENLM_DIR.mkdir(parents=True, exist_ok=True)
    corpus_path = KENLM_DIR / "wikipedia_ptbr_corpus.txt"
    arpa_path   = KENLM_DIR / "wikipedia_ptbr_5gram.arpa"
    binary_path = KENLM_DIR / "wikipedia_ptbr_5gram.binary"

    if binary_path.exists():
        logger.info(f"Modelo KenLM ja existe: {binary_path}")
        return

    logger.info("Construindo modelo KenLM 5-gramas na Wikipedia PT-BR...")
    logger.info(f"  Corpus: {corpus_path}")
    logger.info(f"  Output: {arpa_path}")

    # Escrever corpus de texto para o lmplz
    logger.info(f"  Escrevendo {len(text_corpus):,} artigos em {corpus_path}...")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for text in text_corpus:
            # Normalizar para uma sentença por linha (formato lmplz)
            for line in text.split("\n"):
                line = line.strip()
                if len(line) > 20:
                    f.write(line + "\n")

    # Verificar se lmplz está disponível
    import shutil
    lmplz  = shutil.which("lmplz")
    bin_build = shutil.which("build_binary")

    if not lmplz:
        logger.warning(
            "lmplz nao encontrado no PATH. "
            "Instalar KenLM: https://github.com/kpu/kenlm"
        )
        logger.warning(
            f"Corpus salvo em {corpus_path}. "
            "Executar manualmente: "
            f"lmplz -o 5 < {corpus_path} > {arpa_path}"
        )
        return

    import subprocess

    logger.info("  Executando lmplz (estimacao Kneser-Ney 5-gramas)...")
    try:
        with open(arpa_path, "w") as arpa_out, \
             open(corpus_path, "r") as corpus_in:
            result = subprocess.run(
                ["lmplz", "-o", "5", "--discount_fallback"],
                stdin=corpus_in,
                stdout=arpa_out,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3600,
            )
        if result.returncode != 0:
            logger.error(f"lmplz falhou: {result.stderr[:200]}")
            return
        logger.info(f"  Modelo ARPA criado: {arpa_path}")
    except subprocess.TimeoutExpired:
        logger.error("lmplz timeout (>1h). Executar manualmente.")
        return
    except Exception as e:
        logger.error(f"Falha ao executar lmplz: {e}")
        return

    # Compilar para formato binário (acesso mais rápido)
    if bin_build and arpa_path.exists():
        logger.info("  Compilando modelo para formato binario...")
        try:
            subprocess.run(
                ["build_binary", str(arpa_path), str(binary_path)],
                check=True,
                timeout=600,
            )
            logger.info(f"  Modelo binario criado: {binary_path}")
            # Remover corpus de texto (grande) após compilação
            corpus_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"build_binary falhou (modelo ARPA ainda utilizavel): {e}")

    logger.success(
        f"Modelo KenLM PT-BR pronto: {binary_path if binary_path.exists() else arpa_path}"
    )
    logger.info(
        "  Usar no Script 07 (Common Crawl): "
        f"kenlm.Model('{binary_path if binary_path.exists() else arpa_path}')"
    )


# ── Pipeline principal ────────────────────────────────────────────────────────

def acquire() -> dict[str, Any]:
    """
    Pipeline principal de aquisição da Wikipedia PT-BR.

    Diferenças fundamentais em relação aos Scripts 02-04:

    1. SEM FILTRO DE VARIANTE LINGUÍSTICA:
       A Wikipedia em português é predominantemente PT-BR. Aplicar fastText
       introduziria falsos negativos desnecessários em dados já verificados.
       O campo 'lang' é preenchido diretamente como 'por_Latn_BR'.

    2. LIMPEZA ESPECÍFICA DE WIKIPEDIA:
       A função clean_wikipedia_text() remove artefatos residuais de
       templates e marcações que o WikiExtractor não remove completamente.

    3. FILTROS MAIS RESTRITIVOS:
       MIN_TEXT_LEN = 100 (vs 50 nas demais fontes) e exigência de
       mínimo de 3 sentenças para eliminar stubs e redirects.

    4. BOOTSTRAP DO MODELO KENLM:
       Após a extração, todos os textos são usados para treinar o modelo
       de linguagem KenLM que servirá de filtro de perplexidade no Script 07.

    5. ESTATÍSTICAS DE QUALIDADE:
       Dado o papel especial da Wikipedia como âncora de qualidade,
       coletamos distribuição de scores e comprimentos de artigo.
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
    KENLM_DIR.mkdir(parents=True, exist_ok=True)
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
            f"(~{docs_to_skip:,} docs ja salvos)"
        )
    else:
        logger.info("Iniciando download do zero")

    logger.info("=" * 60)
    logger.info(f"Fonte:         {SOURCE_NAME}")
    logger.info(f"Repositorio:   {HF_REPO_ID} (config='{HF_CONFIG}')")
    logger.info(f"Tokens est.:   {EST_TOKENS_B}B  |  Licenca: {LICENSE}")
    logger.info(f"Referencia:    {REFERENCE}")
    logger.info(f"Output:        {OUTPUT_DIR}")
    logger.info(f"KenLM output:  {KENLM_DIR}")
    logger.info("=" * 60)
    logger.info("Papel duplo: corpus de treinamento + bootstrap modelo KenLM.")
    logger.info("Sem filtro de variante (Wikipedia PT e predominantemente PT-BR).")

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
    # O dataset wikimedia/wikipedia tem: id, url, title, text
    first    = next(iter(dataset))
    text_col = "text"  if "text"  in first else list(first.keys())[0]
    id_col   = "id"    if "id"    in first else None
    title_col = "title" if "title" in first else None

    logger.info(f"Coluna texto:  '{text_col}'")
    logger.info(f"Coluna ID:     '{id_col or 'gerado'}'")
    logger.info(f"Coluna título: '{title_col or 'nao disponivel'}'")
    logger.info(f"Todos os campos: {list(first.keys())}")

    # ── Loop principal ────────────────────────────────────────────────────────
    buffer:       list[dict] = []
    all_texts:    list[str]  = []  # para bootstrap KenLM
    shard_id      = next_shard
    total_seen    = 0
    kept          = 0
    skip_qual     = 0
    t_start       = time.time()

    # Acumuladores para estatísticas de qualidade
    scores:   list[float] = []
    lengths:  list[int]   = []

    for item in tqdm(
        dataset,
        desc="Wikipedia PT-BR",
        unit="artigo",
        mininterval=10,
        file=sys.stderr,
    ):
        total_seen += 1

        # Pular artigos já processados em execução anterior
        if total_seen <= docs_to_skip:
            continue

        raw_text = item.get(text_col, "")
        title    = item.get(title_col, "") if title_col else ""

        # Limpar artefatos de wikitext residual
        text = clean_wikipedia_text(raw_text)

        # Filtro de qualidade (mais restritivo que demais fontes)
        if not passes_filter(text):
            skip_qual += 1
            continue

        score = quality_score(text)
        raw_id = item.get(id_col, "") if id_col else ""

        # Construir documento com título prefixado para melhor contexto
        # A prefixação do título é prática comum em modelos treinados em Wikipedia
        # (Brown et al., 2020 — GPT-3; Raffel et al., 2020 — T5)
        full_text = f"{title}\n\n{text}" if title else text

        buffer.append({
            "text":   full_text,
            "source": SOURCE_NAME,
            "id":     str(raw_id) if raw_id else f"{SOURCE_NAME}_{total_seen}",
            "lang":   "por_Latn_BR",
            "score":  score,
        })
        all_texts.append(text)  # sem título para o KenLM
        kept += 1

        # Acumular estatísticas
        scores.append(score)
        lengths.append(len(text))

        # ── Flush de shard ────────────────────────────────────────────────────
        if len(buffer) >= SHARD_SIZE:
            meta    = write_shard(buffer, shard_id, OUTPUT_DIR)
            elapsed = time.time() - t_start
            logger.info(
                f"Shard {shard_id:04d} | "
                f"{kept:>7,} artigos | "
                f"{meta['size_mb']:5.0f} MB | "
                f"{kept/max(elapsed,1):,.0f} arts/s"
            )
            manifest["sources"][SOURCE_NAME] = {
                "status":         "in_progress",
                "hf_repo":        HF_REPO_ID,
                "hf_config":      HF_CONFIG,
                "license":        LICENSE,
                "total_seen":     total_seen,
                "total_kept":     kept,
                "total_skip":     skip_qual,
                "shards_written": shard_id + 1,
                "estimated_tokens":   kept * 500,  # ~500 tokens/artigo Wikipedia
                "estimated_tokens_b": round(kept * 500 / 1e9, 2),
                "updated_at":     datetime.now(timezone.utc).isoformat(),
            }
            save_manifest(manifest)
            buffer, shard_id = [], shard_id + 1

    # ── Flush do último shard parcial ─────────────────────────────────────────
    if buffer:
        meta = write_shard(buffer, shard_id, OUTPUT_DIR)
        logger.info(
            f"Shard final {shard_id:04d} | "
            f"{len(buffer):,} artigos | "
            f"{meta['size_mb']:.0f} MB"
        )

    # ── Estatísticas de qualidade ─────────────────────────────────────────────
    def percentile(data: list[float], p: float) -> float:
        if not data:
            return 0.0
        s = sorted(data)
        return s[int(len(s) * p / 100)]

    wiki_stats = {
        "source":          SOURCE_NAME,
        "total_articles":  kept,
        "total_skipped":   skip_qual,
        "score_mean":      round(sum(scores) / max(len(scores), 1), 4),
        "score_p10":       round(percentile(scores, 10), 4),
        "score_p50":       round(percentile(scores, 50), 4),
        "score_p90":       round(percentile(scores, 90), 4),
        "length_mean":     round(sum(lengths) / max(len(lengths), 1)),
        "length_p10":      round(percentile(lengths, 10)),
        "length_p50":      round(percentile(lengths, 50)),
        "length_p90":      round(percentile(lengths, 90)),
        "estimated_tokens": kept * 500,
        "generated_at":    datetime.now(timezone.utc).isoformat(),
    }
    stats_path = OUTPUT_DIR / "wikipedia_stats.json"
    stats_path.write_text(
        json.dumps(wiki_stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Estatísticas salvas: {stats_path}")
    logger.info(
        f"  Score medio: {wiki_stats['score_mean']:.4f} "
        f"(p10={wiki_stats['score_p10']:.4f} "
        f"p90={wiki_stats['score_p90']:.4f})"
    )
    logger.info(
        f"  Comprimento medio: {wiki_stats['length_mean']:,} chars "
        f"(p10={wiki_stats['length_p10']:,} "
        f"p90={wiki_stats['length_p90']:,})"
    )

    # ── Relatório final ───────────────────────────────────────────────────────
    elapsed     = time.time() - t_start
    total_bytes = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.parquet"))
    est_tokens  = kept * 500  # estimativa específica para Wikipedia

    final: dict[str, Any] = {
        "status":             "completed",
        "hf_repo":            HF_REPO_ID,
        "hf_config":          HF_CONFIG,
        "license":            LICENSE,
        "reference":          REFERENCE,
        "total_seen":         total_seen,
        "total_kept":         kept,
        "total_skipped":      skip_qual,
        "retention_rate":     round(kept / max(total_seen, 1), 4),
        "shards_written":     shard_id + 1,
        "total_bytes":        total_bytes,
        "total_gb":           round(total_bytes / 1e9, 3),
        "estimated_tokens":   est_tokens,
        "estimated_tokens_b": round(est_tokens / 1e9, 2),
        "elapsed_hours":      round(elapsed / 3600, 2),
        "output_dir":         str(OUTPUT_DIR),
        "kenlm_dir":          str(KENLM_DIR),
        "stats_path":         str(stats_path),
        "completed_at":       datetime.now(timezone.utc).isoformat(),
    }
    manifest["sources"][SOURCE_NAME] = final
    save_manifest(manifest)

    logger.info("=" * 60)
    logger.info(f"CONCLUIDO — {SOURCE_NAME}")
    logger.info(f"  Artigos mantidos:  {kept:,}")
    logger.info(f"  Artigos desc.:     {skip_qual:,}")
    logger.info(f"  Taxa retencao:     {final['retention_rate']:.1%}")
    logger.info(f"  Shards:            {shard_id + 1}")
    logger.info(f"  Tamanho total:     {total_bytes / 1e9:.3f} GB")
    logger.info(f"  Tokens est.:       {est_tokens / 1e9:.2f}B")
    logger.info(f"  Tempo total:       {elapsed / 3600:.2f} horas")
    logger.info(f"  Output:            {OUTPUT_DIR}")
    logger.info("-" * 60)
    logger.info(f"DONE — {SOURCE_NAME}")
    logger.info(f"  Articles kept:     {kept:,}")
    logger.info(f"  Articles dropped:  {skip_qual:,}")
    logger.info(f"  Retention rate:    {final['retention_rate']:.1%}")
    logger.info(f"  Shards:            {shard_id + 1}")
    logger.info(f"  Total size:        {total_bytes / 1e9:.3f} GB")
    logger.info(f"  Est. tokens:       {est_tokens / 1e9:.2f}B")
    logger.info(f"  Total time:        {elapsed / 3600:.2f} hours")
    logger.info(f"  Output:            {OUTPUT_DIR}")
    logger.info("=" * 60)

    # ── Bootstrap do modelo KenLM ─────────────────────────────────────────────
    logger.info("Iniciando bootstrap do modelo KenLM 5-gramas...")
    build_kenlm_model(all_texts)

    logger.info("Proximo passo | Next step: python corpus/scripts/06_acquire_ulysses.py")
    return final


# ── Verificação de integridade pós-download ───────────────────────────────────

def verify() -> None:
    """Verifica integridade dos shards e exibe estatísticas de qualidade."""
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
        f"{total_rows:,} artigos | "
        f"{errors} erro(s)"
    )

    # Estatísticas de qualidade
    stats_path = OUTPUT_DIR / "wikipedia_stats.json"
    if stats_path.exists():
        s = json.loads(stats_path.read_text(encoding="utf-8"))
        logger.info(
            f"Qualidade: score_medio={s.get('score_mean',0):.4f} | "
            f"comprimento_medio={s.get('length_mean',0):,} chars"
        )

    # Verificar modelo KenLM
    binary_path = KENLM_DIR / "wikipedia_ptbr_5gram.binary"
    arpa_path   = KENLM_DIR / "wikipedia_ptbr_5gram.arpa"
    if binary_path.exists():
        logger.info(f"Modelo KenLM binario: {binary_path}")
    elif arpa_path.exists():
        logger.info(f"Modelo KenLM ARPA: {arpa_path}")
    else:
        logger.warning(
            "Modelo KenLM nao encontrado. "
            "Instalar kenlm e executar build_kenlm_model() manualmente."
        )

    # Amostra do último shard
    try:
        t      = pq.read_table(shards[-1], columns=["text", "score"])
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

#!/usr/bin/env python3
"""
Manacá LLM — Script 00: Verificação do Ambiente
================================================
LNCC × NII/LLM-jp | Fase 1 — Corpus PT-BR

Verifica pré-requisitos antes de qualquer extração de corpus.
Executar obrigatoriamente no início de cada sessão de trabalho.

Uso:
    # Docker (recomendado):
    docker compose run --rm corpus python corpus/scripts/00_verify_env.py
    # ou, ja dentro do container 'corpus':
    python corpus/scripts/00_verify_env.py

Saída:
    Tabela de verificação no terminal.
    Código de saída 0 = OK · 1 = falha em alguma verificação crítica.

Autor: Bruno Leonardo Santos Menezes <brunolsm@lncc.br>
Versão: 0.1.0 — Abril 2026
"""

from __future__ import annotations
import os, sys, shutil, socket, time
from pathlib import Path
from datetime import datetime, timezone

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    console = Console()
except ImportError:
    class _C:
        def print(self, *a, **k):
            import re
            print(re.sub(r'\[.*?\]', '', " ".join(str(x) for x in a)))
    console = _C()
    Table = None

# ── Configuração ──────────────────────────────────────────────────────────────
WORK_DIR    = Path(os.environ.get("WORK_DIR",  Path.home() / "manaca-corpus"))
HF_HOME     = Path(os.environ.get("HF_HOME",   Path.home() / "software" / "hf-cache"))
MIN_DISK_GB = 50

PACKAGES = [
    ("datasets",    "4.8",  "datasets"),
    ("datatrove",   "0.9",  "datatrove.pipeline"),
    ("pyarrow",     "15",   "pyarrow"),
    ("datasketch",  "1.9",  "datasketch"),
    ("fasttext",    None,   "fasttext"),
    ("loguru",      "0.7",  "loguru"),
    ("rich",        "14",   "rich"),
    ("numpy",       "2",    "numpy"),
    ("pandas",      "3",    "pandas"),
    ("tokenizers",  "0.19", "tokenizers"),
    ("tqdm",        None,   "tqdm"),
    ("pyyaml",      None,   "yaml"),
    ("xxhash",      None,   "xxhash"),
    ("zstandard",   None,   "zstandard"),
    ("requests",    None,   "requests"),
    ("trafilatura", "1.8",  "trafilatura"),
]

SUBDIRS = ["raw", "processed", "deduped", "final", "stats", "logs", "checkpoints"]


# ── Verificações ──────────────────────────────────────────────────────────────

def check_python():
    v = sys.version_info
    return v.major == 3 and v.minor == 11, f"{v.major}.{v.minor}.{v.micro}"


def check_pkg(import_name, min_ver):
    try:
        m = __import__(import_name)
        ver = getattr(m, "__version__", "OK")
        if min_ver and ver != "OK":
            parts = [int(x) for x in ver.split(".")[:2] if x.isdigit()]
            req   = [int(x) for x in min_ver.split(".")[:2] if x.isdigit()]
            return parts >= req, ver
        return True, ver
    except ImportError:
        return False, "AUSENTE"


def check_parquet():
    try:
        import io
        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = pa.schema([
            pa.field("text",   pa.string()),
            pa.field("source", pa.string()),
            pa.field("id",     pa.string()),
            pa.field("lang",   pa.string()),
            pa.field("score",  pa.float32()),
        ])
        sample = "O manaca-da-serra floresce em branco, rosa e purpura simultaneamente."
        t = pa.table(
            {"text": [sample], "source": ["test"],
             "id": ["0"], "lang": ["por_Latn"], "score": [0.99]},
            schema=schema,
        )
        buf = io.BytesIO()
        pq.write_table(t, buf, compression="zstd")
        buf.seek(0)
        t2 = pq.read_table(buf)
        assert t2["text"][0].as_py() == sample
        return True, f"OK ({buf.tell() / 1024:.1f} KB)"
    except Exception as e:
        return False, f"FALHA: {e}"


def check_minhash():
    try:
        from datasketch import MinHash

        def mh(toks):
            m = MinHash(num_perm=128)
            for t in toks:
                m.update(t.encode())
            return m

        toks  = "o brasil possui uma das maiores biodiversidades do mundo".split()
        s_id  = mh(toks).jaccard(mh(toks))
        s_dif = mh(toks).jaccard(mh(["hello"]))
        assert s_id > 0.99 and s_dif < 0.10
        return True, f"OK (identico={s_id:.2f} diferente={s_dif:.2f})"
    except Exception as e:
        return False, f"FALHA: {e}"


def check_langid():
    try:
        import fasttext  # noqa: F401
        model_path = HF_HOME / "fasttext" / "lid.176.bin"
        if not model_path.exists():
            return True, "OK (modelo lid.176.bin sera baixado no Script 01)"
        model = fasttext.load_model(str(model_path))
        labels, scores = model.predict(
            "O Projeto Manaca desenvolve IA soberana para o Portugues do Brasil.", k=1
        )
        lang = labels[0].replace("__label__", "")
        return lang == "pt", f"{lang} ({float(scores[0]):.2f})"
    except Exception as e:
        return True, f"AVISO: {str(e)[:60]}"


def check_hf():
    try:
        import urllib.request
        t0 = time.time()
        with urllib.request.urlopen("https://huggingface.co", timeout=8) as r:
            code = r.status
        return code == 200, f"HTTP {code} ({time.time() - t0:.2f}s)"
    except Exception as e:
        return False, f"FALHA: {str(e)[:60]}"


def check_storage():
    # Volume de trabalho: bind mount Docker (./data -> /workspace/manaca-corpus)
    # ou NFS/HPC. Definido pela variavel de ambiente WORK_DIR.
    if WORK_DIR.exists() and os.access(WORK_DIR, os.W_OK):
        return True, f"{WORK_DIR} (gravavel)"
    if WORK_DIR.exists():
        return False, f"{WORK_DIR} existe mas sem permissao de escrita"
    return False, f"{WORK_DIR} inexistente — montar o volume (Docker) ou exportar WORK_DIR"


def check_dirs():
    missing = [d for d in SUBDIRS if not (WORK_DIR / d).exists()]
    if missing:
        return False, f"AUSENTES: {', '.join(missing)}"
    return True, str(WORK_DIR)


def check_disk():
    try:
        path = WORK_DIR if WORK_DIR.exists() else Path.home()
        gb = shutil.disk_usage(path).free / 1024 ** 3
        return gb >= MIN_DISK_GB, f"{gb:.0f} GB livres"
    except Exception as e:
        return False, str(e)


# ── Setup de diretórios ───────────────────────────────────────────────────────

def ensure_dirs():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    HF_HOME.mkdir(parents=True, exist_ok=True)
    (HF_HOME / "fasttext").mkdir(exist_ok=True)
    for d in SUBDIRS:
        (WORK_DIR / d).mkdir(exist_ok=True)


# ── Relatório ─────────────────────────────────────────────────────────────────

def main():
    ensure_dirs()

    hostname = socket.gethostname()
    now      = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    console.print()
    console.print(Panel(
        f"[bold]Manaca LLM — Verificacao do Ambiente[/bold]\n"
        f"[dim]{hostname} | {now}[/dim]",
        border_style="cyan",
        expand=False,
    ))
    console.print()

    # Montar lista de verificações
    results = []
    ok, ver = check_python()
    results.append(("Python 3.11", ok, ver, False))

    for name, min_ver, imp in PACKAGES:
        ok, ver = check_pkg(imp, min_ver)
        results.append((name, ok, ver, False))

    results.append(("Parquet + Zstd",  *check_parquet(),  False))
    results.append(("MinHash LSH",     *check_minhash(),   False))
    results.append(("LangID fastText", *check_langid(),    False))
    results.append(("HuggingFace Hub", *check_hf(),        False))
    results.append(("Volume WORK_DIR",  *check_storage(),   True))   # aviso, nao critico
    results.append(("Workspace dirs",  *check_dirs(),      False))
    results.append((f"Disco >={MIN_DISK_GB}GB", *check_disk(), False))

    # Verificações críticas: tudo exceto NFS
    critical_ok = all(ok for name, ok, _, is_warn in results if not is_warn)
    all_ok      = all(ok for _, ok, _, _ in results)

    if Table:
        t = Table(
            show_header=True,
            header_style="bold blue",
            border_style="dim",
            expand=False,
        )
        t.add_column("Verificacao", style="bold", min_width=24)
        t.add_column("Status",      justify="center", min_width=6)
        t.add_column("Detalhe",     style="dim")

        for name, ok, det, is_warn in results:
            if ok:
                icon = "[green]OK[/green]"
                col  = "green"
            elif is_warn:
                icon = "[yellow]AVISO[/yellow]"
                col  = "yellow"
            else:
                icon = "[red]FALHA[/red]"
                col  = "red"
            t.add_row(name, icon, f"[{col}]{det}[/{col}]")

        console.print(t)
    else:
        for name, ok, det, is_warn in results:
            status = "OK   " if ok else ("AVISO" if is_warn else "FALHA")
            print(f"  {status}  {name:<28} {det}")

    console.print()

    if all_ok:
        console.print(Panel(
            "[bold green]AMBIENTE OK[/bold green] — pronto para extracao\n"
            "[dim]Proximo: python corpus/scripts/01_acquire_gigaverbo.py[/dim]",
            border_style="green",
            expand=False,
        ))
    elif critical_ok:
        console.print(Panel(
            "[bold yellow]AMBIENTE OK · VOLUME PENDENTE[/bold yellow]\n"
            "[dim]Pacotes e ferramentas verificados com sucesso.\n"
            "O volume de trabalho (WORK_DIR) nao esta gravavel.\n"
            "Docker: confira o bind mount (./data) em docker-compose.yml.\n"
            "HPC/NFS: exporte WORK_DIR=/caminho/do/nfs/manaca-corpus.[/dim]",
            border_style="yellow",
            expand=False,
        ))
    else:
        n = sum(1 for _, ok, _, is_warn in results if not ok and not is_warn)
        console.print(Panel(
            f"[bold red]{n} VERIFICACAO(OES) CRITICA(S) FALHARAM[/bold red]\n"
            "[dim]Resolver antes de prosseguir.\n"
            "Consultar: corpus/README.md — Secao 2 (Pre-requisitos)[/dim]",
            border_style="red",
            expand=False,
        ))

    console.print()
    return critical_ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

#!/usr/bin/env python3
"""
LLM-BR URL Checker
==================

PT ------------------------------------------------------------------------
Verificador de URLs do LLM-BR
Script automatico para verificar URLs em arquivos Markdown.
Parte do projeto LLM-BR — LNCC × NII.

Uso:
    pip install requests
    python scripts/check-urls.py [arquivo_ou_diretorio_markdown]

Exemplos:
    python scripts/check-urls.py reports/corpora-survey/url-audit.md
    python scripts/check-urls.py reports/
    python scripts/check-urls.py  # verifica todos os arquivos .md do repositorio

Saida:
    Imprime uma tabela resumo com os codigos de status de cada URL encontrada.
    Gera check-urls-report.json com os resultados completos.

EN ------------------------------------------------------------------------
Automated script to verify URLs in Markdown files.
Part of the LLM-BR project — LNCC × NII.

Usage:
    pip install requests
    python scripts/check-urls.py [markdown_file_or_directory]

Examples:
    python scripts/check-urls.py reports/corpora-survey/url-audit.md
    python scripts/check-urls.py reports/
    python scripts/check-urls.py  # checks all .md files in the repo

Output:
    Prints a summary table with status codes for each URL found.
    Generates check-urls-report.json with full results.

Autor | Author: Bruno Menezes (brunolsm@lncc.br)
License: CC BY 4.0
Version: 0.1.0 — March 2026
"""

import re
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    print("Error: 'requests' library not found. Run: pip install requests")
    sys.exit(1)


# ─── Configuration ────────────────────────────────────────────────────────────

TIMEOUT = 15          # seconds per request
DELAY   = 0.5         # seconds between requests (be polite)
MAX_RETRIES = 2
USER_AGENT = "LLM-BR-URL-Checker/0.1 (github.com/lncc-ia/llm-br)"

# URLs to skip (known to block bots or require auth)
SKIP_PATTERNS = [
    "kaggle.com",
    "linguateca.pt",   # SSL issues
    "sketchengine.eu", # paywall
    "openalex.org",    # Cloudflare bot protection (HTTP 403 for automated clients)
]

STATUS_EMOJI = {
    "OK":      "✅",
    "BROKEN":  "❌",
    "SKIP":    "⏭️ ",
    "TIMEOUT": "⏱️ ",
    "ERROR":   "⚠️ ",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def extract_urls(text: str) -> list[str]:
    """Extract all URLs from Markdown text (links and raw URLs)."""
    # Markdown links: [text](url)
    md_links = re.findall(r'\[.*?\]\((https?://[^\)]+)\)', text)
    # Raw URLs in text
    raw_urls = re.findall(r'(?<!\()(https?://[^\s\)\]]+)', text)
    all_urls = list(dict.fromkeys(md_links + raw_urls))  # deduplicate, preserve order
    return all_urls


def should_skip(url: str) -> bool:
    return any(pattern in url for pattern in SKIP_PATTERNS)


def check_url(url: str, session: requests.Session) -> dict:
    if should_skip(url):
        return {"url": url, "status": "SKIP", "code": None, "message": "Skipped (known bot blocker)"}
    try:
        response = session.head(
            url, timeout=TIMEOUT, allow_redirects=True,
            headers={"User-Agent": USER_AGENT}
        )
        # Some servers reject HEAD — fallback to GET
        if response.status_code in (405, 403, 400):
            response = session.get(
                url, timeout=TIMEOUT, allow_redirects=True,
                headers={"User-Agent": USER_AGENT}, stream=True
            )
            response.close()
        
        code = response.status_code
        ok = code < 400
        final_url = response.url if response.url != url else None
        return {
            "url": url,
            "status": "OK" if ok else "BROKEN",
            "code": code,
            "final_url": final_url,
            "message": f"HTTP {code}" + (f" → {final_url}" if final_url else ""),
        }
    except requests.exceptions.Timeout:
        return {"url": url, "status": "TIMEOUT", "code": None, "message": "Request timed out"}
    except requests.exceptions.SSLError as e:
        return {"url": url, "status": "ERROR", "code": None, "message": f"SSL error: {str(e)[:60]}"}
    except requests.exceptions.ConnectionError as e:
        return {"url": url, "status": "BROKEN", "code": None, "message": f"Connection error: {str(e)[:60]}"}
    except Exception as e:
        return {"url": url, "status": "ERROR", "code": None, "message": str(e)[:80]}


def collect_md_files(path_arg: Optional[str]) -> list[Path]:
    if path_arg is None:
        root = Path(".")
        return list(root.rglob("*.md"))
    p = Path(path_arg)
    if p.is_file():
        return [p]
    elif p.is_dir():
        return list(p.rglob("*.md"))
    else:
        print(f"Error: '{path_arg}' is not a file or directory.")
        sys.exit(1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM-BR URL Checker — Verify all URLs in Markdown files."
    )
    parser.add_argument(
        "target", nargs="?", default=None,
        help="Markdown file or directory to check (default: all .md files in current dir)"
    )
    parser.add_argument(
        "--output", default="check-urls-report.json",
        help="Output JSON report file (default: check-urls-report.json)"
    )
    args = parser.parse_args()

    # Setup session with retries
    session = requests.Session()
    retry = Retry(total=MAX_RETRIES, backoff_factor=1, status_forcelist=[500, 502, 503])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    md_files = collect_md_files(args.target)
    if not md_files:
        print("No Markdown files found.")
        sys.exit(0)

    print(f"\n🔍 LLM-BR URL Checker — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"   Checking {len(md_files)} Markdown file(s)...\n")

    all_results = []
    all_urls_seen = set()

    for md_file in sorted(md_files):
        text = md_file.read_text(encoding="utf-8", errors="ignore")
        urls = extract_urls(text)
        new_urls = [u for u in urls if u not in all_urls_seen]
        all_urls_seen.update(new_urls)

        if not new_urls:
            continue

        print(f"📄 {md_file}  ({len(new_urls)} URLs)")
        file_results = []

        for url in new_urls:
            result = check_url(url, session)
            result["source_file"] = str(md_file)
            file_results.append(result)
            all_results.append(result)

            emoji = STATUS_EMOJI.get(result["status"], "❓")
            print(f"   {emoji} {result['status']:7s}  {url[:80]}")
            if result.get("message") and result["status"] != "OK":
                print(f"            ↳ {result['message']}")

            time.sleep(DELAY)
        print()

    # Summary
    counts = {}
    for r in all_results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    print("=" * 70)
    # Resumo final bilingue (EN + PT) — o resultado que o usuario le ao fim da execucao.
    print(f"  SUMMARY — {len(all_results)} unique URLs checked")
    print(f"  RESUMO — {len(all_results)} URLs unicas verificadas")
    print("=" * 70)
    for status, emoji in STATUS_EMOJI.items():
        n = counts.get(status, 0)
        if n:
            print(f"  {emoji} {status:8s}: {n}")
    print()

    broken = [r for r in all_results if r["status"] == "BROKEN"]
    if broken:
        print(f"  ❌ {len(broken)} BROKEN URL(s):")
        for r in broken:
            print(f"     {r['url']}")
            print(f"       File: {r['source_file']}")
            print(f"       Msg:  {r.get('message', '')}")
        print()

    # Save JSON report
    report = {
        "generated_at": datetime.now().isoformat(),
        "total_urls": len(all_results),
        "counts": counts,
        "results": all_results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  📊 Full report saved to: {args.output}")
    print(f"  📊 Relatorio completo salvo em: {args.output}")
    print()

    # Exit with error code if any broken URLs found
    if broken:
        sys.exit(1)


if __name__ == "__main__":
    main()

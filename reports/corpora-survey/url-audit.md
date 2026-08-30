# URL Audit — Brazilian Portuguese Corpora
## 100 Links Verified | 100 Links Verificados — March 2026

**Audited by / Auditado por:** Bruno Leonardo Santos Menezes  
**Date / Data:** March 28, 2026

---

## Summary / Resumo

| Status | Count | Description |
|--------|-------|-------------|
| ✅ Functional | 72 | URL loads correctly |
| ⚠ Functional with caveat | 4 | Accessible but with important operational issue |
| ❌ Broken → corrected | 28 | Original URL broken; correct URL identified |
| **Total** | **100** | **Effective availability of underlying corpora: ~97%** |

---

## Failure Patterns / Padrões de Falha

### 1. Hugging Face namespace migrations (9 cases)
Old community datasets moved to organization namespaces; no automatic redirect.

| Original | Correct |
|----------|---------|
| `datasets/assin2` | `datasets/nilc-nlp/assin2` |
| `datasets/nilc-nlp/mac-morpho` | `datasets/nilc-nlp/mac_morpho` (underscore) |
| `datasets/nilc-nlp/CORAA` | Not on HF — use GitHub |
| `datasets/nilc-nlp/certas_palavras` | `zenodo.org/records/6794924` |
| `datasets/told-br/told-br` | `datasets/JAugusto97/told-br` |
| `datasets/ruanchaves/hate-speech-portuguese` | `datasets/ruanchaves/hatebr` |
| `datasets/xquad` | `datasets/google/xquad` |
| `datasets/porsimplessent` | GitHub only |
| `datasets/HPLT/hplt_monolingual_v2` | `datasets/HPLT/HPLT2.0_cleaned` |

### 2. GitHub organization renames (8 cases)
`Ulysses-NLP` → `ulysses-camara` | `legal-nlp-datasets` → `peluz` | others

| Original | Correct |
|----------|---------|
| `github.com/Ulysses-NLP/ulysses-tesemo` | `github.com/ulysses-camara/ulysses-tesemo` |
| `github.com/Ulysses-NLP/ulysses-ner-br` | `github.com/ulysses-camara/ulysses-ner-br` |
| `github.com/legal-nlp-datasets/victor` | `github.com/peluz/VICTOR-dataset` |
| `github.com/marcossyokese/FaQuAD` | `github.com/liafacom/faquad` |
| `github.com/JAugusto97/ToLD-Br` | `github.com/joaoaleite/ToLD-Br` (redirect) |
| `github.com/uai-ufmg/tolde-br` | `github.com/JAugusto97/told-br` (dataset name issue) |
| `github.com/pasquaini/CorPop` | UFRGS/TEXTECC website |
| `github.com/edilsonacjr/TweetSentBR` | `bitbucket.org/HBrum/tweetsentbr` |

### 3. Academic server restructuring (8 cases)
Brazilian university servers migrated paths, subdomains, or dropped legacy IPs.

| Original | Correct |
|----------|---------|
| `inf.pucrs.br/linatural/wordpress/...blogset-br-english/` | HuggingFace mirror |
| `tycho.iel.unicamp.br/corpus/en/acesso.html` | `tycho.iel.unicamp.br/corpus/` |
| `tycho.iel.unicamp.br/` (root) | SSL error — use subpages |
| `143.107.183.175:21380/sucinto/` | `sites.icmc.usp.br/taspardo/sucinto/cstnews.html` |
| `nurcdigital.fale.ufal.br/` | `fale.ufal.br/projeto/nurcdigital/` |
| `letras.ufrj.br/nurc-rj/` | `nurcrj.letras.ufrj.br/` |
| `cic.unb.br/~teodecampos/LeNER-Br/` (robots.txt) | Use GitHub mirror |
| `opus.nlpl.eu/CCMatrix/` | `opus.nlpl.eu/datasets/CCMatrix` |

---

## Resources with Important Caveats / Recursos com Ressalvas

| Corpus | Caveat |
|--------|--------|
| **mC4** | ⚠ Deprecated — use `allenai/c4` (branch mC4_3.1.0) |
| **OSCAR 23.01** | ⚠ Card accessible but data download temporarily suspended |
| **Common Voice pt** | ⚠ Downloads migrated to Mozilla Data Collective (Oct 2025) |
| **Linguateca** | ⚠ Possible expired SSL cert; use subpages directly |

---

## Full Audit Table / Tabela Completa da Auditoria

For the complete 100-row audit table with all status codes, see the full report in `assets/relatorio_corpora_ptbr_lncc_nii.docx` (Appendix A).

---

## Automated Checker / Verificação Automatizada

A Python script for periodic URL verification is available at [`scripts/check-urls.py`](../../scripts/check-urls.py).

Run it with:
```bash
pip install requests
python scripts/check-urls.py reports/corpora-survey/url-audit.md
```

A GitHub Actions workflow runs this check automatically every month. See [`.github/workflows/url-checker.yml`](../../.github/workflows/url-checker.yml).

---

*Version 0.1.0 — March 2026 | LLM-BR Project — LNCC × NII*

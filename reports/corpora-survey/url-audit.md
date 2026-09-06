# Auditoria de URLs — Corpora em Português do Brasil | URL Audit — Brazilian Portuguese Corpora

**[🇧🇷 Português](#português)** · **[🇬🇧 English](#english)**

## Português

### 100 Links Verificados — Março de 2026

**Auditado por:** Bruno Leonardo Santos Menezes  
**Data:** 28 de março de 2026

---

### Resumo

| Status | Contagem | Descrição |
|--------|-------|-------------|
| ✅ Funcional | 72 | A URL carrega corretamente |
| ⚠ Funcional com ressalva | 4 | Acessível, mas com problema operacional importante |
| ❌ Quebrada → corrigida | 28 | URL original quebrada; URL correta identificada |
| **Total** | **100** | **Disponibilidade efetiva dos corpora subjacentes: ~97%** |

---

### Padrões de Falha

#### 1. Migrações de namespace no Hugging Face (9 casos)
Datasets antigos da comunidade movidos para namespaces de organizações; sem redirecionamento automático.

| Original | Correto |
|----------|---------|
| `datasets/assin2` | `datasets/nilc-nlp/assin2` |
| `datasets/nilc-nlp/mac-morpho` | `datasets/nilc-nlp/mac_morpho` (underscore) |
| `datasets/nilc-nlp/CORAA` | Não está no HF — use o GitHub |
| `datasets/nilc-nlp/certas_palavras` | `zenodo.org/records/6794924` |
| `datasets/told-br/told-br` | `datasets/JAugusto97/told-br` |
| `datasets/ruanchaves/hate-speech-portuguese` | `datasets/ruanchaves/hatebr` |
| `datasets/xquad` | `datasets/google/xquad` |
| `datasets/porsimplessent` | Somente no GitHub |
| `datasets/HPLT/hplt_monolingual_v2` | `datasets/HPLT/HPLT2.0_cleaned` |

#### 2. Renomeações de organizações no GitHub (8 casos)
`Ulysses-NLP` → `ulysses-camara` | `legal-nlp-datasets` → `peluz` | outros

| Original | Correto |
|----------|---------|
| `github.com/Ulysses-NLP/ulysses-tesemo` | `github.com/ulysses-camara/ulysses-tesemo` |
| `github.com/Ulysses-NLP/ulysses-ner-br` | `github.com/ulysses-camara/ulysses-ner-br` |
| `github.com/legal-nlp-datasets/victor` | `github.com/peluz/VICTOR-dataset` |
| `github.com/marcossyokese/FaQuAD` | `github.com/liafacom/faquad` |
| `github.com/JAugusto97/ToLD-Br` | `github.com/joaoaleite/ToLD-Br` (redirecionamento) |
| `github.com/uai-ufmg/tolde-br` | `github.com/JAugusto97/told-br` (problema no nome do dataset) |
| `github.com/pasquaini/CorPop` | Site UFRGS/TEXTECC |
| `github.com/edilsonacjr/TweetSentBR` | `bitbucket.org/HBrum/tweetsentbr` |

#### 3. Reestruturação de servidores acadêmicos (8 casos)
Servidores de universidades brasileiras migraram caminhos, subdomínios ou abandonaram IPs legados.

| Original | Correto |
|----------|---------|
| `inf.pucrs.br/linatural/wordpress/...blogset-br-english/` | Espelho no HuggingFace |
| `tycho.iel.unicamp.br/corpus/en/acesso.html` | `tycho.iel.unicamp.br/corpus/` |
| `tycho.iel.unicamp.br/` (raiz) | Erro de SSL — use as subpáginas |
| `143.107.183.175:21380/sucinto/` | `sites.icmc.usp.br/taspardo/sucinto/cstnews.html` |
| `nurcdigital.fale.ufal.br/` | `fale.ufal.br/projeto/nurcdigital/` |
| `letras.ufrj.br/nurc-rj/` | `nurcrj.letras.ufrj.br/` |
| `cic.unb.br/~teodecampos/LeNER-Br/` (robots.txt) | Use o espelho no GitHub |
| `opus.nlpl.eu/CCMatrix/` | `opus.nlpl.eu/datasets/CCMatrix` |

---

### Recursos com Ressalvas

| Corpus | Ressalva |
|--------|--------|
| **mC4** | ⚠ Descontinuado — use `allenai/c4` (branch mC4_3.1.0) |
| **OSCAR 23.01** | ⚠ Card acessível, mas download dos dados temporariamente suspenso |
| **Common Voice pt** | ⚠ Downloads migrados para o Mozilla Data Collective (out. 2025) |
| **Linguateca** | ⚠ Possível certificado SSL expirado; use as subpáginas diretamente |

---

### Tabela Completa da Auditoria

Para a tabela completa de 100 linhas com todos os códigos de status, veja o relatório completo em `assets/relatorio_corpora_ptbr_lncc_nii.docx` (Apêndice A).

---

### Correções Pós-Auditoria

Links quebrados apontados pela verificação automática mensal depois da auditoria de março de 2026.

| Data | Original | Correto | Motivo |
|------|----------|---------|--------|
| 2026-09 | `github.com/Instituto-IA-LNCC/manaca` | `github.com/Instituto-IA-LNCC/manaca-1b-base` | Repositório renomeado — HTTP 404 |
| 2026-09 | `www.nilc.icmc.usp.br/` | `sites.google.com/view/nilc-usp/` | Servidor legado fora do ar; portal do NILC migrado para o Google Sites |
| 2026-09 | `openalex.org/` | sem alteração | Falso positivo: HTTP 403 por proteção anti-bot. Domínio adicionado a `SKIP_PATTERNS` em `check-urls.py` |

---

### Verificação Automatizada

Um script Python para verificação periódica das URLs está disponível em [`scripts/check-urls.py`](../../scripts/check-urls.py).

Execute com:
```bash
pip install requests
python scripts/check-urls.py reports/corpora-survey/url-audit.md
```

Um workflow do GitHub Actions roda essa verificação automaticamente todo mês. Veja [`.github/workflows/url-checker.yml`](../../.github/workflows/url-checker.yml).

---

*Versão 0.1.0 — Março de 2026 | Projeto LLM-BR — LNCC × NII*

## English

### 100 Links Verified — March 2026

**Audited by:** Bruno Leonardo Santos Menezes  
**Date:** March 28, 2026

---

### Summary

| Status | Count | Description |
|--------|-------|-------------|
| ✅ Functional | 72 | URL loads correctly |
| ⚠ Functional with caveat | 4 | Accessible but with important operational issue |
| ❌ Broken → corrected | 28 | Original URL broken; correct URL identified |
| **Total** | **100** | **Effective availability of underlying corpora: ~97%** |

---

### Failure Patterns

#### 1. Hugging Face namespace migrations (9 cases)
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

#### 2. GitHub organization renames (8 cases)
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

#### 3. Academic server restructuring (8 cases)
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

### Resources with Important Caveats

| Corpus | Caveat |
|--------|--------|
| **mC4** | ⚠ Deprecated — use `allenai/c4` (branch mC4_3.1.0) |
| **OSCAR 23.01** | ⚠ Card accessible but data download temporarily suspended |
| **Common Voice pt** | ⚠ Downloads migrated to Mozilla Data Collective (Oct 2025) |
| **Linguateca** | ⚠ Possible expired SSL cert; use subpages directly |

---

### Full Audit Table

For the complete 100-row audit table with all status codes, see the full report in `assets/relatorio_corpora_ptbr_lncc_nii.docx` (Appendix A).

---

### Post-Audit Corrections

Broken links reported by the monthly automated check after the March 2026 audit.

| Date | Original | Correct | Reason |
|------|----------|---------|--------|
| 2026-09 | `github.com/Instituto-IA-LNCC/manaca` | `github.com/Instituto-IA-LNCC/manaca-1b-base` | Repository renamed — HTTP 404 |
| 2026-09 | `www.nilc.icmc.usp.br/` | `sites.google.com/view/nilc-usp/` | Legacy server unreachable; NILC portal moved to Google Sites |
| 2026-09 | `openalex.org/` | unchanged | False positive: HTTP 403 from bot protection. Domain added to `SKIP_PATTERNS` in `check-urls.py` |

---

### Automated Checker

A Python script for periodic URL verification is available at [`scripts/check-urls.py`](../../scripts/check-urls.py).

Run it with:
```bash
pip install requests
python scripts/check-urls.py reports/corpora-survey/url-audit.md
```

A GitHub Actions workflow runs this check automatically every month. See [`.github/workflows/url-checker.yml`](../../.github/workflows/url-checker.yml).

---

*Version 0.1.0 — March 2026 | LLM-BR Project — LNCC × NII*

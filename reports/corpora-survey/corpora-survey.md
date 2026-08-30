# Levantamento de Corpora em Português do Brasil | Brazilian Portuguese Corpora Survey

**[🇧🇷 Português](#português)** · **[🇬🇧 English](#english)**

## Português

### Subsídio para a Fase 0 do Projeto LLM-BR — Ação 6 do Plano de 90 Dias

**Elaborado por:** Bruno Leonardo Santos Menezes  
**Coordenação:** Prof. Fábio Porto  
**Instituição:** LNCC / Instituto de IA  
**Data:** Março de 2026 | **Versão:** 0.1.0

> **Documento completo:** disponível em formato Word (`.docx`) na pasta `assets/`.  
> Este arquivo Markdown apresenta a síntese navegável para uso no GitHub.

---

### Resumo Executivo

Este relatório é um produto da **Ação 6 do Plano de 90 Dias** (Fase 0 — Preparação e Alinhamento): levantar e documentar os corpora públicos disponíveis em PT-BR.

**Principais achados:**
- Mais de **150 corpus catalogados** em 6 categorias temáticas
- Volume total estimado para pré-treinamento: **> 500 bilhões de tokens**
- **GigaVerbo** (200B tokens) e **Jabuticaba** (139B tokens) são os maiores corpus públicos monolíngues
- Para fala: **> 8.000 horas** de áudio em PT-BR (~1.300h validadas manualmente)
- Domínio jurídico: **Ulysses Tesemõ** com 30,7 GiB de 159 fontes governamentais
- Auditoria de 100 URLs: **72 funcionais** | **4 com ressalvas** | **28 quebradas** (todas corrigidas — ver [url-audit.md](url-audit.md))

---

### 1. Corpus de Texto Geral e Web

| Corpus | Tamanho | Ano | Licença | URL Verificada |
|--------|---------|-----|---------|----------------|
| GigaVerbo (TucanoBR) | 200B tokens; 780 GB | 2024 | Apache 2.0 | [HuggingFace](https://huggingface.co/datasets/TucanoBR/GigaVerbo) |
| Jabuticaba | 139B tokens; 669 GB | 2025 | CC BY-SA 4.0 | [HuggingFace (gated)](https://huggingface.co/datasets/soberania/jabuticaba) |
| ClassiCC-PT | ~120B tokens | 2025 | Variável | [arXiv:2509.08824](https://arxiv.org/abs/2509.08824) |
| brWaC | 2,68B tokens; 3,53M docs | 2017/2018 | Acadêmica | [HuggingFace](https://huggingface.co/datasets/UFRGS/brwac) |
| Corpus Carolina / Ada | ~802M tokens | 2022–2025 | CC BY 4.0 | [HuggingFace](https://huggingface.co/datasets/carolina-c4ai/corpus-carolina) |
| BlogSet-BR | 2,1B palavras; 7,4M posts | 2018 | Apache 2.0 | [HuggingFace](https://huggingface.co/datasets/thegoodfellas/blogset-br) |
| Corpus do Português | 1B+ palavras; NOW: 2,5B | Contínuo | Pesquisa | [corpusdoportugues.org](https://www.corpusdoportugues.org/) |
| Corpus Brasileiro | ~870M palavras | 2010 | Acadêmica | [Sketch Engine](https://www.sketchengine.eu/corpus-brasileiro/) |
| Portuguese-PD | 672M palavras | 2024 | Dom. Público | [HuggingFace](https://huggingface.co/datasets/PleIAs/Portuguese-PD) |
| Pt-Corpus-Instruct | 6,2B tokens | 2024 | Apache 2.0 | [HuggingFace](https://huggingface.co/datasets/nicholasKluge/Pt-Corpus-Instruct) |
| CETENFolha | ~24M palavras | 2000 | Linguateca | [linguateca.pt](https://www.linguateca.pt/cetenfolha/) |
| Mac-Morpho | ~1,1M palavras | 2003+ | CC BY 4.0 | [HuggingFace](https://huggingface.co/datasets/nilc-nlp/mac_morpho) |
| PHPB | Milhares de docs; séc. XVIII–XX | 2002+ | Acadêmica | [Google Sites](https://sites.google.com/site/corporaphpb/) |
| SUBTLEX-PT-BR | 61M palavras | 2015 | CC BY-NC-ND 4.0 | [OSF](https://osf.io/vb5yp/) |
| CorPop | ~685K tokens | 2018 | Acadêmica | [OpenCor](https://opencor.gitlab.io/corpora/pasquaini18corpop/) |
| LexPorBR | >215K entradas | 2017 | CC BY-NC-SA 4.0 | [lexicodoportugues.com](http://www.lexicodoportugues.com/) |

---

### 2. Corpus Científicos e Acadêmicos

| Corpus | Tamanho | Ano | Licença | URL Verificada |
|--------|---------|-----|---------|----------------|
| SciELO Full-Text Parallel | 2,83M pares PT↔EN | 2018 | Pública | [ACL Anthology](https://aclanthology.org/L18-1546/) |
| SciELO (abstracts) | ~86K pares PT/EN | 2016 | Pública | [HuggingFace](https://huggingface.co/datasets/community-datasets/scielo) |
| CAPES Parallel | ~240K pares | 2018 | Pública | [OPUS](https://opus.nlpl.eu/datasets/CAPES) |
| BDTD | ~900K teses/dissertações | Contínuo | Acesso aberto | [bdtd.ibict.br](https://bdtd.ibict.br) |
| Common Corpus (PleIAs) | 1B+ tokens pt | 2024 | Pública | [HuggingFace](https://huggingface.co/datasets/PleIAs/common_corpus) |
| Tycho Brahe | ~3,37M tokens; séc. 1500–1900 | 1998+ | Acadêmica | [Unicamp](https://www.tycho.iel.unicamp.br/corpus/) |
| OpenAlex (API) | Milhões de registros | Contínuo | CC0 | [openalex.org](https://openalex.org/) |

---

### 3. Corpus Jornalísticos e de Notícias

| Corpus | Tamanho | Ano | Licença | URL Verificada |
|--------|---------|-----|---------|----------------|
| CSTNews | 140 textos; 50 clusters | 2008 | Acadêmica | [sites.icmc.usp.br](https://sites.icmc.usp.br/taspardo/sucinto/cstnews.html) |
| TeMário | 250 textos; 61K palavras | 2003 | Acadêmica | [linguateca.pt](https://www.linguateca.pt/Repositorio/TeMario/) |
| Porttinari UD | 167K artigos; 94,6M tokens | 2023+ | CC BY 4.0 | [GitHub UD](https://github.com/UniversalDependencies/UD_Portuguese-Porttinari) |
| Fake.br Corpus | 7.200 notícias (50/50) | 2018 | Acadêmica | [GitHub](https://github.com/roneysco/Fake.br-Corpus) |
| FACTCK.BR | 1.309 claims | 2019 | MIT | [GitHub](https://github.com/jghm-f/FACTCK.BR) |
| FactNews | 6.191 sentenças | 2023 | Acadêmica | [GitHub](https://github.com/franciellevargas/FactNews) |
| News-Crawl-PT (WMT) | Milhões de sentenças | Contínuo | Pública | [statmt.org](https://data.statmt.org/news-crawl/pt/) |

---

### 4. Corpus de Fala e Áudio

| Corpus | Tamanho | Licença | URL Verificada |
|--------|---------|---------|----------------|
| CORAA ASR v1.1 | 290,77h; 400K+ segmentos | CC BY-NC-ND 4.0 | [GitHub](https://github.com/nilc-nlp/CORAA) |
| CORAA NURC-SP Audio | 239,30h; 170K+ segmentos | CC BY-NC-ND 4.0 | [HuggingFace](https://huggingface.co/datasets/nilc-nlp/CORAA-NURC-SP-Audio-Corpus) |
| CORAA NURC-SP Minimal | ~18h; ~155K palavras | CC BY-NC-ND 4.0 | [PORTULAN CLARIN](https://portulanclarin.net/repository/browse/391c9bf232cd11ed84e202420a87010e52130324c1fe4a2981c00cbce6261766/) |
| CORAA Certas Palavras | ~63h; 163 episódios | CC BY-NC 4.0 | [Zenodo](https://zenodo.org/records/6794924) |
| C-ORAL-BRASIL I | ~21h; 208K palavras | Acadêmica | [c-oral-brasil.org](https://www.c-oral-brasil.org/english-site/index.php) |
| MLS Portuguese | ~284h; 62 falantes | CC BY 4.0 | [OpenSLR](https://openslr.org/94/) |
| Common Voice pt | 150–200h+ validadas | CC0/CC BY-SA | [Mozilla Data Collective](https://datacollective.mozillafoundation.org/datasets) |
| Multilingual TEDx pt | ~164h | CC BY-NC-ND 4.0 | [OpenSLR](https://openslr.org/100/) |
| CETUC / FalaBrasil | ~145h | Pública | [GitHub](https://github.com/falabrasil/speech-datasets) |
| BRSD v2 | ~157,5h | Pública | [igormq.github.io](https://igormq.github.io/datasets/) |
| TTS-Portuguese Corpus | 10,5h; 1 falante | Pública | [GitHub](https://github.com/Edresson/TTS-Portuguese-Corpus) |
| FLEURS pt-BR | ~12h | CC BY 4.0 | [HuggingFace](https://huggingface.co/datasets/google/fleurs) |
| NURC Digital (Recife) | 32 inquéritos | Uso acadêmico | [fale.ufal.br](https://fale.ufal.br/projeto/nurcdigital/) |

---

### 5. Corpus Especializados

#### 5.1 Jurídico

| Corpus | Tamanho | Licença | URL Verificada |
|--------|---------|---------|----------------|
| Ulysses Tesemõ | 3,5M+ arquivos; 30,7 GiB | Pública | [GitHub](https://github.com/ulysses-camara/ulysses-tesemo) |
| LegalPT_dedup | ~11,9M registros | Pública | [HuggingFace](https://huggingface.co/datasets/eduagarcia/LegalPT_dedup) |
| LeNER-Br | 70 docs; NER jurídico | Verificar pacote | [GitHub](https://github.com/peluz/lener-br) |
| UlyssesNER-Br | 150 PLs; 138.741 tokens | Acadêmica | [GitHub](https://github.com/ulysses-camara/ulysses-ner-br) |

#### 5.2 Benchmarks e Avaliação

| Corpus | Tarefa | Licença | URL Verificada |
|--------|--------|---------|----------------|
| ASSIN 2 | STS + NLI | Acadêmica | [HuggingFace](https://huggingface.co/datasets/nilc-nlp/assin2) |
| FaQuAD | QA extrativo | Acadêmica | [GitHub](https://github.com/liafacom/faquad) |
| XQuAD (pt) | QA | CC BY-SA 4.0 | [HuggingFace](https://huggingface.co/datasets/google/xquad) |
| ToLD-Br | Toxicidade | Acadêmica | [HuggingFace](https://huggingface.co/datasets/JAugusto97/told-br) |
| HateBR | Discurso de ódio | Acadêmica | [GitHub](https://github.com/franciellevargas/HateBR) |
| B2W-Reviews01 | Sentimento | CC BY 4.0 | [HuggingFace](https://huggingface.co/datasets/ruanchaves/b2w-reviews01) |
| AMR-BP | Semântica AMR | CC BY-NC-SA | [GitHub](https://github.com/nilc-nlp/AMR-BP) |
| PorSimplesSent | Simplificação | Acadêmica | [GitHub](https://github.com/sidleal/porsimplessent) |
| UD Portuguese-Bosque | Treebank UD | CC BY-SA 4.0 | [GitHub](https://github.com/UniversalDependencies/UD_Portuguese-Bosque) |
| UD Portuguese-PetroGold | Treebank UD | CC BY-SA 4.0 | [GitHub](https://github.com/UniversalDependencies/UD_Portuguese-PetroGold) |
| UD Portuguese-GSD | Treebank UD | CC BY-SA 4.0 | [GitHub](https://github.com/UniversalDependencies/UD_Portuguese-GSD) |
| UD Portuguese-Porttinari | Treebank UD | CC BY 4.0 | [GitHub](https://github.com/UniversalDependencies/UD_Portuguese-Porttinari) |

---

### 6. Corpus Multilíngues com Cobertura de PT-BR

| Corpus | Porção pt (est.) | Licença | URL Verificada |
|--------|-----------------|---------|----------------|
| CulturaX | 150–250B tokens | Gated | [HuggingFace](https://huggingface.co/datasets/uonlp/CulturaX) |
| mC4 → allenai/c4 | ~100B tokens | ODC-BY ⚠ depreciado | [HuggingFace](https://huggingface.co/datasets/allenai/c4) |
| FineWeb-2 | Grande | ODC-By 1.0 | [HuggingFace](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) |
| MADLAD-400 | 50–100B+ tokens | Apache 2.0 | [HuggingFace](https://huggingface.co/datasets/allenai/MADLAD-400) |
| CC-100 | ~82 GB | Pública | [statmt.org](https://data.statmt.org/cc-100/) |
| OSCAR 23.01 | ~40–60 GB | Gated ⚠ suspenso | [HuggingFace](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301) |
| HPLT 2.0 cleaned | Grande | CC0 | [HuggingFace](https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned) |
| OpenSubtitles (OPUS) | ~60M pares EN↔ptBR | Pública | [OPUS](https://opus.nlpl.eu/datasets/OpenSubtitles) |
| FLORES+ | 3.001 sentenças pt-BR | CC BY-SA 4.0 | [HuggingFace](https://huggingface.co/datasets/openlanguagedata/flores_plus) |
| NLLB bitext | Grande (en-pt) | CC BY-SA 4.0 | [HuggingFace](https://huggingface.co/datasets/allenai/nllb) |

---

### Portais de Descoberta / Discovery Portals

| Portal | URL | Tipo |
|--------|-----|------|
| Linguateca / AC-DC | [linguateca.pt](https://www.linguateca.pt/) | Portal ⚠ SSL |
| NILC (USP/ICMC) | [nilc.icmc.usp.br](http://www.nilc.icmc.usp.br/) | Portal |
| Hugging Face Hub (nilc-nlp) | [huggingface.co/nilc-nlp](https://huggingface.co/nilc-nlp) | Hub |
| OPUS | [opus.nlpl.eu](https://opus.nlpl.eu/) | Hub paralelo |
| awesome-portuguese-nlp | [github.com/ajdavidl/Portuguese-NLP](https://github.com/ajdavidl/Portuguese-NLP) | Lista curada |
| PORTULAN CLARIN | [portulanclarin.net](https://portulanclarin.net/) | Repositório |
| Projeto Ulysses (org) | [github.com/ulysses-camara](https://github.com/ulysses-camara) | Org GitHub |

---

### Como Contribuir com Novos Corpus / How to Add a New Corpus

Use o template de Issue disponível em [`.github/ISSUE_TEMPLATE/corpus-addition.md`](../../.github/ISSUE_TEMPLATE/corpus-addition.md).

Campos obrigatórios / Required fields:
- Nome oficial / Official name
- Tamanho / Size
- Licença / License
- URL verificada / Verified URL
- Referência bibliográfica / Bibliographic reference
- Breve descrição / Brief description

---

*Versão 0.1.0 — Março de 2026 | Projeto LLM-BR — LNCC × NII*

## English

### Supporting Phase 0 of the LLM-BR Project — Action 6 of the 90-Day Plan

**Prepared by:** Bruno Leonardo Santos Menezes  
**Coordination:** Prof. Fábio Porto  
**Institution:** LNCC / AI Institute  
**Date:** March 2026 | **Version:** 0.1.0

> **Full document:** available in Word format (`.docx`) in the `assets/` folder.  
> This Markdown file provides the navigable summary for GitHub use.

---

### Executive Summary

This report is the product of **Action 6 of the 90-Day Plan** (Phase 0 — Preparation and Alignment): survey and document publicly available PT-BR corpora.

**Key findings:**
- More than **150 corpora catalogued** across 6 thematic categories
- Estimated total volume for pre-training: **> 500 billion tokens**
- **GigaVerbo** (200B tokens) and **Jabuticaba** (139B tokens) are the largest public monolingual corpora
- For speech: **> 8,000 hours** of Brazilian Portuguese audio (~1,300h manually validated)
- Legal domain: **Ulysses Tesemõ** with 30.7 GiB from 159 government sources
- URL audit of 100 links: **72 functional** | **4 with caveats** | **28 broken** (all corrected — see [url-audit.md](url-audit.md))

---

### 1. General Text and Web Corpora

| Corpus | Size | Year | License | Verified URL |
|--------|------|------|---------|-------------|
| GigaVerbo (TucanoBR) | 200B tokens; 780 GB | 2024 | Apache 2.0 | [HuggingFace](https://huggingface.co/datasets/TucanoBR/GigaVerbo) |
| Jabuticaba | 139B tokens; 669 GB | 2025 | CC BY-SA 4.0 | [HuggingFace (gated)](https://huggingface.co/datasets/soberania/jabuticaba) |
| ClassiCC-PT | ~120B tokens | 2025 | Variable | [arXiv:2509.08824](https://arxiv.org/abs/2509.08824) |
| brWaC | 2.68B tokens; 3.53M docs | 2017/2018 | Academic | [HuggingFace](https://huggingface.co/datasets/UFRGS/brwac) |
| Corpus Carolina / Ada | ~802M tokens | 2022–2025 | CC BY 4.0 | [HuggingFace](https://huggingface.co/datasets/carolina-c4ai/corpus-carolina) |
| BlogSet-BR | 2.1B words; 7.4M posts | 2018 | Apache 2.0 | [HuggingFace](https://huggingface.co/datasets/thegoodfellas/blogset-br) |
| Corpus do Português | 1B+ words; NOW: 2.5B | Ongoing | Research | [corpusdoportugues.org](https://www.corpusdoportugues.org/) |
| Corpus Brasileiro | ~870M words | 2010 | Academic | [Sketch Engine](https://www.sketchengine.eu/corpus-brasileiro/) |
| Portuguese-PD | 672M words | 2024 | Public Domain | [HuggingFace](https://huggingface.co/datasets/PleIAs/Portuguese-PD) |
| Pt-Corpus-Instruct | 6.2B tokens | 2024 | Apache 2.0 | [HuggingFace](https://huggingface.co/datasets/nicholasKluge/Pt-Corpus-Instruct) |
| CETENFolha | ~24M words | 2000 | Linguateca | [linguateca.pt](https://www.linguateca.pt/cetenfolha/) |
| Mac-Morpho | ~1.1M words | 2003+ | CC BY 4.0 | [HuggingFace](https://huggingface.co/datasets/nilc-nlp/mac_morpho) |
| PHPB | Thousands of docs; 18th–20th c. | 2002+ | Academic | [Google Sites](https://sites.google.com/site/corporaphpb/) |
| SUBTLEX-PT-BR | 61M words | 2015 | CC BY-NC-ND 4.0 | [OSF](https://osf.io/vb5yp/) |
| CorPop | ~685K tokens | 2018 | Academic | [OpenCor](https://opencor.gitlab.io/corpora/pasquaini18corpop/) |
| LexPorBR | >215K entries | 2017 | CC BY-NC-SA 4.0 | [lexicodoportugues.com](http://www.lexicodoportugues.com/) |

---

### 2. Scientific and Academic Corpora

| Corpus | Size | Year | License | Verified URL |
|--------|------|------|---------|-------------|
| SciELO Full-Text Parallel | 2.83M PT↔EN pairs | 2018 | Public | [ACL Anthology](https://aclanthology.org/L18-1546/) |
| SciELO (abstracts) | ~86K PT/EN pairs | 2016 | Public | [HuggingFace](https://huggingface.co/datasets/community-datasets/scielo) |
| CAPES Parallel | ~240K pairs | 2018 | Public | [OPUS](https://opus.nlpl.eu/datasets/CAPES) |
| BDTD | ~900K theses/dissertations | Ongoing | Open access | [bdtd.ibict.br](https://bdtd.ibict.br) |
| Common Corpus (PleIAs) | 1B+ tokens pt | 2024 | Public | [HuggingFace](https://huggingface.co/datasets/PleIAs/common_corpus) |
| Tycho Brahe | ~3.37M tokens; 16th–19th c. | 1998+ | Academic | [Unicamp](https://www.tycho.iel.unicamp.br/corpus/) |
| OpenAlex (API) | Millions of records | Ongoing | CC0 | [openalex.org](https://openalex.org/) |

---

### 3. Journalistic and News Corpora

| Corpus | Size | Year | License | Verified URL |
|--------|------|------|---------|-------------|
| CSTNews | 140 texts; 50 clusters | 2008 | Academic | [sites.icmc.usp.br](https://sites.icmc.usp.br/taspardo/sucinto/cstnews.html) |
| TeMário | 250 texts; 61K words | 2003 | Academic | [linguateca.pt](https://www.linguateca.pt/Repositorio/TeMario/) |
| Porttinari UD | 167K articles; 94.6M tokens | 2023+ | CC BY 4.0 | [GitHub UD](https://github.com/UniversalDependencies/UD_Portuguese-Porttinari) |
| Fake.br Corpus | 7,200 news (50/50) | 2018 | Academic | [GitHub](https://github.com/roneysco/Fake.br-Corpus) |
| FACTCK.BR | 1,309 claims | 2019 | MIT | [GitHub](https://github.com/jghm-f/FACTCK.BR) |
| FactNews | 6,191 sentences | 2023 | Academic | [GitHub](https://github.com/franciellevargas/FactNews) |
| News-Crawl-PT (WMT) | Millions of sentences | Ongoing | Public | [statmt.org](https://data.statmt.org/news-crawl/pt/) |

---

### 4. Speech and Audio Corpora

| Corpus | Size | License | Verified URL |
|--------|------|---------|-------------|
| CORAA ASR v1.1 | 290.77h; 400K+ segments | CC BY-NC-ND 4.0 | [GitHub](https://github.com/nilc-nlp/CORAA) |
| CORAA NURC-SP Audio | 239.30h; 170K+ segments | CC BY-NC-ND 4.0 | [HuggingFace](https://huggingface.co/datasets/nilc-nlp/CORAA-NURC-SP-Audio-Corpus) |
| CORAA NURC-SP Minimal | ~18h; ~155K words | CC BY-NC-ND 4.0 | [PORTULAN CLARIN](https://portulanclarin.net/repository/browse/391c9bf232cd11ed84e202420a87010e52130324c1fe4a2981c00cbce6261766/) |
| CORAA Certas Palavras | ~63h; 163 episodes | CC BY-NC 4.0 | [Zenodo](https://zenodo.org/records/6794924) |
| C-ORAL-BRASIL I | ~21h; 208K words | Academic | [c-oral-brasil.org](https://www.c-oral-brasil.org/english-site/index.php) |
| MLS Portuguese | ~284h; 62 speakers | CC BY 4.0 | [OpenSLR](https://openslr.org/94/) |
| Common Voice pt | 150–200h+ validated | CC0/CC BY-SA | [Mozilla Data Collective](https://datacollective.mozillafoundation.org/datasets) |
| Multilingual TEDx pt | ~164h | CC BY-NC-ND 4.0 | [OpenSLR](https://openslr.org/100/) |
| CETUC / FalaBrasil | ~145h | Public | [GitHub](https://github.com/falabrasil/speech-datasets) |
| BRSD v2 | ~157.5h | Public | [igormq.github.io](https://igormq.github.io/datasets/) |
| TTS-Portuguese Corpus | 10.5h; 1 speaker | Public | [GitHub](https://github.com/Edresson/TTS-Portuguese-Corpus) |
| FLEURS pt-BR | ~12h | CC BY 4.0 | [HuggingFace](https://huggingface.co/datasets/google/fleurs) |
| NURC Digital (Recife) | 32 inquiries | Academic use | [fale.ufal.br](https://fale.ufal.br/projeto/nurcdigital/) |

---

### 5. Specialized Corpora

#### 5.1 Legal

| Corpus | Size | License | Verified URL |
|--------|------|---------|-------------|
| Ulysses Tesemõ | 3.5M+ files; 30.7 GiB | Public | [GitHub](https://github.com/ulysses-camara/ulysses-tesemo) |
| LegalPT_dedup | ~11.9M records | Public | [HuggingFace](https://huggingface.co/datasets/eduagarcia/LegalPT_dedup) |
| LeNER-Br | 70 legal docs; NER | Check package | [GitHub](https://github.com/peluz/lener-br) |
| UlyssesNER-Br | 150 bills; 138,741 tokens | Academic | [GitHub](https://github.com/ulysses-camara/ulysses-ner-br) |

#### 5.2 Benchmarks and Evaluation

| Corpus | Task | License | Verified URL |
|--------|------|---------|-------------|
| ASSIN 2 | STS + NLI | Academic | [HuggingFace](https://huggingface.co/datasets/nilc-nlp/assin2) |
| FaQuAD | Extractive QA | Academic | [GitHub](https://github.com/liafacom/faquad) |
| XQuAD (pt) | QA | CC BY-SA 4.0 | [HuggingFace](https://huggingface.co/datasets/google/xquad) |
| ToLD-Br | Toxicity | Academic | [HuggingFace](https://huggingface.co/datasets/JAugusto97/told-br) |
| HateBR | Hate speech | Academic | [GitHub](https://github.com/franciellevargas/HateBR) |
| B2W-Reviews01 | Sentiment | CC BY 4.0 | [HuggingFace](https://huggingface.co/datasets/ruanchaves/b2w-reviews01) |
| AMR-BP | AMR semantics | CC BY-NC-SA | [GitHub](https://github.com/nilc-nlp/AMR-BP) |
| PorSimplesSent | Simplification | Academic | [GitHub](https://github.com/sidleal/porsimplessent) |
| UD Portuguese-Bosque | UD treebank | CC BY-SA 4.0 | [GitHub](https://github.com/UniversalDependencies/UD_Portuguese-Bosque) |
| UD Portuguese-GSD | UD treebank | CC BY-SA 4.0 | [GitHub](https://github.com/UniversalDependencies/UD_Portuguese-GSD) |
| UD Portuguese-PetroGold | UD treebank | CC BY-SA 4.0 | [GitHub](https://github.com/UniversalDependencies/UD_Portuguese-PetroGold) |
| UD Portuguese-Porttinari | UD treebank | CC BY 4.0 | [GitHub](https://github.com/UniversalDependencies/UD_Portuguese-Porttinari) |

---

### 6. Multilingual Corpora with PT-BR Coverage

| Corpus | PT portion (est.) | License | Verified URL |
|--------|------------------|---------|-------------|
| CulturaX | 150–250B tokens | Gated | [HuggingFace](https://huggingface.co/datasets/uonlp/CulturaX) |
| mC4 → allenai/c4 | ~100B tokens | ODC-BY ⚠ deprecated | [HuggingFace](https://huggingface.co/datasets/allenai/c4) |
| FineWeb-2 | Large | ODC-By 1.0 | [HuggingFace](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) |
| MADLAD-400 | 50–100B+ tokens | Apache 2.0 | [HuggingFace](https://huggingface.co/datasets/allenai/MADLAD-400) |
| CC-100 | ~82 GB | Public | [statmt.org](https://data.statmt.org/cc-100/) |
| OSCAR 23.01 | ~40–60 GB | Gated ⚠ suspended | [HuggingFace](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301) |
| HPLT 2.0 cleaned | Large | CC0 | [HuggingFace](https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned) |
| OpenSubtitles (OPUS) | ~60M EN↔ptBR pairs | Public | [OPUS](https://opus.nlpl.eu/datasets/OpenSubtitles) |
| FLORES+ | 3,001 sentences pt-BR | CC BY-SA 4.0 | [HuggingFace](https://huggingface.co/datasets/openlanguagedata/flores_plus) |
| NLLB bitext | Large (en-pt) | CC BY-SA 4.0 | [HuggingFace](https://huggingface.co/datasets/allenai/nllb) |

---

### How to Add a New Corpus

Use the Issue template available at [`.github/ISSUE_TEMPLATE/corpus-addition.md`](../../.github/ISSUE_TEMPLATE/corpus-addition.md).

Required fields:
- Official name
- Size (tokens, words, or audio hours)
- License
- Verified URL
- Bibliographic reference
- Brief description

---

*Version 0.1.0 — March 2026 | LLM-BR Project — LNCC × NII*

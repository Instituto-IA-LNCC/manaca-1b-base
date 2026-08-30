# Contributing to LLM-BR | Guia de Contribuição

🇧🇷 [Português](#-português) | 🇬🇧 [English](#-english)

---

## 🇧🇷 Português

Obrigado pelo interesse em contribuir com o **Manacá-1B**! Este repositório é mantido por **Bruno Menezes**, **Carlos Cardoso** e **Prof. Fábio Porto** (LNCC) como mantenedores principais, e está aberto a contribuições de pesquisadores, alunos e colaboradores do projeto.

### Como contribuir

#### 1. Adicionando um novo corpus ao levantamento

Use o template de Issue **"Corpus Addition"** disponível em `.github/ISSUE_TEMPLATE/corpus-addition.md`.

Inclua obrigatoriamente:
- Nome oficial do corpus
- Tamanho (tokens, palavras ou horas de áudio)
- Ano de publicação
- Licença exata
- URL verificada e funcional
- Referência bibliográfica (se houver DOI, inclua)
- Breve descrição do conteúdo e uso

#### 2. Corrigindo URLs quebradas

Se você identificar um link quebrado no [relatório de auditoria](reports/corpora-survey/url-audit.md):
1. Abra uma Issue com o label `broken-url`
2. Informe a URL original, o status e a URL correta
3. Ou submeta diretamente um Pull Request corrigindo o arquivo

#### 3. Atualizando documentos

Para atualizar o Relatório de Corpora ou a documentação de avaliação:
1. Faça um fork do repositório
2. Crie uma branch descritiva: `git checkout -b update/corpus-survey-2026-q2`
3. Faça suas alterações
4. Abra um Pull Request com descrição clara do que foi alterado e por quê

#### 4. Propondo novas seções ou relatórios

Use o template de Issue **"Feature Request"** e descreva a proposta. Os mantenedores avaliarão e, se aprovada, abrirão uma branch dedicada.

### Padrões de qualidade

- Todos os links devem ser verificados antes de serem adicionados
- Referências bibliográficas devem ter DOI ou URL permanente sempre que possível
- Documentos em português devem ter versão em inglês (e vice-versa) — mesmo que resumida
- Commits devem seguir o padrão: `type: short description` (ex: `docs: add SUBTLEX-PT-BR to corpus survey`)

### Governança

Os mantenedores principais têm direito de revisão e merge em qualquer PR. Contribuições de alunos e colaboradores são revisadas antes de serem incorporadas ao branch `main`.

---

## 🇬🇧 English

Thank you for your interest in contributing to **Manacá-1B**! This repository is maintained by **Bruno Menezes**, **Carlos Cardoso**, and **Prof. Fábio Porto** (LNCC) as lead maintainers, and is open to contributions from researchers, students, and project collaborators.

### How to contribute

#### 1. Adding a new corpus to the survey

Use the **"Corpus Addition"** Issue template available at `.github/ISSUE_TEMPLATE/corpus-addition.md`.

Required information:
- Official corpus name
- Size (tokens, words, or audio hours)
- Publication year
- Exact license
- Verified and functional URL
- Bibliographic reference (include DOI if available)
- Brief description of content and use case

#### 2. Fixing broken URLs

If you identify a broken link in the [URL audit report](reports/corpora-survey/url-audit.md):
1. Open an Issue with the label `broken-url`
2. Provide the original URL, status, and the correct URL
3. Or directly submit a Pull Request fixing the file

#### 3. Updating documents

To update the Corpora Survey or the evaluation documentation:
1. Fork the repository
2. Create a descriptive branch: `git checkout -b update/corpus-survey-2026-q2`
3. Make your changes
4. Open a Pull Request with a clear description of what was changed and why

#### 4. Proposing new sections or reports

Use the **"Feature Request"** Issue template and describe your proposal. Maintainers will evaluate it and, if approved, open a dedicated branch.

### Quality standards

- All links must be verified before being added
- Bibliographic references should include a DOI or permanent URL whenever possible
- Portuguese documents should have an English version (and vice versa) — even if summarized
- Commits should follow the pattern: `type: short description` (e.g., `docs: add SUBTLEX-PT-BR to corpus survey`)

### Governance

Lead maintainers have review and merge rights on all PRs. Contributions from students and collaborators are reviewed before being merged into the `main` branch.

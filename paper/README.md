# Preprint (arXiv) do Manacá-1B

`manaca1b.tex` é o manuscrito, escrito para submissão no arXiv: um único `.tex`
self-contained (bibliografia embutida em `thebibliography`, só pacotes padrão do
TeXLive) mais uma figura.

## Compilar

```bash
cd paper
pdflatex manaca1b
pdflatex manaca1b      # segunda passada resolve referencias e citacoes
```

A figura vem de `../docs/evaluation/benchmarks_paper_en.{pdf,png}` (o
`\graphicspath` já aponta para lá). O PDF vetorial é gerado por
`python scripts/eval/plot_benchmarks_paper.py`; se ele não existir, o pdflatex usa
o PNG commitado. Para a versão final, gere o PDF (vetorial).

## Submeter no arXiv

O arXiv compila com pdfLaTeX (sem `-shell-escape`). Monte o pacote de submissão
com os arquivos ao lado do `.tex`:

```bash
mkdir -p arxiv-submit
cp manaca1b.tex arxiv-submit/
python ../scripts/eval/plot_benchmarks_paper.py          # painel de benchmarks (fig:panel)
python ../scripts/eval/plot_training_dynamics.py         # dinamica de treino (fig:dyn)
cp ../docs/evaluation/benchmarks_paper_en.pdf arxiv-submit/
cp ../docs/training/training_dynamics_en.pdf arxiv-submit/
# no manaca1b.tex, o \graphicspath ja inclui './', entao as figuras ao lado resolvem.
cd arxiv-submit && zip ../manaca1b-arxiv.zip *            # suba este zip no arXiv
```

O manuscrito usa duas figuras: `benchmarks_paper_en` (painel 2x2 dos quatro
benchmarks) e `training_dynamics_en` (loss/grad-norm/LR do pre-treino). Ambas saem
em PDF vetorial + PNG; o `\graphicspath` aponta para `../docs/evaluation/` e
`../docs/training/`.

Categoria sugerida: `cs.CL` (primária), `cs.LG` (secundária).

## Antes de submeter (pendências marcadas no `.tex`)

- Preencher a lista de **autores** e afiliações (bloco `\author`/`\affil`, ver
  `% TODO(authors)`), incluindo os colaboradores do LNCC e do NII / LLM-jp.
- Preencher os **Acknowledgments** (financiamento, computação, colaboração).
- Conferir os **identificadores/anos** das referências (alguns arXiv ids foram
  preenchidos por memória; valide cada um) e trocar por DOI/venue quando houver.
- Test-compile num TeX completo (Overleaf ou TeXLive local): este ambiente não
  tinha engine LaTeX, então a checagem aqui foi de sintaxe/estrutura, não de build.

## Notas de estilo

- Escrito **sem travessão** (em dash), a pedido; en dash (`--`) só em intervalos.
- Voz de pesquisador sênior, honesto quanto ao escopo: modelo base, ~41,9 B tokens
  de atualização (~24 tok/param, ~2 épocas de um corpus curado de ~20,1 B, próximo
  do compute-optimal), quatro benchmarks, com o achado do tokenizador e a conversão
  Megatron→HF em destaque e trabalho futuro explícito.

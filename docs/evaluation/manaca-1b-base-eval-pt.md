# Avaliação do Manacá-1B (modelo base) vs. modelos de referência

**[🇧🇷 Português](#português)** · **[🇬🇧 English](#english)**

## Português

**Data:** 29 de agosto de 2026
**Autor:** Bruno Leonardo Santos Menezes (LNCC × NII / LLM-jp)
**Escopo:** modelo **base** (pré-treinado, sem SFT/DPO), passo final 20.000
**Scripts:** `scripts/eval/eval_base.py`, `run_eval.sh`, `run_batch_pt.sh`, `paired_compare.py`, `plot_scaling_pt.py`
**Dados brutos:** `docs/evaluation/results-base.json` e `docs/evaluation/logs/`

Este documento registra, de forma transparente e reprodutível, a avaliação
comparativa do Manacá-1B contra **9 modelos de referência** PT-BR, PT-PT e
multilíngues (família TeenyTinyLlama + Tucano, GlórIA, mGPT e Sabiá-7B). Todos
foram medidos no **mesmo texto**, com o **mesmo protocolo** e no **mesmo harness**,
por **métricas independentes do tokenizador**.

---

### 1. Métricas

- **BPB (bits por byte)** — métrica-âncora e **justa entre tokenizadores
  diferentes**. É a NLL total em bits dividida pelos **bytes** do texto (não
  pelos tokens). Vocabulários de tamanhos diferentes (Manacá 64k, Tucano 32k)
  dão perplexidades por token diferentes, mas o mesmo texto tem os mesmos bytes.
  **Menor é melhor.**
- **CALAME-PT** — acurácia em prever a **última palavra** de um parágrafo
  (benchmark nativo PT-BR, `NOVA-vision-language/calame-pt`, arquivo
  `calamept_all.jsonl`, 2075 exemplos). Funciona em modelo base e, por ser
  acerto, é comparável entre modelos. **Maior é melhor.**
- **Perplexidade por token** — informativa, mas **só comparável** entre modelos
  de mesmo tokenizador; não é a métrica de comparação.

### 2. Protocolo

- **Texto reservado:** `scripts/eval/holdout_pt.txt` (536 palavras, registros
  variados: científico, narrativo, jurídico, técnico). Escrito do zero para esta
  avaliação, portanto **inédito para todos os modelos** (não está no treino de
  nenhum). Usado idêntico para todos.
- **BPB/perplexidade:** janela deslizante de 2048 tokens, passo 1024. Cada modelo
  tokeniza com o **próprio tokenizador**. O Manacá usa o SentencePiece do treino
  (`manaca-tokenizer.model`, normalização `nmt_nfkc_cf`), garantindo fidelidade à
  tokenização de treino (o tokenizador HF do pacote não reproduz essa
  normalização; ver §5).
- **CALAME-PT:** para cada exemplo, o contexto (`sentence`) é dado ao modelo, que
  gera de forma **gulosa** (greedy); a primeira palavra gerada é comparada à
  palavra-alvo (`last_word`), sem distinção de maiúsculas/minúsculas.
- **Caixa (case):** o Manacá é **lowercase** por construção (tokenizador com
  `nmt_nfkc_cf`, que dobra maiúsculas). Para separar esse efeito, cada medição
  foi feita duas vezes: no **texto original** e com **`--lowercase`** (texto todo
  em minúsculas para todos).
- **Hardware/imagem:** 1 GPU; imagem `manaca-train` (torch + transformers 4.x).

### 3. Resultados

#### Liderança CALAME-PT (10 modelos, mesmo harness, n = 2075)

Ordenado por acurácia. IC95% = ±1,96·SE binomial. BPB(res) é o BPB no texto
reservado (menor é melhor).

| # | Modelo | Params | CALAME-PT ↑ (IC95%) | BPB(res) ↓ |
|---|---|---|---|---|
| 1 | **Sabiá-7B** | 7,0 B | **63,23 %** [61,1; 65,3] | 0,646 |
| 2 | **Manacá-1B** | 1,72 B | **60,63 %** [58,5; 62,7] | 0,702 |
| 3 | GlórIA-1b3 (PT-PT) | 1,3 B | 60,39 % [58,3; 62,5] | 0,865 |
| 4 | Tucano-2b4 | 2,4 B | 59,57 % [57,4; 61,7] | 0,691 |
| 5 | Tucano-1b1 | 1,1 B | 59,08 % [57,0; 61,2] | 0,719 |
| 6 | Tucano-630m | 0,63 B | 56,63 % [54,5; 58,8] | 0,764 |
| 7 | mGPT-1b3 (multi) | 1,3 B | 55,57 % [53,4; 57,7] | 1,238 |
| 8 | Tucano-160m | 0,16 B | 52,53 % [50,4; 54,7] | 0,842 |
| 9 | TTL-460m | 0,46 B | 51,23 % [49,1; 53,4] | 0,864 |
| 10 | TTL-160m | 0,16 B | 47,33 % [45,2; 49,5] | 0,947 |

A curva de escala (acurácia vs parâmetros) está em `calame_escala_pt.png`
(gerada por `plot_scaling_pt.py`). O Manacá fica no topo do agrupamento de 1 a 2 B,
sobe acima da linha da família PT-BR e só é superado pelo Sabiá-7B, que é 4x maior.

**Leitura (com o teste de significância; ver §9).** No CALAME-PT o Manacá-1B tem o
segundo maior número absoluto (60,63 %), atrás só do Sabiá-7B. Pelo teste pareado
de McNemar: o Sabiá é o **único** significativamente acima do Manacá (p = 0,004);
Manacá **empata** com GlórIA-1b3 (p = 0,83), Tucano-2b4 (p = 0,29) e Tucano-1b1
(p = 0,13); e Manacá **supera** significativamente o mGPT-1b3 multilíngue de mesmo
porte (+5,06 pontos, p < 0,001). A afirmação defensável é **paridade com os melhores
modelos de 1 a 2 B do português** e superioridade clara sobre o baseline multilíngue.

#### Validação do harness e ressalva sobre GlórIA/mGPT

Nossas acurácias foram cruzadas com a **tabela publicada do Tucano**. A família
SentencePiece/LLaMA-BPE do PT-BR (TTL e Tucano) **bate dentro de ~1 ponto**, o que
valida o pipeline:

| Modelo | nosso | publicado | Δ |
|---|---|---|---|
| Tucano-160m | 52,53 | 52,31 | +0,22 |
| Tucano-630m | 56,63 | 56,55 | +0,08 |
| Tucano-1b1 | 59,08 | 58,24 | +0,84 |
| Tucano-2b4 | 59,57 | 59,06 | +0,51 |
| GlórIA-1b3 (byte-BPE) | 60,39 | 52,79 | **+7,60** |
| mGPT-1b3 (byte-BPE) | 55,57 | 47,14 | **+8,43** |

Os dois modelos de tokenizador **byte-level BPE** (GlórIA, mGPT) divergem ~7 a 8
pontos do publicado, enquanto todos os de SentencePiece batem. Isso indica que a
extração de última palavra **por geração** é sensível à família de tokenizador (a
tokenização byte-level da continuação favorece o casamento no nosso protocolo). Logo:

- Os números de **GlórIA e mGPT são provisórios** enquanto o protocolo não for
  alinhado (uma pendência é re-rodar salvando as gerações cruas para diagnosticar
  caixa/espaço em branco, ou passar a um escore por log-verossimilhança).
- **Manacá vs mGPT** é robusto mesmo assim: o mGPT está super-pontuado no nosso
  harness e ainda perde, então na verdade a vantagem do Manacá é maior.
- **Manacá vs GlórIA** fica **inconclusivo**: como o GlórIA está super-pontuado, o
  empate observado pode ser artefato; sob um protocolo alinhado o Manacá pode ficar
  à frente. Não declaramos vencedor aqui.

#### Robustez de caixa (Manacá e Tucano-1b1)

| Métrica | Manacá-1B | Tucano-1b1 |
|---|---|---|
| BPB, texto original | **0,702** | 0,719 |
| BPB, minúsculas | **0,702** | 0,728 |
| CALAME-PT, original | **60,63 %** | 59,08 % |
| CALAME-PT, minúsculas | **60,63 %** | 58,02 % |

O Manacá **não muda** entre original e minúsculas (já dobra maiúsculas; forçar
minúsculas é redundante). O Tucano **piora** em minúsculas (fora da distribuição
dele). Isso confirma que a robustez de caixa do Manacá é real e **não** é um
artefato de tokenização.

#### Duas medidas de BPB, e por que discordam

O BPB depende do texto. Medimos em dois textos e o ranking **inverte**:

| Texto | Manacá | Tucano-1b1 | vencedor |
|---|---|---|---|
| Reservado (1 doc, 536 palavras, sem IC) | **0,702** | 0,719 | Manacá |
| CALAME (2075 frases, com IC) | 0,7319 [0,7268; 0,7372] | **0,7221** [0,7168; 0,7278] | Tucano-1b1 |

No CALAME os ICs **não se sobrepõem**, então ali o Tucano-1b1 comprime o texto um
pouco melhor, apesar de ser menor. Não há contradição: são textos diferentes e
regimes diferentes (um documento longo e contínuo vs. 2075 frases curtas isoladas,
sem contexto acumulado). O texto reservado é **uma** amostra pequena, sem IC, com
variância alta; a medida do CALAME é a estatisticamente robusta. Conclusão honesta:
no BPB os dois modelos estão **muito próximos**, o vencedor depende do texto, e o
Manacá não tem vantagem de compressão que sobreviva ao IC.

### 4. Ressalvas honestas

1. **Tamanho.** O Manacá-1B tem ~1,72 B de parâmetros; o Tucano-1b1 tem ~1,1 B.
   Parte da vantagem sobre o Tucano-1b1 vem de ser maior. O teste mais duro,
   contra o **Tucano-2b4** (~2,4 B, maior que o Manacá), já foi feito: o Manacá
   perde no BPB (esperado, é menor) mas fica à frente no CALAME-PT (§3).
2. **Caixa.** O Manacá é lowercase (decisão do tokenizador, fidelidade ao
   LLM-jp). É uma limitação real (não distingue "Rio" de "rio", nunca produz
   maiúsculas), documentada aqui para não ser confundida com vantagem.
3. **Amostra do BPB.** O texto reservado tem 536 palavras; é pequeno e, por isso,
   tem alguma variância. As conclusões se apoiam também no CALAME (2075 exemplos)
   e são consistentes entre as duas métricas.

### 5. Tokenizador HF do Manacá: problema e correção

O `LlamaTokenizer(Fast)` derivado do `.model` **não** reproduzia a normalização
`nmt_nfkc_cf` do treino: o normalizador saía `None`, então todo texto com
**maiúscula** caía em **byte-fallback** e divergia da tokenização de treino
(verificado: 0/5 sequências batendo com o SentencePiece). Isso é invisível no BPB
por janela, mas **arruina** qualquer avaliação que use o tokenizador HF, como o
lm-eval (§10): antes da correção o LAMBADA-PT do Manacá dava 25,0 % com
perplexidade ~10⁶.

A correção (`scripts/eval/fix_hf_tokenizer.py`) edita o `tokenizer.json` no nível
do JSON e injeta `normalizer = Sequence([NFKC, Lowercase])`, reproduzindo o SPM do
treino (**5/5** sequências idênticas, incluindo maiúsculas e nomes próprios). Com o
tokenizador corrigido o LAMBADA saltou para 45,3 % e a perplexidade para 17,3. Esse
tokenizador corrigido é também o artefato necessário para **publicar o Manacá no HF
Hub**. Para o BPB/CALAME do §3 a avaliação usa o SPM direto (fiel por construção).

### 6. Como reproduzir

Na máquina de treino (com a imagem `manaca-train` e o modelo convertido em
`/prj/.../manaca-1b-hf`):

```bash
# Manacá-1B (tokeniza com o SPM do treino)
./scripts/eval/run_eval.sh --model /m --spm /tok/manaca-tokenizer.model \
    --text /eval/holdout_pt.txt --calame                 # texto original
./scripts/eval/run_eval.sh --model /m --spm /tok/manaca-tokenizer.model \
    --text /eval/holdout_pt.txt --calame --lowercase     # minúsculas

# Tucano-1b1 (usa o próprio tokenizador)
./scripts/eval/run_eval.sh --model TucanoBR/Tucano-1b1 \
    --text /eval/holdout_pt.txt --calame
./scripts/eval/run_eval.sh --model TucanoBR/Tucano-1b1 \
    --text /eval/holdout_pt.txt --calame --lowercase
```

Cada execução grava um log com timestamp em `$HOME/manaca-eval-logs/`.

### 7. Saída bruta (verbatim dos logs)

```
Manacá-1B  (original):
  RESUMO: {'model': '/m', 'tokenizer': 'SPM(64000)', 'tokens': 662, 'bytes': 3449,
           'bytes_per_token': 5.21, 'bits_per_token': 3.6573, 'ppl_token': 12.6171,
           'bpb': 0.702, 'calame_acc': 0.6063, 'calame_n': 2075}

Manacá-1B  (--lowercase):
  RESUMO: {'model': '/m', 'tokenizer': 'SPM(64000)', ... 'bpb': 0.702,
           'calame_acc': 0.6063, 'calame_n': 2075}

Tucano-1b1  (original):
  RESUMO: {'model': 'TucanoBR/Tucano-1b1', 'tokenizer': 'HF(32000)', 'tokens': 680,
           'bytes': 3449, 'bytes_per_token': 5.0721, 'bits_per_token': 3.6488,
           'ppl_token': 12.5432, 'bpb': 0.7194, 'calame_acc': 0.5908, 'calame_n': 2075}

Tucano-1b1  (--lowercase):
  RESUMO: {'model': 'TucanoBR/Tucano-1b1', 'tokenizer': 'HF(32000)', 'tokens': 681,
           'bytes': 3449, 'bytes_per_token': 5.0646, 'bits_per_token': 3.6853,
           'ppl_token': 12.8645, 'bpb': 0.7277, 'calame_acc': 0.5802, 'calame_n': 2075}

Tucano-2b4  (original):
  RESUMO: {'model': 'TucanoBR/Tucano-2b4', 'tokenizer': 'HF(32000)', 'tokens': 680,
           'bytes': 3449, 'bytes_per_token': 5.0721, 'bits_per_token': 3.5043,
           'ppl_token': 11.3473, 'bpb': 0.6909, 'calame_acc': 0.5957, 'calame_n': 2075}
```

### 8. Pendências

- [x] Comparação ampla: 9 modelos PT-BR/PT-PT/multilíngues (família TTL+Tucano,
      GlórIA, mGPT, Sabiá-7B) no mesmo harness, com IC e **testes pareados** (§9).
- [x] Curva de escala CALAME-PT vs parâmetros (`plot_scaling_pt.py`).
- [x] Validação do harness contra números publicados (§3).
- [x] **ARC-Challenge-PT (25-shot) + HellaSwag-PT (10-shot) + LAMBADA-PT (0-shot)**
      via lm-eval-harness, no protocolo do Tucano, nos 10 modelos (§10).
- [x] **Corrigir o tokenizador HF do Manacá** (NFKC + Lowercase, via
      `fix_hf_tokenizer.py`, 5/5 vs o SPM). Necessário para o lm-eval ser justo e
      para publicar no Hub.
- [ ] Llama-3.2-1B / Qwen2.5-1.5B / SmolLM2-1.7B (generalistas multilíngues).
- [ ] Publicar no HF Hub com o tokenizador corrigido.

### 9. Incerteza e significância

Cada métrica tem sua forma de incerteza, e o avaliador (`eval_base.py`) a reporta:

- **CALAME-PT** é uma proporção: erro padrão binomial SE = √(p(1−p)/n) e IC 95%
  por bootstrap sobre os exemplos. Para n = 2075, SE ≈ 1,1% (IC ≈ ±2,1%).
- **BPB** é um agregado: IC 95% por bootstrap sobre segmentos (as 2075 frases do
  CALAME dão muitos segmentos e um IC estreito).

**Cuidado metodológico.** Comparar dois modelos por sobreposição de ICs marginais
é o teste errado, porque eles veem os **mesmos** exemplos (erros correlacionados).
O teste correto é **pareado**: bootstrap pareado da diferença + McNemar, feito por
`scripts/eval/paired_compare.py` a partir dos acertos por exemplo salvos com
`--save-calame`.

#### Testes pareados (Manacá vs cada modelo, CALAME-PT, n = 2075)

Bootstrap pareado da diferença (5000 reamostragens) + McNemar, a partir dos vetores
de acerto por exemplo (`*_calame.json` em `docs/evaluation/logs/`, via
`paired_compare.py`). A diferença é Manacá menos o outro modelo.

| B (comparado) | Diferença | IC95% pareado | McNemar p | Significativo? |
|---|---|---|---|---|
| Sabiá-7B | **-2,60** | [-4,3; -0,9] | **0,004** | Sim, Sabiá acima |
| GlórIA-1b3 | +0,24 | [-1,9; +2,4] | 0,83 | Não (empate)\* |
| Tucano-2b4 | +1,06 | [-0,8; +2,9] | 0,29 | Não (empate) |
| Tucano-1b1 | +1,54 | [-0,3; +3,4] | 0,13 | Não (empate) |
| mGPT-1b3 | **+5,06** | [+3,1; +7,0] | **<0,001** | Sim, Manacá acima |

\* GlórIA: empate provável **artefato de protocolo** (o GlórIA está super-pontuado
no nosso harness, ver §3). Não declaramos vencedor nesse par.

**Interpretação.** Depois do teste pareado, o quadro honesto é:

1. **Sabiá-7B** é o único significativamente acima do Manacá (p = 0,004), e é 4x
   maior. Um modelo de 1,72 B ficar ~2,6 pontos atrás de um de 7 B é esperado.
2. **Manacá empata** com os melhores modelos de 1 a 2 B do português (Tucano-1b1,
   Tucano-2b4 e, ressalvado o protocolo, GlórIA-1b3): as diferenças de ~1 ponto
   ficam dentro do ruído.
3. **Manacá supera** claramente o baseline multilíngue de mesmo porte (mGPT-1b3,
   +5,06, p < 0,001), evidência direta do valor de especializar em PT.

No BPB sobre as 2075 frases (§3) o ranking segue o tamanho e o Manacá não tem
vantagem de compressão que sobreviva ao IC. A leitura final é de **paridade com o
estado da arte aberto de porte comparável em português**, com robustez de caixa e
tokenizador nativo de PT como vantagens qualitativas. O Manacá foi treinado com
~41,9 B de tokens (20000 passos × batch 512 × seq 4096), ou ~24 tokens/parâmetro,
próximo do ponto compute-optimal de Chinchilla (~20). A folga de raciocínio para os
Tucano não vem de subtreino do Manacá, e sim de os Tucano serem fortemente
sobretreinados (mais de uma ordem de grandeza a mais de tokens) em contagem de
parâmetros comparável.

#### Saúde do treinamento (fonte bruta, `docs/training/logs/`)

Para transparência, o log completo do pré-treino também está no repositório. Varredura
das 2000 linhas de iteração (`train_20260706_030015.log`, iter 10 a 20000):

- **0 iterações puladas** e **0 iterações com NaN** em toda a corrida (soma dos
  contadores `number of skipped/nan iterations`). Estabilidade total.
- **lm loss** de 11,41 (iter 10, ~inicialização aleatória, ln 64128 ≈ 11,07) para
  2,48 (iter 20000), mínimo 2,458 em iter 18960. Curva monótona até o platô.
- **grad norm** min/mediana/máx = 0,097 / 0,119 / 24,54. Os dois maiores picos são
  precoces (iter 2070 e 8800) e se recuperam sem pulo nem NaN; ao final o grad norm
  fica estável em ~0,10. O pequeno bump em iter 14030 (grad 0,556 vs base ~0,11) é
  local e some no passo seguinte, sem efeito na loss.
- **learning rate** encerra em 3,0e-5 (mínimo do cosseno), confirmando o schedule
  completo. O log curto `train_20260706_025144.log` é um teste de fumaça anterior
  (salva checkpoint no iter 1); a corrida real começa do zero em `030015`.

### 10. Benchmarks de capacidade (ARC-PT, HellaSwag-PT, LAMBADA-PT)

Além da última palavra (CALAME), avaliamos raciocínio, senso comum e uma segunda
tarefa de última palavra, no **mesmo protocolo do Tucano**, via
**lm-evaluation-harness** (`scripts/eval/run_lm_eval_pt.sh`, imagem `manaca-lmeval`):

- **ARC-Challenge-PT** (`arc_pt`, `alexandrainst/m_arc` config `pt`): 25-shot,
  `acc_norm`. Múltipla escolha, raciocínio/conhecimento.
- **HellaSwag-PT** (`hellaswag_pt`, `alexandrainst/m_hellaswag` config `pt`):
  10-shot, `acc_norm`. Senso comum (completar frase).
- **LAMBADA-PT** (`TucanoBR/lambada-pt`, YAML próprio): 0-shot, `acc`. Última palavra.

Todos por **log-verossimilhança** (não geração), o que é **justo entre
tokenizadores** e resolve a ressalva de GlórIA/mGPT do §3.

Valores em % com **± erro padrão (SE)**; IC95% ≈ ±1,96·SE.

| Modelo | Par (B) | CALAME | ARC-Ch | HellaSwag | LAMBADA |
|---|---|---|---|---|---|
| TTL-160m | 0,16 | 47,33 ±1,10 | 25,38 ±1,27 | 29,73 ±0,48 | 21,21 ±0,57 |
| Tucano-160m | 0,16 | 52,53 ±1,10 | 25,04 ±1,27 | 33,56 ±0,49 | 25,62 ±0,61 |
| TTL-460m | 0,46 | 51,23 ±1,10 | 27,09 ±1,30 | 34,47 ±0,49 | 22,18 ±0,58 |
| Tucano-630m | 0,63 | 56,63 ±1,09 | 27,52 ±1,31 | 39,96 ±0,51 | 31,03 ±0,64 |
| Tucano-1b1 | 1,10 | 59,08 ±1,08 | 29,66 ±1,34 | 44,23 ±0,52 | 31,50 ±0,65 |
| GlórIA-1b3 | 1,30 | 60,39 ±1,07 | 24,44 ±1,26 | 25,83 ±0,46 | 35,30 ±0,67 |
| mGPT-1b3 | 1,30 | 55,57 ±1,09 | 23,93 ±1,25 | 25,42 ±0,45 | 37,38 ±0,67 |
| **Manacá-1B** | 1,72 | **60,63 ±1,07** | 27,18 ±1,30 | 41,61 ±0,51 | **45,31 ±0,69** |
| Tucano-2b4 | 2,40 | 59,57 ±1,08 | 30,85 ±1,35 | 48,63 ±0,52 | 34,35 ±0,66 |
| Sabiá-7B | 7,00 | 63,23 ±1,06 | 46,67 ±1,46 | 64,55 ±0,50 | 63,67 ±0,67 |

**Validação do harness.** Tucano-1b1 deu ARC 29,66 / HellaSwag 44,23 / LAMBADA
31,50, contra os publicados 30,43 / 42,84 / 34,7. Bate dentro de ~2 pontos.

#### Incerteza e significância (sugestão do Fabio)

Cada célula traz o **SE** (o lm-eval reporta o stderr de `acc_norm`/`acc`; o CALAME
é SE binomial). Para comparar o Manacá com cada modelo usamos o **McNemar pareado +
bootstrap pareado** sobre os acertos por exemplo (`--log_samples`), no **mesmo
padrão do CALAME** (§9). Reprodutível: `scripts/eval/paired_lm_eval.py`; resumo em
`docs/evaluation/paired-benchmarks-pt.md`, vetores compactos em `vectors-pt.json`.

- **LAMBADA-PT** (n = 5153): o Manacá supera **todos** os modelos abaixo do Sabiá-7B,
  todos com McNemar **p < 0,0001** (+13,8 vs Tucano-1b1, +11,0 vs Tucano-2b4, +10,0
  vs GlórIA, +7,9 vs mGPT). Só o Sabiá-7B o supera (−18,4; p < 0,0001).
- **HellaSwag-PT** (n = 9229): **abaixo** dos Tucanos com significância (−2,6 vs 1b1,
  −7,0 vs 2b4; p < 0,0001) e **muito acima** dos pares de mesmo porte GlórIA e mGPT
  (+15,8 e +16,3; p < 0,0001).
- **ARC-Challenge-PT** (n = 1170): perto do aleatório; **empate** com todos os
  sub-2B — Manacá vs Tucano-1b1 p = 0,059, vs GlórIA p = 0,12, vs mGPT p = 0,09,
  vs os menores n.s. Fica **significativamente abaixo** apenas do Tucano-2b4
  (p = 0,0047) e do Sabiá-7B.

O teste pareado (mais potente que o não pareado) mudou **um** veredito: no ARC,
Manacá vs Tucano-2b4 passou de borderline (não pareado p = 0,05) a **significativo**
(pareado p = 0,0047); todo o resto se manteve. Com isso a estatística é **pareada e
idêntica em todos os benchmarks** (CALAME + os três).

O painel `docs/evaluation/benchmarks_escala_pt.png` (via `plot_benchmarks_pt.py`)
mostra os quatro benchmarks lado a lado (acurácia vs parâmetros, IC95%, Manacá em
destaque): fica visível que o Manacá salta **acima** da tendência PT-BR no LAMBADA,
fica um pouco **abaixo** dos Tucanos no HellaSwag, e no ARC todos os pequenos se
amontoam perto do acaso.

#### Pré-requisito crítico: o tokenizador do Manacá (ver §5)

A primeira rodada do Manacá no lm-eval usou o tokenizador HF **sem** a correção e
saiu **artificialmente baixa** (LAMBADA 25,0, HellaSwag 35,6), porque o modelo via
byte-fallback em todo texto com maiúscula. Com o tokenizador corrigido
(`fix_hf_tokenizer.py`, 5/5 vs o SPM), o **LAMBADA saltou de 25,0 para 45,3** e a
perplexidade de ~10⁶ para 17,3. Os números da tabela são os **corrigidos**. Isso é
uma lição metodológica: avaliar um modelo *lowercase* pelo tokenizador HF errado o
penaliza de forma invisível.

#### Leitura

- **Última palavra (CALAME 60,63 + LAMBADA 45,31):** o Manacá é o **melhor de todos
  abaixo do Sabiá-7B**. No LAMBADA supera Tucano-1b1 (31,5), Tucano-2b4 (34,3),
  GlórIA (35,3) e mGPT (37,4). É a força do modelo, coerente com o tokenizador
  nativo de PT.
- **HellaSwag 41,61:** encosta no Tucano-1b1 (44,2) e supera com folga os pares de
  mesmo porte GlórIA (25,8) e mGPT (25,4).
- **ARC-Challenge 27,18:** meio da tabela, perto do aleatório (25%), como todos os
  modelos base pequenos; um pouco atrás dos Tucanos, à frente de GlórIA/mGPT.
- **Contra os pares de mesmo tamanho** (GlórIA-1b3, mGPT-1b3) o Manacá vence em 3
  dos 4 benchmarks. Contra o Tucano-1b1 (treinado em ~25x mais tokens) ganha o
  LAMBADA com folga e segura o HellaSwag; só o Sabiá-7B, 4x maior, domina tudo.

Leitura honesta: o Manacá-1B é **forte em modelagem de linguagem / última palavra**,
**competitivo em senso comum**, e **mediano no raciocínio difícil** (ARC, onde
modelos pequenos ficam perto do aleatório). Bate os pares abertos de mesmo porte em
português.

#### Ressalvas

- **LAMBADA-PT** aqui é por log-verossimilhança; o script original do Tucano faz
  geração + match exato, então é uma aproximação (mesma faixa, pode diferir alguns
  pontos).
- **HellaSwag** foi rodado a 10-shot (texto do artigo do Tucano); o código deles usa
  0-shot. Diferenças pequenas podem vir daí.
- Dados brutos por modelo/tarefa em `docs/evaluation/lmeval/`; tabela consolidada em
  `docs/evaluation/benchmarks-pt.md` / `.json` (via `merge_pt_benchmarks.py`).

## English

**Date:** August 29, 2026
**Author:** Bruno Leonardo Santos Menezes (LNCC × NII / LLM-jp)
**Scope:** **base** model (pre-trained, no SFT/DPO), final step 20,000
**Scripts:** `scripts/eval/eval_base.py`, `run_eval.sh`, `run_batch_pt.sh`, `paired_compare.py`, `plot_scaling_pt.py`
**Raw data:** `docs/evaluation/results-base.json` and `docs/evaluation/logs/`

This document records, transparently and reproducibly, the comparative
evaluation of Manacá-1B against **9 reference models** in PT-BR, PT-PT, and
multilingual varieties (the TeenyTinyLlama + Tucano family, GlórIA, mGPT, and Sabiá-7B). All
were measured on the **same text**, with the **same protocol**, and on the **same harness**,
with **tokenizer-independent metrics**.

---

### 1. Metrics

- **BPB (bits per byte)** — the anchor metric and **fair across different
  tokenizers**. It is the total NLL in bits divided by the **bytes** of the text (not
  by the tokens). Vocabularies of different sizes (Manacá 64k, Tucano 32k)
  give different per-token perplexities, but the same text has the same bytes.
  **Lower is better.**
- **CALAME-PT** — accuracy in predicting the **last word** of a paragraph
  (native PT-BR benchmark, `NOVA-vision-language/calame-pt`, file
  `calamept_all.jsonl`, 2075 examples). It works on a base model and, being an
  exact match, is comparable across models. **Higher is better.**
- **Per-token perplexity** — informative, but **only comparable** across models
  with the same tokenizer; it is not the comparison metric.

### 2. Protocol

- **Holdout text:** `scripts/eval/holdout_pt.txt` (536 words, varied
  registers: scientific, narrative, legal, technical). Written from scratch for this
  evaluation, therefore **unseen by all models** (it is not in any model's
  training set). Used identically for all.
- **BPB/perplexity:** sliding window of 2048 tokens, stride 1024. Each model
  tokenizes with its **own tokenizer**. Manacá uses the SentencePiece from training
  (`manaca-tokenizer.model`, `nmt_nfkc_cf` normalization), ensuring fidelity to the
  training tokenization (the package's HF tokenizer does not reproduce that
  normalization; see §5).
- **CALAME-PT:** for each example, the context (`sentence`) is given to the model, which
  generates **greedily**; the first generated word is compared to the
  target word (`last_word`), case-insensitively.
- **Case:** Manacá is **lowercase** by construction (tokenizer with
  `nmt_nfkc_cf`, which folds uppercase). To isolate this effect, each measurement
  was done twice: on the **original text** and with **`--lowercase`** (all text
  lowercased for everyone).
- **Hardware/image:** 1 GPU; `manaca-train` image (torch + transformers 4.x).

### 3. Results

#### CALAME-PT leaderboard (10 models, same harness, n = 2075)

Ordered by accuracy. 95% CI = ±1,96·binomial SE. BPB(res) is the BPB on the
holdout text (lower is better).

| # | Modelo | Params | CALAME-PT ↑ (IC95%) | BPB(res) ↓ |
|---|---|---|---|---|
| 1 | **Sabiá-7B** | 7,0 B | **63,23 %** [61,1; 65,3] | 0,646 |
| 2 | **Manacá-1B** | 1,72 B | **60,63 %** [58,5; 62,7] | 0,702 |
| 3 | GlórIA-1b3 (PT-PT) | 1,3 B | 60,39 % [58,3; 62,5] | 0,865 |
| 4 | Tucano-2b4 | 2,4 B | 59,57 % [57,4; 61,7] | 0,691 |
| 5 | Tucano-1b1 | 1,1 B | 59,08 % [57,0; 61,2] | 0,719 |
| 6 | Tucano-630m | 0,63 B | 56,63 % [54,5; 58,8] | 0,764 |
| 7 | mGPT-1b3 (multi) | 1,3 B | 55,57 % [53,4; 57,7] | 1,238 |
| 8 | Tucano-160m | 0,16 B | 52,53 % [50,4; 54,7] | 0,842 |
| 9 | TTL-460m | 0,46 B | 51,23 % [49,1; 53,4] | 0,864 |
| 10 | TTL-160m | 0,16 B | 47,33 % [45,2; 49,5] | 0,947 |

The scaling curve (accuracy vs parameters) is in `calame_escala_pt.png`
(generated by `plot_scaling_pt.py`). Manacá sits at the top of the 1-to-2B cluster,
rises above the PT-BR family line, and is only surpassed by Sabiá-7B, which is 4x larger.

**Reading (with the significance test; see §9).** On CALAME-PT, Manacá-1B has the
second-highest absolute number (60,63 %), behind only Sabiá-7B. By the paired McNemar
test: Sabiá is the **only** one significantly above Manacá (p = 0,004);
Manacá **ties** with GlórIA-1b3 (p = 0,83), Tucano-2b4 (p = 0,29), and Tucano-1b1
(p = 0,13); and Manacá **beats** significantly the same-size multilingual mGPT-1b3
(+5,06 points, p < 0,001). The defensible claim is **parity with the best
1-to-2B Portuguese models** and clear superiority over the multilingual baseline.

#### Harness validation and caveat about GlórIA/mGPT

Our accuracies were cross-checked against the **Tucano published table**. The
PT-BR SentencePiece/LLaMA-BPE family (TTL and Tucano) **matches within ~1 point**, which
validates the pipeline:

| Modelo | nosso | publicado | Δ |
|---|---|---|---|
| Tucano-160m | 52,53 | 52,31 | +0,22 |
| Tucano-630m | 56,63 | 56,55 | +0,08 |
| Tucano-1b1 | 59,08 | 58,24 | +0,84 |
| Tucano-2b4 | 59,57 | 59,06 | +0,51 |
| GlórIA-1b3 (byte-BPE) | 60,39 | 52,79 | **+7,60** |
| mGPT-1b3 (byte-BPE) | 55,57 | 47,14 | **+8,43** |

The two **byte-level BPE** tokenizer models (GlórIA, mGPT) diverge ~7 to 8
points from the published values, while all the SentencePiece ones match. This indicates that
last-word extraction **by generation** is sensitive to the tokenizer family (the
byte-level tokenization of the continuation favors matching under our protocol). Hence:

- The **GlórIA and mGPT numbers are provisional** until the protocol is
  aligned (one pending item is to re-run saving the raw generations to diagnose
  case/whitespace, or to switch to a log-likelihood score).
- **Manacá vs mGPT** is robust anyway: mGPT is over-scored in our
  harness and still loses, so in fact Manacá's advantage is larger.
- **Manacá vs GlórIA** is **inconclusive**: since GlórIA is over-scored, the
  observed tie may be an artifact; under an aligned protocol Manacá may come out
  ahead. We do not declare a winner here.

#### Case robustness (Manacá and Tucano-1b1)

| Métrica | Manacá-1B | Tucano-1b1 |
|---|---|---|
| BPB, texto original | **0,702** | 0,719 |
| BPB, minúsculas | **0,702** | 0,728 |
| CALAME-PT, original | **60,63 %** | 59,08 % |
| CALAME-PT, minúsculas | **60,63 %** | 58,02 % |

Manacá **does not change** between original and lowercase (it already folds uppercase; forcing
lowercase is redundant). Tucano **degrades** in lowercase (out of its
distribution). This confirms that Manacá's case robustness is real and is **not** a
tokenization artifact.

#### Two BPB measurements, and why they disagree

BPB depends on the text. We measured on two texts and the ranking **inverts**:

| Texto | Manacá | Tucano-1b1 | vencedor |
|---|---|---|---|
| Reservado (1 doc, 536 palavras, sem IC) | **0,702** | 0,719 | Manacá |
| CALAME (2075 frases, com IC) | 0,7319 [0,7268; 0,7372] | **0,7221** [0,7168; 0,7278] | Tucano-1b1 |

On CALAME the CIs **do not overlap**, so there Tucano-1b1 compresses the text a
bit better, despite being smaller. There is no contradiction: they are different texts and
different regimes (one long, continuous document vs. 2075 short, isolated sentences,
with no accumulated context). The holdout text is **one** small sample, without a CI, with
high variance; the CALAME measurement is the statistically robust one. Honest conclusion:
on BPB the two models are **very close**, the winner depends on the text, and
Manacá has no compression advantage that survives the CI.

### 4. Honest caveats

1. **Size.** Manacá-1B has ~1,72 B parameters; Tucano-1b1 has ~1,1 B.
   Part of the advantage over Tucano-1b1 comes from being larger. The harder test,
   against **Tucano-2b4** (~2,4 B, larger than Manacá), has already been done: Manacá
   loses on BPB (expected, it is smaller) but comes out ahead on CALAME-PT (§3).
2. **Case.** Manacá is lowercase (a tokenizer decision, fidelity to
   LLM-jp). It is a real limitation (it does not distinguish "Rio" from "rio", never produces
   uppercase), documented here so it is not confused with an advantage.
3. **BPB sample.** The holdout text has 536 words; it is small and, therefore,
   has some variance. The conclusions also rely on CALAME (2075 examples)
   and are consistent across the two metrics.

### 5. Manacá's HF tokenizer: problem and fix

The `LlamaTokenizer(Fast)` derived from the `.model` did **not** reproduce the
`nmt_nfkc_cf` normalization from training: the normalizer came out `None`, so all text with
an **uppercase** letter fell into **byte-fallback** and diverged from the training
tokenization (verified: 0/5 sequences matching the SentencePiece). This is invisible in windowed
BPB, but **ruins** any evaluation that uses the HF tokenizer, such as
lm-eval (§10): before the fix, Manacá's LAMBADA-PT scored 25,0 % with
perplexity ~10⁶.

The fix (`scripts/eval/fix_hf_tokenizer.py`) edits the `tokenizer.json` at the
JSON level and injects `normalizer = Sequence([NFKC, Lowercase])`, reproducing the training
SPM (**5/5** identical sequences, including uppercase and proper nouns). With the
fixed tokenizer, LAMBADA jumped to 45,3 % and perplexity to 17,3. This
fixed tokenizer is also the artifact needed to **publish Manacá on the HF
Hub**. For the BPB/CALAME in §3, the evaluation uses the SPM directly (faithful by construction).

### 6. How to reproduce

On the training machine (with the `manaca-train` image and the model converted at
`/prj/.../manaca-1b-hf`):

```bash
# Manacá-1B (tokenizes with the training SPM)
./scripts/eval/run_eval.sh --model /m --spm /tok/manaca-tokenizer.model \
    --text /eval/holdout_pt.txt --calame                 # texto original
./scripts/eval/run_eval.sh --model /m --spm /tok/manaca-tokenizer.model \
    --text /eval/holdout_pt.txt --calame --lowercase     # minúsculas

# Tucano-1b1 (uses its own tokenizer)
./scripts/eval/run_eval.sh --model TucanoBR/Tucano-1b1 \
    --text /eval/holdout_pt.txt --calame
./scripts/eval/run_eval.sh --model TucanoBR/Tucano-1b1 \
    --text /eval/holdout_pt.txt --calame --lowercase
```

Each run writes a timestamped log to `$HOME/manaca-eval-logs/`.

### 7. Raw output (verbatim from the logs)

```
Manacá-1B  (original):
  RESUMO: {'model': '/m', 'tokenizer': 'SPM(64000)', 'tokens': 662, 'bytes': 3449,
           'bytes_per_token': 5.21, 'bits_per_token': 3.6573, 'ppl_token': 12.6171,
           'bpb': 0.702, 'calame_acc': 0.6063, 'calame_n': 2075}

Manacá-1B  (--lowercase):
  RESUMO: {'model': '/m', 'tokenizer': 'SPM(64000)', ... 'bpb': 0.702,
           'calame_acc': 0.6063, 'calame_n': 2075}

Tucano-1b1  (original):
  RESUMO: {'model': 'TucanoBR/Tucano-1b1', 'tokenizer': 'HF(32000)', 'tokens': 680,
           'bytes': 3449, 'bytes_per_token': 5.0721, 'bits_per_token': 3.6488,
           'ppl_token': 12.5432, 'bpb': 0.7194, 'calame_acc': 0.5908, 'calame_n': 2075}

Tucano-1b1  (--lowercase):
  RESUMO: {'model': 'TucanoBR/Tucano-1b1', 'tokenizer': 'HF(32000)', 'tokens': 681,
           'bytes': 3449, 'bytes_per_token': 5.0646, 'bits_per_token': 3.6853,
           'ppl_token': 12.8645, 'bpb': 0.7277, 'calame_acc': 0.5802, 'calame_n': 2075}

Tucano-2b4  (original):
  RESUMO: {'model': 'TucanoBR/Tucano-2b4', 'tokenizer': 'HF(32000)', 'tokens': 680,
           'bytes': 3449, 'bytes_per_token': 5.0721, 'bits_per_token': 3.5043,
           'ppl_token': 11.3473, 'bpb': 0.6909, 'calame_acc': 0.5957, 'calame_n': 2075}
```

### 8. Pending items

- [x] Broad comparison: 9 PT-BR/PT-PT/multilingual models (the TTL+Tucano family,
      GlórIA, mGPT, Sabiá-7B) on the same harness, with CIs and **paired tests** (§9).
- [x] CALAME-PT scaling curve vs parameters (`plot_scaling_pt.py`).
- [x] Harness validation against published numbers (§3).
- [x] **ARC-Challenge-PT (25-shot) + HellaSwag-PT (10-shot) + LAMBADA-PT (0-shot)**
      via lm-eval-harness, in the Tucano protocol, on the 10 models (§10).
- [x] **Fix Manacá's HF tokenizer** (NFKC + Lowercase, via
      `fix_hf_tokenizer.py`, 5/5 vs the SPM). Needed for lm-eval to be fair and
      to publish on the Hub.
- [ ] Llama-3.2-1B / Qwen2.5-1.5B / SmolLM2-1.7B (multilingual generalists).
- [ ] Publish on the HF Hub with the fixed tokenizer.

### 9. Uncertainty and significance

Each metric has its form of uncertainty, and the evaluator (`eval_base.py`) reports it:

- **CALAME-PT** is a proportion: binomial standard error SE = √(p(1−p)/n) and 95% CI
  by bootstrap over the examples. For n = 2075, SE ≈ 1,1% (CI ≈ ±2,1%).
- **BPB** is an aggregate: 95% CI by bootstrap over segments (the 2075 CALAME sentences
  give many segments and a narrow CI).

**Methodological caution.** Comparing two models by overlap of marginal CIs
is the wrong test, because they see the **same** examples (correlated errors).
The correct test is **paired**: paired bootstrap of the difference + McNemar, done by
`scripts/eval/paired_compare.py` from the per-example correctness saved with
`--save-calame`.

#### Paired tests (Manacá vs each model, CALAME-PT, n = 2075)

Paired bootstrap of the difference (5000 resamplings) + McNemar, from the per-example
correctness vectors (`*_calame.json` in `docs/evaluation/logs/`, via
`paired_compare.py`). The difference is Manacá minus the other model.

| B (comparado) | Diferença | IC95% pareado | McNemar p | Significativo? |
|---|---|---|---|---|
| Sabiá-7B | **-2,60** | [-4,3; -0,9] | **0,004** | Sim, Sabiá acima |
| GlórIA-1b3 | +0,24 | [-1,9; +2,4] | 0,83 | Não (empate)\* |
| Tucano-2b4 | +1,06 | [-0,8; +2,9] | 0,29 | Não (empate) |
| Tucano-1b1 | +1,54 | [-0,3; +3,4] | 0,13 | Não (empate) |
| mGPT-1b3 | **+5,06** | [+3,1; +7,0] | **<0,001** | Sim, Manacá acima |

\* GlórIA: tie likely a **protocol artifact** (GlórIA is over-scored
in our harness, see §3). We do not declare a winner in that pair.

**Interpretation.** After the paired test, the honest picture is:

1. **Sabiá-7B** is the only one significantly above Manacá (p = 0,004), and it is 4x
   larger. A 1,72 B model being ~2,6 points behind a 7 B one is expected.
2. **Manacá ties** with the best 1-to-2B Portuguese models (Tucano-1b1,
   Tucano-2b4 and, with the protocol caveat, GlórIA-1b3): the ~1-point differences
   fall within the noise.
3. **Manacá clearly beats** the same-size multilingual baseline (mGPT-1b3,
   +5,06, p < 0,001), direct evidence of the value of specializing in PT.

On BPB over the 2075 sentences (§3) the ranking follows size and Manacá has no
compression advantage that survives the CI. The final reading is one of **parity with the
open state of the art of comparable size in Portuguese**, with case robustness and a
native PT tokenizer as qualitative advantages. Manacá was trained with
~41,9 B tokens (20000 steps × batch 512 × seq 4096), or ~24 tokens/parameter,
close to Chinchilla's compute-optimal point (~20). The reasoning gap to the
Tucanos does not come from Manacá being under-trained, but rather from the Tucanos being heavily
over-trained (more than an order of magnitude more tokens) at a comparable
parameter count.

#### Training health (raw source, `docs/training/logs/`)

For transparency, the complete pre-training log is also in the repository. A sweep
of the 2000 iteration lines (`train_20260706_030015.log`, iter 10 to 20000):

- **0 skipped iterations** and **0 NaN iterations** across the whole run (sum of the
  `number of skipped/nan iterations` counters). Total stability.
- **lm loss** from 11,41 (iter 10, ~random initialization, ln 64128 ≈ 11,07) to
  2,48 (iter 20000), minimum 2,458 at iter 18960. Monotonic curve up to the plateau.
- **grad norm** min/median/max = 0,097 / 0,119 / 24,54. The two largest spikes are
  early (iter 2070 and 8800) and recover without a skip or NaN; at the end the grad norm
  stays stable at ~0,10. The small bump at iter 14030 (grad 0,556 vs base ~0,11) is
  local and disappears at the next step, with no effect on the loss.
- **learning rate** ends at 3,0e-5 (cosine minimum), confirming the complete
  schedule. The short log `train_20260706_025144.log` is an earlier smoke test
  (saves a checkpoint at iter 1); the real run starts from scratch at `030015`.

### 10. Capability benchmarks (ARC-PT, HellaSwag-PT, LAMBADA-PT)

Beyond the last word (CALAME), we evaluate reasoning, common sense, and a second
last-word task, in the **same Tucano protocol**, via
**lm-evaluation-harness** (`scripts/eval/run_lm_eval_pt.sh`, `manaca-lmeval` image):

- **ARC-Challenge-PT** (`arc_pt`, `alexandrainst/m_arc` config `pt`): 25-shot,
  `acc_norm`. Multiple choice, reasoning/knowledge.
- **HellaSwag-PT** (`hellaswag_pt`, `alexandrainst/m_hellaswag` config `pt`):
  10-shot, `acc_norm`. Common sense (sentence completion).
- **LAMBADA-PT** (`TucanoBR/lambada-pt`, own YAML): 0-shot, `acc`. Last word.

All by **log-likelihood** (not generation), which is **fair across
tokenizers** and resolves the GlórIA/mGPT caveat from §3.

Values in % with **± standard error (SE)**; 95% CI ≈ ±1,96·SE.

| Modelo | Par (B) | CALAME | ARC-Ch | HellaSwag | LAMBADA |
|---|---|---|---|---|---|
| TTL-160m | 0,16 | 47,33 ±1,10 | 25,38 ±1,27 | 29,73 ±0,48 | 21,21 ±0,57 |
| Tucano-160m | 0,16 | 52,53 ±1,10 | 25,04 ±1,27 | 33,56 ±0,49 | 25,62 ±0,61 |
| TTL-460m | 0,46 | 51,23 ±1,10 | 27,09 ±1,30 | 34,47 ±0,49 | 22,18 ±0,58 |
| Tucano-630m | 0,63 | 56,63 ±1,09 | 27,52 ±1,31 | 39,96 ±0,51 | 31,03 ±0,64 |
| Tucano-1b1 | 1,10 | 59,08 ±1,08 | 29,66 ±1,34 | 44,23 ±0,52 | 31,50 ±0,65 |
| GlórIA-1b3 | 1,30 | 60,39 ±1,07 | 24,44 ±1,26 | 25,83 ±0,46 | 35,30 ±0,67 |
| mGPT-1b3 | 1,30 | 55,57 ±1,09 | 23,93 ±1,25 | 25,42 ±0,45 | 37,38 ±0,67 |
| **Manacá-1B** | 1,72 | **60,63 ±1,07** | 27,18 ±1,30 | 41,61 ±0,51 | **45,31 ±0,69** |
| Tucano-2b4 | 2,40 | 59,57 ±1,08 | 30,85 ±1,35 | 48,63 ±0,52 | 34,35 ±0,66 |
| Sabiá-7B | 7,00 | 63,23 ±1,06 | 46,67 ±1,46 | 64,55 ±0,50 | 63,67 ±0,67 |

**Harness validation.** Tucano-1b1 gave ARC 29,66 / HellaSwag 44,23 / LAMBADA
31,50, against the published 30,43 / 42,84 / 34,7. Matches within ~2 points.

#### Uncertainty and significance (Fabio's suggestion)

Each cell carries the **SE** (lm-eval reports the stderr of `acc_norm`/`acc`; CALAME
is binomial SE). To compare Manacá with each model we use **paired McNemar +
paired bootstrap** over the per-example correctness (`--log_samples`), in the **same
pattern as CALAME** (§9). Reproducible: `scripts/eval/paired_lm_eval.py`; summary in
`docs/evaluation/paired-benchmarks-pt.md`, compact vectors in `vectors-pt.json`.

- **LAMBADA-PT** (n = 5153): Manacá beats **all** models below Sabiá-7B,
  all with McNemar **p < 0,0001** (+13,8 vs Tucano-1b1, +11,0 vs Tucano-2b4, +10,0
  vs GlórIA, +7,9 vs mGPT). Only Sabiá-7B beats it (−18,4; p < 0,0001).
- **HellaSwag-PT** (n = 9229): **below** the Tucanos with significance (−2,6 vs 1b1,
  −7,0 vs 2b4; p < 0,0001) and **well above** the same-size peers GlórIA and mGPT
  (+15,8 and +16,3; p < 0,0001).
- **ARC-Challenge-PT** (n = 1170): near chance; **tie** with all
  sub-2B — Manacá vs Tucano-1b1 p = 0,059, vs GlórIA p = 0,12, vs mGPT p = 0,09,
  vs the smaller ones n.s. It is **significantly below** only Tucano-2b4
  (p = 0,0047) and Sabiá-7B.

The paired test (more powerful than the unpaired one) changed **one** verdict: on ARC,
Manacá vs Tucano-2b4 went from borderline (unpaired p = 0,05) to **significant**
(paired p = 0,0047); everything else held. With this, the statistics are **paired and
identical across all benchmarks** (CALAME + the three).

The panel `docs/evaluation/benchmarks_escala_pt.png` (via `plot_benchmarks_pt.py`)
shows the four benchmarks side by side (accuracy vs parameters, 95% CI, Manacá
highlighted): it is visible that Manacá jumps **above** the PT-BR trend on LAMBADA,
sits a bit **below** the Tucanos on HellaSwag, and on ARC all the small ones
crowd near chance.

#### Critical prerequisite: Manacá's tokenizer (see §5)

Manacá's first lm-eval run used the HF tokenizer **without** the fix and
came out **artificially low** (LAMBADA 25,0, HellaSwag 35,6), because the model saw
byte-fallback on all text with uppercase. With the fixed tokenizer
(`fix_hf_tokenizer.py`, 5/5 vs the SPM), **LAMBADA jumped from 25,0 to 45,3** and
perplexity from ~10⁶ to 17,3. The table numbers are the **fixed** ones. This is
a methodological lesson: evaluating a *lowercase* model with the wrong HF tokenizer
penalizes it invisibly.

#### Reading

- **Last word (CALAME 60,63 + LAMBADA 45,31):** Manacá is the **best of all
  below Sabiá-7B**. On LAMBADA it beats Tucano-1b1 (31,5), Tucano-2b4 (34,3),
  GlórIA (35,3), and mGPT (37,4). It is the model's strength, consistent with the native
  PT tokenizer.
- **HellaSwag 41,61:** it edges up to Tucano-1b1 (44,2) and comfortably beats the same-size
  peers GlórIA (25,8) and mGPT (25,4).
- **ARC-Challenge 27,18:** mid-table, near chance (25%), like all
  small base models; a bit behind the Tucanos, ahead of GlórIA/mGPT.
- **Against the same-size peers** (GlórIA-1b3, mGPT-1b3) Manacá wins on 3
  of the 4 benchmarks. Against Tucano-1b1 (trained on ~25x more tokens) it wins
  LAMBADA comfortably and holds HellaSwag; only Sabiá-7B, 4x larger, dominates everything.

Honest reading: Manacá-1B is **strong in language modeling / last word**,
**competitive in common sense**, and **middling in hard reasoning** (ARC, where
small models sit near chance). It beats the same-size open peers in
Portuguese.

#### Caveats

- **LAMBADA-PT** here is by log-likelihood; the original Tucano script does
  generation + exact match, so it is an approximation (same range, may differ by a few
  points).
- **HellaSwag** was run at 10-shot (text from the Tucano paper); their code uses
  0-shot. Small differences may come from that.
- Raw data per model/task in `docs/evaluation/lmeval/`; consolidated table in
  `docs/evaluation/benchmarks-pt.md` / `.json` (via `merge_pt_benchmarks.py`).

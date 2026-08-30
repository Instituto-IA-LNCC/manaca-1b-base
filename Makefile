# =============================================================================
# [PT] Manacá — atalhos Docker (equivalem a conda + screen + sbatch do fluxo HPC)
# [EN] Manacá — Docker shortcuts (equivalent to conda + screen + sbatch of the HPC flow)
# =============================================================================
# [PT] Usa '>' como prefixo de receita (em vez de TAB) para robustez.
# [EN] Uses '>' as the recipe prefix (instead of TAB) for robustness.
.RECIPEPREFIX = >
.DEFAULT_GOAL := help
SHELL := /bin/bash

DC := docker compose

.PHONY: help env build build-corpus build-train build-eval \
        verify check-dirs gigaverbo madlad fineweb2 hplt2 wikipedia ulysses ulysses-gdrive acquire \
        cc-pipeline cc-test dedup validate characterize logs own trim-gigaverbo \
        tokenizer-shell tokenizer-sample tokenizer-train tokenizer-convert tokenizer \
        preprocess-megatron pretrain prune-checkpoints autoprune stop-autoprune eval-shell \
        shell-corpus shell-train down clean

help:  ## Lista os alvos disponíveis | Lists the available targets
> @grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
>   | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

env:  ## Cria .env a partir de .env.example (se ainda não existir) | Creates .env from .env.example (if it does not exist yet)
> @test -f .env || (cp .env.example .env && echo "Criado .env — edite os segredos.")

# ── Build das imagens ────────────────────────────────────────────────────────
build: build-corpus build-train build-eval  ## Constrói todas as imagens | Builds all images

build-corpus: env  ## Imagem CPU (Fase 1 + Fase 2a) | CPU image (Phase 1 + Phase 2a)
> $(DC) build corpus

build-train: env  ## Imagem GPU de pré-treino (Fase 2b, Megatron-LM) | Pretraining GPU image (Phase 2b, Megatron-LM)
> $(DC) build train

build-eval: env  ## Imagem GPU de avaliação (Fase 4) | Evaluation GPU image (Phase 4)
> $(DC) build eval

# ── Fase 1: Corpus (equivale a `screen -S <fonte> ... conda activate ...`) ───
verify: env  ## Verifica o ambiente (Script 00) | Verifies the environment (Script 00)
> $(DC) run --rm corpus python corpus/scripts/00_verify_env.py

check-dirs: env  ## Verifica se os bind mounts (DATA_DIR/HF_CACHE_DIR/CKPT_DIR) caem no disco certo (/data) | Checks whether the bind mounts (DATA_DIR/HF_CACHE_DIR/CKPT_DIR) land on the right disk (/data)
> @echo "── 1) Config no .env ──"
> @grep -E '^(DATA_DIR|HF_CACHE_DIR|CKPT_DIR)=' .env || echo "  (defaults: ./data ./hf-cache ./checkpoints)"
> @echo "── 2) Escreve arquivo de teste de DENTRO do container ──"
> @$(DC) run --rm train sh -c 'for d in /workspace/manaca-corpus /workspace/hf-cache /workspace/checkpoints; do echo manaca-mount-test > "$$d/_mount_test.txt" && echo "  OK escrever: $$d"; done'
> @echo "── 3) Confere no HOST + espaço em disco (df) ──"
> @dd=$$(grep -E '^DATA_DIR=' .env | cut -d= -f2-); hf=$$(grep -E '^HF_CACHE_DIR=' .env | cut -d= -f2-); ck=$$(grep -E '^CKPT_DIR=' .env | cut -d= -f2-); \
>   for p in "$${dd:-./data}" "$${hf:-./hf-cache}" "$${ck:-./checkpoints}"; do \
>     if [ -f "$$p/_mount_test.txt" ]; then echo "  ✓ apareceu no host: $$p"; else echo "  ✗ NAO apareceu: $$p"; fi; \
>     df -h "$$p" 2>/dev/null | tail -1 | awk '{print "     disco:",$$1,"| livre:",$$4,"| montado:",$$6}'; \
>   done
> @echo "── 4) Limpa os arquivos de teste ──"
> @$(DC) run --rm train sh -c 'rm -f /workspace/manaca-corpus/_mount_test.txt /workspace/hf-cache/_mount_test.txt /workspace/checkpoints/_mount_test.txt' && echo "  OK"

gigaverbo:  ## Baixa GigaVerbo (destacado; logs: make logs SRC=gigaverbo) | Downloads GigaVerbo (detached; logs: make logs SRC=gigaverbo)
> $(DC) run -d --name gigaverbo corpus python corpus/scripts/01_acquire_gigaverbo.py

madlad:  ## Baixa MADLAD-400 (destacado) | Downloads MADLAD-400 (detached)
> $(DC) run -d --name madlad corpus python corpus/scripts/02_acquire_madlad400.py

fineweb2:  ## Baixa FineWeb-2 (destacado) | Downloads FineWeb-2 (detached)
> $(DC) run -d --name fineweb2 corpus python corpus/scripts/03_acquire_fineweb2.py

hplt2:  ## Baixa HPLT 2.0 (destacado) | Downloads HPLT 2.0 (detached)
> $(DC) run -d --name hplt2 corpus python corpus/scripts/04_acquire_hplt2.py

wikipedia:  ## Baixa Wikipedia PT-BR (destacado) | Downloads Wikipedia PT-BR (detached)
> $(DC) run -d --name wikipedia corpus python corpus/scripts/05_acquire_wikipedia.py

ulysses:  ## Baixa Ulysses Tesemõ via git (obsoleto — dados não estão no git; use ulysses-gdrive) | Downloads Ulysses Tesemõ via git (obsolete — data is not in git; use ulysses-gdrive)
> $(DC) run -d --name ulysses corpus python corpus/scripts/06_acquire_ulysses.py

ulysses-gdrive:  ## Baixa Ulysses Tesemõ do Google Drive (Script 06b, destacado) | Downloads Ulysses Tesemõ from Google Drive (Script 06b, detached)
> -docker rm ulysses 2>/dev/null || true
> $(DC) run -d --name ulysses corpus python corpus/scripts/06b_acquire_ulysses_gdrive.py

acquire:  ## Roda um script arbitrário: make acquire SCRIPT=corpus/scripts/NN_....py | Runs an arbitrary script: make acquire SCRIPT=corpus/scripts/NN_....py
> $(DC) run --rm corpus python $(SCRIPT)

logs:  ## Segue os logs de um container destacado: make logs SRC=gigaverbo | Follows the logs of a detached container: make logs SRC=gigaverbo
> docker logs -f $(SRC)

own:  ## Corrige a posse dos arquivos gerados (evita root-owned; use após cada fonte) | Fixes ownership of generated files (avoids root-owned; use after each source)
> $(DC) run --rm corpus chown -R $$(id -u):$$(id -g) /workspace/manaca-corpus /workspace/hf-cache

trim-gigaverbo:  ## Mantém só os primeiros N shards do GigaVerbo: make trim-gigaverbo KEEP=600 | Keeps only the first N GigaVerbo shards: make trim-gigaverbo KEEP=600
> $(DC) run --rm corpus sh -c 'cd /workspace/manaca-corpus/raw/gigaverbo && ls shard_*.parquet | sort | tail -n +$$(( $(KEEP) + 1 )) | xargs -r rm -f && echo "Restam: $$(ls shard_*.parquet | wc -l) shards" && du -sh .'

cc-test:  ## Pipeline Common Crawl — teste local (1 snapshot, 4 tarefas) | Common Crawl pipeline — local test (1 snapshot, 4 tasks)
> $(DC) run --rm corpus python corpus/scripts/07_cc_pipeline.py \
>   --snapshot CC-MAIN-2025-08 --num-tasks 4 --executor local

cc-pipeline:  ## Pipeline Common Crawl — todos os snapshots (executor local) | Common Crawl pipeline — all snapshots (local executor)
> $(DC) run --rm corpus python corpus/scripts/07_cc_pipeline.py --executor local --num-tasks 8

dedup:  ## Deduplicação global cross-source (Script 08) | Global cross-source deduplication (Script 08)
> $(DC) run --rm corpus python corpus/scripts/08_global_dedup.py

validate:  ## Validação e estatísticas do corpus (Script 09; DIR=raw e meta ~20B do Manacá-1B; a dedup global é pulada nas 3 fontes curadas) | Corpus validation and statistics (Script 09; DIR=raw and ~20B target of Manacá-1B; global dedup is skipped for the 3 curated sources)
> $(DC) run --rm corpus python corpus/scripts/09_validate_corpus.py --corpus-dir /workspace/manaca-corpus/$(or $(DIR),raw) --target-tokens-b $(or $(TARGET_B),20)

characterize:  ## Caracterização do corpus (Script 10) | Corpus characterization (Script 10)
> $(DC) run --rm corpus python corpus/scripts/10_characterize_corpus.py

tokenizer-shell:  ## Shell na imagem para o pipeline do tokenizador (Fase 2a) | Shell into the image for the tokenizer pipeline (Phase 2a)
> $(DC) run --rm tokenizer /bin/sh

tokenizer-sample:  ## Fase 2a.1 — amostra balanceada p/ tokenizador: make tokenizer-sample GB=5 | Phase 2a.1 — balanced sample for tokenizer: make tokenizer-sample GB=5
> $(DC) run --rm tokenizer python tokenizer/scripts/01_sample_corpus.py --gb $(or $(GB),5)

tokenizer-train:  ## Fase 2a.2 — treina SentencePiece: make tokenizer-train VOCAB=64000 | Phase 2a.2 — trains SentencePiece: make tokenizer-train VOCAB=64000
> $(DC) run --rm tokenizer python tokenizer/scripts/02_train_spm.py --vocab-size $(or $(VOCAB),64000)

tokenizer-convert:  ## Fase 2a.3 — converte SPM -> HuggingFace | Phase 2a.3 — converts SPM -> HuggingFace
> $(DC) run --rm tokenizer python tokenizer/scripts/03_convert_to_hf.py

tokenizer: tokenizer-sample tokenizer-train tokenizer-convert  ## Fase 2a completa (amostra+treino+conversão) | Full Phase 2a (sample+train+convert)

# ── Fase 2b/4: treino e avaliação em GPU (equivale a `sbatch ... srun`) ───────
preprocess-megatron:  ## Fase 2b prep — Parquet -> Megatron .bin/.idx (imagem train; requer 'make build-train' + tokenizer). Ex.: make preprocess-megatron ARGS="--workers 16" | Phase 2b prep — Parquet -> Megatron .bin/.idx (train image; requires 'make build-train' + tokenizer). E.g.: make preprocess-megatron ARGS="--workers 16"
> $(DC) run --rm train python corpus/scripts/11_prepare_megatron.py $(ARGS)

pretrain:  ## Pré-treino Megatron (single/multinó): make pretrain ARGS="..." | Megatron pretraining (single/multi-node): make pretrain ARGS="..."
> ./scripts/docker/run_pretrain.sh $(ARGS)

prune-checkpoints:  ## Mantém só os últimos KEEP checkpoints (o fork NÃO limpa!): make prune-checkpoints KEEP=3 MODEL=manaca-1b | Keeps only the last KEEP checkpoints (the fork does NOT clean up!): make prune-checkpoints KEEP=3 MODEL=manaca-1b
> $(DC) run --rm train sh -c 'cd /workspace/checkpoints/$(or $(MODEL),manaca-1b) 2>/dev/null && ls -d iter_* 2>/dev/null | sort | head -n -$(or $(KEEP),3) | xargs -r rm -rf; echo "Restam $$(ls -d iter_* 2>/dev/null | wc -l) checkpoints;"; du -sh . 2>/dev/null'

autoprune:  ## Pruner em background: a cada 30min mantém só KEEP checkpoints. make autoprune KEEP=3 MODEL=manaca-1b | Background pruner: every 30min keeps only KEEP checkpoints. make autoprune KEEP=3 MODEL=manaca-1b
> -docker rm manaca-prune 2>/dev/null || true
> $(DC) run -d --name manaca-prune train sh -c 'while true; do cd /workspace/checkpoints/$(or $(MODEL),manaca-1b) 2>/dev/null && ls -d iter_* 2>/dev/null | sort | head -n -$(or $(KEEP),3) | xargs -r rm -rf; sleep 1800; done'
> @echo "Pruner iniciado (KEEP=$(or $(KEEP),3)). Pare com: make stop-autoprune"

stop-autoprune:  ## Para o pruner em background | Stops the background pruner
> -docker stop manaca-prune 2>/dev/null && docker rm manaca-prune 2>/dev/null || true

eval-shell:  ## Shell na imagem de avaliação (Fase 4) | Shell into the evaluation image (Phase 4)
> $(DC) run --rm eval /bin/bash

# ── Shells interativos ───────────────────────────────────────────────────────
shell-corpus:  ## Shell na imagem de corpus (CPU) | Shell into the corpus image (CPU)
> $(DC) run --rm corpus /bin/sh

shell-train:  ## Shell na imagem de pré-treino (GPU) | Shell into the pretraining image (GPU)
> $(DC) run --rm train /bin/bash

# ── Limpeza ──────────────────────────────────────────────────────────────────
down: env  ## Para e remove containers do projeto | Stops and removes the project containers
> $(DC) down --remove-orphans

clean: env  ## Remove containers e imagens do Manacá (mantém ./data) | Removes Manacá containers and images (keeps ./data)
> $(DC) down --remove-orphans --rmi local

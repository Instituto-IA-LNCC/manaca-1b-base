# Guia de Configuração do Ambiente Docker

**[🇧🇷 Português](#português)** · **[🇬🇧 English](#english)**

## Português

### Manacá-1B — LNCC (Instituto de IA) × NII/LLM-jp

**Instituição:** LNCC / Instituto de IA
**Status:** Ambiente Docker — pipeline reprodutível de corpus, pré-treino e avaliação

> O Manacá-1B roda todo o pipeline de treinamento em **Docker**, com imagens que
> congelam SO, CUDA, Python e dependências. O mesmo `docker compose` roda num
> servidor GPU local ou na nuvem; um apêndice mostra como converter as imagens
> para **Apptainer/Singularity** em clusters HPC que não permitem Docker.

---

### Sumário

1. [Por que Docker](#1-por-que-docker)
2. [Pré-requisitos](#2-pré-requisitos)
3. [Arquitetura de Imagens](#3-arquitetura-de-imagens)
4. [Modelo de Armazenamento](#4-modelo-de-armazenamento)
5. [Configuração Inicial](#5-configuração-inicial)
6. [Build das Imagens](#6-build-das-imagens)
7. [Verificação do Ambiente](#7-verificação-do-ambiente)
8. [Execução por Fase](#8-execução-por-fase)
9. [Treinamento Multinó](#9-treinamento-multinó)
10. [Mapeamento de Comandos HPC → Docker](#10-mapeamento-de-comandos-hpc--docker)
11. [Variáveis de Ambiente](#11-variáveis-de-ambiente)
12. [Solução de Problemas](#12-solução-de-problemas)
13. [Apêndice: HPC rootless (Apptainer)](#13-apêndice-hpc-rootless-apptainer)

---

### 1. Por que Docker

Ambientes de HPC costumam fazer o isolamento de software com
**Singularity/Apptainer**, a orquestração com **SLURM** e a gestão de dependências
com **conda** carregado via `module load`.

O Manacá-1B usa **Docker** no lugar dessa pilha, com três objetivos:

- **Reprodutibilidade:** a imagem congela SO, CUDA, Python e todas as
  dependências. Não depende de `module load` nem de um `conda env` mantido à mão.
- **Portabilidade:** o mesmo `docker compose` roda no laptop, num servidor GPU
  local ou na nuvem, sem depender de NFS nem da fila SLURM.
- **Simplicidade operacional:** `docker compose run` substitui a tríade
  `conda activate` + `screen` + `sbatch`.

O código-fonte **não é copiado** para dentro das imagens: as imagens carregam só
o *ambiente*; o repositório é montado em `/workspace/manaca` via *bind mount*.
Assim você edita os scripts no host e executa no container sem rebuild.

---

### 2. Pré-requisitos

| Componente | Versão mínima | Observação |
|------------|---------------|------------|
| Docker Engine | 24.x | `docker --version` |
| Docker Compose | v2.20+ | `docker compose version` (plugin, não `docker-compose`) |
| NVIDIA driver | compatível com CUDA 12.x | apenas para Fases 2b/4 (GPU) |
| NVIDIA Container Toolkit | 1.14+ | habilita `--gpus` no Docker (GPU) |
| Disco | ≥ 2 TB recomendado | corpus da Fase 1 cresce para vários TB |

Instalação do NVIDIA Container Toolkit (Ubuntu), necessário só para GPU:

```bash
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
# Teste:
docker run --rm --gpus all nvcr.io/nvidia/pytorch:24.12-py3 nvidia-smi
```

---

### 3. Arquitetura de Imagens

Três imagens cobrem as fases deste repositório. Cada uma tem um Dockerfile em
`docker/` e um arquivo de dependências em `requirements/`.

| Serviço (compose) | Imagem | Base | Fase | Dockerfile |
|-------------------|--------|------|------|------------|
| `corpus` | `manaca-corpus` | `python:3.11-slim` (CPU) | 1 — Corpus | `docker/Dockerfile.corpus` |
| `tokenizer` | `manaca-corpus` | idem (CPU) | 2a — Tokenizador | `docker/Dockerfile.corpus` |
| `train` | `manaca-train` | `nvcr.io/nvidia/pytorch` (GPU) | 2b — Pré-treino | `docker/Dockerfile.train` |
| `eval` | `manaca-eval` | `nvcr.io/nvidia/pytorch` (GPU) | 4 — avaliação | `docker/Dockerfile.eval` |

A imagem `manaca-corpus` (Python 3.11) traz a pilha de dados: `datasets`,
`datatrove`, `pyarrow`, `datasketch`, `fasttext`, `sentencepiece`, `trafilatura`,
etc. (ver `requirements/corpus.txt`).

As imagens GPU partem de um container **NGC PyTorch**, que já traz CUDA, cuDNN,
NCCL, PyTorch, Apex, Transformer-Engine e FlashAttention compilados — o mesmo
runtime que em HPC seria empacotado como `.sif`. Se `nvcr.io` estiver bloqueado
pela política de rede, troque a base:

```bash
# .env
GPU_BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
```

---

### 4. Modelo de Armazenamento

O armazenamento usa **bind mounts** para diretórios do host. Dentro do container
os caminhos são fixos:

| Host (configurável no `.env`) | Container | Papel |
|-------------------------------|-----------|-------|
| `${DATA_DIR:-./data}` | `/workspace/manaca-corpus` | `WORK_DIR` — corpus, logs, checkpoints de jobs |
| `${HF_CACHE_DIR:-./hf-cache}` | `/workspace/hf-cache` | `HF_HOME` — cache HuggingFace |
| `${CKPT_DIR:-./checkpoints}` | `/workspace/checkpoints` | checkpoints de treino (Fase 2b) |
| `.` (raiz do repo) | `/workspace/manaca` | código (bind mount, editável) |

Todos os scripts de corpus leem `WORK_DIR` da variável de ambiente — o mesmo
mecanismo funciona para um volume local e para um NFS. Para apontar a um NFS num
host HPC com Docker, basta `DATA_DIR=/caminho/do/nfs/manaca-corpus`.

---

### 5. Configuração Inicial

```bash
git clone https://github.com/brunoleomenezes/manaca-1b-base.git
cd manaca-1b-base

# Cria o .env a partir do exemplo (ou: make env)
cp .env.example .env
# Edite .env: HF_TOKEN, WANDB_API_KEY, DATA_DIR, GPU_BASE_IMAGE, ...
```

Variáveis principais do `.env` estão documentadas na Seção 11.

---

### 6. Build das Imagens

```bash
make build-corpus         # imagem CPU (Fases 1 e 2a)
make build-train          # imagem GPU de pré-treino (Fase 2b)
make build-eval           # imagem GPU de avaliação (Fase 4)
# ou tudo de uma vez:
make build
```

Equivalente sem Makefile:

```bash
docker compose build corpus
docker compose build train
```

Para habilitar o KenLM (Fase 1.5 — rescoring; Script 10, futuro):

```bash
docker compose build corpus --build-arg INSTALL_KENLM=true
```

---

### 7. Verificação do Ambiente

Executa o Script 00 dentro do container (equivale ao `python
corpus/scripts/00_verify_env.py` do fluxo conda):

```bash
make verify
# ou:
docker compose run --rm corpus python corpus/scripts/00_verify_env.py
```

A verificação de armazenamento agora checa o **volume WORK_DIR** (montado), não
mais o NFS. Todos os 16 pacotes devem retornar OK.

---

### 8. Execução por Fase

#### Fase 1 — Corpus (CPU)

Aquisição de fontes. Cada download longo roda em um **container destacado**
(equivalente ao `screen -S <fonte>` do fluxo HPC):

```bash
make gigaverbo                 # docker compose run -d --name gigaverbo corpus ...
make logs SRC=gigaverbo        # segue o log (equivale a tail -f download.log)

make madlad
make fineweb2
make hplt2
make wikipedia
make ulysses
```

Pipeline Common Crawl (Script 07):

```bash
make cc-test                   # teste local: 1 snapshot, 4 tarefas
make cc-pipeline               # todos os snapshots (executor local)
```

> **Nota sobre SLURM:** os Scripts 07 e 08 ainda suportam
> `--executor slurm` (via `datatrove.SlurmPipelineExecutor`) para quem roda num
> cluster com SLURM. Em Docker use o executor `local`; para escala, distribua
> vários containers `corpus` ou monte o volume num nó com mais CPUs.

Deduplicação e validação:

```bash
make dedup                     # Script 08 — dedup global cross-source
make validate                  # Script 09 — validação + estatísticas
```

#### Fase 2a — Tokenizador (CPU)

Usa a mesma imagem `corpus` (que já inclui `sentencepiece`):

```bash
docker compose run --rm tokenizer python tokenizer/scripts/01_sample.py
# ... demais estágios do pipeline do tokenizador (Fase 2a).
```

#### Fase 2b — Pré-treinamento (GPU)

Executa o `pretrain_gpt.py` do Megatron com `torchrun` num container GPU, com
**resume automático** do último checkpoint:

```bash
make build-train
make pretrain                                  # Manacá-1B (2 GPUs por padrão)
# argumentos extras vão direto ao pretrain_gpt.py:
make pretrain ARGS="--train-iters 50000 --micro-batch-size 2"
```

Antes é preciso gerar os dados Megatron (`.bin`/`.idx`) e o tokenizador — ver
`make preprocess-megatron` e o pipeline em `tokenizer/`.

#### Fase 4 — avaliação (GPU)

```bash
make build-eval
make eval-shell                # shell na imagem com vLLM + métricas
```

Os benchmarks em português (CALAME-PT, ARC-Challenge-PT, HellaSwag-PT,
LAMBADA-PT) e os testes pareados ficam em `scripts/eval/` — ver o
[README principal](../../README.md).

---

### 9. Treinamento Multinó

Em HPC, o SLURM aloca N nós e o `srun` lança um processo por GPU. Em Docker, roda-se
o **mesmo container em cada nó físico**, com `--network host` (para o NCCL) e
`torchrun` fazendo o *rendezvous*. O `run_pretrain.sh` já faz isso:

```bash
# Nó 0 (MASTER_ADDR = IP interno do nó 0):
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 GPUS_PER_NODE=8 \
  ./scripts/docker/run_pretrain.sh

# Nó 1 (mesmo comando, NODE_RANK=1):
NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 GPUS_PER_NODE=8 \
  ./scripts/docker/run_pretrain.sh
```

O paralelismo (TP × PP × DP) é passado via flags do Megatron (`ARGS`). Ajuste
`NCCL_SOCKET_IFNAME`/`NCCL_IB_*` no `.env` conforme a rede (InfiniBand) do cluster.

---

### 10. Mapeamento de Comandos HPC → Docker

| Fluxo HPC / Singularity | Equivalente Docker |
|-------------------------|--------------------|
| `conda activate manaca-corpus` | (nada — o ambiente já está na imagem) |
| `module load cuda/12.1 cudnn/9.0` | CUDA embutido na base NGC PyTorch |
| `python corpus/scripts/00_verify_env.py` | `docker compose run --rm corpus python corpus/scripts/00_verify_env.py` |
| `screen -S gigaverbo -dm ... python 01_...py` | `docker compose run -d --name gigaverbo corpus python corpus/scripts/01_...py` |
| `tail -f $WORK_DIR/raw/<fonte>/download.log` | `docker logs -f <fonte>` (container destacado; ou o mesmo `tail` no volume) |
| `sbatch job.slurm` / `srun python pretrain_gpt.py` | `./scripts/docker/run_pretrain.sh` (`torchrun` em container GPU) |
| `--dependency=afterok` (chain job) | resume automático do último `iter_*` no `run_pretrain.sh` |
| `datatrove SlurmPipelineExecutor` | `--executor local` no container (ou SLURM se disponível) |
| `singularity build img.sif docker://...` | `docker build` (nativo) |
| `singularity exec img.sif <cmd>` | `docker run <imagem> <cmd>` |
| NFS `/caminho/do/nfs/...` | bind mount `./data` → `/workspace/manaca-corpus` |
| `squeue -u $USER` | `docker ps` |

---

### 11. Variáveis de Ambiente

Definidas no `.env` (ver `.env.example`):

| Variável | Padrão | Papel |
|----------|--------|-------|
| `DATA_DIR` | `./data` | Host → `/workspace/manaca-corpus` (WORK_DIR) |
| `HF_CACHE_DIR` | `./hf-cache` | Host → `/workspace/hf-cache` (HF_HOME) |
| `CKPT_DIR` | `./checkpoints` | Host → `/workspace/checkpoints` |
| `HF_TOKEN` | — | Token HuggingFace (fontes gated) |
| `WANDB_API_KEY` | — | Weights & Biases |
| `GPU_BASE_IMAGE` | `nvcr.io/nvidia/pytorch:24.12-py3` | Base das imagens GPU |
| `MEGATRON_REPO` / `MEGATRON_REF` | `llm-jp/Megatron-LM` / `nii-geniac` | Fork do Megatron-LM |
| `NNODES` / `NODE_RANK` / `MASTER_ADDR` / `GPUS_PER_NODE` | `1` / `0` / `127.0.0.1` / `8` | Topologia multinó |

---

### 12. Solução de Problemas

| Sintoma | Causa provável | Correção |
|---------|----------------|----------|
| `could not select device driver "nvidia"` | NVIDIA Container Toolkit ausente | Instale o toolkit (Seção 2) |
| Build GPU falha ao puxar `nvcr.io/...` | Política de rede bloqueia NGC | `GPU_BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` |
| Arquivos no `./data` pertencem ao `root` | Container roda como root | Descomente `user:` no compose e defina `HOST_UID`/`HOST_GID` |
| `WORK_DIR sem permissao de escrita` | Bind mount não montado ou sem permissão | Confira `DATA_DIR` no `.env` e as permissões do diretório |
| OOM/NCCL em multinó | `--network host`/IB mal configurados | Ajuste `NCCL_SOCKET_IFNAME`, `--shm-size`, `--ipc host` |

---

### 13. Apêndice: HPC rootless (Apptainer)

Muitos supercomputadores (inclusive o **Santos Dumont**) **não permitem Docker**
(precisa de root) e usam **Apptainer/Singularity**. As mesmas imagens deste repo
convertem-se para `.sif` sem reescrever nada:

```bash
# No host com Docker, gere a imagem e exporte:
docker build -f docker/Dockerfile.train -t manaca-train:latest .
docker save manaca-train:latest -o manaca-train.tar

# No cluster (Apptainer):
apptainer build manaca-train.sif docker-archive://manaca-train.tar
srun apptainer exec --nv manaca-train.sif \
     torchrun ... /opt/Megatron-LM/pretrain_gpt.py ...
```

Assim o desenvolvimento e a validação acontecem em Docker, e a produção em larga
escala pode ocorrer em HPC via Apptainer — sem divergência de ambiente.

---

*Projeto Manacá-1B — LNCC (Instituto de IA) × NII/LLM-jp*

## English

### Manacá-1B — LNCC (AI Institute) × NII/LLM-jp

**Institution:** LNCC / AI Institute
**Status:** Docker environment — reproducible corpus, pre-training, and evaluation pipeline

> Manacá-1B runs the entire training pipeline on **Docker**, with images that
> freeze the OS, CUDA, Python, and dependencies. The same `docker compose` runs on a
> local GPU server or in the cloud; an appendix shows how to convert the images
> to **Apptainer/Singularity** on HPC clusters that do not allow Docker.

---

### Table of Contents

1. [Why Docker](#1-why-docker)
2. [Prerequisites](#2-prerequisites)
3. [Image Architecture](#3-image-architecture)
4. [Storage Model](#4-storage-model)
5. [Initial Setup](#5-initial-setup)
6. [Building the Images](#6-building-the-images)
7. [Environment Verification](#7-environment-verification)
8. [Running by Phase](#8-running-by-phase)
9. [Multinode Training](#9-multinode-training)
10. [HPC → Docker Command Mapping](#10-hpc--docker-command-mapping)
11. [Environment Variables](#11-environment-variables)
12. [Troubleshooting](#12-troubleshooting)
13. [Appendix: Rootless HPC (Apptainer)](#13-appendix-rootless-hpc-apptainer)

---

### 1. Why Docker

HPC environments usually handle software isolation with
**Singularity/Apptainer**, orchestration with **SLURM**, and dependency management
with **conda** loaded via `module load`.

Manacá-1B uses **Docker** in place of that stack, with three goals:

- **Reproducibility:** the image freezes the OS, CUDA, Python, and all
  dependencies. It does not depend on `module load` or a hand-maintained `conda env`.
- **Portability:** the same `docker compose` runs on a laptop, on a local GPU
  server, or in the cloud, without depending on NFS or the SLURM queue.
- **Operational simplicity:** `docker compose run` replaces the trio
  `conda activate` + `screen` + `sbatch`.

The source code is **not copied** into the images: the images carry only
the *environment*; the repository is mounted at `/workspace/manaca` via *bind mount*.
This way you edit the scripts on the host and run them in the container without a rebuild.

---

### 2. Prerequisites

| Component | Minimum version | Note |
|------------|---------------|------------|
| Docker Engine | 24.x | `docker --version` |
| Docker Compose | v2.20+ | `docker compose version` (plugin, not `docker-compose`) |
| NVIDIA driver | compatible with CUDA 12.x | only for Phases 2b/4 (GPU) |
| NVIDIA Container Toolkit | 1.14+ | enables `--gpus` in Docker (GPU) |
| Disk | ≥ 2 TB recommended | the Phase 1 corpus grows to several TB |

Installing the NVIDIA Container Toolkit (Ubuntu), needed only for GPU:

```bash
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
# Test:
docker run --rm --gpus all nvcr.io/nvidia/pytorch:24.12-py3 nvidia-smi
```

---

### 3. Image Architecture

Three images cover the phases of this repository. Each one has a Dockerfile in
`docker/` and a dependencies file in `requirements/`.

| Service (compose) | Image | Base | Phase | Dockerfile |
|-------------------|--------|------|------|------------|
| `corpus` | `manaca-corpus` | `python:3.11-slim` (CPU) | 1 — Corpus | `docker/Dockerfile.corpus` |
| `tokenizer` | `manaca-corpus` | same (CPU) | 2a — Tokenizer | `docker/Dockerfile.corpus` |
| `train` | `manaca-train` | `nvcr.io/nvidia/pytorch` (GPU) | 2b — Pre-training | `docker/Dockerfile.train` |
| `eval` | `manaca-eval` | `nvcr.io/nvidia/pytorch` (GPU) | 4 — evaluation | `docker/Dockerfile.eval` |

The `manaca-corpus` image (Python 3.11) brings the data stack: `datasets`,
`datatrove`, `pyarrow`, `datasketch`, `fasttext`, `sentencepiece`, `trafilatura`,
etc. (see `requirements/corpus.txt`).

The GPU images start from an **NGC PyTorch** container, which already ships CUDA, cuDNN,
NCCL, PyTorch, Apex, Transformer-Engine, and FlashAttention compiled — the same
runtime that on HPC would be packaged as `.sif`. If `nvcr.io` is blocked
by the network policy, swap the base:

```bash
# .env
GPU_BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
```

---

### 4. Storage Model

Storage uses **bind mounts** to host directories. Inside the container
the paths are fixed:

| Host (configurable in `.env`) | Container | Role |
|-------------------------------|-----------|-------|
| `${DATA_DIR:-./data}` | `/workspace/manaca-corpus` | `WORK_DIR` — corpus, logs, job checkpoints |
| `${HF_CACHE_DIR:-./hf-cache}` | `/workspace/hf-cache` | `HF_HOME` — HuggingFace cache |
| `${CKPT_DIR:-./checkpoints}` | `/workspace/checkpoints` | training checkpoints (Phase 2b) |
| `.` (repo root) | `/workspace/manaca` | code (bind mount, editable) |

All corpus scripts read `WORK_DIR` from the environment variable — the same
mechanism works for a local volume and for an NFS. To point to an NFS on an
HPC host with Docker, just set `DATA_DIR=/path/to/nfs/manaca-corpus`.

---

### 5. Initial Setup

```bash
git clone https://github.com/brunoleomenezes/manaca-1b-base.git
cd manaca-1b-base

# Create the .env from the example (or: make env)
cp .env.example .env
# Edit .env: HF_TOKEN, WANDB_API_KEY, DATA_DIR, GPU_BASE_IMAGE, ...
```

The main `.env` variables are documented in Section 11.

---

### 6. Building the Images

```bash
make build-corpus         # CPU image (Phases 1 and 2a)
make build-train          # GPU pre-training image (Phase 2b)
make build-eval           # GPU evaluation image (Phase 4)
# or all at once:
make build
```

Equivalent without the Makefile:

```bash
docker compose build corpus
docker compose build train
```

To enable KenLM (Phase 1.5 — rescoring; Script 10, future):

```bash
docker compose build corpus --build-arg INSTALL_KENLM=true
```

---

### 7. Environment Verification

Runs Script 00 inside the container (equivalent to `python
corpus/scripts/00_verify_env.py` in the conda flow):

```bash
make verify
# or:
docker compose run --rm corpus python corpus/scripts/00_verify_env.py
```

The storage check now verifies the **WORK_DIR volume** (mounted), no
longer the NFS. All 16 packages should return OK.

---

### 8. Running by Phase

#### Phase 1 — Corpus (CPU)

Source acquisition. Each long download runs in a **detached container**
(equivalent to `screen -S <source>` in the HPC flow):

```bash
make gigaverbo                 # docker compose run -d --name gigaverbo corpus ...
make logs SRC=gigaverbo        # follows the log (equivalent to tail -f download.log)

make madlad
make fineweb2
make hplt2
make wikipedia
make ulysses
```

Common Crawl pipeline (Script 07):

```bash
make cc-test                   # local test: 1 snapshot, 4 tasks
make cc-pipeline               # all snapshots (local executor)
```

> **Note on SLURM:** Scripts 07 and 08 still support
> `--executor slurm` (via `datatrove.SlurmPipelineExecutor`) for those running on a
> cluster with SLURM. On Docker use the `local` executor; for scale, distribute
> multiple `corpus` containers or mount the volume on a node with more CPUs.

Deduplication and validation:

```bash
make dedup                     # Script 08 — global cross-source dedup
make validate                  # Script 09 — validation + statistics
```

#### Phase 2a — Tokenizer (CPU)

Uses the same `corpus` image (which already includes `sentencepiece`):

```bash
docker compose run --rm tokenizer python tokenizer/scripts/01_sample.py
# ... remaining stages of the tokenizer pipeline (Phase 2a).
```

#### Phase 2b — Pre-training (GPU)

Runs Megatron's `pretrain_gpt.py` with `torchrun` in a GPU container, with
**automatic resume** from the last checkpoint:

```bash
make build-train
make pretrain                                  # Manacá-1B (2 GPUs by default)
# extra arguments go straight to pretrain_gpt.py:
make pretrain ARGS="--train-iters 50000 --micro-batch-size 2"
```

Beforehand you must generate the Megatron data (`.bin`/`.idx`) and the tokenizer — see
`make preprocess-megatron` and the pipeline in `tokenizer/`.

#### Phase 4 — evaluation (GPU)

```bash
make build-eval
make eval-shell                # shell in the image with vLLM + metrics
```

The Portuguese benchmarks (CALAME-PT, ARC-Challenge-PT, HellaSwag-PT,
LAMBADA-PT) and the paired tests are in `scripts/eval/` — see the
[main README](../../README.md).

---

### 9. Multinode Training

On HPC, SLURM allocates N nodes and `srun` launches one process per GPU. On Docker, you run
the **same container on each physical node**, with `--network host` (for NCCL) and
`torchrun` doing the *rendezvous*. `run_pretrain.sh` already does this:

```bash
# Node 0 (MASTER_ADDR = node 0's internal IP):
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 GPUS_PER_NODE=8 \
  ./scripts/docker/run_pretrain.sh

# Node 1 (same command, NODE_RANK=1):
NNODES=2 NODE_RANK=1 MASTER_ADDR=10.0.0.1 GPUS_PER_NODE=8 \
  ./scripts/docker/run_pretrain.sh
```

Parallelism (TP × PP × DP) is passed via Megatron flags (`ARGS`). Adjust
`NCCL_SOCKET_IFNAME`/`NCCL_IB_*` in the `.env` according to the cluster's network (InfiniBand).

---

### 10. HPC → Docker Command Mapping

| HPC / Singularity flow | Docker equivalent |
|-------------------------|--------------------|
| `conda activate manaca-corpus` | (nothing — the environment is already in the image) |
| `module load cuda/12.1 cudnn/9.0` | CUDA built into the NGC PyTorch base |
| `python corpus/scripts/00_verify_env.py` | `docker compose run --rm corpus python corpus/scripts/00_verify_env.py` |
| `screen -S gigaverbo -dm ... python 01_...py` | `docker compose run -d --name gigaverbo corpus python corpus/scripts/01_...py` |
| `tail -f $WORK_DIR/raw/<source>/download.log` | `docker logs -f <source>` (detached container; or the same `tail` on the volume) |
| `sbatch job.slurm` / `srun python pretrain_gpt.py` | `./scripts/docker/run_pretrain.sh` (`torchrun` in a GPU container) |
| `--dependency=afterok` (chain job) | automatic resume from the last `iter_*` in `run_pretrain.sh` |
| `datatrove SlurmPipelineExecutor` | `--executor local` in the container (or SLURM if available) |
| `singularity build img.sif docker://...` | `docker build` (native) |
| `singularity exec img.sif <cmd>` | `docker run <image> <cmd>` |
| NFS `/path/to/nfs/...` | bind mount `./data` → `/workspace/manaca-corpus` |
| `squeue -u $USER` | `docker ps` |

---

### 11. Environment Variables

Defined in the `.env` (see `.env.example`):

| Variable | Default | Role |
|----------|--------|-------|
| `DATA_DIR` | `./data` | Host → `/workspace/manaca-corpus` (WORK_DIR) |
| `HF_CACHE_DIR` | `./hf-cache` | Host → `/workspace/hf-cache` (HF_HOME) |
| `CKPT_DIR` | `./checkpoints` | Host → `/workspace/checkpoints` |
| `HF_TOKEN` | — | HuggingFace token (gated sources) |
| `WANDB_API_KEY` | — | Weights & Biases |
| `GPU_BASE_IMAGE` | `nvcr.io/nvidia/pytorch:24.12-py3` | Base of the GPU images |
| `MEGATRON_REPO` / `MEGATRON_REF` | `llm-jp/Megatron-LM` / `nii-geniac` | Megatron-LM fork |
| `NNODES` / `NODE_RANK` / `MASTER_ADDR` / `GPUS_PER_NODE` | `1` / `0` / `127.0.0.1` / `8` | Multinode topology |

---

### 12. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|----------------|----------|
| `could not select device driver "nvidia"` | NVIDIA Container Toolkit missing | Install the toolkit (Section 2) |
| GPU build fails to pull `nvcr.io/...` | Network policy blocks NGC | `GPU_BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` |
| Files in `./data` are owned by `root` | Container runs as root | Uncomment `user:` in the compose and set `HOST_UID`/`HOST_GID` |
| `WORK_DIR sem permissao de escrita` | Bind mount not mounted or without permission | Check `DATA_DIR` in the `.env` and the directory permissions |
| OOM/NCCL in multinode | `--network host`/IB misconfigured | Adjust `NCCL_SOCKET_IFNAME`, `--shm-size`, `--ipc host` |

---

### 13. Appendix: Rootless HPC (Apptainer)

Many supercomputers (including **Santos Dumont**) **do not allow Docker**
(it requires root) and use **Apptainer/Singularity**. The same images from this repo
convert to `.sif` without rewriting anything:

```bash
# On the host with Docker, build the image and export it:
docker build -f docker/Dockerfile.train -t manaca-train:latest .
docker save manaca-train:latest -o manaca-train.tar

# On the cluster (Apptainer):
apptainer build manaca-train.sif docker-archive://manaca-train.tar
srun apptainer exec --nv manaca-train.sif \
     torchrun ... /opt/Megatron-LM/pretrain_gpt.py ...
```

This way development and validation happen on Docker, and large-scale
production can run on HPC via Apptainer — with no environment drift.

---

*Manacá-1B Project — LNCC (AI Institute) × NII/LLM-jp*

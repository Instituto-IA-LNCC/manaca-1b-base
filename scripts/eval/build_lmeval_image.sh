#!/usr/bin/env bash
# =============================================================================
# Manaca-1B - Constroi a imagem manaca-lmeval SEM `docker build`
# -----------------------------------------------------------------------------
# Em alguns hosts o `docker build` nao aceita --sysctl e o DNS quebra com IPv6 (o
# pip nao resolve o pypi). Aqui instalamos o lm-eval DENTRO da manaca-train (que ja
# tem torch + transformers==4.46.3, compatíveis) via `docker run` com IPv6
# desabilitado (onde o DNS funciona) e commitamos a imagem, preservando o
# entrypoint da base. Nada de novo build.
#
# Uso:
#   ./scripts/eval/build_lmeval_image.sh
# Overrides: BASE_IMAGE (default manaca-train:latest), OUT_IMAGE (manaca-lmeval:latest).
# =============================================================================
set -euo pipefail

BASE="${BASE_IMAGE:-manaca-train:latest}"
OUT="${OUT_IMAGE:-manaca-lmeval:latest}"
# Em hosts onde se ROTEIA ate o pypi (Fastly) mas nao se RESOLVE o nome, fixamos o IP anycast
# do Fastly no /etc/hosts do container p/ contornar o DNS. Se mudar, sobrescreva
# com PYPI_IP=<ip> (descubra com: python3 -c "import socket;print(socket.gethostbyname('pypi.org'))").
PYPI_IP="${PYPI_IP:-151.101.0.223}"
NAME="lmeval-build-$$"
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT

echo "[build] instalando lm-eval dentro de $BASE (IPv6 off + /etc/hosts p/ pypi=$PYPI_IP)..."
# bash -c (NAO -lc): o shell de login zerava PYPI_IP antes do echo no /etc/hosts.
# PIP_EXTRA_INDEX_URL vazio remove o indice pypi.ngc.nvidia.com (so resolve IPv6).
docker run --name "$NAME" --gpus all \
  --sysctl net.ipv6.conf.all.disable_ipv6=1 \
  --sysctl net.ipv6.conf.default.disable_ipv6=1 \
  -e PYPI_IP="$PYPI_IP" \
  -e PIP_EXTRA_INDEX_URL= \
  -e PIP_DISABLE_PIP_VERSION_CHECK=1 \
  "$BASE" \
  bash -c '
    set -e
    IP="${PYPI_IP:-151.101.0.223}"
    printf "%s pypi.org\n%s files.pythonhosted.org\n" "$IP" "$IP" >> /etc/hosts
    echo "[hosts] pypi.org -> $(python3 -c "import socket;print(socket.gethostbyname(\"pypi.org\"))")"
    # Mata o indice extra pypi.ngc.nvidia.com (so resolve IPv6 e trava o pip): vem
    # de pip.conf da imagem e/ou de variaveis de ambiente. Removemos os dois.
    find /etc /root /usr -name pip.conf -not -path "*/site-packages/*" -delete 2>/dev/null || true
    unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL
    for i in 1 2 3 4 5; do
      echo "== tentativa pip $i =="
      # --break-system-packages: instala no ambiente do container (PEP 668); container
      # descartavel que sera commitado, entao e o comportamento desejado.
      pip install --break-system-packages --index-url https://pypi.org/simple \
        --no-cache-dir --retries 20 --timeout 120 \
        lm-eval==0.4.5 transformers==4.46.3 datasets==3.1.0 && break
      echo "pip falhou; nova tentativa em 8s..."; sleep 8
    done
    # Verificacao: o que importa e o import funcionar (lm_eval nao expoe __version__).
    python -c "import lm_eval, transformers, torch, importlib.metadata as m; print(\"OK: lm_eval\", m.version(\"lm_eval\"), \"| transformers\", transformers.__version__, \"| torch\", torch.__version__)"
  '

echo "[build] commit -> $OUT"
docker commit "$NAME" "$OUT"
echo "[build] pronto: $OUT   (rode ./scripts/eval/run_lm_eval_pt.sh)"

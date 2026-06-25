#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-neuralrecon-sentinel}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINT_DIR="${REPO_ROOT}/third_party/reconstruction/NeuralRecon/checkpoints"
FILE_ID="${NEURALRECON_GDRIVE_ID:-1zKuWqm9weHSm98SZKld1PbEddgLOQkQV}"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

mkdir -p "${CHECKPOINT_DIR}"
cd "${CHECKPOINT_DIR}"

if find . -maxdepth 1 -type f -name '*.ckpt' | grep -q .; then
  echo "[checkpoint] existing checkpoint:"
  find . -maxdepth 1 -type f -name '*.ckpt' -printf '  %f\n'
  exit 0
fi

python -m pip install gdown
if gdown --help 2>&1 | grep -q -- "--id"; then
  gdown --id "${FILE_ID}"
else
  gdown "${FILE_ID}"
fi

echo "[checkpoint] downloaded files:"
find . -maxdepth 1 -type f -printf '  %f\n'

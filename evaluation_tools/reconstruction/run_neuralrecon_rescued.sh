#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-neuralrecon-sentinel}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${REPO_ROOT}/dataset/reconstruction/neuralrecon/2026-03-30-21-31-03_rescued_extfix"
CHECKPOINT_DIR="${REPO_ROOT}/third_party/reconstruction/NeuralRecon/checkpoints"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"
source "${REPO_ROOT}/evaluation_tools/reconstruction/neuralrecon_env.sh"

cd "${REPO_ROOT}"

python evaluation_tools/reconstruction/run_neuralrecon_demo.py \
  --data-dir "${DATA_DIR}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --n-views 9

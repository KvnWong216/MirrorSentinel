#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-neuralrecon-sentinel}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TORCHSPARSE_DIR="${REPO_ROOT}/third_party/reconstruction/torchsparse"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"
source "${REPO_ROOT}/evaluation_tools/reconstruction/neuralrecon_env.sh"

if [[ ! -d "${TORCHSPARSE_DIR}/.git" ]]; then
  git -c http.proxy= -c https.proxy= clone --depth 1 https://github.com/mit-han-lab/torchsparse.git "${TORCHSPARSE_DIR}"
fi

cd "${TORCHSPARSE_DIR}"
python setup.py build_ext --inplace -v

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
try:
    import torchsparse
    print("torchsparse", torchsparse.__file__)
except RuntimeError as exc:
    print("torchsparse import reached runtime init but CUDA is not available:", exc)
PY

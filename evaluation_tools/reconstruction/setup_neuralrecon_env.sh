#!/usr/bin/env bash
set -euo pipefail

# Create an isolated NeuralRecon environment for Sentinel-LIO reconstruction priors.
#
# Why not use NeuralRecon's original environment.yaml verbatim?
# - The official CVPR 2021 environment pins PyTorch 1.6 + CUDA 10.2.
# - The current workstation uses an RTX 4060 Laptop GPU, which is better served
#   by a newer CUDA/PyTorch stack.
# - NeuralRecon's critical dependency is torchsparse; this script installs a
#   conda-provided CUDA toolkit and then builds torchsparse inside the
#   environment.

ENV_NAME="${ENV_NAME:-neuralrecon-sentinel}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_VERSION="${TORCH_VERSION:-2.1.2}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.16.2}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.1.2}"
CUDA_VERSION="${CUDA_VERSION:-11.8}"
TORCH_WHEEL_INDEX="${TORCH_WHEEL_INDEX:-https://download.pytorch.org/whl/cu118}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NEURALRECON_ROOT="${REPO_ROOT}/third_party/reconstruction/NeuralRecon"

source ~/anaconda3/etc/profile.d/conda.sh

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[setup] conda env ${ENV_NAME} already exists"
else
  echo "[setup] creating conda env ${ENV_NAME}"
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

conda activate "${ENV_NAME}"

echo "[setup] installing conda CUDA toolkit ${CUDA_VERSION}"
conda install -y -c nvidia "cuda-toolkit=${CUDA_VERSION}"

echo "[setup] installing sparsehash headers from conda-forge"
conda install -y -c conda-forge sparsehash

export CUDA_HOME="${CONDA_PREFIX}"
export CUDACXX="${CONDA_PREFIX}/bin/nvcc"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/include:${CPLUS_INCLUDE_PATH:-}"
# The workstation uses an RTX 4060 Laptop GPU (Ada, sm_89). CUDA 11.8 knows
# this target and matches the cu118 PyTorch wheels installed below.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export FORCE_CUDA="${FORCE_CUDA:-1}"
export MAX_JOBS="${MAX_JOBS:-2}"

echo "[setup] installing Python deps"
python -m pip install --upgrade "pip<26" "setuptools<70" wheel
echo "[setup] installing PyTorch ${TORCH_VERSION} wheels from ${TORCH_WHEEL_INDEX}"
python -m pip uninstall -y torch torchvision torchaudio || true
conda remove -y --force pytorch torchvision torchaudio pytorch-cuda || true
python -m pip install \
  --index-url "${TORCH_WHEEL_INDEX}" \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}"

python -m pip install \
  "numpy<2" \
  ray \
  transforms3d \
  tqdm \
  "opencv-python==4.9.0.80" \
  numba \
  tensorboardX \
  scikit-image \
  "trimesh~=3.9.18" \
  yacs \
  h5py \
  loguru \
  gdown \
  "pyrender~=0.1.45" \
  pyglet \
  open3d

# pycuda is not needed for the NeuralRecon demo path we use.  It is omitted to
# avoid an extra CUDA toolchain failure point on systems without system nvcc.

echo "[setup] installing torchsparse"
TORCHSPARSE_DIR="${REPO_ROOT}/third_party/reconstruction/torchsparse"
if [[ ! -d "${TORCHSPARSE_DIR}/.git" ]]; then
  git -c http.proxy= -c https.proxy= clone --depth 1 https://github.com/mit-han-lab/torchsparse.git "${TORCHSPARSE_DIR}"
fi
cd "${TORCHSPARSE_DIR}"
rm -rf build
if [[ ! -f "${CUDA_HOME}/include/cuda_runtime.h" ]]; then
  echo "[setup] missing ${CUDA_HOME}/include/cuda_runtime.h after CUDA install" >&2
  exit 2
fi
python -m pip install --no-build-isolation -v .

echo "[setup] validating imports"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
import torchsparse
print("torchsparse", torchsparse.__file__)
import cv2, open3d, yacs
print("basic deps ok")
PY

echo "[setup] done"
echo "Activate with: conda activate ${ENV_NAME}"
echo "NeuralRecon root: ${NEURALRECON_ROOT}"

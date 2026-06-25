#!/usr/bin/env bash
# Shared runtime/build environment for the local NeuralRecon wrapper.
#
# Source this file after activating the conda environment. It exposes the
# in-tree TorchSparse checkout and, when the conda env only contains runtime
# CUDA libs, builds a lightweight CUDA_HOME under /tmp from conda's package
# cache so torch extensions can find CUDA headers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TORCHSPARSE_DIR="${REPO_ROOT}/third_party/reconstruction/torchsparse"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "[neuralrecon-env] CONDA_PREFIX is empty; activate neuralrecon-sentinel first" >&2
  return 2 2>/dev/null || exit 2
fi

if [[ -f "${CONDA_PREFIX}/include/cuda_runtime.h" && -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
  export CUDA_HOME="${CONDA_PREFIX}"
else
  CUDA_CACHE_ROOT="${CUDA_CACHE_ROOT:-/tmp/neuralrecon_cuda117_root}"
  CUDA_NVCC_PKG="${CUDA_NVCC_PKG:-${HOME}/anaconda3/pkgs/cuda-nvcc-11.7.99-0}"
  CUDA_CUDART_DEV_PKG="${CUDA_CUDART_DEV_PKG:-${HOME}/anaconda3/pkgs/cuda-cudart-dev-11.7.99-0}"
  CUDA_CCCL_PKG="${CUDA_CCCL_PKG:-${HOME}/anaconda3/pkgs/cuda-cccl-11.7.91-0}"
  CUDA_CUSPARSE_DEV_PKG="${CUDA_CUSPARSE_DEV_PKG:-${HOME}/anaconda3/pkgs/libcusparse-dev-11.7.4.91-0}"
  CUDA_CUBLAS_DEV_PKG="${CUDA_CUBLAS_DEV_PKG:-${HOME}/anaconda3/pkgs/libcublas-dev-11.10.3.66-0}"
  CUDA_CUSOLVER_DEV_PKG="${CUDA_CUSOLVER_DEV_PKG:-${HOME}/anaconda3/pkgs/libcusolver-dev-11.4.0.1-0}"
  CUDA_CURAND_DEV_PKG="${CUDA_CURAND_DEV_PKG:-${HOME}/anaconda3/pkgs/libcurand-dev-10.2.10.91-0}"
  CUDA_CUFFT_DEV_PKG="${CUDA_CUFFT_DEV_PKG:-${HOME}/anaconda3/pkgs/libcufft-dev-10.7.2.91-0}"

  for required in \
    "${CUDA_NVCC_PKG}/bin/nvcc" \
    "${CUDA_CUDART_DEV_PKG}/include/cuda_runtime.h" \
    "${CUDA_CCCL_PKG}/include/thrust" \
    "${CUDA_CUSPARSE_DEV_PKG}/include/cusparse.h" \
    "${CUDA_CUBLAS_DEV_PKG}/include/cublas_v2.h" \
    "${CUDA_CUSOLVER_DEV_PKG}/include/cusolverDn.h" \
    "${CUDA_CURAND_DEV_PKG}/include/curand.h" \
    "${CUDA_CUFFT_DEV_PKG}/include/cufft.h"; do
    if [[ ! -e "${required}" ]]; then
      echo "[neuralrecon-env] missing CUDA cache component: ${required}" >&2
      echo "[neuralrecon-env] install cuda-toolkit=11.8 in ${CONDA_PREFIX}, or run setup_neuralrecon_env.sh with network access." >&2
      return 2 2>/dev/null || exit 2
    fi
  done

  mkdir -p "${CUDA_CACHE_ROOT}/include" "${CUDA_CACHE_ROOT}/lib"
  if [[ ! -x "${CUDA_CACHE_ROOT}/bin/nvcc" ]]; then
    cp -a "${CUDA_NVCC_PKG}/bin" "${CUDA_CACHE_ROOT}/"
    cp -a "${CUDA_NVCC_PKG}/nvvm" "${CUDA_CACHE_ROOT}/"
  fi
  cp -a "${CUDA_NVCC_PKG}/include/." "${CUDA_CACHE_ROOT}/include/"
  cp -a "${CUDA_CUDART_DEV_PKG}/include/." "${CUDA_CACHE_ROOT}/include/"
  cp -a "${CUDA_CCCL_PKG}/include/." "${CUDA_CACHE_ROOT}/include/"
  cp -a "${CUDA_CUSPARSE_DEV_PKG}/include/." "${CUDA_CACHE_ROOT}/include/"
  cp -a "${CUDA_CUBLAS_DEV_PKG}/include/." "${CUDA_CACHE_ROOT}/include/"
  cp -a "${CUDA_CUSOLVER_DEV_PKG}/include/." "${CUDA_CACHE_ROOT}/include/"
  cp -a "${CUDA_CURAND_DEV_PKG}/include/." "${CUDA_CACHE_ROOT}/include/"
  cp -a "${CUDA_CUFFT_DEV_PKG}/include/." "${CUDA_CACHE_ROOT}/include/"
  cp -a "${CUDA_CUDART_DEV_PKG}/lib/." "${CUDA_CACHE_ROOT}/lib/"
  if [[ ! -e "${CUDA_CACHE_ROOT}/lib/libcudart.so.11.7.99" && -e "${CONDA_PREFIX}/lib/libcudart.so.11.7.99" ]]; then
    cp -a "${CONDA_PREFIX}/lib/libcudart.so.11.7.99" "${CUDA_CACHE_ROOT}/lib/"
  fi

  export CUDA_HOME="${CUDA_CACHE_ROOT}"
fi

export CUDA_PATH="${CUDA_HOME}"
export PYTORCH_NVCC="${CUDA_HOME}/bin/nvcc"
export PATH="${CUDA_HOME}/bin:${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export CPLUS_INCLUDE_PATH="${CUDA_HOME}/include:${CONDA_PREFIX}/include:${CPLUS_INCLUDE_PATH:-}"
export CPATH="${CUDA_HOME}/include:${CONDA_PREFIX}/include:${CPATH:-}"

# cuda-nvcc 11.7 does not know Ada sm_89. If a full CUDA 11.8 toolkit is in the
# env, setup_neuralrecon_env.sh overrides this to sm_89; the cache fallback uses
# Ampere SASS + PTX so RTX 4060 can JIT kernels.
if "${CUDA_HOME}/bin/nvcc" --version 2>/dev/null | grep -q "release 11.7"; then
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6+PTX}"
else
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
fi
export FORCE_CUDA="${FORCE_CUDA:-1}"
export MAX_JOBS="${MAX_JOBS:-2}"
export PYTHONPATH="${TORCHSPARSE_DIR}:${PYTHONPATH:-}"

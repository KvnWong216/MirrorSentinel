#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-MirrSLAM_rtdepth}"
REPO_URL="${REPO_URL:-https://github.com/ByteDance-Seed/Depth-Anything-3.git}"
THIRD_PARTY_DIR="${THIRD_PARTY_DIR:-$(pwd)/third_party}"
DA3_DIR="${DA3_DIR:-${THIRD_PARTY_DIR}/Depth-Anything-3}"
DA3_SOURCE_DIR="${DA3_SOURCE_DIR:-}"

if [[ ! -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  echo "Cannot find conda.sh under ${HOME}/anaconda3/etc/profile.d/conda.sh" >&2
  exit 2
fi

source "${HOME}/anaconda3/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"
python -m pip install -U pip setuptools wheel

mkdir -p "${THIRD_PARTY_DIR}"
if [[ -n "${DA3_SOURCE_DIR}" ]]; then
  if [[ ! -d "${DA3_SOURCE_DIR}" ]]; then
    echo "DA3_SOURCE_DIR does not exist: ${DA3_SOURCE_DIR}" >&2
    exit 2
  fi
  rm -rf "${DA3_DIR}"
  cp -a "${DA3_SOURCE_DIR}" "${DA3_DIR}"
elif [[ ! -d "${DA3_DIR}/.git" ]]; then
  git -c http.proxy= -c https.proxy= clone "${REPO_URL}" "${DA3_DIR}"
else
  git -C "${DA3_DIR}" -c http.proxy= -c https.proxy= pull --ff-only || true
fi

python -m pip install -e "${DA3_DIR}"

cat <<EOF

Depth Anything 3 environment is ready.

Use it through the existing ROS2 runner, for example:

  SENTINEL_LIO_CONDA_ENV=${ENV_NAME} python3 evaluation_tools/run_ros2_slam_eval.py \\
    --bag dataset/rosbag2/2026-03-30-21-31-03 \\
    --sequence 2026-03-30-21-31-03_fullbag_da3_depth_only \\
    --method sentinel_rt_depth \\
    --annotation evaluation_tools/data/annotations/self_collected/2026-03-30-21-31-03_rescued.yaml \\
    --play-rate 1.0 --startup-wait 5 --shutdown-wait 10 --timeout 240 \\
    --run-da3-depth \\
    --da3-backend module \\
    --da3-model depth_anything_3.api:DepthAnything3 \\
    --da3-checkpoint depth-anything/DA3Metric-Large \\
    --frame-prior-post-clean

The first DA3 run may download model weights from HuggingFace.
EOF

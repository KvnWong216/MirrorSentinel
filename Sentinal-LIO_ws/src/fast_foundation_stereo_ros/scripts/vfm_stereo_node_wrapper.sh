#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f /opt/ros/humble/setup.bash ]]; then
  source /opt/ros/humble/setup.bash
fi

if [[ -n "${SENTINEL_LIO_CONDA_SH:-}" && -f "${SENTINEL_LIO_CONDA_SH}" ]]; then
  source "${SENTINEL_LIO_CONDA_SH}"
  conda activate "${SENTINEL_LIO_CONDA_ENV:-Sentinel-LIO}"
fi

TARGET_SCRIPT="vfm_stereo_node.py"
if [[ $# -gt 0 && "${1}" == *.py ]]; then
  TARGET_SCRIPT="${1}"
  shift
fi

exec python3 "${SCRIPT_DIR}/${TARGET_SCRIPT}" "$@"

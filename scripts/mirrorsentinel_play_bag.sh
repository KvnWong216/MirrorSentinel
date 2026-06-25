#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
DEFAULT_BAG="${REPO_ROOT}/dataset/rosbag2/2026-03-30-21-31-03_rescued"

BAG="${DEFAULT_BAG}"
if [[ $# -gt 0 && "${1}" != -* ]]; then
  BAG="${1}"
  shift
fi

set +u
source /opt/ros/humble/setup.bash
set -u
exec ros2 bag play "${BAG}" --clock --read-ahead-queue-size 100 "$@"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
WS_ROOT="${REPO_ROOT}/Sentinal-LIO_ws"

cd "${WS_ROOT}"
set +u
source /opt/ros/humble/setup.bash
set -u

if [[ ! -f install/setup.bash ]]; then
  echo "Missing ${WS_ROOT}/install/setup.bash. Build first:" >&2
  echo "  cd ${WS_ROOT} && source /opt/ros/humble/setup.bash && colcon build --symlink-install" >&2
  exit 2
fi

set +u
source install/setup.bash
set -u
exec ros2 launch sentinel_lio run_sentinel_ouster.launch.py rviz:=true "$@"

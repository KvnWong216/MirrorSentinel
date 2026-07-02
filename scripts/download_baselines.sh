#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE_DIR="${BASELINE_DIR:-${ROOT_DIR}/third_party/baselines}"
GIT_BASE=(git -c http.proxy= -c https.proxy=)

mkdir -p "${BASELINE_DIR}"

clone_or_update() {
  local name="$1"
  local url="$2"
  local dst="${BASELINE_DIR}/${name}"

  if [[ -d "${dst}/.git" ]]; then
    echo "[baselines] updating ${name}"
    "${GIT_BASE[@]}" -C "${dst}" pull --ff-only
  else
    echo "[baselines] cloning ${name}"
    "${GIT_BASE[@]}" clone --depth 1 "${url}" "${dst}"
  fi
}

clone_or_update "LVI-SAM" "https://github.com/TixiaoShan/LVI-SAM.git"
clone_or_update "Reflection_detection" "https://github.com/STAR-Center/Reflection_detection.git"
clone_or_update "FAST-LIVO2-ROS2" "https://github.com/v4rl-ucy/FAST-LIVO2-ROS2.git"
clone_or_update "GRASS" "https://github.com/wpshao/GRASS.git"

cat <<EOF

[baselines] done

Local paths:
  ${BASELINE_DIR}/LVI-SAM
  ${BASELINE_DIR}/Reflection_detection
  ${BASELINE_DIR}/FAST-LIVO2-ROS2
  ${BASELINE_DIR}/GRASS

Note:
  LVI-SAM and Reflection_detection are ROS1/catkin reference repositories.
  FAST-LIVO2-ROS2 is the preferred ROS2 Humble LIV baseline candidate, but keep
  it outside the MirrorSentinel workspace until its external dependencies and
  sensor config are installed intentionally. GRASS currently provides paper/data
  pointers rather than a ready-to-run ROS2 baseline.
EOF

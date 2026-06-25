#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -x "${SCRIPT_DIR}/build/pcl_room_annotator" ]]; then
  "${SCRIPT_DIR}/build.sh"
fi

# PCL/VTK on Ubuntu links against system Qt/OpenGL. Conda and ROS library paths
# can shadow those libraries at runtime, so launch the viewer in a clean process.
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL PYTHONPATH
export LD_LIBRARY_PATH="${PCL_ROOM_ANNOTATOR_LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-0}"

exec "${SCRIPT_DIR}/build/pcl_room_annotator" "$@"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CMAKE_PREFIX_PATH="/usr:${CMAKE_PREFIX_PATH:-}"
cmake -S "${SCRIPT_DIR}" -B "${SCRIPT_DIR}/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBoost_NO_BOOST_CMAKE=ON \
  -DBOOST_ROOT=/usr
cmake --build "${SCRIPT_DIR}/build" -j"$(nproc)"

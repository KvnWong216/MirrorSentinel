#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/home/wangyg}"
LIMIT="${LIMIT:-200}"

echo "[1/3] ROS2 bags containing /vfm priors under ${ROOT}"
find "${ROOT}" -path '*/.git' -prune -o -path '*/anaconda3' -prune -o -path '*/.cache' -prune -o \
  -name metadata.yaml -print 2>/dev/null \
  | while read -r meta; do
      if grep -qE '/vfm/(depth_image|mirror_mask)' "${meta}"; then
        dirname "${meta}"
      fi
    done \
  | sort -u \
  | sed -n "1,${LIMIT}p"

echo
echo "[2/3] Candidate DA3 / depth / mirror / reflection / VFM model assets"
find "${ROOT}" \
  -path '*/.git' -prune -o \
  -path '*/anaconda3' -prune -o \
  -path '*/.cache' -prune -o \
  -path '*/.local' -prune -o \
  -path '*/snap' -prune -o \
  -type f \( -name '*.engine' -o -name '*.onnx' -o -name '*.pth' -o -name '*.pt' -o -name '*.ckpt' \) -print 2>/dev/null \
  | grep -Ei 'da3|depth-anything|depth|mirror|reflect|seg|vfm|gemdepth|videodepth' \
  | sed -n "1,${LIMIT}p" \
  || true

echo
echo "[3/3] DA3 / depth / reflection directory hints"
find "${ROOT}" \
  -path '*/.git' -prune -o \
  -path '*/anaconda3' -prune -o \
  -path '*/.cache' -prune -o \
  -type d \
  | awk -F/ '{name=tolower($NF); if (name ~ /(depth-anything|^da3$|^vfm$|reflect|reconstruction|mast3r|slam3r|neuralrecon|gemdepth|videodepth)/) print $0}' \
  | sed -n "1,${LIMIT}p" \
  || true

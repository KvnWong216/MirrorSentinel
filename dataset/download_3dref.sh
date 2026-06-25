#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/dataset/public/3DRef"
ARCHIVE_DIR="${DATA_DIR}/archives"

REFLECTION_URL="https://robotics.shanghaitech.edu.cn/static/datasets/3DRef/reflection.tar.bz2"
SEQ1_URL="https://robotics.shanghaitech.edu.cn/static/datasets/3DRef/seq1.tar.bz2"
SEQ2_URL="https://robotics.shanghaitech.edu.cn/static/datasets/3DRef/seq2.tar.bz2"
SEQ3_URL="https://robotics.shanghaitech.edu.cn/static/datasets/3DRef/seq3.tar.bz2"

mkdir -p "${ARCHIVE_DIR}"

download_file() {
  local url="$1"
  local out="$2"
  echo "[3DRef] downloading ${url}"
  echo "[3DRef] output: ${out}"
  wget -c --show-progress -O "${out}" "${url}"
}

extract_file() {
  local archive="$1"
  local outdir="$2"
  if [[ ! -f "${archive}" ]]; then
    echo "[3DRef] missing archive: ${archive}" >&2
    exit 1
  fi
  mkdir -p "${outdir}"
  echo "[3DRef] extracting ${archive} -> ${outdir}"
  tar -xjf "${archive}" -C "${outdir}"
}

check_layout() {
  echo "[3DRef] data dir: ${DATA_DIR}"
  du -sh "${DATA_DIR}" 2>/dev/null || true
  find "${DATA_DIR}" -maxdepth 4 -type d | sort | sed -n '1,120p'
  echo
  echo "[3DRef] archives:"
  find "${ARCHIVE_DIR}" -maxdepth 1 -type f -printf '%p %s bytes\n' | sort || true
}

case "${1:-help}" in
  reflection)
    download_file "${REFLECTION_URL}" "${ARCHIVE_DIR}/reflection.tar.bz2"
    ;;
  seq1)
    download_file "${SEQ1_URL}" "${ARCHIVE_DIR}/seq1.tar.bz2"
    ;;
  seq2)
    download_file "${SEQ2_URL}" "${ARCHIVE_DIR}/seq2.tar.bz2"
    ;;
  seq3)
    download_file "${SEQ3_URL}" "${ARCHIVE_DIR}/seq3.tar.bz2"
    ;;
  extract-reflection)
    extract_file "${ARCHIVE_DIR}/reflection.tar.bz2" "${DATA_DIR}/reflection"
    ;;
  extract-seq1)
    extract_file "${ARCHIVE_DIR}/seq1.tar.bz2" "${DATA_DIR}/raw"
    ;;
  extract-seq2)
    extract_file "${ARCHIVE_DIR}/seq2.tar.bz2" "${DATA_DIR}/raw"
    ;;
  extract-seq3)
    extract_file "${ARCHIVE_DIR}/seq3.tar.bz2" "${DATA_DIR}/raw"
    ;;
  check)
    check_layout
    ;;
  help|*)
    cat <<EOF
Usage:
  bash dataset/download_3dref.sh reflection          # download benchmark RGB/network/scripts package
  bash dataset/download_3dref.sh extract-reflection  # extract reflection.tar.bz2
  bash dataset/download_3dref.sh check               # inspect local layout

Optional raw sequences, for later:
  bash dataset/download_3dref.sh seq1
  bash dataset/download_3dref.sh seq2
  bash dataset/download_3dref.sh seq3
EOF
    ;;
esac

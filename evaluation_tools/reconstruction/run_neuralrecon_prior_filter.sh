#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-neuralrecon-sentinel}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MAP_PCD="${MAP_PCD:-${REPO_ROOT}/evaluation_tools/results/slam/self_collected/2026-03-30-21-31-03_fastlio2_equiv_v3_sparse/fast_lio2_equiv/map.pcd}"
PRIOR_NPZ="${PRIOR_NPZ:-${REPO_ROOT}/third_party/reconstruction/NeuralRecon/results/scene_demo_checkpoints_fusion_eval_47/2026-03-30-21-31-03_rescued_extfix.npz}"
PRIOR_MESH="${PRIOR_MESH:-${REPO_ROOT}/third_party/reconstruction/NeuralRecon/results/scene_demo_checkpoints_fusion_eval_47/2026-03-30-21-31-03_rescued_extfix.ply}"
ANNOTATION="${ANNOTATION:-${REPO_ROOT}/evaluation_tools/data/annotations/self_collected/2026-03-30-21-31-03_rescued.yaml}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/evaluation_tools/results/reconstruction_prior/2026-03-30-21-31-03_fastlio2_neuralrecon_prior}"

source ~/anaconda3/etc/profile.d/conda.sh
conda activate "${ENV_NAME}"

cd "${REPO_ROOT}"

python evaluation_tools/reconstruction/inspect_neuralrecon_result.py \
  --neuralrecon-npz "${PRIOR_NPZ}" \
  --mesh "${PRIOR_MESH}" \
  --map "${MAP_PCD}" \
  --output-json "${OUT_DIR}/inspect_summary.json"

python evaluation_tools/reconstruction/apply_neuralrecon_prior_to_map.py \
  --map "${MAP_PCD}" \
  --neuralrecon-npz "${PRIOR_NPZ}" \
  --mesh "${PRIOR_MESH}" \
  --annotation "${ANNOTATION}" \
  --out-dir "${OUT_DIR}" \
  --coverage-mode aabb \
  --reject-distance 0.35 \
  --soft-distance 0.15 \
  --aabb-padding 0.05

python evaluation_tools/map_ghost_eval.py \
  --map "${MAP_PCD}" \
  --annotation "${ANNOTATION}" \
  --output-json "${OUT_DIR}/metrics_before.json" \
  --output-csv "${OUT_DIR}/metrics_before.csv"

python evaluation_tools/map_ghost_eval.py \
  --map "${OUT_DIR}/filtered_map.pcd" \
  --annotation "${ANNOTATION}" \
  --output-json "${OUT_DIR}/metrics_after.json" \
  --output-csv "${OUT_DIR}/metrics_after.csv"

python evaluation_tools/reconstruction/export_prior_filter_quicklook.py \
  --filtered-map "${OUT_DIR}/filtered_map.pcd" \
  --rejected "${OUT_DIR}/rejected_points.pcd" \
  --mesh "${PRIOR_MESH}" \
  --output "${OUT_DIR}/quicklook_prior_filter.ply"

echo "NeuralRecon prior filter outputs:"
echo "  ${OUT_DIR}/filtered_map.pcd"
echo "  ${OUT_DIR}/rejected_points.pcd"
echo "  ${OUT_DIR}/prior_candidate_points.pcd"
echo "  ${OUT_DIR}/quicklook_prior_filter.ply"
echo "  ${OUT_DIR}/summary.json"
echo "  ${OUT_DIR}/metrics_before.json"
echo "  ${OUT_DIR}/metrics_after.json"

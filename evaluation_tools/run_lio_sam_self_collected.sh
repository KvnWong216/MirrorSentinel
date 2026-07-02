#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIO_SAM_WS="${REPO_ROOT}/baselines/lio_sam_ws"
GTSAM_PREFIX="${LIO_SAM_WS}/local_deps/opt/ros/humble"
BAG_DIR="${1:-${REPO_ROOT}/dataset/rosbag2/2026-03-30-21-31-03}"
RUN_NAME="${2:-2026-03-30-21-31-03_lio_sam}"
PLAY_RATE="${3:-1.0}"
RESULT_DIR="${REPO_ROOT}/evaluation_tools/results/slam/self_collected/${RUN_NAME}/lio_sam"
OUT_BAG="${RESULT_DIR}/outputs_bag"
QOS_FILE="${RESULT_DIR}/lio_sam_record_qos.yaml"
SAVE_DIR_REL="/yugong_slam_lio_sam_export/${RUN_NAME}"
SAVE_DIR="${HOME}${SAVE_DIR_REL}"
MAP_PATH="${SAVE_DIR}/GlobalMap.pcd"
FINAL_MAP="${RESULT_DIR}/map.pcd"
PARAMS_FILE="${REPO_ROOT}/evaluation_tools/configs/lio_sam_self_collected.yaml"
ANNOTATION="${REPO_ROOT}/evaluation_tools/data/annotations/self_collected/2026-03-30-21-31-03_rescued.yaml"
REFERENCE_MAP="${REPO_ROOT}/evaluation_tools/data/gt_reference/self_collected/2026-03-30-21-31-03_rescued/room_gt_reference.pcd"

mkdir -p "${RESULT_DIR}"
rm -rf "${OUT_BAG}" "${SAVE_DIR}"

cat >"${QOS_FILE}" <<'YAML'
/lio_sam/mapping/odometry:
  reliability: best_effort
  durability: volatile
  history: keep_last
  depth: 50
/lio_sam/mapping/odometry_incremental:
  reliability: best_effort
  durability: volatile
  history: keep_last
  depth: 50
/lio_sam/mapping/cloud_registered:
  reliability: best_effort
  durability: volatile
  history: keep_last
  depth: 10
/lio_sam/mapping/cloud_registered_raw:
  reliability: best_effort
  durability: volatile
  history: keep_last
  depth: 10
/lio_sam/mapping/map_global:
  reliability: best_effort
  durability: volatile
  history: keep_last
  depth: 10
/lio_sam/mapping/path:
  reliability: best_effort
  durability: volatile
  history: keep_last
  depth: 10
YAML

# Keep ROS2 runtime on system libraries. Anaconda in LD_LIBRARY_PATH can shadow
# Boost/TBB/Qt and break LIO-SAM at runtime.
export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES-}"
export AMENT_PYTHON_EXECUTABLE=/usr/bin/python3
export COLCON_PYTHON_EXECUTABLE=/usr/bin/python3
export CMAKE_PREFIX_PATH="${GTSAM_PREFIX}:/opt/ros/humble"
export PYTHONPATH=""
export LD_LIBRARY_PATH=""

set +u
source /opt/ros/humble/setup.bash
source "${LIO_SAM_WS}/install/setup.bash"
set -u

export PYTHONUNBUFFERED=1
export ROS_LOG_DIR="${RESULT_DIR}/ros_logs"
export LD_LIBRARY_PATH="${GTSAM_PREFIX}/lib/x86_64-linux-gnu:${LIO_SAM_WS}/install/lio_sam/lib:/opt/ros/humble/lib:/usr/lib/x86_64-linux-gnu"

cleanup() {
  set +e
  [[ -n "${PLAY_PID:-}" ]] && kill -INT "-${PLAY_PID}" 2>/dev/null || true
  [[ -n "${FIRST_CLOUD_PID:-}" ]] && kill -INT "${FIRST_CLOUD_PID}" 2>/dev/null || true
  [[ -n "${RECORD_PID:-}" ]] && kill -INT "-${RECORD_PID}" 2>/dev/null || true
  [[ -n "${LAUNCH_PID:-}" ]] && kill -INT "-${LAUNCH_PID}" 2>/dev/null || true
  wait "${PLAY_PID:-}" 2>/dev/null || true
  wait "${FIRST_CLOUD_PID:-}" 2>/dev/null || true
  wait "${RECORD_PID:-}" 2>/dev/null || true
  wait "${LAUNCH_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT

setsid ros2 launch "${REPO_ROOT}/evaluation_tools/launch/lio_sam_self_collected.launch.py" \
  params_file:="${PARAMS_FILE}" \
  use_sim_time:=true \
  >"${RESULT_DIR}/lio_sam_launch.log" 2>&1 &
LAUNCH_PID=$!

sleep 6

setsid ros2 bag record -o "${OUT_BAG}" \
  --qos-profile-overrides-path "${QOS_FILE}" \
  /lio_sam/mapping/odometry \
  /lio_sam/mapping/odometry_incremental \
  /lio_sam/mapping/cloud_registered \
  /lio_sam/mapping/cloud_registered_raw \
  /lio_sam/mapping/map_global \
  /lio_sam/mapping/path \
  >"${RESULT_DIR}/record.log" 2>&1 &
RECORD_PID=$!

sleep 2

timeout 120s ros2 topic echo --once /lio_sam/mapping/cloud_registered \
  --qos-reliability best_effort \
  --qos-durability volatile \
  --no-arr \
  --no-str \
  >"${RESULT_DIR}/first_registered_cloud.log" 2>&1 &
FIRST_CLOUD_PID=$!

setsid ros2 bag play "${BAG_DIR}" \
  --clock \
  --rate "${PLAY_RATE}" \
  --read-ahead-queue-size 50 \
  --disable-keyboard-controls \
  --topics /ouster/points /ouster/imu \
  >"${RESULT_DIR}/bag_play.log" 2>&1 &
PLAY_PID=$!

wait "${PLAY_PID}"
sleep 8

if ! wait "${FIRST_CLOUD_PID}" 2>/dev/null; then
  kill -INT "-${RECORD_PID}" 2>/dev/null || true
  wait "${RECORD_PID}" 2>/dev/null || true
  echo "LIO-SAM did not publish /lio_sam/mapping/cloud_registered; skip save_map to avoid empty-map crash." >&2
  echo "See logs in ${RESULT_DIR}" >&2
  exit 3
fi

timeout 60s ros2 service call /lio_sam/save_map lio_sam/srv/SaveMap \
  "{resolution: 0.2, destination: '${SAVE_DIR_REL}'}" \
  >"${RESULT_DIR}/save_map.log" 2>&1 || true
sleep 2

kill -INT "-${RECORD_PID}" 2>/dev/null || true
wait "${RECORD_PID}" 2>/dev/null || true

if [[ ! -s "${MAP_PATH}" ]]; then
  echo "LIO-SAM did not export a non-empty map: ${MAP_PATH}" >&2
  echo "See logs in ${RESULT_DIR}" >&2
  exit 2
fi

cp "${MAP_PATH}" "${FINAL_MAP}"

/usr/bin/python3 "${REPO_ROOT}/evaluation_tools/map_ghost_eval.py" \
  --map "${FINAL_MAP}" \
  --annotation "${ANNOTATION}" \
  --output-json "${RESULT_DIR}/metrics_mapping.json" \
  --output-csv "${RESULT_DIR}/metrics_mapping.csv"

if [[ -f "${REFERENCE_MAP}" ]]; then
  /usr/bin/python3 "${REPO_ROOT}/evaluation_tools/reference_map_eval.py" \
    --map "${FINAL_MAP}" \
    --reference "${REFERENCE_MAP}" \
    --output-json "${RESULT_DIR}/metrics_reference_map.json" \
    --output-csv "${RESULT_DIR}/metrics_reference_map.csv"
fi

/usr/bin/python3 "${REPO_ROOT}/evaluation_tools/ros2_bag_topic_rate.py" \
  "${OUT_BAG}" \
  --topic /lio_sam/mapping/cloud_registered \
  --requested-play-rate "${PLAY_RATE}" \
  --output-json "${RESULT_DIR}/metrics_runtime.json" \
  --output-csv "${RESULT_DIR}/metrics_runtime.csv"

echo "LIO-SAM run finished:"
echo "  result dir: ${RESULT_DIR}"
echo "  map: ${FINAL_MAP}"
echo "  exported map dir: ${SAVE_DIR}"
echo "  output bag: ${OUT_BAG}"

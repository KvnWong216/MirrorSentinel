#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RTABMAP_WS="${REPO_ROOT}/baselines/rtabmap_ws"
BAG_DIR="${1:-${REPO_ROOT}/dataset/rosbag2/2026-03-30-21-31-03}"
RUN_NAME="${2:-2026-03-30-21-31-03_rtabmap_min}"
RESULT_DIR="${REPO_ROOT}/evaluation_tools/results/slam/self_collected/${RUN_NAME}/rtabmap_min"
CALIB_DIR="${RESULT_DIR}/calib"
OUT_BAG="${RESULT_DIR}/outputs_bag"
DB_PATH="${RESULT_DIR}/rtabmap.db"
ANNOTATION="${REPO_ROOT}/evaluation_tools/data/annotations/self_collected/2026-03-30-21-31-03_rescued.yaml"
REFERENCE_MAP="${REPO_ROOT}/evaluation_tools/data/gt_reference/self_collected/2026-03-30-21-31-03_rescued/room_gt_reference.pcd"

mkdir -p "${RESULT_DIR}"
rm -rf "${OUT_BAG}"

/usr/bin/python3 "${REPO_ROOT}/evaluation_tools/rtabmap_make_zed_stereo_calib.py" \
  --out-dir "${CALIB_DIR}" >/dev/null

export AMENT_TRACE_SETUP_FILES="${AMENT_TRACE_SETUP_FILES-}"
export AMENT_PYTHON_EXECUTABLE="${AMENT_PYTHON_EXECUTABLE-/usr/bin/python3}"
export COLCON_TRACE="${COLCON_TRACE-}"
export COLCON_PREFIX_PATH="${COLCON_PREFIX_PATH-}"
export COLCON_PYTHON_EXECUTABLE="${COLCON_PYTHON_EXECUTABLE-/usr/bin/python3}"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH-}"
export PYTHONPATH="${PYTHONPATH-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH-}"
set +u
source /opt/ros/humble/setup.bash
source "${RTABMAP_WS}/install/setup.bash"
set -u

export PYTHONUNBUFFERED=1
export ROS_LOG_DIR="${RESULT_DIR}/ros_logs"
export LD_LIBRARY_PATH="${RTABMAP_WS}/compat_libs:${LD_LIBRARY_PATH-}"

cleanup() {
  set +e
  [[ -n "${PLAY_PID:-}" ]] && kill -INT "-${PLAY_PID}" 2>/dev/null || true
  [[ -n "${RECORD_PID:-}" ]] && kill -INT "-${RECORD_PID}" 2>/dev/null || true
  [[ -n "${LAUNCH_PID:-}" ]] && kill -INT "-${LAUNCH_PID}" 2>/dev/null || true
  wait "${PLAY_PID:-}" 2>/dev/null || true
  wait "${RECORD_PID:-}" 2>/dev/null || true
  wait "${LAUNCH_PID:-}" 2>/dev/null || true
}
trap cleanup EXIT

setsid ros2 launch "${REPO_ROOT}/evaluation_tools/launch/rtabmap_self_collected_min.launch.py" \
  database_path:="${DB_PATH}" \
  use_sim_time:=true \
  >"${RESULT_DIR}/rtabmap_launch.log" 2>&1 &
LAUNCH_PID=$!

sleep 8

setsid ros2 bag record -o "${OUT_BAG}" \
  /mapData \
  /mapGraph \
  /cloud_map \
  /map \
  /rtabmap/info \
  /zed2/zed_node/odom \
  >"${RESULT_DIR}/record.log" 2>&1 &
RECORD_PID=$!

sleep 3

PLAY_CMD=(ros2 bag play "${BAG_DIR}" --clock --read-ahead-queue-size 100)
setsid "${PLAY_CMD[@]}" >"${RESULT_DIR}/bag_play.log" 2>&1 &
PLAY_PID=$!

wait "${PLAY_PID}"
sleep 8

timeout 20s ros2 service call /rtabmap/publish_map rtabmap_msgs/srv/PublishMap \
  "{global_map: true, optimized: true, graph_only: false}" \
  >"${RESULT_DIR}/publish_map.log" 2>&1 || true
sleep 5

kill -INT "-${RECORD_PID}" 2>/dev/null || true
wait "${RECORD_PID}" 2>/dev/null || true

/usr/bin/python3 "${REPO_ROOT}/evaluation_tools/ros2_bag_pointcloud_to_pcd.py" \
  "${OUT_BAG}" \
  --topic /cloud_map \
  --out "${RESULT_DIR}/map.pcd"

/usr/bin/python3 "${REPO_ROOT}/evaluation_tools/map_ghost_eval.py" \
  --map "${RESULT_DIR}/map.pcd" \
  --annotation "${ANNOTATION}" \
  --output-json "${RESULT_DIR}/metrics_mapping.json" \
  --output-csv "${RESULT_DIR}/metrics_mapping.csv"

if [[ -f "${REFERENCE_MAP}" ]]; then
  /usr/bin/python3 "${REPO_ROOT}/evaluation_tools/reference_map_eval.py" \
    --map "${RESULT_DIR}/map.pcd" \
    --reference "${REFERENCE_MAP}" \
    --output-json "${RESULT_DIR}/metrics_reference_map.json" \
    --output-csv "${RESULT_DIR}/metrics_reference_map.csv"
fi

/usr/bin/python3 "${REPO_ROOT}/evaluation_tools/ros2_bag_topic_rate.py" \
  "${OUT_BAG}" \
  --topic /cloud_map \
  --output-json "${RESULT_DIR}/metrics_runtime.json" \
  --output-csv "${RESULT_DIR}/metrics_runtime.csv"

echo "RTAB-Map run finished:"
echo "  result dir: ${RESULT_DIR}"
echo "  map: ${RESULT_DIR}/map.pcd"
echo "  output bag: ${OUT_BAG}"

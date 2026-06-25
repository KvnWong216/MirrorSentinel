# MirrorSentinel

MirrorSentinel is the minimal ROS2 Humble system for the YuGong mirror/glass
SLAM work.  Its purpose is not mirror segmentation by itself; the target task is
to keep LiDAR-inertial mapping stable while suppressing mirror/glass ghost map
points.

## Minimal Core

The current minimum viable version has two parts:

- `Sentinal-LIO_ws/src/sentinel_lio`: ROS2 LiDAR-inertial mapping, RViz
  visualization, and online visual/depth prior consumption.
- `evaluation_tools/mirrorsentinel_run_pipeline.py`: bag replay, output
  recording, automatic reflective-plane detection, historical prior voting, and
  map-level evaluation.

The paper-facing map outputs are:

```text
map_raw.pcd
map_marker_clean.pcd
map_vote_clean.pcd
metrics_mapping_ablation.{json,csv,md}
```

`map_vote_clean.pcd` is the current MirrorSentinel backend result.  The backend
keeps a 100k-level map by default and removes points that are both behind
reflective geometry and contradicted by historical depth/mask prior frames.

## Build

```bash
git clone <your-mirrorsentinel-repo-url>
cd MirrorSentinel
scripts/build_ros2.sh
```

## Live RViz Bag View

Terminal 1:

```bash
cd MirrorSentinel
scripts/mirrorsentinel_rviz.sh
```

Terminal 2:

```bash
cd MirrorSentinel
scripts/mirrorsentinel_play_bag.sh
```

The RViz panel shows `/cloud_registered`, `/Odometry`, `/path`, TF, and optional
MirrorSentinel prior/debug topics such as `/vfm/depth_image`,
`/vfm/mirror_mask`, and `/mirror_sentinel/ghost_points`.

## Main Experiment

Run the current MVP pipeline on the rescued elevator bag:

```bash
cd MirrorSentinel
python3 evaluation_tools/mirrorsentinel_run_pipeline.py
```

Open RViz during the run:

```bash
python3 evaluation_tools/mirrorsentinel_run_pipeline.py --rviz
```

The wrapper delegates to `evaluation_tools/run_ros2_slam_eval.py` with the
current MirrorSentinel backend settings.  The longer runner remains available
for ablations and baselines.

## User-Facing Tool Names

Use the `mirrorsentinel_*` entry points for new experiments:

```text
evaluation_tools/mirrorsentinel_run_pipeline.py
evaluation_tools/mirrorsentinel_detect_planes.py
evaluation_tools/mirrorsentinel_clean_map.py
evaluation_tools/mirrorsentinel_replay_votes.py
```

The older implementation files are still present for compatibility with
previous experiment logs:

```text
evaluation_tools/detect_reflective_planes.py
evaluation_tools/filter_map_by_frame_prior_votes.py
evaluation_tools/apply_frame_prior_vote_threshold.py
evaluation_tools/run_ros2_slam_eval.py
```

Automatic reflective-plane detection does not use an elevator/room absolute
size prior by default.  It uses vertical wall-line support, local odometry
coverage, and compactness after coverage saturation.

## More Docs

```text
SLAM_EVAL_PROTOCOL_CN.md
evaluation_tools/README.md
evaluation_tools/FRAME_PRIOR_BACKEND_CN.md
Sentinal-LIO_ws/src/sentinel_lio/README.md
```

## Data and Large Dependencies

The GitHub repository intentionally does not vendor local bags, benchmark
results, checkpoints, or large third-party reconstruction/baseline repositories.
Use the dataset and reconstruction docs when those experiments are needed:

```text
dataset/README_DATASETS_CN.md
evaluation_tools/reconstruction/README.md
```

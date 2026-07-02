# MirrorSentinel

MirrorSentinel is a ROS 2 Humble LiDAR-inertial mapping system for mirror and
glass scenes. Its target is the SLAM map itself: suppress reflected or
behind-wall ghost points caused by non-Lambertian surfaces while preserving a
usable map and online playback throughput.

This repository contains the minimal research code for the current paper
prototype:

- a FAST-LIO-style ROS 2 mapper;
- a MirrorSentinel backend for visual-prior point gating and map cleanup;
- a monocular Depth Anything 3 compatible depth-prior node;
- optional mirror/glass mask-prior nodes;
- RViz visualization for online bag replay;
- evaluation tools for reflection residuals, wall thickness, map retention,
  reference-map F-score, and runtime FPS.

The current formal visual upstream is **DA3 / Depth Anything 3 monocular
depth**, not a stereo-specific frontend. The `/vfm/*` topic namespace is kept
only as a generic "visual foundation model" interface for compatibility.

## Repository Layout

```text
Sentinal-LIO_ws/src/sentinel_lio/
  ROS 2 mapping package, MirrorSentinel backend, launch files, RViz config.

Sentinal-LIO_ws/src/mirrorsentinel_visual_prior/
  DA3 depth-prior node, optional reflection mask node, and mask-prior utilities.

evaluation_tools/
  Bag replay, map export, reflection/GT evaluation, baseline adapters,
  annotation tools, and result summarization scripts.

evaluation_tools/data/annotations/
  Lightweight room-boundary annotations for mirror/glass map evaluation.

evaluation_tools/data/gt_reference/
  A small manually annotated cuboid reference map for the current self-collected
  elevator sequence.

scripts/
  Build, RViz, bag playback, DA3 environment, and baseline download helpers.

docs/
  Research review, baseline notes, draft material, and team handoff notes.

dataset/
  Dataset notes and download helpers. Large bags are intentionally not tracked.
```

Local-only directories such as `third_party/`, `baselines/`, ROS build folders,
model checkpoints, generated maps, and bag files are excluded from Git.

## Requirements

Tested target environment:

```text
Ubuntu 22.04
ROS 2 Humble
Python 3.10 through /usr/bin/python3 for ROS tooling
PCL, OpenCV, Eigen, Ceres, and tf2 from apt/ROS packages
CUDA GPU recommended for DA3 online depth inference
```

Install common ROS dependencies:

```bash
sudo apt update
sudo apt install -y \
  ros-humble-desktop \
  ros-humble-pcl-ros \
  ros-humble-pcl-conversions \
  ros-humble-tf2-eigen \
  ros-humble-cv-bridge \
  ros-humble-image-transport \
  ros-humble-rviz2 \
  python3-colcon-common-extensions \
  python3-rosdep \
  libpcl-dev \
  libeigen3-dev \
  libceres-dev
```

The ROS 2 build should use the system Python. The helper build script clears
active conda variables before invoking `colcon`.

## Build

```bash
git clone https://github.com/KvnWong216/MirrorSentinel.git
cd MirrorSentinel
bash scripts/build_ros2.sh
```

This builds:

```text
sentinel_lio
mirrorsentinel_visual_prior
```

On resource-constrained machines:

```bash
export MAKEFLAGS=-j2
export CMAKE_BUILD_PARALLEL_LEVEL=2
bash scripts/build_ros2.sh
```

## Data And Large Files

Large runtime assets are not committed:

```text
dataset/rosbag2/
dataset/public/
evaluation_tools/results/
third_party/
baselines/
Sentinal-LIO_ws/models/
*.bag, *.db3, *.mcap, *.pcd, *.pth, *.pt, *.onnx, *.engine
```

To reproduce the current self-collected experiment, place the converted ROS 2
bag at:

```text
dataset/rosbag2/2026-03-30-21-31-03
```

See [DATASET_AND_LARGE_FILES.md](DATASET_AND_LARGE_FILES.md) and
[dataset/README_DATASETS_CN.md](dataset/README_DATASETS_CN.md) for the local
data policy and dataset notes.

## Quick RViz Playback

Terminal 1:

```bash
cd MirrorSentinel
bash scripts/mirrorsentinel_rviz.sh
```

Terminal 2:

```bash
cd MirrorSentinel
bash scripts/mirrorsentinel_play_bag.sh dataset/rosbag2/2026-03-30-21-31-03
```

The RViz panel visualizes the online mapping stream:

```text
/cloud_registered
/Odometry
/path
/tf
/mirror_sentinel/ghost_points          optional deleted/marked points
/vfm/depth_image                       optional visual depth prior
/vfm/mirror_mask                       optional mirror/glass prior
```

## Depth Anything 3 Environment

The DA3 code and weights are not vendored. Prepare a separate conda environment:

```bash
cd MirrorSentinel
bash scripts/setup_depth_anything3_env.sh
```

By default this creates `MirrSLAM_rtdepth` and installs the official
Depth-Anything-3 repository under `third_party/Depth-Anything-3`. The first run
may download model weights from HuggingFace.

If you already have a DA3 checkout:

```bash
DA3_SOURCE_DIR=/path/to/Depth-Anything-3 bash scripts/setup_depth_anything3_env.sh
```

## Run MirrorSentinel With DA3

The lower-level runner launches SLAM, optionally launches DA3, plays the bag,
records output topics, exports maps, and runs evaluation:

```bash
HF_ENDPOINT=https://huggingface.co \
SENTINEL_LIO_CONDA_ENV=MirrSLAM_rtdepth \
python3 evaluation_tools/run_ros2_slam_eval.py \
  --bag dataset/rosbag2/2026-03-30-21-31-03 \
  --sequence 2026-03-30-21-31-03_fullbag_da3_depth_only \
  --method sentinel_rt_depth \
  --annotation evaluation_tools/data/annotations/self_collected/2026-03-30-21-31-03_rescued.yaml \
  --play-rate 1.0 \
  --startup-wait 5 \
  --shutdown-wait 10 \
  --timeout 240 \
  --run-da3-depth \
  --da3-backend module \
  --da3-model depth_anything_3.api:DepthAnything3 \
  --da3-checkpoint depth-anything/DA3Metric-Large \
  --frame-prior-post-clean
```

Main outputs are written under:

```text
evaluation_tools/results/slam/self_collected/<sequence>/<method>/
```

Paper-facing map outputs:

```text
map_raw.pcd
map_marker_clean.pcd
map_vote_clean.pcd
metrics_mapping_ablation.json
metrics_mapping_ablation.csv
metrics_runtime.json
metrics_reference_map_raw.json
metrics_reference_map_vote_clean.json
```

`map_raw.pcd` is the uncleaned accumulated map. `map_marker_clean.pcd` removes
points marked as ghost candidates. `map_vote_clean.pcd` applies historical
frame-prior voting and is the current main MirrorSentinel output.

## Run With A Precomputed Prior Bag

If `/vfm/depth_image` and `/vfm/mirror_mask` have already been recorded in a
separate ROS 2 bag, replay that prior bag instead of running DA3 online:

```bash
python3 evaluation_tools/run_ros2_slam_eval.py \
  --bag dataset/rosbag2/2026-03-30-21-31-03 \
  --prior-bag /path/to/prior_outputs_bag \
  --sequence 2026-03-30-21-31-03_priorbag \
  --method sentinel_full \
  --annotation evaluation_tools/data/annotations/self_collected/2026-03-30-21-31-03_rescued.yaml \
  --play-rate 1.0 \
  --timeout 240 \
  --frame-prior-post-clean
```

Do not combine `--prior-bag` with `--run-da3-depth`,
`--run-reflection-mask`, or `--run-mask-prior`.

## Evaluation

Evaluate reflection residuals and wall thickness for a saved map:

```bash
python3 evaluation_tools/map_ghost_eval.py \
  --map evaluation_tools/results/slam/self_collected/<sequence>/<method>/map_vote_clean.pcd \
  --annotation evaluation_tools/data/annotations/self_collected/2026-03-30-21-31-03_rescued.yaml \
  --output-json evaluation_tools/results/slam/self_collected/<sequence>/<method>/metrics_mapping_vote_clean.json \
  --output-csv evaluation_tools/results/slam/self_collected/<sequence>/<method>/metrics_mapping_vote_clean.csv
```

Evaluate against the current manual cuboid reference map:

```bash
python3 evaluation_tools/reference_map_eval.py \
  --map evaluation_tools/results/slam/self_collected/<sequence>/<method>/map_vote_clean.pcd \
  --reference evaluation_tools/data/gt_reference/self_collected/2026-03-30-21-31-03_rescued/room_gt_reference.pcd \
  --output-json evaluation_tools/results/slam/self_collected/<sequence>/<method>/metrics_reference_map_vote_clean.json \
  --output-csv evaluation_tools/results/slam/self_collected/<sequence>/<method>/metrics_reference_map_vote_clean.csv
```

Refresh the method summary table:

```bash
python3 evaluation_tools/summarize_self_collected_methods.py
```

Summary outputs:

```text
evaluation_tools/results/slam/self_collected/all_method_eval_summary.md
evaluation_tools/results/slam/self_collected/all_method_eval_summary.csv
```

## Annotation Tools

For elevator-like scenes where the four side walls are mirror/glass, annotate
the true room cuboid rather than drawing semantic image ROIs. The evaluator
derives the reflective wall planes from the cuboid boundary.

Use the lightweight Python annotator:

```bash
python3 evaluation_tools/room_gt_annotator.py \
  --output-dir evaluation_tools/data/gt_reference/self_collected/<sequence>
```

Or inspect and render an existing annotation:

```bash
python3 evaluation_tools/annotation_helper.py inspect \
  --map evaluation_tools/results/slam/self_collected/<sequence>/<method>/map_raw.pcd \
  --annotation evaluation_tools/data/annotations/self_collected/2026-03-30-21-31-03_rescued.yaml \
  --output-dir evaluation_tools/results/annotation_preview/<sequence>
```

A PCL/VTK annotator is also included:

```bash
evaluation_tools/pcl_room_annotator/build.sh
evaluation_tools/pcl_room_annotator/run.sh --map /path/to/map.pcd --output-yaml /path/to/annotation.yaml
```

On some Wayland/Xwayland setups, the PCL window may crash inside the VTK event
loop. Use the Python annotator in that case.

## Method Summary

For every LiDAR point in a frame, MirrorSentinel projects the point into the
camera view and compares its LiDAR depth with a visual foundation-model depth
prior. Points that are much farther than the visual prior, especially near
reflective regions or reflective wall geometry, are treated as ghost-point
candidates. Instead of relying on a single-frame hard deletion, the backend
keeps frame-level candidate markers and performs history-vote cleanup on the
accumulated map. This produces three maps for ablation:

```text
raw map              baseline accumulated map before cleanup
marker-clean map     direct removal of marked ghost candidates
vote-clean map       history-vote cleanup, current main output
```

The system is framed as a SLAM reconstruction problem, not as a pure image
segmentation task. Reflection masks, if used, are priors that constrain where
depth inconsistency should be trusted.

## Baselines

The current internal comparison uses:

```text
FAST-LIO2-equivalent raw mapping
FAST-LIVO2 ROS 2 raw export
RTAB-Map ICP mapping
LIO-SAM recovered historical output
Mapping with Reflection adapted baseline
Manual geometry oracle upper bound
MirrorSentinel DA3 depth-only variants
```

Optional external baselines can be downloaded locally:

```bash
bash scripts/download_baselines.sh
```

Downloaded third-party repositories and generated baseline maps should remain
outside Git.

## Current Internal Snapshot

On the self-collected elevator mirror/glass sequence
`2026-03-30-21-31-03`, the latest internal evaluation indicates that
MirrorSentinel DA3 depth-only vote-clean reduces reflection residual and wall
thickness relative to FAST-LIO2-equivalent, LIO-SAM, RTAB-Map, FAST-LIVO2 raw
export, and the adapted geometry baseline, while keeping a 100k-point map
budget. The manual cuboid geometry oracle is treated only as an upper bound,
not as an online method.

The exact generated result tables are local experiment artifacts and are not
tracked in this repository. Reproduce them by restoring the bag under
`dataset/rosbag2/` and running the commands above.

## Useful Documents

Recommended reading order for teammates:

```text
docs/TEAM_HANDOFF_CN.md
docs/ROBIO_PAPER_DRAFT_CN.md
SLAM_EVAL_PROTOCOL_CN.md
docs/REFLECTION_SLAM_BASELINES_CN.md
docs/MIRROR_SLAM_RESEARCH_REVIEW_CN.md
```

Large-file policy:

```text
DATASET_AND_LARGE_FILES.md
```

ROS 2 package details:

```text
Sentinal-LIO_ws/ROS2_PORT.md
Sentinal-LIO_ws/src/sentinel_lio/README.md
```

## Citation

This repository is an active research prototype. If you use it before a formal
paper release, please cite the repository and contact the authors for the most
recent experiment protocol.

## License

The MirrorSentinel research code in this repository is intended for academic
research use. The `sentinel_lio` package is derived from FAST-LIO-style open
source components; please check package-level license files and upstream
licenses before redistribution or commercial use.

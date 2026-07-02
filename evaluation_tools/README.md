# Sentinel-LIO Evaluation Tools

This directory contains the evaluation harness for the ROBIO mirror/glass SLAM work.

The current protocol is documented in:

```text
SLAM_EVAL_PROTOCOL_CN.md
evaluation_tools/configs/benchmark_plan.yaml
```

The key separation is:

1. Main SLAM task: ROS2 bag replay, online mapping, trajectory/map/runtime metrics.
2. Upstream prior task: depth/reflection prior accuracy on 3DRef, reported as an auxiliary experiment only.

## 1. Main Mirror/Glass SLAM Evaluation

Main experiments must run on ROS2 Humble rosbag2 data with synchronized LiDAR, IMU, and image streams.

Prototype bag currently available:

```bash
source /opt/ros/humble/setup.bash
ros2 bag info dataset/rosbag2/2026-03-30-21-31-03_rescued
```

Expected main outputs per method:

```text
evaluation_tools/results/slam/self_collected/<sequence>/<method>/
  trajectory_tum.txt
  map.pcd
  frame_stats.csv
  run_metadata.yaml
  metrics_mapping.json
  metrics_no_gt.json
```

Primary main-task metrics:

```text
Reflection Removal Rate or Removal vs Raw
Reflection Residual Rate
Valid Structure Precision / Indoor Precision
Plane Thickness P95
Map Retention Rate
Pipeline FPS / Hz
```

These metrics are defined in `SLAM_EVAL_PROTOCOL_CN.md`.  The mirror/glass
metrics are aligned with 3DRef / reflection LiDAR SLAM reflection-removal
protocols and with SLAM map / 3D reconstruction map-quality metrics.  They are
not reported as a pure segmentation score; IoU/F1 on 3DRef are auxiliary prior
metrics only.

## 2. Trajectory Evaluation

Use trajectory metrics only when a GT or defensible reference trajectory exists.

TUM-style ATE/RPE:

```bash
python3 evaluation_tools/tum_trajectory_eval.py all \
  --groundtruth evaluation_tools/data/gt_tum.txt \
  --estimate evaluation_tools/results/slam/self_collected/<sequence>/<method>/trajectory_tum.txt \
  --max-difference 0.02 \
  --output-dir evaluation_tools/results/slam/self_collected/<sequence>/<method>/tum_eval
```

No-GT consistency proxy for self-collected loop/revisit sequences:

```bash
python3 evaluation_tools/no_gt_trajectory_eval.py \
  --trajectory evaluation_tools/results/slam/self_collected/<sequence>/<method>/trajectory_tum.txt \
  --revisit-pairs evaluation_tools/data/self_collected/revisit_pairs.txt \
  --output-json evaluation_tools/results/slam/self_collected/<sequence>/<method>/metrics_no_gt.json \
  --output-csv evaluation_tools/results/slam/self_collected/<sequence>/<method>/metrics_no_gt.csv
```

No-GT results must be described as loop/revisit consistency, not as ground-truth ATE.

Export a ROS2-recorded trajectory topic to TUM format:

```bash
python3 evaluation_tools/ros2_bag_to_tum.py \
  dataset/rosbag2/2026-03-30-21-31-03_rescued \
  --topic /zed2/zed_node/odom \
  --out evaluation_tools/results/slam/self_collected/2026-03-30-21-31-03_rescued/zed_odom_tum.txt
```

For Sentinel-LIO runs, use the recorded `/Odometry` topic instead.

## 3. Map Reflection-Removal/Thickness Evaluation

After a method saves `map.pcd`, evaluate reflective-plane ghost points and local map thickness:

```bash
python3 evaluation_tools/map_ghost_eval.py \
  --map evaluation_tools/results/slam/self_collected/<sequence>/<method>/map.pcd \
  --annotation evaluation_tools/data/annotations/self_collected/<sequence>.yaml \
  --output-json evaluation_tools/results/slam/self_collected/<sequence>/<method>/metrics_mapping.json \
  --output-csv evaluation_tools/results/slam/self_collected/<sequence>/<method>/metrics_mapping.csv
```

The annotation YAML must define `reflective_planes` and optional `planar_regions`.
See `SLAM_EVAL_PROTOCOL_CN.md` and the placeholder annotation under
`evaluation_tools/data/annotations/self_collected/`.

The evaluator keeps legacy fields such as `ghost_rate` for older logs, but new
paper-facing JSON fields are:

```text
reflection_residual_points
reflection_residual_rate
reflection_residual_density_m3
valid_structure_precision_proxy
reflective_plane_thickness_p95_m
```

When raw/marker-clean/vote-clean maps are evaluated together, the runner also
writes relative ablation fields:

```text
reflection_removal_rate_vs_raw
map_retention_rate_vs_raw
thickness_reduction_rate_vs_raw
```

For elevator-like scenes where nearly all walls are mirror/glass, annotate the
room cuboid boundary instead of semantic mirror ROIs. Pick the four floor
corners of the elevator room and set the vertical extent. The YAML stores
`room_bounds`; the evaluator derives the four mirror walls from those true
boundary lines. Any wall-thickness slab used for ghost counting is generated
internally and is not the manual annotation.

The robust annotation path uses the matplotlib helper and writes only the true
room cuboid:

```bash
python3 evaluation_tools/annotation_helper.py room-bounds \
  --map evaluation_tools/results/slam/self_collected/2026-03-30-21-31-03_fastlio2_equiv_v3_sparse/fast_lio2_equiv/map.pcd \
  --z-min -0.3 --z-max 2.2 \
  --sequence 2026-03-30-21-31-03_rescued \
  --bag dataset/rosbag2/2026-03-30-21-31-03_rescued \
  --map-frame camera_init \
  --output-yaml evaluation_tools/data/annotations/self_collected/2026-03-30-21-31-03_rescued.yaml \
  --output-dir evaluation_tools/results/annotation_preview/2026-03-30-21-31-03_fastlio2/room_bounds_manual \
  --crop-min -6 -3.5 -0.5 \
  --crop-max 4.5 3.5 2.5
```

For reproducibility, pass the four XY corners directly:

```bash
python3 evaluation_tools/annotation_helper.py room-bounds \
  --map evaluation_tools/results/slam/self_collected/2026-03-30-21-31-03_fastlio2_equiv_v3_sparse/fast_lio2_equiv/map.pcd \
  --z-min -0.3 --z-max 2.2 \
  --sequence 2026-03-30-21-31-03_rescued \
  --bag dataset/rosbag2/2026-03-30-21-31-03_rescued \
  --map-frame camera_init \
  --output-yaml evaluation_tools/data/annotations/self_collected/2026-03-30-21-31-03_rescued.yaml \
  --output-dir evaluation_tools/results/annotation_preview/2026-03-30-21-31-03_fastlio2/room_bounds_manual \
  --corners 3.715335 0.388164 1.360079 0.574105 1.205127 -0.820455 3.653355 -1.083872
```

There is also a PCL cuboid annotator. Its non-interactive YAML writer is useful
for checking the same `room_bounds` schema. On some Ubuntu Wayland/Xwayland
setups, the interactive PCL/VTK window can segfault inside the X11 event loop;
use the `room-bounds` helper above if that happens.

```bash
evaluation_tools/pcl_room_annotator/build.sh

evaluation_tools/pcl_room_annotator/run.sh \
  --map evaluation_tools/results/slam/self_collected/2026-03-30-21-31-03_fastlio2_equiv_v3_sparse/fast_lio2_equiv/map.pcd \
  --z-min -0.3 --z-max 2.2 \
  --sequence 2026-03-30-21-31-03_rescued \
  --bag dataset/rosbag2/2026-03-30-21-31-03_rescued \
  --map-frame camera_init \
  --output-yaml evaluation_tools/data/annotations/self_collected/2026-03-30-21-31-03_rescued.yaml
```

PCL interaction:

```text
Shift + left click: pick one floor corner in the point cloud
u: undo the last corner
r: reset all corners
[: lower z_min, ]: raise z_min
;: lower z_max, ': raise z_max
s: save YAML after four corners are selected
q or Esc: quit
```

After saving the YAML, render the true wall boundary lines and inspect the
derived evaluation crops before using the metrics:

```bash
python3 evaluation_tools/annotation_helper.py render \
  --map evaluation_tools/results/slam/self_collected/2026-03-30-21-31-03_fastlio2_equiv_v3_sparse/fast_lio2_equiv/map.pcd \
  --annotation evaluation_tools/data/annotations/self_collected/2026-03-30-21-31-03_rescued.yaml \
  --output-dir evaluation_tools/results/annotation_preview/2026-03-30-21-31-03_fastlio2/room_bounds_manual_render \
  --prefix room_bounds \
  --crop-min -6 -3.5 -0.5 \
  --crop-max 4.5 3.5 2.5 \
  --max-points 120000
```

```bash
python3 evaluation_tools/annotation_helper.py inspect \
  --map evaluation_tools/results/slam/self_collected/2026-03-30-21-31-03_fastlio2_equiv_v3_sparse/fast_lio2_equiv/map.pcd \
  --annotation evaluation_tools/data/annotations/self_collected/2026-03-30-21-31-03_rescued.yaml \
  --output-dir evaluation_tools/results/annotation_preview/2026-03-30-21-31-03_fastlio2/room_bounds_manual_inspect
```

### Manual Cuboid Reference Map

If a room is close to a cuboid, create a stronger geometric reference map from
the same four floor corners and `z_min/z_max`.  This is useful for local
MapEval/ETH3D-style map-quality evidence, but it must be described as a manual
geometric reference, not scanner-grade GT.

Interactive mode:

```bash
python3 evaluation_tools/room_gt_annotator.py \
  --map evaluation_tools/results/slam/self_collected/<sequence>/<method>/map_raw.pcd \
  --z-min -0.3 --z-max 2.2 \
  --sequence <sequence> \
  --bag dataset/rosbag2/<sequence> \
  --map-frame camera_init \
  --spacing 0.03 \
  --include-floor-ceiling \
  --output-dir evaluation_tools/data/gt_reference/self_collected/<sequence>
```

Reproducible non-interactive mode:

```bash
python3 evaluation_tools/room_gt_annotator.py \
  --corners 3.715335 0.388164 1.360079 0.574105 1.205127 -0.820455 3.653355 -1.083872 \
  --z-min -0.3 --z-max 2.2 \
  --sequence 2026-03-30-21-31-03_rescued \
  --bag dataset/rosbag2/2026-03-30-21-31-03_rescued \
  --map-frame camera_init \
  --spacing 0.03 \
  --include-floor-ceiling \
  --output-dir evaluation_tools/data/gt_reference/self_collected/2026-03-30-21-31-03_rescued
```

Outputs:

```text
room_gt_annotation.yaml      # usable by map_ghost_eval.py
room_gt_reference.pcd        # dense six-face reference cloud
room_gt_mesh.ply             # cuboid mesh
room_gt_face_labels.npy      # face id per reference point
gt_top_view.png / gt_xz_view.png
```

If a high-end scanner reference is available, prefer it for global
accuracy/completeness.  The manual cuboid reference is the lightweight fallback
for mirror elevators and rectangular rooms.

Evaluate a SLAM map against the reference cloud:

```bash
python3 evaluation_tools/reference_map_eval.py \
  --map evaluation_tools/results/slam/self_collected/<sequence>/<method>/map_vote_clean.pcd \
  --reference evaluation_tools/data/gt_reference/self_collected/<sequence>/room_gt_reference.pcd \
  --thresholds-m 0.05,0.10,0.20 \
  --output-json evaluation_tools/results/slam/self_collected/<sequence>/<method>/metrics_reference_map.json \
  --output-csv evaluation_tools/results/slam/self_collected/<sequence>/<method>/metrics_reference_map.csv
```

This reports map-to-reference precision/accuracy, reference-to-map
completeness/recall, and F-score at each threshold.

## 4. One-Command ROS2 Bag Pipeline

The preferred MirrorSentinel MVP entry point is:

```bash
cd /home/wangyg/yugong_slam
python3 evaluation_tools/mirrorsentinel_run_pipeline.py
```

Add `--rviz` if you want to watch the live map while the bag is being replayed:

```bash
python3 evaluation_tools/mirrorsentinel_run_pipeline.py --rviz
```

This wrapper runs Sentinel-LIO on one ROS2 bag, records outputs, collects
`map_raw.pcd` / `map_marker_clean.pcd` / `map_vote_clean.pcd`, exports
`/Odometry` to TUM, and runs the available map/trajectory evaluators.
It also writes `metrics_runtime.json` with `/cloud_registered`, `/Odometry`,
`/vfm/depth_image`, and `/vfm/mirror_mask` FPS measured from the recorded output
bag.

The lower-level runner is still available for baselines and ablations:

```bash
python3 evaluation_tools/run_ros2_slam_eval.py \
  --bag dataset/rosbag2/2026-03-30-21-31-03_rescued \
  --sequence 2026-03-30-21-31-03_rescued \
  --method sentinel_full \
  --annotation evaluation_tools/data/annotations/self_collected/2026-03-30-21-31-03_rescued.yaml \
  --timeout 180
```

Use `--method fast_lio2_equiv`, `sentinel_no_mask`, `sentinel_no_depth`, or
`sentinel_hard_reject` for ablations on the same bag.

## 5. 3DRef-Style Point Filtering Metrics

For official 3DRef-style point labels, use segmentation IoU/F1 only for the
upstream prior:

```bash
python3 evaluation_tools/mirror_segmentation_eval.py semantickitti \
  --gt-dir dataset/public/3DRef/labels \
  --pred-dir evaluation_tools/results/3dref/predictions/labels \
  --positive-labels 2,3,4,5 \
  --output-json evaluation_tools/results/3dref/eval/prior_semantic.json
```

For the SLAM/map-cleaning question, evaluate whether the kept map points removed
3DRef label `5` reflection/virtual points while retaining indoor structure:

```bash
python3 evaluation_tools/reflection_removal_eval.py \
  --labels dataset/public/3DRef/labels/000000.label \
  --kept evaluation_tools/results/3dref/kept_indices/000000.npy \
  --output-json evaluation_tools/results/3dref/eval/removal_000000.json \
  --output-csv evaluation_tools/results/3dref/eval/removal_000000.csv
```

Default labels follow 3DRef:

```text
reflection / virtual points: 5
indoor retained structure: 1,2,3,4,6
```

The current recommended MirrorSentinel backend is geometry-constrained
historical-prior map reintegration. It exports the three paper ablation maps:

```text
map_raw.pcd
map_marker_clean.pcd
map_vote_clean.pcd
```

Expanded command for the current self-collected elevator sequence with the saved
depth/mask prior bag and a 100k-point vote-clean map:

```bash
python3 evaluation_tools/run_ros2_slam_eval.py \
  --bag dataset/rosbag2/2026-03-30-21-31-03_rescued \
  --prior-bag evaluation_tools/results/slam/self_collected/2026-03-30-21-31-03_sentinel_full_calib_oracle_maponly_v5_calib50_qos/sentinel_full/outputs_bag \
  --sequence 2026-03-30-21-31-03_rescued_priorbag_annotation_backend_100k \
  --method sentinel_full \
  --params-base /tmp/sentinel_history_marker_dense_r008_rate05.yaml \
  --annotation evaluation_tools/data/annotations/self_collected/2026-03-30-21-31-03_rescued.yaml \
  --play-rate 0.5 \
  --shutdown-wait 20 \
  --timeout 240 \
  --frame-prior-post-clean \
  --auto-reflective-planes \
  --auto-plane-z-min -0.3 \
  --auto-plane-z-max 2.2 \
  --auto-plane-crop-radius 2.3 \
  --auto-plane-compactness-weight 180000 \
  --frame-prior-frame-stride 5 \
  --frame-prior-max-pair-dt 0.30 \
  --frame-prior-selection-mode annotation_score_budget \
  --frame-prior-target-kept-points 100000 \
  --frame-prior-score-mode ghost_votes_ratio \
  --frame-prior-annotation-candidate-mode behind \
  --frame-prior-annotation-budget-weight candidate_count \
  --frame-prior-min-observations 1 \
  --frame-prior-min-ghost-votes 1 \
  --frame-prior-min-ghost-ratio 0.0
```

Automatic reflective-plane detection is scale-free by default with respect to
room/elevator size: no absolute maximum room side length is passed unless
`--auto-plane-max-side-length` is explicitly set to a positive value.

See `FRAME_PRIOR_BACKEND_CN.md` for the current method notes, related-work
framing, and the latest ablation numbers.

## 5. ROS2 Bag Rescue

The old self-collected `.bag` had a damaged rosbag1 index. It was converted without ROS1 by sequentially scanning readable chunks:

```bash
python3 evaluation_tools/rescue_rosbag1_to_ros2.py \
  --src dataset/rosbag/2026-03-30-21-31-03.bag \
  --dst dataset/rosbag2/2026-03-30-21-31-03_rescued \
  --overwrite
```

This path is for data rescue only. New paper sequences should be recorded directly as ROS2 bags.

## 6. Upstream Reflection Prior Evaluation

3DRef is an auxiliary perception benchmark. It evaluates mirror/glass/reflection prior quality; it does not evaluate the online SLAM system.

Image-mask evaluation:

```bash
python3 evaluation_tools/mirror_segmentation_eval.py image \
  --gt-dir dataset/public/3DRef/reflection/rgb/alllabel/test/mask \
  --pred-dir evaluation_tools/results/3dref/predictions/alllabel \
  --pred-threshold 127 \
  --output-json evaluation_tools/results/3dref/eval/alllabel.json \
  --output-csv evaluation_tools/results/3dref/eval/alllabel.csv
```

Batch 3DRef-style table:

```bash
python3 evaluation_tools/batch_reflection_eval.py \
  --config evaluation_tools/configs/benchmark_plan.yaml
```

Official baseline wrappers currently present:

```text
evaluation_tools/baselines/run_satnet_3dref_infer.py
evaluation_tools/baselines/run_eblnet_3dref_infer.py
evaluation_tools/summarize_3dref_baselines.py
```

## 7. Current Tool Coverage

Implemented:

```text
tum_trajectory_eval.py
no_gt_trajectory_eval.py
mirror_segmentation_eval.py
batch_tum_eval.py
batch_reflection_eval.py
rescue_rosbag1_to_ros2.py
ros2_bag_to_tum.py
map_ghost_eval.py
run_ros2_slam_eval.py
mirrorsentinel_run_pipeline.py
mirrorsentinel_detect_planes.py
mirrorsentinel_clean_map.py
mirrorsentinel_replay_votes.py
```

Still needed for the main protocol:

```text
batch_slam_table.py     aggregate per-method metrics into paper tables
```

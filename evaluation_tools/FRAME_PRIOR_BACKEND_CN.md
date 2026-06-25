# MirrorSentinel Map Backend Notes

本文档记录当前 MirrorSentinel 镜面/玻璃 ghost 点清理后端的定位、参考工作脉络和复现实验命令。

## 1. 参考工作怎么处理类似问题

相关工作大致分成两条线。

第一条是重建融合线：不要相信单帧深度或单帧 mask，而是把多帧 depth/range evidence 融合进地图。

- Curless & Levoy 的 volumetric/TSDF fusion 用累计加权 signed distance 集成多张 range image，强调 incremental update、方向不确定性和 outlier robustness。参考：https://graphics.stanford.edu/papers/volrange/
- BundleFusion 在在线重建中维护历史帧，并在全局位姿优化后做 surface reintegration。对我们的启发是：地图不是一次性生成后不动，而是应该能用历史观测重新整理。参考：https://graphics.stanford.edu/projects/bundlefusion/
- RoutedFusion 在 real-time depth map fusion 中用 learned routing/fusion 处理 noisy/outlier depth，尤其关注传统 fusion 的 thickening artifacts。对我们的启发是：上游深度先验会有噪声，后端需要按置信度和多帧一致性更新，而不是硬阈值一刀切。参考：https://arxiv.org/abs/2001.04388

第二条是反射/镜面 LiDAR 线：通常先建立或检测反射平面，再根据几何一致性区分真实点、穿透点和镜像 ghost 点。

- Zhao 等的 Mapping with Reflection 通过 intensity、dual return、plane fitting、boundary 等线索检测玻璃/镜面反射，并把反射点按平面镜像关系分类。参考：https://arxiv.org/abs/1909.12483
- Detection and Utilization of Reflections in LiDAR Scans 扩展为全局优化 reflective plane map，然后在地图级分类 LiDAR readings。参考：https://www.mdpi.com/1424-8220/24/15/4794
- Virtual point removal / glass reflection artifact suppression 这类工作通常遵循“玻璃/镜面平面提取 + virtual point discrimination”的两阶段范式。参考：https://pubmed.ncbi.nlm.nih.gov/31398108/ 和 https://www.mdpi.com/2072-4292/18/2/332

因此，我们现在的算法定位应写成：

> 用上游视觉/深度先验提供候选表面深度，在 SLAM 后端对累计地图点做历史帧重投影一致性投票；在已知或可检测的反射平面约束下，优先清理平面后方且被多帧 depth prior 证伪的 ghost map points。

这不是纯 CV segmentation。mask/depth 是辅助观测，主任务指标仍然是 SLAM map 中镜面墙后的 ghost 点数、ghost rate 和 reflective-plane thickness。

## 2. 当前后端模块

新的用户可见入口：

```text
evaluation_tools/mirrorsentinel_run_pipeline.py
evaluation_tools/mirrorsentinel_detect_planes.py
evaluation_tools/mirrorsentinel_clean_map.py
evaluation_tools/mirrorsentinel_replay_votes.py
```

底层兼容实现：

```text
evaluation_tools/run_ros2_slam_eval.py
evaluation_tools/detect_reflective_planes.py
evaluation_tools/filter_map_by_frame_prior_votes.py
evaluation_tools/apply_frame_prior_vote_threshold.py
```

后续新实验优先使用 `mirrorsentinel_*` 名称；旧脚本保留是为了不破坏已经跑完的实验日志和命令。

核心输入：

- `map_raw.pcd`：逐帧累计 raw map。
- `outputs_bag`：包含 `/Odometry`、`/vfm/depth_image`、`/vfm/mirror_mask`。
- `params.yaml`：相机内参和 LiDAR-camera 外参。
- `annotation.yaml`：当前自采电梯用 room cuboid boundary，四面墙为 mirror。

核心输出：

- `map_raw.pcd`
- `map_marker_clean.pcd`
- `map_vote_clean.pcd`
- `frame_prior_post_clean/point_vote_stats.npz`
- `metrics_mapping_ablation.{json,csv,md}`

新增选择模式：

```text
threshold
score_budget
annotation_score_budget
```

其中 `annotation_score_budget` 是当前主推模式：

1. 先找 annotated reflective planes 后方的候选点。
2. 候选点必须被历史帧 depth/mask prior 投票支持，即不是凭 annotation 直接删所有墙后点。
3. 优先删除这些“反射平面后方 + 历史深度证伪”的点。
4. 若仍未达到点数预算，用全局 score-budget 填充，清掉地图中其他高置信离群候选。

这版保留 10 万级地图点，符合用户当前目标。

## 3. 当前主结果

结果目录：

```text
evaluation_tools/results/slam/self_collected/2026-03-30-21-31-03_rescued_priorbag_history_marker_dense_r008_rate05_wait20/sentinel_full
```

当前 manual-geometry ablation：

| label | points | ROI | behind | ghost_rate | thickness_p95_m |
|---|---:|---:|---:|---:|---:|
| raw | 110566 | 5628 | 3229 | 0.573738 | 0.961600 |
| marker_clean | 100804 | 5102 | 3021 | 0.592121 | 0.969458 |
| vote_clean | 100000 | 3697 | 1298 | 0.351095 | 0.621725 |

相对 `raw`：

- behind-plane ghost points: `3229 -> 1298`
- ghost_rate: `0.573738 -> 0.351095`
- reflective thickness p95: `0.961600 m -> 0.621725 m`
- map size: `110566 -> 100000`

相对旧的 global score-budget：

- old vote_clean: behind `2772`, ghost_rate `0.536067`, p95 `0.757093 m`
- new annotation_score_budget: behind `1298`, ghost_rate `0.351095`, p95 `0.621725 m`

自动反射平面检测版已经接入，用户入口是：

```text
evaluation_tools/mirrorsentinel_detect_planes.py
```

当前第一版默认不使用电梯/房间绝对尺度先验。它在累计地图中检测竖直墙线，沿 odometry 局部窗口搜索 cuboid，并用三类通用准则选房间边界：

- wall-line support：墙线点数、长度、高度覆盖和残差。
- local odometry coverage：候选 cuboid 应解释局部机器人可行空间。
- compactness：在 odometry 覆盖饱和后，优先选择更紧的边界，避免把走廊/外部 ghost 区域包进房间。

实现上，`--max-side-length <= 0` 表示不启用绝对房间边长上限；只有显式传正数时才作为调试约束。

自动平面用于清理，最终仍用人工 room-boundary 协议做外部评测，避免自证循环。当前结果：

| label | points | ROI | behind | ghost_rate | thickness_p95_m |
|---|---:|---:|---:|---:|---:|
| raw | 110566 | 5628 | 3229 | 0.573738 | 0.961600 |
| marker_clean | 100804 | 5102 | 3021 | 0.592121 | 0.969458 |
| auto_plane_vote_clean | 100000 | 4111 | 1722 | 0.418876 | 0.620508 |
| manual_geometry_vote_clean | 100000 | 3697 | 1298 | 0.351095 | 0.621725 |

解释：

- `auto_plane_vote_clean` 是当前真正自动系统版本。
- `manual_geometry_vote_clean` 是 oracle/manual geometry upper-bound，用来说明如果反射边界更准，后端还能继续下降。
- 自动版已经显著优于 raw 和 marker_clean，但和 manual geometry 上限仍有差距，主要来自自动 cuboid 边界位置偏差。

按墙诊断：

| wall | ROI | behind | depth_checked_behind | ghost_voted_behind | rejected_behind | rejected_behind_rate |
|---|---:|---:|---:|---:|---:|---:|
| room_wall_0 | 1285 | 804 | 750 | 692 | 692 | 0.861 |
| room_wall_1 | 827 | 559 | 559 | 361 | 361 | 0.646 |
| room_wall_2 | 1822 | 918 | 902 | 635 | 635 | 0.692 |
| room_wall_3 | 1694 | 948 | 745 | 243 | 243 | 0.256 |

解释：wall3 的历史 depth-prior ghost evidence 仍弱，后续需要靠更好的视角覆盖、外参/时间同步、或自动反射平面检测后的局部投票补强。

## 4. 推荐完整 pipeline 命令

最小可行版入口：

```bash
cd /home/wangyg/yugong_slam
python3 evaluation_tools/mirrorsentinel_run_pipeline.py
```

带 RViz 实时观察：

```bash
python3 evaluation_tools/mirrorsentinel_run_pipeline.py --rviz
```

展开后的等价命令：

```bash
cd /home/wangyg/yugong_slam

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

只复用已有 votes 快速调参：

```bash
python3 evaluation_tools/apply_frame_prior_vote_threshold.py \
  --map evaluation_tools/results/slam/self_collected/2026-03-30-21-31-03_rescued_priorbag_history_marker_dense_r008_rate05_wait20/sentinel_full/map_raw.pcd \
  --votes evaluation_tools/results/slam/self_collected/2026-03-30-21-31-03_rescued_priorbag_history_marker_dense_r008_rate05_wait20/sentinel_full/frame_prior_score_budget/target100k_gv_ratio/point_vote_stats.npz \
  --out-dir evaluation_tools/results/slam/self_collected/2026-03-30-21-31-03_rescued_priorbag_history_marker_dense_r008_rate05_wait20/sentinel_full/frame_prior_annotation_score_budget/target100k_behind_fill_global \
  --annotation evaluation_tools/data/annotations/self_collected/2026-03-30-21-31-03_rescued.yaml \
  --selection-mode annotation_score_budget \
  --target-kept-points 100000 \
  --score-mode ghost_votes_ratio \
  --annotation-candidate-mode behind \
  --annotation-budget-weight candidate_count \
  --min-observations 1 \
  --min-mask-hits 1 \
  --min-depth-checks 1 \
  --min-ghost-votes 1 \
  --min-ghost-ratio 0.0
```

## 5. 下一步算法工作

短期：

- 继续改自动平面检测/跟踪，缩小 `auto_plane_vote_clean` 与 `manual_geometry_vote_clean` 的差距。
- 针对 wall3 做局部补强：检查相机视角、深度 prior 覆盖、时间同步和外参；必要时对每面墙维护独立的 frame coverage confidence。
- 增加“保面约束”：删除墙后 ghost 时尽量保留墙面附近真实平面点，减少 ROI 点过度变少导致的指标解释问题。
- 论文方法名可写成 `MirrorSentinel: geometry-constrained frame-prior reintegration`，manual geometry 和 global score-budget 作为 ablation。

中期：

- 新采至少 2 条 ROS2 Humble bag：一个电梯/镜墙，一个玻璃门/橱窗；复用同一 pipeline。
- 跑 FAST-LIO2/LIO-SAM 基线，统一输出 `map_raw.pcd` 并用同一 annotation/eval。
- 将 DA3/NeuralRecon/GemDepth 类上游只作为 prior provider；主表仍按 SLAM map metrics 报告。

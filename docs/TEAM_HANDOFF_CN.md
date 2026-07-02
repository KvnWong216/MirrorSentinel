# MirrorSentinel 队友交接说明

更新时间：2026-07-02

这份文档给参与 ROBIO 投稿的同学看。它不替代代码 README，而是说明当前科研进度、系统方法、论文结构、实验状态和接下来需要大家补的事情。

## 1. 当前一句话进度

我们已经有一个可以跑完整自采 ROS2 bag 的 MirrorSentinel 最小系统：基于 ROS2 Humble 的 FAST-LIO 风格建图，接入 Depth Anything 3 单目深度先验，输出 `map_raw.pcd`、`map_marker_clean.pcd`、`map_vote_clean.pcd` 三份地图，并用镜面房间边界标注评估 reflection residual、墙厚、保留率、reference-map F-score 和 FPS。

当前主结果在自采电梯镜面场景上优于 FAST-LIO2、LIO-SAM、RTAB-Map、FAST-LIVO2 和一个 reflection geometry adapted baseline。还缺多序列验证、高质量图、消融补齐和论文文字收敛。

## 2. 论文问题定义

目标问题不是“镜子分割”，而是：

```text
在 LiDAR/视觉 SLAM 建图过程中，镜子/玻璃等非朗伯表面会导致虚假穿墙点、重复结构和墙面厚化。
我们希望系统在 ROS2 bag 回放建图时稳定拒绝这些 ghost points，同时保留正常结构和实时性。
```

论文里要避免把工作写成纯 CV segmentation。3DRef 或 mask IoU 只能作为上游 prior 的辅助实验。主实验必须围绕 SLAM map quality。

## 3. 当前方法

### 3.1 输入

自采 ROS2 bag 包含：

```text
/ouster/points
/ouster/imu
/zed2/zed_node/left/image_rect_color
/tf
/tf_static
```

视觉先验话题：

```text
/vfm/depth_image
/vfm/mirror_mask    # 可选，不再阻塞主线
```

### 3.2 核心思想

1. 用 LiDAR-IMU 前端生成实时位姿和累计地图。
2. 把 LiDAR 点投影到相机图像。
3. 用 DA3 单目 metric depth 给出视觉深度先验。
4. 比较 LiDAR 深度和视觉深度的一致性。
5. 镜面/玻璃附近或反射平面后方的异常点先标记，不立即激进删除。
6. 后端对逐帧累计地图做历史投票清理，输出：

```text
map_raw.pcd           原始累计地图
map_marker_clean.pcd  基于单次/标记清理的地图
map_vote_clean.pcd    基于历史投票的最终地图
```

当前论文主结果使用 `map_vote_clean.pcd`。

### 3.3 为什么 mask 不是当前阻塞项

最初设想需要 `/vfm/mirror_mask`，但后续实验说明主线可以先用 DA3 depth-only：

```text
镜面鬼影的本质是错误重建/虚假几何点，不是单独的图像分割。
只要视觉深度 prior 与 LiDAR 点深度存在稳定冲突，就能作为后端拒绝依据。
```

因此 learned reflection mask 可以作为增强变体或辅助实验，不要阻塞主论文。

## 4. 当前完成了什么

### 4.1 系统工程

已完成：

```text
ROS2 Humble Sentinel-LIO/MirrorSentinel 主包
RViz 可视化配置
DA3 depth node 适配
mask prior node 和 reflection mask 训练/推理脚本
bag replay + pipeline + eval 一键脚本
rosbag1 到 rosbag2 救援/转换工具
room-boundary 标注工具
manual cuboid reference map 工具
reflection map quality evaluator
reference-map F-score evaluator
baseline 汇总脚本
LIO-SAM 历史输出导入脚本
```

核心代码位置：

```text
Sentinal-LIO_ws/src/sentinel_lio/
Sentinal-LIO_ws/src/mirrorsentinel_visual_prior/
evaluation_tools/mirrorsentinel_run_pipeline.py
evaluation_tools/run_ros2_slam_eval.py
evaluation_tools/map_ghost_eval.py
evaluation_tools/reference_map_eval.py
evaluation_tools/summarize_self_collected_methods.py
```

### 4.2 数据和标注

当前主序列：

```text
dataset/2026-03-30-21-31-03.bag
dataset/rosbag2/2026-03-30-21-31-03
```

对应电梯/镜面房间 annotation：

```text
evaluation_tools/data/annotations/self_collected/2026-03-30-21-31-03_rescued.yaml
```

手动几何 reference：

```text
evaluation_tools/data/gt_reference/self_collected/2026-03-30-21-31-03_rescued/room_gt_reference.pcd
```

注意：这个 reference 是手工长方体几何参考，不是激光扫描仪级 GT。论文里要称为 manual geometric reference 或 cuboid reference，不要写成 scanner-grade ground truth。

### 4.3 Baseline 和主结果

当前总表：

```text
evaluation_tools/results/slam/self_collected/all_method_eval_summary.md
```

关键结果：

| Method | Role | Points | RER ↓ | Thickness P95 ↓ | F@5cm ↑ | FPS ↑ | 论文定位 |
|---|---|---:|---:|---:|---:|---:|---|
| FAST-LIO2 equiv raw | baseline | 157402 | 0.5233 | 0.8729 | 0.0278 | 5.43 | 主 baseline |
| FAST-LIVO2 ROS2 raw | visual-lidar baseline | 2973949 | 0.4148 | 0.5206 | 0.1148 | 19.80 | 视觉-LiDAR baseline |
| RTAB-Map ICP | mapping baseline | 165280 | 0.8144 | 1.1112 | 0.0167 | 1.11 | 传统 mapping baseline |
| LIO-SAM ROS1 recovered | LIO baseline | 84906 | 0.7396 | 0.9152 | 0.0169 | 5.03 | 历史恢复 baseline |
| Mapping with Reflection adapted | geometry baseline | 152450 | 0.3376 | 0.6085 | - | - | 反射几何 baseline |
| MirrorSentinel DA3 vote-clean | main method | 100000 | 0.2499 | 0.2916 | 0.0362 | 4.61 | 当前主方法 |
| Manual geometry oracle | oracle | 153184 | 0.0000 | 0.1898 | - | - | 上界 |

解释：

- `RER` 是 Reflection Residual Rate，越低越好。
- `Thickness P95` 是镜面墙附近点云厚化程度，越低越好。
- `F@5cm` 是 map 与手工 reference 的 5 cm F-score，辅助看重建几何贴合度。
- `FPS` 是 pipeline 输出频率或历史结果推导频率。

## 5. 论文建议结构

### 5.1 题目

建议题目：

```text
MirrorSentinel: Visual-Prior Assisted LiDAR-Inertial Mapping for Mirror/Glass Ghost Suppression
```

### 5.2 Abstract

必须包含：

```text
镜子/玻璃导致 LiDAR/camera 非朗伯伪影
传统 SLAM 会产生 behind-wall ghost points 和 map thickening
我们用视觉基础模型深度先验约束 LiDAR 点
用历史投票做地图后端清理
主指标是 SLAM map quality，不是 segmentation IoU
在自采 mirror/glass ROS2 bag 上降低 RER 和 thickness，同时保持实时性
```

### 5.3 Introduction

逻辑链条：

1. 镜子/玻璃是 SLAM 的真实硬问题。
2. LiDAR ToF 和相机成像都会被反射/透射破坏。
3. FAST-LIO/LIO-SAM/FAST-LIVO/RTAB-Map 等传统系统会建出鬼影地图。
4. 单纯 segmentation 不能直接解决 SLAM map consistency。
5. 视觉 foundation model 提供深度/几何先验，但需要和 LiDAR/SLAM 后端耦合。
6. 本文贡献是一个 system-level stitching：把视觉先验接到 LIO mapping 和 map cleanup。

### 5.4 Related Work

建议分四块：

```text
LiDAR-inertial / visual-LiDAR SLAM:
  FAST-LIO2, LIO-SAM, FAST-LIVO2, R3LIVE/类似系统，说明它们不是 reflection-aware。

Transparent/reflective perception:
  mirror/reflection detection, 3DRef, transparent object depth completion。

Reflection artifact removal:
  geometry-based reflection removal, multi-echo/ray-casting/symmetry methods，
  说明我们借鉴的是 system stitching，不是重写物理模型。

Foundation-model depth/reconstruction:
  Depth Anything 系列、NeuralRecon/MASt3R-SLAM/SLAM3R 作为启发或 baseline 背景。
```

### 5.5 Method

建议小节：

```text
System Overview
LiDAR-Camera Projection
Visual Depth Prior Consistency
Reflective Region / Plane Candidate Handling
Historical Vote-Based Map Cleanup
Map Outputs and Runtime Design
```

### 5.6 Experiments

建议表格：

```text
Table 1: Main SLAM map-quality comparison
Table 2: Ablation, raw vs marker-clean vs vote-clean
Table 3: Runtime and map size
Table 4: Auxiliary 3DRef prior quality, optional
```

建议图：

```text
Fig. 1: System pipeline
Fig. 2: RViz/map top-view, baseline vs ours
Fig. 3: mirror-wall close-up, ghost points removed
Fig. 4: ablation visualization
Fig. 5: failure cases / limitations
```

## 6. 现在缺什么

### 6.1 必须补

1. 至少再采 2 到 3 条 ROS2 mirror/glass bag。
2. 每条 bag 做 room-boundary annotation 或更强 GT/reference。
3. 对每条 bag 跑 FAST-LIO2 equivalent、FAST-LIVO2、ours。
4. 补 raw / marker-clean / vote-clean 消融图和表。
5. 生成论文定性图：top-view + 局部镜面墙放大。
6. 写完 related work，并把“为什么不是 segmentation”讲清楚。

### 6.2 最好补

1. learned `/vfm/mirror_mask` 轻量头，基于 3DRef 微调，作为 optional full variant。
2. 官方/公开数据上的 auxiliary prior 结果，比如 3DRef IoU/F1/FPS。
3. 更高质量 GT：用手持扫描仪、贴遮挡膜后重扫、或人工 CAD/CloudCompare 精修 reference。

### 6.3 可以不阻塞投稿

```text
NeuralRecon / SLAM3R / MASt3R-SLAM 完整部署
完全自动反射平面检测
官方 reflection-specific SLAM benchmark 全量复现
```

这些可以写成后续工作或 appendix，不要让它们拖住主线。

## 7. 需要队友具体做什么

### 数据同学

采集 3 到 5 条原生 ROS2 bag：

```text
mirror_elevator_02
glass_corridor_01
mirror_room_loop_01
normal_indoor_control_01
mixed_glass_mirror_01
```

每条至少包含：

```text
/ouster/points
/ouster/imu
/zed2/zed_node/left/image_rect_color
/tf
/tf_static
```

要求：

```text
30 到 90 秒
有明显镜子/玻璃
至少一次回环或重复观察
最好包含普通非镜面墙作为对照
```

### 标注同学

对每条主 bag 做：

```text
room_bounds annotation
reference map annotation 或 cuboid reference
annotation preview 检查
```

入口：

```bash
python3 evaluation_tools/annotation_helper.py room-bounds ...
python3 evaluation_tools/room_gt_annotator.py ...
```

### 实验同学

对每条 bag 跑：

```text
FAST-LIO2 equivalent
FAST-LIVO2 ROS2
RTAB-Map ICP
MirrorSentinel raw/marker/vote
geometry baseline, if time allows
```

刷新总表：

```bash
python3 evaluation_tools/summarize_self_collected_methods.py
```

### 写作同学

优先写：

```text
Introduction
Related Work
Evaluation Protocol
Experiment Setup
Limitations
```

不要把当前方法写成“提出全新视觉大模型”。定位更准确：

```text
foundation-model prior + LiDAR SLAM backend stitching for mirror/glass ghost suppression
```

## 8. 当前风险

1. 数据量不足：目前只有一条主要自采序列，投稿风险最大。
2. GT 不够硬：现在是 manual cuboid reference，不是扫描仪 GT。
3. DA3 depth-only 主线已经能讲，但 learned mask 还不是完整强结果。
4. LIO-SAM 当前 ROS2 复现失败，只能用历史恢复结果，并必须写 provenance。
5. 评估协议是我们针对任务组织的，不是一个完全官方统一 benchmark；需要在 related work 和实验协议里把指标来源讲清楚。

## 9. 预计时间

如果只做“可投 ROBIO workshop/短文强度”的系统论文：

```text
2 到 3 天：补 1 到 2 条序列 + 图 + 消融表
5 到 7 天：完整论文初稿
7 到 10 天：较稳妥的投稿版，包括补实验和润色
```

如果要更强的正会长文质量：

```text
2 到 3 周：多场景数据、强 GT、更多 baseline、完整相关工作和失败案例
```

我的建议是按 7 天节奏冲一版：先把主线结果、消融、图和问题定义打穿，再决定是否继续拉大实验规模。

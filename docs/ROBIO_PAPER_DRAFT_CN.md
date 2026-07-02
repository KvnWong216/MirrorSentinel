# MirrorSentinel ROBIO 论文草稿

> 这是一份写作骨架，不是最终稿。当前已有官方 DA3 metric depth 的 depth-only 主候选结果。下一步是补多序列验证、消融和论文图。

## 暂定题目

MirrorSentinel: Visual-Prior Assisted LiDAR-Inertial Mapping for Mirror/Glass Ghost Suppression

## 摘要草稿

镜子和玻璃等非朗伯表面会在 LiDAR-inertial SLAM 中产生穿墙鬼影点和重复结构，使 scan-to-map 配准和地图表达同时受污染。本文提出 MirrorSentinel，一个运行在 ROS2 Humble 上的视觉先验辅助 LiDAR-inertial mapping 系统。系统将上游视觉基础模型产生的深度/反射先验投影到 LiDAR 点，并通过深度一致性、反射区域约束和历史帧投票对地图点进行在线/后端抑制。与把问题建模为纯反射分割不同，本文直接评估 SLAM 地图中的 reflection residual、平面厚度、有效结构保留率和实时性。在自采镜面电梯 rosbag 上，FAST-LIO2-equivalent baseline 产生明显 behind-wall residual；人工几何 oracle 表明当前标注与后端清理可完全移除该类 residual。基于官方 Depth Anything 3 metric depth 的 depth-only 版本已经能显著降低 reflection residual 和平面厚度，后续 learned reflection mask 可作为增强变体。

## 贡献

1. 提出一个面向镜子/玻璃场景的 ROS2 LiDAR-inertial mapping 框架，将视觉深度/反射先验用于 SLAM 地图鬼影点拒绝，而不是仅输出图像分割。
2. 设计 depth-mask consistency 与历史帧投票的地图清理后端，输出 `map_raw.pcd`、`map_marker_clean.pcd`、`map_vote_clean.pcd` 以支持清晰消融。
3. 建立一套面向非朗伯 SLAM 地图质量的评估协议，结合 reflection residual、平面厚度、map retention、reference-map F-score 和 pipeline FPS。
4. 在自采 ROS2 mirror/glass bag 上给出 baseline、DA3 depth-only 主候选、oracle upper-bound、runtime 和后续 mask 增强变体。

## 方法

### 系统输入

系统输入为同步的 LiDAR、IMU 和 RGB 图像流：

```text
/ouster/points
/ouster/imu
/zed2/zed_node/left/image_rect_color
```

上游视觉先验输出：

```text
/vfm/depth_image
/vfm/mirror_mask
```

### 点投影

对 LiDAR 当前帧点 \(p_l\)，通过外参变换到相机坐标：

\[
p_c = R_{cl} p_l + t_{cl}
\]

再用相机内参投影到图像：

\[
u = f_x x_c/z_c + c_x,\quad v = f_y y_c/z_c + c_y
\]

### 深度一致性

LiDAR range/depth 与视觉先验深度比较：

\[
r_d = z_c - D(u,v)
\]

当点落在反射 mask 或其边界附近，且 \(r_d\) 超过绝对/相对阈值时，该点被视为 ghost candidate 或降低置信度。

### 历史投票地图清理

系统保留逐帧累计地图和每个地图点的历史先验观测统计。若一个点在多个历史帧中持续满足：

```text
inside reflective/behind-wall candidate region
and visual prior contradicts LiDAR depth
```

则在后端导出时从 `map_vote_clean.pcd` 中移除。论文消融使用：

```text
map_raw.pcd
map_marker_clean.pcd
map_vote_clean.pcd
```

## 实验协议

主实验不使用 segmentation IoU 作为最终指标，而评价地图：

```text
RER ↓
Removal vs Raw ↑
Valid Precision Proxy ↑
Plane Thickness P95 ↓
Map Retention ↑
Pipeline FPS ↑
```

3DRef 只作为上游 reflection/depth prior 辅助实验，可报告：

```text
IoU / F1 / Precision / Recall / Prior FPS
```

自采序列没有动捕 GT，因此轨迹只报告 no-GT loop consistency，不写成 ATE。

## 当前可写结果

| Method | Role | RER ↓ | Residual ↓ | Thickness P95 ↓ | FPS ↑ | 论文定位 |
|---|---|---:|---:|---:|---:|---|
| FAST-LIO2 equiv raw | baseline | 0.5233 | 4218 | 0.8729 m | 5.43 | 主 baseline |
| FAST-LIVO2 ROS2 raw | visual-lidar baseline | 0.4148 | 219983 | 0.5206 m | 19.80 | 视觉-LiDAR baseline |
| LIO-SAM ROS1 recovered | LIO baseline | 0.7396 | 4928 | 0.9152 m | 5.03 | 历史恢复 baseline |
| Mapping with Reflection adapted | geometry baseline | 0.3376 | 1506 | 0.6085 m | - | 反射几何 baseline |
| DA3 depth-only vote-clean | main candidate | 0.2499 | 1172 | 0.2916 m | 4.61 | 当前主方法候选 |
| Manual geometry clean | oracle | 0.0000 | 0 | 0.1898 m | - | 上限，不是在线方法 |

## 当前不能写的 claim

- 不能说 manual geometric reference 是 scanner-grade GT。
- 不能说已经完成 depth+mask 完整变体；目前完成的是 DA3 depth-only 主候选。
- 不能把 3DRef segmentation 表当成 SLAM 主实验。

## 还缺的实验

1. `raw` / `marker-clean` / `vote-clean` 消融图和表。
2. 至少一条额外自采 mirror/glass ROS2 bag，或 3DRef 上游辅助结果。
3. RViz/地图定性图。

## 图表清单

| Figure/Table | 内容 | 状态 |
|---|---|---|
| Fig. 1 | 系统图：LiDAR/IMU + visual prior + map cleanup | 待画 |
| Fig. 2 | baseline vs DA3 depth-only top-view map | 待出图 |
| Fig. 3 | reflective ROI ghost focus | baseline/oracle/DA3 已有，待出图 |
| Table 1 | 主 SLAM 地图指标 | baseline/DA3/oracle 已有 |
| Table 2 | 消融 | 待补 |
| Table 3 | runtime | baseline/ours 已有，待整理成论文表 |
| Table 4 | 3DRef 上游辅助 | 待补 |

## 最小投稿判断

当前 DA3 depth-only 主候选已经在 RER 和 Thickness P95 上明显优于 FAST-LIO2-equivalent，同时保留 10 万级地图点并保持约 5 Hz SLAM 输出。接下来若补齐消融、图和至少一条泛化序列，就可以进入一周成稿节奏；learned mirror mask 可以作为增强变体而不是阻塞项。

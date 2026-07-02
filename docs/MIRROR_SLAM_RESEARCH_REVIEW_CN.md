# MirrorSentinel 调研备忘：问题定义、相关工作、方法定位与 GT 方案

生成日期：2026-06-30

这份备忘用于支撑 ROBIO 论文的 Introduction、Related Work、Method Motivation 和 Evaluation Protocol。核心判断是：MirrorSentinel 应被定位为 **reflection-aware LiDAR-inertial mapping / reconstruction**，而不是一个纯 RGB mirror segmentation 方法。

## 1. 问题定义

镜子、玻璃、抛光金属等非朗伯表面对 SLAM 的破坏来自两个层面：

1. 对视觉 SLAM：镜面区域中的纹理来自虚像，几何上不属于当前真实可占据空间；多视图匹配会把镜中内容当作真实场景点，导致地图中真实结构和虚像结构混叠。
2. 对 LiDAR SLAM：镜面/玻璃会改变激光路径，产生 virtual/ghost returns、点云 dropout 或 behind-plane points；scan-to-map 假设局部几何一致，一旦这些点进入配准和地图，就会污染残差和后续地图。

可以引用的关键证据：

- MirrorDrift 直接证明镜子/玻璃的 specular reflection 会产生 ghost points，并使 LiDAR SLAM 的 scan-matching correspondence 系统性偏移；其 2026 实验中能把生产级安全 LiDAR 上的定位误差推到米级。
- Mapping with Reflection / Detection and Utilization of Reflections in 3D Lidar Scans 明确把反射点分类、玻璃平面检测和 virtual point 处理放进 3D LiDAR mapping 框架。
- MirrEnv 说明现有视觉 SLAM 在镜面环境中轨迹可能只是中等退化，但重建 mesh 会非常糟糕，真实反射和虚像会在重建中碰撞。这一点特别适合支持“我们的任务是建图质量，不是只看 ATE”。

论文问题定义建议：

> Given a ROS2 LiDAR-inertial-image stream in mirror/glass-rich indoor scenes, estimate a stable trajectory and produce a point-cloud map where reflection-induced virtual points behind reflective boundaries are suppressed while real indoor structures are preserved.

## 2. 为什么传统 SLAM 基线不够

### 2.1 FAST-LIO / FAST-LIO2

FAST-LIO 系列是强 LiDAR-inertial baseline，优势是实时、稳健、工程成熟。但它默认 LiDAR returns 是真实几何观测，点到面残差没有非朗伯物理模型。镜面 ghost point 如果局部看起来可配准，就可能被吸收到 ikd-tree/map 中。

在本文中，`fast_lio2_equiv` 是最公平的基础 baseline，因为它和我们共用同一 ROS2 预处理、外参和地图导出路径，只关闭 MirrorSentinel gating。

### 2.2 LIO-SAM

LIO-SAM 的价值在因子图、IMU 预积分和回环，但同样不显式处理 specular reflection / virtual points。它适合作为传统 LIO baseline，但部署到当前 ROS2 Humble/Ouster/ZED bag 会有额外工程风险。若时间紧，论文主 baseline 优先保留 `fast_lio2_equiv`。

### 2.3 ORB-SLAM3 / LVI-SAM / R3LIVE

这些方法面向视觉、视觉惯性或 LiDAR-visual fusion。它们可以作为泛化对照，但不是最直接回答“LiDAR map ghost point rejection”的 baseline。

- ORB-SLAM3：可用于 MirrEnv 这类 RGB-D/visual benchmark；但我们的主数据是 Ouster+IMU+ZED rosbag2，不是单纯 RGB-D visual SLAM。
- LVI-SAM/R3LIVE：融合视觉和 LiDAR，但视觉主要帮助 odometry/texture/colorization，不等于显式识别和拒绝镜面虚点。

建议：论文主表中不强行塞太多古老/不同任务 baseline。主表聚焦：

```text
fast_lio2_equiv
sentinel_no_depth
sentinel_no_mask
sentinel_hard_reject
sentinel_full
manual_geometry_oracle
```

若要加专门 reflection baseline，应优先加：

```text
Mapping with Reflection / reflection plane + virtual point handling
3DRef lidar reflection detector / SATNet / EBLNet retrained prior
GRASS / GRAR style TLS reflection artifact removal, if可复现
```

## 3. 专门处理镜面/玻璃的工作

### 3.1 Reflection Removal for Large-Scale 3D Point Clouds / Virtual Point Removal

这类工作通常面向 terrestrial laser scanner 或 large-scale 3D point cloud，常见套路是：

1. 估计 glass / mirror plane。
2. 利用反射对称关系寻找虚点和真实对应点。
3. 用几何相似性或学习特征判断 virtual points。

启发：我们的 backend 不应该只依赖 mask，而应该把“反射平面 + behind-plane candidate + 深度/几何一致性”作为核心。

局限：多数是离线 TLS/LS3DPC，不是 ROS2 online LiDAR-inertial SLAM；传感器是多站位扫描仪，不是移动机器人实时 Ouster。

### 3.2 Mapping with Reflection

这类 3D LiDAR mapping 工作检测玻璃/反射并利用反射信息建图，特征包括 intensity peak、dual return、plane fitting、boundary detection 等。

启发：如果当前 Ouster 有 multi-return/intensity 信息，应该纳入候选平面和 ghost candidate 的判据。当前 MirrorSentinel 只用视觉先验和几何 annotation，后续可以补 `intensity/multi-return cue`。

局限：依赖较强的 LiDAR 反射特征和几何规则；对普通单回波/低质量 intensity、复杂玻璃/镜子材质鲁棒性有限。

### 3.3 3DRef

3DRef 是目前最直接相关的公开 benchmark：多传感器 LiDAR、RGB、2D/3D semantic labels、textured GT mesh，覆盖 glass、mirror、whiteboard、monitor 等反射物。

适合本文的用法：

- 上游 reflection prior 辅助实验：IoU/F1/Precision/Recall/FPS。
- 学习或验证 `/vfm/mirror_mask` / `/vfm/reflection_mask`。
- 不能直接当 ROS2 SLAM 主 benchmark，因为它不是我们的 online LIO replay task。

### 3.4 2026 GRASS / GRAR

GRASS/GRAR 一类最新工作明显说明研究趋势正在从纯规则转向：

```text
vision foundation model initial glass mask
plus geometric refinement/completion
plus physics-driven reflection geometry descriptor
```

这和我们思路高度一致：VFM 不是最终答案，而是给 LiDAR/geometric backend 提供先验。

## 4. 为什么视觉大模型合理

### 4.1 从 MirrorNet 到 RGB-D mirror detection

MirrorNet / MSD 的贡献是把镜子检测作为一个明确视觉问题提出，并证明镜面可以通过上下文差异、语义和低层纹理边界被检测。后续 Depth-Aware Mirror Segmentation、SATNet、RGB-D Video Mirror Detection 继续引入深度、对称性、时序。

这些工作提供：

- mirror/reflection prior 的可学习性。
- 可用于训练或蒸馏轻量 reflection prior head 的数据和模型。

但它们的不足也很明确：

- 输出是 2D mask，不直接保证 3D reconstruction 正确。
- 没有利用 LiDAR scan geometry / range consistency。
- 没有解决 SLAM map 中历史污染点的删除和保留问题。

因此 mask 在我们系统中不应是唯一核心，而应是 **candidate region / uncertainty prior**。

### 4.2 Depth Anything / DA3

Depth Anything 系列证明大规模数据训练的 monocular depth foundation model 具备强 zero-shot generalization。DA3 进一步把任务从单张深度推进到 arbitrary views / spatially consistent geometry / camera pose / point maps，并释放 metric/mono/nested 多系列模型。

对我们的启发：

- 先验不是“分类镜子”，而是给出真实空间的 depth/ray/geometry hypothesis。
- 对每个 LiDAR 点，判断它是否和视觉模型的真实空间深度一致，比只看 mask 更贴近重建问题。
- DA3 支持 metric depth 和 multi-view/streaming，更适合做单目或视频流上游，摆脱双目限制。

限制：

- 单目深度仍有尺度漂移和域外失败；必须用 LiDAR 非反射区域做尺度/偏置校准。
- 视觉模型可能把镜中虚像也当成合理视觉结构，所以不能盲信 depth；必须用 LiDAR 反射平面/历史一致性做过滤。
- DA3/GemDepth 等模型计算量仍需实测，论文实时性要报告真实 FPS。

### 4.3 视频流重建模型：GemDepth / FlashDepth / MASt3R-SLAM / SLAM3R / VGGT

最新趋势说明：视觉大模型正在从单帧 depth 转向视频/多视图一致的 dense reconstruction。

可作为相关工作和潜在 baseline：

- MASt3R-SLAM：monocular dense SLAM with 3D reconstruction priors，报告约 15 FPS，证明“把重建先验做成 SLAM 前端/后端”是合理方向。
- SLAM3R：从 RGB video 直接回归 pointmaps 并全局对齐，20+ FPS，适合作为视觉重建 baseline，而不是 LiDAR map cleaning baseline。
- VGGT：feed-forward 输出 camera parameters、depth maps、point maps、tracks，说明统一 3D foundation model 已经成为强趋势。
- GemDepth：显式引入 camera motion 和 global 3D structure，针对 video depth 的空间精度和时序一致性，和我们“视频流重建先验”方向非常贴。
- FlashDepth：强调 real-time streaming video depth，适合作为实时性相关工作。

本文定位不应是和这些视觉 SLAM 竞争，而是：

> use visual reconstruction priors as a geometry hypothesis to regularize LiDAR-inertial mapping in non-Lambertian scenes.

## 5. 我们方法是否合理

当前构想：

```text
单目/视频深度大模型 -> 与 LiDAR 点投影对齐 -> 非反射/可信区域做尺度匹配 -> 在反射候选区域拒绝与真实深度不一致的 LiDAR 点 -> 历史投票清理地图
```

这是合理的，但需要三处修正：

### 5.1 尺度匹配必须成为正式模块

DA3/monocular depth 输出可能是 relative 或 metric-but-biased。应在每帧或滑窗内用非反射高置信 LiDAR 点拟合：

\[
z_{lidar} \approx s z_{vfm} + b
\]

推荐使用 Huber/RANSAC/trimmed least squares，并记录：

```text
calibration_points
scale
shift
raw_residual
calibrated_residual
```

当前代码已有 depth calibration 字段，但论文中要明确为 visual-depth-to-LiDAR scale alignment。

### 5.2 mask 可以弱化，但不应完全删除

如果一直没有真实 mirror mask，主方法可以退化为：

```text
depth-only consistency + reflective plane proposal + behind-plane candidate
```

但 mask 仍有价值：

- 限定哪些图像区域允许 aggressive rejection。
- 防止普通物体深度误差导致误删。
- 给 reflection plane detection 提供候选。

建议架构上保留 mask input，但论文主方法可以写成 mask-optional：

```text
reflection prior M(u,v) may come from a learned reflection detector, a VFM prompt, or a geometry-derived candidate region.
```

如果短期没有 mask checkpoint：

1. 主实验先用 DA3 depth + 手工/自动反射平面候选。
2. 3DRef 训练轻量 reflection head 作为 auxiliary。
3. 消融中把 mask 作为可插拔增强，而不是系统唯一核心。

### 5.3 关键不是前端删点，而是地图级历史清理

镜面鬼影点并非每帧都可靠可判，单帧直接硬删容易误伤。更好的论文卖点是：

```text
soft front-end confidence
plus backend temporal voting
plus map export ablation
```

这与文献中的 reflection plane + virtual point removal 思路一致，也与现代 dense reconstruction 的 temporal consistency 趋势一致。

## 6. 目前系统缺什么

按优先级：

1. 真实 DA3 / GemDepth / VideoDepthAnything 上游 depth prior。
2. depth-to-LiDAR scale alignment 的可视化和数值日志。
3. mask 可选化：没有 mask 时不能阻塞主方法。
4. 自动反射平面检测或半自动平面 GT，减少手工 annotation 争议。
5. 高质量 GT map，而不是只用 manual cuboid reference。
6. 与 reflection-specific baseline 的对照，至少离线比较一个 virtual point removal / 3DRef-trained prior。

## 7. GT / Benchmark 怎么做

### 7.1 最强方案：survey-grade TLS / scanner map

参考 MapEval 和 VECtor 的做法：

1. 用 Leica RTC360 / FARO / RIEGL 等站式扫描仪采集场景高精度点云。
2. 多站位配准，得到 scanner-grade reference map。
3. 对 mirror/glass 区域人工清理或建模真实边界，避免 GT 自身包含反射伪点。
4. 将 SLAM map ICP/point-to-plane 对齐到 reference map。
5. 报告 Accuracy、Completeness、F-score、local consistency、reflection residual。

优点：论文最硬。
缺点：需要设备。

### 7.2 可行替代：静态多站 Ouster/Livox + CloudCompare 精配准

参考 MapEval 提到的低成本方案：

1. 机器人不动，多站位静态扫描，手动或标靶辅助配准。
2. CloudCompare/ICP 精配准得到 high-quality reference map。
3. 对镜子/玻璃背后区域手工删除，保留真实墙面/边界。
4. 标注 reflective planes / room cuboid / clean mesh。

优点：我们现在设备可能能做。
缺点：不是毫米级 GT，但比 manual cuboid 强很多。

### 7.3 3DRef 式 GT：textured mesh + 自动 raycast label

3DRef 的做法值得模仿：

1. 先获得 textured 3D mesh。
2. 手动在 texture / mesh 上标注 glass、mirror、other reflective。
3. 根据 LiDAR pose 和 mesh raycast 自动给每帧点云贴标签。

这是最适合 reflection detection benchmark 的 GT 形式。我们如果要做小型自采 benchmark，应尽量按这个范式做：

```text
reference_mesh.ply
reflective_faces/texture labels
lidar_poses.txt
per-frame point labels
```

### 7.4 MirrEnv 式 GT：受控机械臂轨迹 + RGB-D + mirror state

MirrEnv 通过机器人臂轨迹提供 camera localization GT，并设置 mirror visible / covered / removed 对照。这对 visual SLAM 很好，但缺 LiDAR/IMU 和点云地图清理任务。

对我们的启发：

- 自采时最好做 visible / covered 对照：同一轨迹，镜子裸露一遍、遮住一遍。
- covered run 可以近似作为 clean reference / sanity check。

## 8. 论文主线建议

推荐叙事：

1. 传统 LIO/VIO/VSLAM 默认观测来自真实几何，镜子/玻璃破坏这个假设。
2. 反射专门方法能做 virtual point removal，但多是 TLS/离线/规则依赖，不是 ROS2 online LIO。
3. 视觉大模型提供强几何先验，但单独做视觉重建/segmentation 不知道 LiDAR 观测是否物理一致。
4. 我们把 visual reconstruction prior 和 LiDAR consistency 耦合：用可信 LiDAR 校准视觉尺度，用视觉/反射先验拒绝反射 ghost LiDAR 点，用历史投票稳定地图。
5. 评估不只看 ATE，而直接看 map ghost residual、plane thickness、map retention、FPS 和 reference-map F-score。

## 9. 立即工程路线

短期最小可行方法：

```text
DA3 depth only
+ LiDAR scale/shift calibration
+ manual/auto reflective plane candidate
+ behind-plane + depth inconsistency vote
+ map_vote_clean export
```

mask 的处理：

```text
保留接口，但不阻塞主方法；
没有真实 mask 时不再用 heuristic mask 写主结果；
后续用 3DRef 训练轻量 reflection head 或接 SATNet/EBLNet 作辅助。
```

下一步应做：

1. 部署 DA3，先跑离线 image -> `/vfm/depth_image` prior bag。
2. 在我们的 full ROS2 bag 上跑 `sentinel_full` depth-only + reflective plane。
3. 如果 RER/Thickness 有提升，立刻补 no-depth/no-mask/hard/full 消融。
4. 如果 DA3 深度在镜面区域也 hallucinate 虚像，改用 video consistency / GemDepth 或增加 LiDAR-driven scale/rejection。

## 10. 关键文献/项目链接

- MirrorDrift: https://arxiv.org/abs/2603.11364
- Mapping with Reflection: https://arxiv.org/abs/1909.12483
- 3DRef: https://arxiv.org/abs/2403.06538 and https://3dref.github.io/
- GRAR: https://arxiv.org/abs/2606.10541
- MirrorNet: https://mhaiyang.github.io/ICCV2019_MirrorNet/index
- Depth Anything: https://depth-anything.github.io/
- Depth Anything 3: https://github.com/ByteDance-Seed/Depth-Anything-3
- GemDepth: https://github.com/yuecheng919/gemdepth
- MASt3R-SLAM: https://arxiv.org/abs/2412.12392
- SLAM3R: https://arxiv.org/abs/2412.09401
- VGGT: https://arxiv.org/abs/2503.11651
- MapEval: https://arxiv.org/html/2411.17928
- MirrEnv: https://research-data.cardiff.ac.uk/articles/media/MirrEnv_a_benchmarking_dataset_for_visual_SLAM_in_mirror_environments/27054073
- VECtor GT protocol: https://star-datasets.github.io/vector/about/ground_truth/

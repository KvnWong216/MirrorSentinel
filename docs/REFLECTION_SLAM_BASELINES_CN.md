# MirrorSentinel Baseline 备忘：LVI-SAM 与反射/玻璃 SLAM 对照

生成日期：2026-07-01

本文档记录本轮新增 baseline 源码与论文对照思路。核心判断：LVI-SAM 值得作为传统多模态 SLAM 强基线/反证基线，但 MirrorSentinel 的真正同问题对照应优先来自 glass/mirror/reflection-aware mapping、reflection artifact removal 和 dense reconstruction。

## 已拉取源码

```text
third_party/baselines/LVI-SAM
third_party/baselines/Reflection_detection
third_party/baselines/FAST-LIVO2-ROS2
third_party/baselines/GRASS
```

`third_party/` 当前被 `.gitignore` 排除，因此这些仓库是本地复现实验资产，不会直接进入 MirrorSentinel GitHub 主仓库。后续如果需要“下载即可用”，应提供下载脚本或 git submodule，而不是直接 vendor 大型第三方工程。

## 为什么要用 LVI-SAM

LVI-SAM 是 ICRA 2021 的 LiDAR-Visual-Inertial smoothing and mapping 系统，官方 README 明确其把 LIO-SAM 和 VINS-Mono 在系统层面结合，目标是实时状态估计和建图。它适合作为本文的传统多模态强基线，原因是：

1. 它是经典 LiDAR + camera + IMU 融合路线，能回答审稿人可能会问的“视觉已经进 SLAM 了，为什么还需要 MirrorSentinel”。
2. 它代表一种常见融合范式：视觉主要贡献 odometry、初始化、约束或鲁棒性，而不是显式建模 mirror/glass 造成的 virtual/ghost LiDAR returns。
3. 如果 LVI-SAM 或同类 LIV 方法在镜面电梯中仍保留 behind-plane ghost map points，可以支撑本文论点：普通多模态融合并不会自动解决非朗伯反射建图污染。
4. 它和 LIO-SAM 同源，便于在 related work 中把纯 LIO、LVI fusion、reflection-aware mapping 三条线分开。

但 LVI-SAM 不应被包装成最核心的同问题 baseline：

1. 官方工程是 ROS1/catkin 生态，README 标注测试环境为 kinetic/melodic；当前 MirrorSentinel 主系统是 ROS2 Humble，直接集成成本高。
2. 它的论文任务是高精度实时 odometry/mapping，不是 mirror/glass virtual point detection/removal。
3. 它不提供反射平面、behind-plane candidate、地图历史删除、reflection residual 等机制。

建议实验定位：

```text
FAST-LIO2-equivalent: 主公平 LIO baseline，同 ROS2 pipeline。
LVI-SAM: 传统 LIV fusion 代表，若时间允许用 ROS1/docker/离线转换跑；若来不及，作为 related-work 反证基线讨论。
MirrorSentinel: depth/reconstruction prior + LiDAR consistency + map-level historical voting。
Reflection_detection / Mapping with Reflection: 反射专用几何 baseline 或实现参考。
```

## 更同问题的 SLAM / Mapping 工作

### 1. Mapping with Reflection / Reflection_detection

本地源码：

```text
third_party/baselines/Reflection_detection
```

对应工作：Zhao, Yang, Schwertfeger, “Mapping with Reflection - Detection and Utilization of Reflection in 3D Lidar Scans,” SSRR 2020。

它使用 intensity peak、dual return、plane fitting、boundary detection 等线索检测 3D LiDAR 反射，并将反射点基于检测到的玻璃/镜面平面进行分类或镜像利用。这是当前最接近我们“反射点识别 + 地图质量改善”的开源几何 baseline。

与 MirrorSentinel 的关系：

```text
它证明 reflection plane + virtual/reflection point handling 是已有合理范式；
我们可以把它改造成离线几何 baseline，或把它的 plane/intensity/multi-return cue 接进 MirrorSentinel 自动反射平面检测。
```

局限：

```text
ROS1/catkin；
传感器/数据格式偏 3DRef/Velodyne/Hesai；
不是 ROS2 online LIO 后端；
没有视觉大模型 depth consistency。
```

### 2. Cartographer_glass

对应工作：“Cartographer_glass: 2D Graph SLAM Framework using LiDAR for Glass Environments,” 2022。

它明确把 glass object detection 放入 Google Cartographer 的优化式 2D Graph SLAM/occupancy grid 流程，目标是解决 LiDAR 在玻璃环境中不能正确注册的问题。

与 MirrorSentinel 的关系：

```text
同属 glass-aware SLAM；
可作为“玻璃环境中修改 SLAM 地图表示”的相关工作；
适合说明已有方法多在 2D occupancy/submap 层面，而我们处理 3D LiDAR-inertial point-cloud map ghost residual。
```

局限：

```text
主要是 2D LiDAR/occupancy grid；
不直接处理 3D Ouster + IMU + image 的点云地图；
不提供视觉重建先验。
```

### 3. Detection and Utilization of Reflections through Plane Optimization and Plane SLAM

对应工作：Sensors 2024 的 reflective-plane global optimization 扩展工作。

它从多帧 LiDAR scan 中优化全局 reflective plane map，然后在末端对 LiDAR readings 分类。这条线很重要，因为它把“反射平面不是单帧局部规则，而应在 SLAM/map 中全局优化”说清楚。

与 MirrorSentinel 的关系：

```text
支撑我们后端历史投票/地图级清理的合理性；
可以作为自动反射平面检测的理论支撑；
也提醒我们后续要从手工 cuboid annotation 走向多帧平面检测与优化。
```

### 4. 3DRef

对应工作：“3DRef: 3D Dataset and Benchmark for Reflection Detection in RGB and Lidar Data,” 2024。

3DRef 提供多回波 LiDAR、RGB、2D/3D semantic labels 和 textured GT meshes，覆盖 mirror、glass、whiteboard、monitor 等反射物。它不是在线 SLAM rosbag benchmark，但非常适合作为 reflection prior / reflection point removal 的辅助 benchmark。

与 MirrorSentinel 的关系：

```text
用来训练或评估 /vfm/reflection_mask；
用其 reflection removal / non-reflection precision 口径支撑我们的 RRR/RER 指标；
不能替代主 SLAM bag replay + map quality 实验。
```

### 5. MirrEnv

对应工作：“Benchmarking visual SLAM methods in mirror environments,” 2024。

MirrEnv 是 RGB-D visual SLAM mirror environment 数据集，包含 mirror present / covered / removed 变体和 GT camera localization。论文结论对我们很有用：轨迹 ATE 可能只是中等退化，但 mesh map 会明显错误，真实结构与虚像会冲突。

与 MirrorSentinel 的关系：

```text
支撑本文不要只看 ATE，而要看 map ghost residual / reconstruction quality；
可作为视觉 SLAM/重建泛化验证；
不是 ROS2 LiDAR-inertial 主实验。
```

### 6. MirrorDrift

对应工作：“MirrorDrift: Actuated Mirror-Based Attacks on LiDAR SLAM,” 2026。

该工作直接证明 specular reflection 可以在无需信号注入的情况下产生 ghost points，并系统性偏置 scan matching correspondences，使多个 LiDAR SLAM 系统出现米级误差。

与 MirrorSentinel 的关系：

```text
作为问题定义最强证据之一；
证明现代 LiDAR SLAM 的 scan-matching 几何一致性假设会被镜面反射破坏；
可用于 introduction/motivation，不一定能作为可复现实验 baseline。
```

### 7. GRASS / GRAR / LS3DPC reflection removal

对应工作包括 CVPR 2018 “Reflection Removal for Large-Scale 3D Point Clouds”、GRASS/GRAR 等 TLS 点云反射伪影去除方法。

与 MirrorSentinel 的关系：

```text
它们把问题定义为 reconstruction/map artifact removal，而不是 segmentation；
常用 glass plane estimation、multi-echo cue、laser trajectory tracing、real/virtual symmetry 等思路；
可作为离线点云清理 baseline 或设计灵感。
```

局限：

```text
多数面向 terrestrial laser scanner / 静态多站点点云；
不直接处理在线 ROS2 LiDAR-inertial SLAM；
往往依赖 scanner position 或多回波质量。
```

当前可复现性：

```text
GRASS GitHub 当前更像论文/数据入口，未提供完整可运行算法代码；
GRAR arXiv HTML 说明 VFM + 几何 refinement + descriptor 的最新趋势，但暂未形成可直接接 ROS2 bag 的 baseline；
因此短期主表不把 GRASS/GRAR 当可跑 baseline，related work 中重点引用其思想。
```

对我们的启发：

```text
1. VFM/mask 只负责给反射区域或玻璃面候选，不应单独成为最终任务；
2. 后端必须保留反射平面、射线路径、对称几何或历史一致性；
3. 我们当前 Mapping-with-Reflection adapted baseline 覆盖了平面/behind-plane 几何，MirrorSentinel 进一步加入 DA3 depth consistency 和时间投票。
```

### 8. 新一代 LIV / dense reconstruction 基线

这类不是专门处理镜子/玻璃，但可作为“上层任务类似”的现代基线：

```text
R3LIVE / R3LIVE++: LiDAR-inertial-visual tightly coupled mapping，偏彩色/辐射重建。
FAST-LIVO2: fast direct LiDAR-inertial-visual odometry，偏实时鲁棒状态估计。
MASt3R-SLAM / SLAM3R: 视觉重建先验驱动的 dense SLAM / RGB video reconstruction baseline。
LPVIMO-SAM / PFLIO-SAM: 引入偏振视觉等传感器改善退化/低纹理环境，可作为“硬件解决路线”的相关工作。
```

与 MirrorSentinel 的关系：

```text
这些方法说明 SLAM 正在向多模态、dense reconstruction 和 foundation prior 发展；
但它们多数没有显式输出 reflection residual cleanup；
适合放 related work，不应挤掉 reflection-aware baseline。
```

### 9. FAST-LIVO2-ROS2

本地源码：

```text
third_party/baselines/FAST-LIVO2-ROS2
```

这是当前最值得优先部署的 ROS2 相关工作基线。它是 FAST-LIVO2 的 ROS2 Humble 版本，README 明确定位为 fast direct LiDAR-inertial-visual odometry，并提供 `mapping_ouster_ntu.launch.py` 和 Ouster 参数配置。

为什么适合我们：

```text
ROS2 Humble，可避免 LVI-SAM 的 ROS1 环境冲突；
上层任务是 LiDAR + image + IMU 的实时建图/定位；
有 Ouster 输入配置，和我们的 Ouster+ZED bag 更接近；
能回答“现代 LIV fusion 是否已经解决镜面建图污染”这个审稿问题。
```

风险和待改：

```text
依赖 vikit_common / vikit_ros / livox_ros_driver2 / Sophus；
默认 NTU_VIRAL topic 是 /os1_cloud_node1/points, /imu/imu, /left/image_raw；
我们的 topic 是 /ouster/points, /ouster/imu, /zed2/zed_node/left/image_rect_color；
需要填写 ZED pinhole intrinsics 和 LiDAR-camera extrinsic；
它本身没有 reflection-aware cleanup，预期会保留部分镜面 ghost map points。
```

论文定位：

```text
FAST-LIVO2-ROS2 is the preferred modern ROS2 LIV baseline.
It is stronger and more current than LVI-SAM for our system setting, but it is
not reflection-aware; therefore it tests whether general direct LIV fusion alone
can suppress mirror/glass ghost structures.
```

### 10. NeRFReN / Mirror-NeRF / reflected neural rendering

这类工作从神经渲染或新视角合成角度处理反射/镜面，常见做法是把 transmitted/reflected components 分解，或者显式建模镜面反射射线。它们能支持本文“反射导致虚拟几何，需要物理/几何约束”的论点，但不适合作为主 SLAM baseline：

```text
输入通常是多视角 RGB / posed images，而不是 ROS2 LiDAR+IMU+camera stream；
输出是 NeRF/渲染场或反射分解，不是实时 LiDAR-inertial point-cloud map；
计算量和实时性目标与 ROBIO 在线建图系统不一致。
```

放法：

```text
Related work 中作为 reflective reconstruction / neural rendering 分支；
不进入主表，除非后续做纯视觉重建泛化实验。
```

### 11. ERASOR / Removert / OctoMap visibility

这类工作原本面向动态物体删除或静态地图更新，不是镜面专用，但“可见性冲突 / ray casting / pseudo occupancy”思想与 ghost removal 有交集。它适合作为一个后端清图类 baseline：

```text
输入：多帧 LiDAR scan + pose；
输出：cleaned static map；
优点：不需要视觉、不需要反射 mask；
缺点：没有 reflective plane 物理模型，可能把真实遮挡和镜面虚点混淆。
```

短期建议：

```text
如果要再加一个 ROS2/离线强 baseline，优先考虑 ERASOR/Removert-style visibility cleanup；
它和 MirrorSentinel 的历史投票模块相近，但缺少 DA3 depth 和 reflection-plane condition。
```

## 推荐 baseline 优先级

三天出结果优先级：

```text
P0: fast_lio2_equiv，同 ROS2 pipeline，已经可跑。
P0: MirrorSentinel DA3 depth-only / vote-clean，主方法。
P0: manual_geometry_oracle，上限，不作为真实在线方法。
P1: Reflection_detection 离线几何 baseline，尝试接我们的点云或 3DRef。
P1: FAST-LIVO2-ROS2，现代 ROS2 LIV baseline，优先部署到 Ouster+ZED bag。
P1: MASt3R-SLAM / SLAM3R，视觉重建 baseline，展示纯视觉重建在镜面场景的局限。
P2: LVI-SAM，传统 LIV fusion baseline。优先 docker/ROS1 独立跑，避免污染 ROS2 Humble 主仓库。
P2: Cartographer_glass / GRASS / GRAR，作为相关工作或离线点云清理对照。
```

论文中最稳妥的表述：

```text
We compare against a strong LiDAR-inertial baseline sharing the same ROS2 mapping pipeline,
and discuss representative LiDAR-visual-inertial systems such as LVI-SAM as general multi-modal
fusion baselines. Since general LIV fusion does not explicitly model specularly induced virtual
returns, we further relate MirrorSentinel to reflection-aware mapping methods such as Mapping
with Reflection and Cartographer_glass.
```

## 下一步执行

1. 保持 LVI-SAM 为独立 ROS1 baseline，不进入 `Sentinal-LIO_ws`。
2. 给 `Reflection_detection` 写一个 wrapper：输入我们导出的 `map_raw.pcd` 或逐帧 cloud，输出 reflection candidate labels / clean map。
3. 把 `Mapping with Reflection` 的 plane/intensity/multi-return cue 加入 MirrorSentinel 自动反射平面检测设计。
4. 主实验先不等 LVI-SAM，继续以 `fast_lio2_equiv` vs `MirrorSentinel DA3 depth-only` vs `oracle` 出第一版图表。
5. 如果需要投论文补强，优先补一个 reflection-specific baseline，而不是优先把 ROS1 LVI-SAM 硬移植到 ROS2。

## 参考链接

- LVI-SAM official GitHub: https://github.com/TixiaoShan/LVI-SAM
- LVI-SAM paper: https://ieeexplore.ieee.org/document/9561996/
- FAST-LIVO2-ROS2 GitHub: https://github.com/v4rl-ucy/FAST-LIVO2-ROS2
- FAST-LIVO2 arXiv: https://arxiv.org/abs/2408.14035
- Mapping with Reflection arXiv: https://arxiv.org/abs/1909.12483
- Reflection_detection official GitHub: https://github.com/STAR-Center/Reflection_detection
- Cartographer_glass arXiv: https://arxiv.org/abs/2212.08633
- 3DRef official site: https://3dref.github.io/
- 3DRef arXiv: https://arxiv.org/abs/2403.06538
- MirrEnv dataset: https://research-data.cardiff.ac.uk/articles/media/MirrEnv_a_benchmarking_dataset_for_visual_SLAM_in_mirror_environments/27054073
- MirrEnv paper: https://link.springer.com/article/10.1007/s41095-022-0329-x
- MirrorDrift arXiv: https://arxiv.org/abs/2603.11364
- Plane optimization / Plane SLAM reflection work: https://www.mdpi.com/1424-8220/24/15/4794
- GRASS official GitHub: https://github.com/wpshao/GRASS
- GRAR arXiv HTML: https://arxiv.org/html/2606.10541v1
- Reflection Removal for Large-Scale 3D Point Clouds: https://openaccess.thecvf.com/content_cvpr_2018/papers/Yun_Reflection_Removal_for_CVPR_2018_paper.pdf
- LPVIMO-SAM arXiv: https://arxiv.org/abs/2504.20380

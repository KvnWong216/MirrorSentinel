# Mirror/Glass SLAM 评估协议

本文的评估拆成两条互不混淆的线：

1. 主 SLAM 任务：评价在线建图系统是否在镜子/玻璃场景中减少鬼影点、墙体增厚、轨迹漂移，并保持实时性。
2. 上游先验任务：评价 depth / reflection prior 本身的质量，只作为辅助实验，不参与 SLAM 主指标定义。

这个协议的原则是：轨迹指标采用 TUM RGB-D、KITTI Odometry、Hilti SLAM Challenge 这类 SLAM benchmark 的通用做法；建图指标采用 3D reconstruction benchmark 常用的 precision / recall / F-score / cloud-to-reference 思路；镜子/玻璃鬼影指标是建图 precision / outlier metric 在反射 ROI 内的特化版本。

参考 benchmark 角色：

| Benchmark family | 本文采用的部分 | 本文不直接采用的部分 |
| --- | --- | --- |
| TUM RGB-D / evo | ATE、RPE、时间戳关联、轨迹对齐 | RGB-D 数据本身 |
| KITTI Odometry | 分段 drift、trans/rot relative drift 的报告习惯 | KITTI 车载场景作为主数据 |
| Hilti SLAM Challenge | failure rate、真实工程场景鲁棒性报告习惯 | Hilti 作为镜/玻璃主数据 |
| Tanks and Temples / ETH3D / reconstruction benchmark | precision、recall、F-score、cloud-to-reference | 大规模 MVS 场景设置 |
| SLAMBench family | runtime、FPS、系统资源 | 它的特定传感器/实现 |

## 1. 数据要求

### 1.1 主 SLAM 数据

主实验必须使用可回放的时间同步传感器流。推荐统一为 ROS2 Humble rosbag2：

```text
required:
  /ouster/points                         sensor_msgs/msg/PointCloud2
  /ouster/imu                            sensor_msgs/msg/Imu
  /zed2/zed_node/left/image_rect_color   sensor_msgs/msg/Image
  /zed2/zed_node/right/image_rect_color  sensor_msgs/msg/Image
  /tf
  /tf_static

recommended:
  /zed2/zed_node/left/camera_info
  /zed2/zed_node/right/camera_info
```

可以额外记录缓存 topic 以加速复现实验：

```text
/vfm/depth_image
/vfm/mirror_mask
/mirror_sentinel/frame_stats
```

但主方法和 baseline 必须基于同一原始 bag、同一外参、同一 LiDAR/IMU/Image 时间流运行。

当前已可用的 prototype bag：

```text
dataset/rosbag2/2026-03-30-21-31-03_rescued
```

这个 bag 由损坏索引的旧 `.bag` 顺序扫描抢救而来，ROS2 Humble `ros2 bag info` 可读。它适合跑通 pipeline 和生成第一版定量结果，但不建议作为论文唯一主序列。

### 1.2 上游辅助数据

3DRef 只用于 reflection prior 的辅助评估。它不是在线 SLAM bag benchmark，因此不用于 FAST-LIO / LIO-SAM / Sentinel-LIO 的公平主比较。

## 2. 方法集合

主 SLAM 实验至少包含：

```text
fast_lio2_equiv        Sentinel-LIO 中关闭 mirror/depth gating，作为同代码基线
sentinel_no_mask       关闭 explicit mask，只保留 depth consistency
sentinel_no_depth      关闭 depth consistency，只保留 reflection mask
sentinel_hard_reject   反射/不一致点硬剔除
sentinel_full          soft confidence weighting 完整方法
```

可选：

```text
lio_sam_ros2           只有在 ROS2 版本、Ouster topic、IMU 格式和外参都稳定后加入
```

最公平的基础 LIO baseline 是 `fast_lio2_equiv`，因为它与主方法共享同一套点云预处理、外参和发布接口，只关闭本文新增 gating。

## 3. 主指标

### 3.1 轨迹指标

有 GT 或 reference trajectory 时，采用 TUM / KITTI / Hilti 风格指标：

| Metric | 来源/惯例 | 用途 |
| --- | --- | --- |
| ATE RMSE | TUM RGB-D / evo 常用 | 全局轨迹误差 |
| ATE Mean / Median | TUM RGB-D / evo 常用 | 误差分布稳定性 |
| RPE Translation | TUM RGB-D | 局部相对运动误差 |
| RPE Rotation | TUM RGB-D | 局部相对姿态误差 |
| KITTI Segment Drift | KITTI Odometry | 固定长度片段上的 translational / rotational drift |
| Failure Rate | Hilti / SLAM challenge 常用报告方式 | 是否丢轨、崩溃或无法输出完整轨迹 |

有完整 GT 时，论文表优先报告：

```text
ATE RMSE ↓
RPE Trans RMSE ↓
RPE Rot Mean ↓
Failure Rate ↓
```

无动捕 GT 的自采主序列不报告 ATE 主结果，只报告 no-GT consistency：

| Metric | 定义 | 适用条件 |
| --- | --- | --- |
| Loop Translation Error | 起点和终点回到同一物理位置时的位移差 | 闭环路线 |
| Loop Rotation Error | 起点和终点回到同一物理朝向时的姿态差 | 闭环路线 |
| Drift Ratio | loop translation error / trajectory length | 闭环路线 |
| Revisit Consistency | 人工标注重访时间对之间的相对 pose 差 | 重访路线 |

这些 no-GT 指标只能称为 consistency / drift proxy，不能称为 ground-truth ATE。

### 3.2 建图指标

建图质量采用 reconstruction benchmark 中常见的 cloud-to-reference / precision / recall / F-score 思路。

如果有 reference map、CAD、mesh 或高质量离线 fused map，报告：

| Metric | 定义 | 解释 |
| --- | --- | --- |
| Accuracy / Precision Distance | 估计点云到 reference 最近邻距离 | 估计图是否有虚假点、噪声、鬼影 |
| Completeness / Recall Distance | reference 到估计点云最近邻距离 | 地图是否漏建 |
| F-score@τ | precision@τ 与 recall@τ 的调和平均 | 统一衡量准确性与完整性 |

阈值 `τ` 建议按室内 LiDAR 建图尺度设置：

```text
tau_map: 0.05 m, 0.10 m, 0.20 m
```

如果没有全局 reference map，则主报告局部几何指标：

| Metric | 定义 | 解释 |
| --- | --- | --- |
| Plane Thickness Std | 目标平面 ROI 内点到拟合平面的标准差 | 墙/玻璃/镜面是否变厚 |
| Plane Thickness P90 / P95 | 点到平面距离绝对值的高分位 | 是否存在拖影和重复面 |
| Inlier Ratio@τ | ROI 内距离平面小于 τ 的点比例 | 平面局部建图质量 |
| Outlier Ratio@τ | ROI 内距离平面大于 τ 的点比例 | 局部伪点比例 |

推荐阈值：

```text
tau_plane: 0.05 m, 0.10 m, 0.20 m
```

### 3.3 反射鬼影指标

反射鬼影指标不是凭空定义的新 benchmark，而是 reconstruction precision / outlier ratio 在镜子/玻璃平面 ROI 内的特化。

每个反射物体标注一个平面和 ROI：

```text
plane: n_x x + n_y y + n_z z + d = 0
roi:   map frame 下的 3D bounding box 或 polygon/frustum
front_side: 真实可占据空间一侧
behind_side: 镜子/玻璃背后不应出现实体点的一侧
```

主指标：

| Metric | 定义 | 解释 |
| --- | --- | --- |
| Behind-Plane Point Count ↓ | ROI 内落在 reflective plane 背后的点数 | 直接统计鬼影点 |
| Ghost Rate ↓ | behind-plane points / reflective ROI points | 不受地图点数规模影响 |
| Behind-Plane Density ↓ | behind-plane points / ROI volume | 比较不同 ROI 尺寸 |
| Ghost Distance Mean/P95 ↓ | behind-plane 点到 reflective plane 的距离 | 鬼影深度和严重程度 |
| Mirror/Glass Plane Thickness ↓ | reflective plane 附近点到平面距离分布 | 镜面/玻璃面是否被建厚 |

推荐主表只放 2 到 3 个最核心指标：

```text
Ghost Rate ↓
Behind-Plane Points ↓
Plane Thickness P95 ↓
```

其余指标放附表或 supplement。

### 3.4 运行效率指标

实时性必须单独报告，因为本文上游选择 FastStereoFoundation / DAv3 剪枝路线的动机之一就是在线性。

| Metric | 来源 | 说明 |
| --- | --- | --- |
| LiDAR Frame Runtime | SLAMBench 风格 runtime 报告 | 每帧 LIO 处理耗时 |
| VFM Runtime | 上游视觉模型 | depth/mask 生成耗时 |
| Total Pipeline Latency | 系统指标 | 从输入到地图更新 |
| FPS / Hz | 系统指标 | 是否满足 10Hz LiDAR |
| GPU Memory | 系统指标 | 工程可部署性 |

运行效率和上游精度解耦报告。主 SLAM 表中只保留 `Runtime / Hz`；上游辅助表中报告 `Depth/Mask FPS`。

## 4. 推荐主表

### 4.1 主 SLAM 表：镜子/玻璃建图

```text
Method | Ghost Rate ↓ | Behind-Plane Points ↓ | Thickness P95 ↓ | Loop Drift ↓ | Runtime ↓
```

### 4.2 有 GT 泛化表：普通困难 SLAM

```text
Method | ATE RMSE ↓ | RPE Trans RMSE ↓ | RPE Rot Mean ↓ | Failure Rate ↓ | Runtime ↓
```

### 4.3 消融表

```text
Variant | Ghost Rate ↓ | Thickness P95 ↓ | Loop Drift ↓ | Rejected Ratio | Runtime ↓
```

### 4.4 上游辅助表

```text
Method | Split | IoU ↑ | F1 ↑ | Precision ↑ | Recall ↑ | FPS ↑
```

## 5. 标注规范

每条自采主序列需要保存一个 annotation YAML：

```yaml
sequence: mirror_corridor_01
bag: dataset/rosbag2/mirror_corridor_01
map_frame: camera_init
reflective_planes:
  - id: mirror_wall_0
    type: mirror
    normal: [0.0, 1.0, 0.0]
    d: -2.35
    front_side: positive
    roi:
      min: [-3.0, 2.20, -0.2]
      max: [ 3.0, 2.80,  2.2]
    thresholds_m: [0.05, 0.10, 0.20]
planar_regions:
  - id: wall_left_0
    normal: [1.0, 0.0, 0.0]
    d: 1.20
    roi:
      min: [-1.35, -3.0, -0.2]
      max: [-1.05,  3.0,  2.2]
```

平面可以通过人工测量、点云中手工选点拟合、或 AprilTag/标定板辅助确定。论文中必须说明标注方式。

## 6. 结果目录

```text
evaluation_tools/results/slam/
  self_collected/
    <sequence>/
      <method>/
        trajectory_tum.txt
        map.pcd
        frame_stats.csv
        run_metadata.yaml
        metrics_mapping.json
        metrics_no_gt.json
  public_slam/
    <dataset>/<sequence>/<method>/
      trajectory_tum.txt
      metrics_tum.json

evaluation_tools/data/annotations/
  self_collected/
    <sequence>.yaml
```

每次运行必须保存：

```text
git commit/hash
params yaml
bag path
method name
start/end time
hardware
runtime log
```

## 7. 最小可投稿协议

最低要求：

1. 3 条以上自采 ROS2 mirror/glass bag。
2. 每条至少一个 reflective plane ROI 标注。
3. 每条跑 `fast_lio2_equiv`、`sentinel_no_mask`、`sentinel_no_depth`、`sentinel_full`。
4. 主表报告 Ghost Rate、Behind-Plane Points、Thickness P95、Runtime。
5. 若路线闭环，补 Loop Drift / Drift Ratio。
6. 3DRef 只作为上游 reflection prior 辅助表。
7. M2DGR/Hilti 只作为普通困难 SLAM 泛化表。

这样写的重点是：本文的主证据来自在线 SLAM 建图结果，而不是离线 segmentation。

# 数据集本地部署说明

当前实验不再把 3DRef 当作主 SLAM benchmark。数据角色如下：

| 数据 | 角色 | 是否用于主 SLAM 评估 |
| --- | --- | --- |
| 自采 ROS2 mirror/glass bag | 主实验：bag 回放、在线建图、鬼影/厚度/漂移评估 | 是 |
| 3DRef | 上游 reflection/depth prior 辅助实验 | 否 |
| M2DGR / Hilti | 普通困难 SLAM 泛化验证，有 GT 时算 ATE/RPE | 否，作为泛化 |

主评估协议见：

```text
SLAM_EVAL_PROTOCOL_CN.md
evaluation_tools/configs/benchmark_plan.yaml
```

## 1. 主 SLAM 数据

主 SLAM 实验需要 ROS2 Humble rosbag2，至少包含：

```text
/ouster/points
/ouster/imu
/zed2/zed_node/left/image_rect_color
/zed2/zed_node/right/image_rect_color
/tf
/tf_static
```

当前已经可用的 prototype bag：

```text
dataset/rosbag2/2026-03-30-21-31-03_rescued
```

检查：

```bash
source /opt/ros/humble/setup.bash
ros2 bag info dataset/rosbag2/2026-03-30-21-31-03_rescued
```

该 bag 由旧 `.bag` 顺序扫描抢救得到，适合 pipeline validation，不建议作为论文唯一主序列。

正式论文建议重新采集 3 到 5 条原生 ROS2 bag：

```text
mirror_corridor_01
glass_wall_01
mixed_reflection_01
normal_indoor_01
long_loop_reflection_01
```

录制示例：

```bash
ros2 bag record -o dataset/rosbag2/mirror_corridor_01 \
  /ouster/points \
  /ouster/imu \
  /zed2/zed_node/left/image_rect_color \
  /zed2/zed_node/right/image_rect_color \
  /zed2/zed_node/left/camera_info \
  /zed2/zed_node/right/camera_info \
  /tf \
  /tf_static
```

## 2. 3DRef 辅助实验

3DRef 用于证明 reflection prior，不用于 SLAM 主任务。它可以输出：

```text
IoU
F1
Precision
Recall
FPS
```

本地目录：

```text
dataset/public/3DRef/reflection/
```

下载和解包：

```bash
bash dataset/download_3dref.sh reflection
bash dataset/download_3dref.sh extract-reflection
bash dataset/download_3dref.sh check
```

## 3. 泛化数据

M2DGR / Hilti 用于普通困难 SLAM 泛化，定位是：

```text
证明本文方法不破坏常规困难 SLAM 表现，而不是证明镜子/玻璃鬼影抑制。
```

有 GT 时报告：

```text
ATE RMSE
RPE Translation
RPE Rotation
Failure Rate
Runtime
```


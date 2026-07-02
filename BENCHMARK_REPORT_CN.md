# Sentinel-LIO 官方 Benchmark 部署报告

## 1. 结论

这个仓库现在已经补上了一套面向 ROBIO 投稿的 benchmark harness：

1. `evaluation_tools/tum_trajectory_eval.py`
   - TUM RGB-D 官方格式兼容的轨迹评测。
   - 支持 ATE、RPE、时间戳关联、SVD 刚体对齐、JSON 结果导出。
   - 输入格式为 `timestamp tx ty tz qx qy qz qw`。

2. `evaluation_tools/rosbag_to_tum.py`
   - ROS1 rosbag 中 `/Odometry`、`/path` 或 `PoseStamped` 到 TUM trajectory txt 的导出脚本。
   - 需要在 ROS1 环境中运行，因为当前机器 shell 里没有 `rosbag` Python 模块。

3. `evaluation_tools/mirror_segmentation_eval.py`
   - 3DRef 风格反射检测评测入口。
   - 支持 RGB mask 文件夹与 SemanticKITTI 风格 point label 文件夹。
   - 输出 IoU、F1、precision、recall、accuracy、balanced accuracy。

4. `evaluation_tools/README.md`
   - 放了可直接复制执行的 benchmark 命令、输入格式、输出文件说明。

当前仓库自带一条 8GB 自采 rosbag，可作为真实反射场景实验：

```text
dataset/rosbag/2026-03-30-21-31-03.bag
```

但当前 shell 只有 `/opt/ros/humble`，仓库本身是 ROS1/catkin 风格，并且 launch/wrapper 中仍然写着 `/opt/ros/noetic`。因此我没有直接从这条 bag 里导出 `/Odometry`。这不是评测脚本的问题，而是本机当前环境没有 ROS1 `rosbag` 命令。

另外，尝试用 `rosbags-convert` 直接读取该 ROS1 bag 时发现索引损坏：

```text
Bag index looks damaged
```

后续需要在 ROS1 环境中先执行 `rosbag reindex`，再转 rosbag2 或导出轨迹。

## 2. 为什么选这两条官方线

### 2.1 SLAM 主指标：TUM RGB-D trajectory benchmark 格式

对 ROBIO 来说，SLAM 方法必须有标准轨迹精度指标。最稳妥的是使用 TUM RGB-D benchmark 的轨迹文件格式和经典指标：

1. ATE: Absolute Trajectory Error
2. RPE: Relative Pose Error

这条线的优点：

1. 格式简单，跨视觉 SLAM、LiDAR SLAM、VIO/LIO 都能复用。
2. ATE/RPE 是审稿人熟悉的定位精度指标。
3. 你们的系统已经发布 `/Odometry`，天然能导出成 TUM 格式。
4. 不强依赖某一个数据集，只要有 ground truth 就能算。

### 2.2 非朗伯主题指标：3DRef 风格反射检测

本文的核心贡献不是一般 LIO，而是利用视觉大模型深度/反射先验抑制非朗伯体干扰。只报 ATE/RPE 会不够锋利，因为审稿人会问：

1. 镜面/玻璃区域到底识别得怎么样？
2. 反射区域抑制是否真的来自视觉先验，而不是调参碰巧提升？
3. 分割或点级置信度是否和定位提升有相关性？

因此新增 3DRef 风格的 mask / point-label 评测入口，用 IoU、F1、precision、recall 支撑“反射感知模块”的可信度。

## 3. 已验证情况

已在当前机器上完成：

```bash
python3 -m py_compile \
  evaluation_tools/tum_trajectory_eval.py \
  evaluation_tools/rosbag_to_tum.py \
  evaluation_tools/mirror_segmentation_eval.py
```

结果：通过。

合成轨迹测试：

```bash
python3 evaluation_tools/tum_trajectory_eval.py all \
  --groundtruth /tmp/sentinel_bench_test/gt.txt \
  --estimate /tmp/sentinel_bench_test/est.txt \
  --max-difference 0.02 \
  --output-dir /tmp/sentinel_bench_test/tum_eval
```

输出中 ATE/RPE 都接近 0，符合“估计轨迹只相对真值整体平移”的预期，因为 ATE 会做刚体对齐，RPE 不受整体平移影响。

合成 mask 测试：

```bash
python3 evaluation_tools/mirror_segmentation_eval.py image \
  --gt-dir /tmp/sentinel_bench_test/masks/gt \
  --pred-dir /tmp/sentinel_bench_test/masks/pred \
  --gt-threshold 0 \
  --pred-threshold 127 \
  --output-json /tmp/sentinel_bench_test/mask.json \
  --output-csv /tmp/sentinel_bench_test/mask.csv
```

输出：

```text
IoU/F1/Precision/Recall/Accuracy: 0.500000 0.666667 0.666667 0.666667 0.666667
```

## 4. 推荐实验协议

### 4.1 轨迹精度

每个序列至少跑以下方法：

1. FAST-LIO2 原版
2. LIO-SAM
3. LVI-SAM 或 R3LIVE
4. Sentinel-LIO without mirror mask
5. Sentinel-LIO without depth prior
6. Sentinel-LIO full

每条序列记录：

1. `/Odometry`
2. `/path`
3. `/mirror_sentinel/frame_stats`
4. `/cloud_registered`
5. 最终 `PCD/scans.pcd`

导出轨迹：

```bash
source /opt/ros/noetic/setup.bash
source Sentinal-LIO_ws/devel/setup.bash

python3 evaluation_tools/rosbag_to_tum.py result.bag \
  --topic /Odometry \
  --out evaluation_tools/results/sentinel_full_tum.txt
```

计算 ATE/RPE：

```bash
python3 evaluation_tools/tum_trajectory_eval.py all \
  --groundtruth evaluation_tools/data/gt_tum.txt \
  --estimate evaluation_tools/results/sentinel_full_tum.txt \
  --max-difference 0.02 \
  --output-dir evaluation_tools/results/sentinel_full_tum_eval
```

论文主表建议列：

```text
Method | ATE RMSE ↓ | ATE Mean ↓ | RPE Trans RMSE ↓ | RPE Rot Mean ↓ | Runtime ↑/↓
```

### 4.2 反射感知精度

对 3DRef 或自采标注集跑：

```bash
python3 evaluation_tools/mirror_segmentation_eval.py image \
  --gt-dir /path/to/gt_masks \
  --pred-dir evaluation_tools/results/pred_masks \
  --pred-threshold 127 \
  --output-json evaluation_tools/results/3dref_alllabel.json \
  --output-csv evaluation_tools/results/3dref_alllabel_per_sample.csv
```

论文表建议按类别拆：

```text
Class | IoU ↑ | F1 ↑ | Precision ↑ | Recall ↑
mirror
glass
other-reflective
all-reflective
```

### 4.3 建图质量

当前仓库还没有完整地图质量指标脚本，建议补：

1. Cloud-to-cloud distance
2. Cloud-to-mesh distance
3. Ghost-point rate inside mirror/glass frustum
4. Map thickness / wall duplication metric
5. Occupancy inconsistency around reflective planes

如果没有高精 mesh，可以先做“反射区域伪点率”：

```text
ghost_rate = number of mapped points behind annotated reflective plane / total points in reflective frustum
```

这会比单纯看 PCD 截图更能支撑论文标题里的“非朗伯体干扰抑制”。

## 5. 当前仓库风险点

1. `evaluation_tools/` 原本为空，说明之前还没有真正可复现实验闭环。
2. 根目录原 `BENCHMARK_REPORT_CN.md` 文件内容已经被 NUL 字节覆盖，我已重写。
3. 当前视觉上游已切换到 `mirrorsentinel_visual_prior` 的 DA3 单目深度 prior；旧双目深度 wrapper 不再是论文主线。
4. 自采 rosbag 没有动捕 ground truth，因此轨迹只能写 no-GT loop/revisit consistency，不能写成 ATE。
5. `MirrorSentinel_paper.tex` 还只是草稿模板，不满足 ROBIO 完整论文要求。

## 6. 最小可投稿实验闭环

当前已经进一步拆成三条实验线：

1. M2DGR：公共 SLAM 精度，输出 ATE/RPE。
2. 3DRef：公共非朗伯感知，输出 mirror/glass/all-reflective IoU/F1。
3. 自采 bag：真实反射场景，无 GT 时输出 loop drift、revisit consistency、ghost rate、map thickness。

若时间紧，至少完成以下闭环：

1. 选择 3 到 5 条自采镜面/玻璃序列。
2. 每条序列提供 ground truth：
   - motion capture、total station、AprilTag/SLAM reference、或人工测量闭环误差均可，但要写清楚。
3. 跑 FAST-LIO2、LIO-SAM、Sentinel-LIO full、两个消融版本。
4. 用 `tum_trajectory_eval.py` 输出 ATE/RPE。
5. 手工标 100 到 300 帧反射 mask，或使用 3DRef 公共数据补 mask 精度。
6. 用 `mirror_segmentation_eval.py` 输出 IoU/F1。
7. 保存 PCD 并做 3 到 4 张地图对比图。
8. 整理 runtime：FPS、平均每帧耗时、GPU 型号、CPU 型号。

完成这些，论文实验主线才算从“系统 demo”进入“可审稿验证”的状态。

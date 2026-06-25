# ROBIO 投稿前 TODO 清单

## A. 必须完成

- [ ] 确认 ROS 运行环境。当前仓库是 ROS1/catkin 风格，launch 和 wrapper 写的是 Noetic；当前机器只有 `/opt/ros/humble`。需要准备 ROS1 Noetic 环境，或把系统完整迁到 ROS2。
- [ ] 修掉视觉节点绝对路径。`vfm_stereo_node.py` 和 `vfm_stereo_node_wrapper.sh` 里仍有 `/home/zjt/yugong_slam`、`/home/zjt/miniconda3`。
- [ ] 补 `/vfm/mirror_mask` 生成链路。主 LIO 已订阅 mask，但当前 stereo node 只发布 `/vfm/depth_image`。
- [ ] 给 `dataset/rosbag/2026-03-30-21-31-03.bag` 建立 metadata：场景、传感器、时长、topic、外参、是否有真值。
- [ ] 准备至少 3 条有 ground truth 的非朗伯序列：镜面、玻璃、混合反光/弱纹理。
- [ ] 用 `evaluation_tools/tum_trajectory_eval.py` 产出 ATE/RPE 表。
- [ ] 用 `evaluation_tools/mirror_segmentation_eval.py` 产出反射 mask 或点级反射检测表。
- [ ] 完成消融：无 mask、无 depth prior、硬删除点、软 confidence weighting、full model。
- [ ] 完成与 FAST-LIO2、LIO-SAM、LVI-SAM/R3LIVE 的对比。
- [ ] 整理 runtime：平均处理时间、最大处理时间、帧率、GPU/CPU 配置、显存占用。

## B. 论文实验图表

- [ ] 主结果表：各序列 ATE RMSE / RPE translational / RPE rotational。
- [ ] 消融表：证明视觉深度先验和 mirror mask 各自有效。
- [ ] 反射检测表：mirror/glass/other-reflective/all-reflective 的 IoU、F1、precision、recall。
- [ ] 地图质量图：同一场景中 FAST-LIO2 与 Sentinel-LIO 的 ghost points、墙体重复、反射伪结构对比。
- [ ] 轨迹图：top-down trajectory overlay，包含 ground truth、baseline、ours。
- [ ] 定性图：RGB、mask、depth、weighted point cloud、final map 五联图。
- [ ] 失败案例图：mask 漏检、强反射过曝、深度模型 hallucination、外参误差导致的误筛。

## C. 方法叙事需要补强

- [ ] 明确定义问题：非朗伯表面导致 LiDAR ghost points 和 scan-to-map residual 被污染。
- [ ] 明确本文贡献：
  - [ ] 一种视觉基础模型辅助的 mirror-aware LIO 框架。
  - [ ] 一种 depth-mask consistency 的点级置信度分配策略。
  - [ ] 一种软权重残差进入 ESKF/scan-to-map update 的实现。
  - [ ] 面向反射场景的 benchmark/数据与消融验证。
- [ ] 给出关键公式：
  - [ ] LiDAR 点投影到相机。
  - [ ] LiDAR range 与 visual depth 的一致性残差。
  - [ ] mask/boundary/depth-validity 到 confidence 的映射。
  - [ ] confidence 对 point-to-plane residual 和 Jacobian 的加权方式。
- [ ] 把 `MirrorSentinel.hpp` 中已有逻辑写成论文算法框。
- [ ] 给出复杂度和实时性分析。

## D. 工程复现

- [ ] 写根目录 README：安装、模型权重、TensorRT engine、运行 launch、评测命令。
- [ ] 提供 `requirements.txt` 或 conda `environment.yml`。
- [ ] 提供 `scripts/run_benchmark.sh`，一键跑某条 bag 并导出结果。
- [ ] 提供 `configs/` 中每条实验序列的外参和 topic mapping。
- [ ] 固定随机种子与模型版本。
- [ ] 保存每次实验的 commit/hash、参数 YAML、日志和输出 JSON。

## E. 当前仓库中最应优先修的文件

- [ ] `Sentinal-LIO_ws/src/fast_foundation_stereo_ros/scripts/vfm_stereo_node.py`
  - 绝对路径。
  - topic 写死。
  - 没有 mirror mask output。
  - 缺少参数化 baseline、engine path、model path。

- [ ] `Sentinal-LIO_ws/src/fast_foundation_stereo_ros/scripts/vfm_stereo_node_wrapper.sh`
  - 绝对路径。
  - ROS/conda 环境写死。
  - `LD_PRELOAD` 写死 libffi 版本。

- [ ] `Sentinal-LIO_ws/src/sentinel_lio/launch/run_sentinel_ouster.launch`
  - 默认启动 RViz，不适合 batch benchmark。
  - 默认启动 VFM 节点，不利于做 ablation。
  - 需要增加 `run_vfm`、`run_rviz`、`record_outputs` 等开关。

- [ ] `Sentinal-LIO_ws/src/sentinel_lio/config/sentinel_ouster.yaml`
  - 外参需要标明坐标系定义。
  - baseline 参数缺失，但 stereo node 会读取 `/sentinel/baseline`。
  - 需要为每条序列单独保存配置，避免覆盖。

- [ ] `evaluation_tools/`
  - 已补 ATE/RPE 和 3DRef 风格指标。
  - 仍需补地图 ghost metric 和 batch runner。

## F. 投稿前判断标准

可以进入写作定稿的最低标准：

- [ ] 至少 3 条非朗伯序列，且每条有真值或可辩护的 reference trajectory。
- [ ] 主方法在多数序列上 ATE/RPE 优于 FAST-LIO2 和 LIO-SAM。
- [ ] 消融能证明 full model 明显优于 no-mask/no-depth。
- [ ] mask 或点级反射检测有定量结果，不只展示可视化。
- [ ] runtime 仍接近在线，不是离线慢处理。
- [ ] 失败案例被诚实呈现，并有合理解释。

如果只剩很少时间，优先顺序是：

1. 跑通 ROS1 环境与 `/Odometry` 导出。
2. 拿到 ground truth 并出 ATE/RPE。
3. 补 `/vfm/mirror_mask`。
4. 做 full/no-mask/no-depth 消融。
5. 做 baseline 对比。
6. 画地图和轨迹图。
7. 最后才是大规模 polish。

# ROBIO 实验部署说明

## 1. 实验分工

第一阶段先把 **3DRef** 作为主评估数据集跑通。等方法在 3DRef 上成功后，再拉 M2DGR 小序列做泛化和有 GT 的轨迹评估。

完整论文阶段建议按三条线部署实验：

| 线 | 数据 | 角色 | 主指标 |
| --- | --- | --- | --- |
| 公共 SLAM | M2DGR | 证明轨迹精度，有 reference/GT 可算 | ATE, RPE |
| 公共非朗伯感知 | 3DRef | 证明 mirror/glass/reflective prior 有效 | IoU, F1, Precision, Recall |
| 自采真实场景 | `dataset/rosbag/2026-03-30-21-31-03.bag` | 证明真实镜面/玻璃场景下地图污染减少 | loop drift, revisit consistency, ghost rate, map thickness |

自采数据没有动捕时，不作为唯一 ATE/RPE 证据。论文里应称为 real-world reflective-scene experiment，而不是 ground-truth trajectory benchmark。

## 2. 目录规范

建议按以下路径落数据和结果：

```text
dataset/
  public/
    M2DGR/
    3DRef/
  rosbag/
    2026-03-30-21-31-03.bag

evaluation_tools/
  data/
    m2dgr/
      groundtruth_tum/
        hall_01.txt
    self_collected/
      revisit_pairs.txt
  results/
    m2dgr/
      trajectories/
        sentinel_full/hall_01.txt
      eval/
        summary.csv
    3dref/
      predictions/
        alllabel/
      eval/
        summary.csv
    self_collected/
      trajectories/
        sentinel_full/2026-03-30-21-31-03.txt
      eval/
        sentinel_full/no_gt_metrics.json
```

公共实验计划在：

```text
evaluation_tools/configs/benchmark_plan.yaml
```

新增序列或方法时，先改这个 YAML。

## 3. M2DGR 部署

M2DGR 用来出论文主轨迹表。推荐先跑：

```text
hall_01
hall_02
lift_01
door_01
room_01
```

每个方法导出 TUM 格式：

```text
timestamp tx ty tz qx qy qz qw
```

路径示例：

```text
evaluation_tools/results/m2dgr/trajectories/sentinel_full/hall_01.txt
evaluation_tools/results/m2dgr/trajectories/fast_lio2/hall_01.txt
evaluation_tools/data/m2dgr/groundtruth_tum/hall_01.txt
```

批量评测：

```bash
python3 evaluation_tools/batch_tum_eval.py \
  --config evaluation_tools/configs/benchmark_plan.yaml
```

输出：

```text
evaluation_tools/results/m2dgr/eval/summary.csv
evaluation_tools/results/m2dgr/eval/summary.json
```

论文主表列：

```text
Method | ATE RMSE | ATE Mean | RPE Trans RMSE | RPE Rot Mean | Runtime
```

## 4. 3DRef 部署

3DRef 用来证明非朗伯区域识别/先验有效。

下载主 benchmark 包：

```bash
bash dataset/download_3dref.sh reflection
bash dataset/download_3dref.sh extract-reflection
bash dataset/download_3dref.sh check
```

把模型预测 mask 放到：

```text
evaluation_tools/results/3dref/predictions/mirror
evaluation_tools/results/3dref/predictions/glass
evaluation_tools/results/3dref/predictions/others
evaluation_tools/results/3dref/predictions/alllabel
```

按配置中��� GT 路径检查：

```text
dataset/public/3DRef/reflection/rgb/mirror/test/mask
dataset/public/3DRef/reflection/rgb/glass/test/mask
dataset/public/3DRef/reflection/rgb/others/test/mask
dataset/public/3DRef/reflection/rgb/alllabel/test/mask
```

批量评测：

```bash
python3 evaluation_tools/batch_reflection_eval.py \
  --config evaluation_tools/configs/benchmark_plan.yaml
```

输出：

```text
evaluation_tools/results/3dref/eval/summary.csv
evaluation_tools/results/3dref/eval/<split>.json
```

论文表列：

```text
Class | IoU | F1 | Precision | Recall
```

## 5. 自采真实场景实验

自采 bag：

```text
dataset/rosbag/2026-03-30-21-31-03.bag
```

当前该 bag 的 ROS1 index 看起来损坏，`rosbags-convert` 报：

```text
Bag index looks damaged
```

需要先在 ROS1 环境中修复：

```bash
source /opt/ros/noetic/setup.bash
rosbag reindex dataset/rosbag/2026-03-30-21-31-03.bag
```

如果 `rosbag reindex` 生成 `.orig.bag`，保留原始文件，不要删除。

每个方法跑完后导出 TUM：

```text
evaluation_tools/results/self_collected/trajectories/fast_lio2/2026-03-30-21-31-03.txt
evaluation_tools/results/self_collected/trajectories/sentinel_no_mask/2026-03-30-21-31-03.txt
evaluation_tools/results/self_collected/trajectories/sentinel_no_depth/2026-03-30-21-31-03.txt
evaluation_tools/results/self_collected/trajectories/sentinel_full/2026-03-30-21-31-03.txt
```

无 GT 轨迹一致性：

```bash
python3 evaluation_tools/no_gt_trajectory_eval.py \
  --trajectory evaluation_tools/results/self_collected/trajectories/sentinel_full/2026-03-30-21-31-03.txt \
  --output-json evaluation_tools/results/self_collected/eval/sentinel_full/no_gt_metrics.json \
  --output-csv evaluation_tools/results/self_collected/eval/sentinel_full/no_gt_metrics.csv
```

如果路线闭环，默认用首尾 pose 计算 loop drift。若起止不是首尾，可以指定时间：

```bash
--start-time 1774877001.25 --end-time 1774877102.80
```

重访一致性需要人工标时间对：

```text
evaluation_tools/data/self_collected/revisit_pairs.txt
```

格式：

```text
# timestamp_a timestamp_b
1774877001.25 1774877050.92
1774877013.40 1774877062.11
```

然后运行：

```bash
python3 evaluation_tools/no_gt_trajectory_eval.py \
  --trajectory evaluation_tools/results/self_collected/trajectories/sentinel_full/2026-03-30-21-31-03.txt \
  --revisit-pairs evaluation_tools/data/self_collected/revisit_pairs.txt \
  --output-json evaluation_tools/results/self_collected/eval/sentinel_full/no_gt_metrics.json
```

## 6. 自采实验论文表

建议表格：

```text
Method | Loop Trans. Error | Loop Rot. Error | Drift Ratio | Revisit Trans. Error | Ghost Rate | Map Thickness
```

其中：

1. Loop drift 和 revisit consistency 由 `no_gt_trajectory_eval.py` 得到。
2. Ghost rate 需要标注镜面/玻璃平面后统计。
3. Map thickness 需要对墙面/镜面附近平面拟合点到平面的标准差。
4. Runtime 从 ROS log 或单独 timing node 统计。

论文中建议写：

```text
Since motion-capture ground truth is unavailable in the self-collected reflective scene,
we use it as a real-world consistency and mapping-quality experiment rather than an
absolute trajectory benchmark.
```

## 7. 消融设置

至少保留：

```text
fast_lio2
sentinel_no_mask
sentinel_no_depth
sentinel_full
```

如果时间允许，再加：

```text
sentinel_hard_remove
sentinel_soft_weight_only
sentinel_depth_only
```

每个实验必须保存：

```text
trajectory txt
parameter yaml
runtime log
frame_stats
final pcd
```

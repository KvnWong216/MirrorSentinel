# MirrorSentinel 开源发布清单

这份清单用于把当前科研仓库整理成 GitHub 可发布版本。

## 1. 必须保留

代码：

```text
Sentinal-LIO_ws/src/sentinel_lio/
Sentinal-LIO_ws/src/mirrorsentinel_visual_prior/
evaluation_tools/*.py
evaluation_tools/baselines/*.py
evaluation_tools/launch/*.py
evaluation_tools/reconstruction/*.py
evaluation_tools/pcl_room_annotator/
scripts/*.sh
```

配置和轻量数据：

```text
Sentinal-LIO_ws/src/sentinel_lio/config/
Sentinal-LIO_ws/src/sentinel_lio/rviz_cfg/
evaluation_tools/configs/
evaluation_tools/data/annotations/
evaluation_tools/data/gt_reference/
dataset/download_3dref.sh
dataset/README_DATASETS_CN.md
```

文档：

```text
README.md
DATASET_AND_LARGE_FILES.md
SLAM_EVAL_PROTOCOL_CN.md
docs/TEAM_HANDOFF_CN.md
docs/ROBIO_PAPER_DRAFT_CN.md
docs/MIRROR_SLAM_RESEARCH_REVIEW_CN.md
docs/REFLECTION_SLAM_BASELINES_CN.md
evaluation_tools/README.md
Sentinal-LIO_ws/src/sentinel_lio/README.md
```

## 2. 不要提交

本地数据：

```text
dataset/*.bag
dataset/rosbag2/
dataset/public/
dataset/reconstruction/
```

运行结果和中间产物：

```text
evaluation_tools/results/
**/build/
**/install/
**/log/
**/__pycache__/
*.pcd
*.ply
*.db3
*.mcap
*.log
```

模型和第三方大仓：

```text
third_party/
baselines/
*.pth
*.pt
*.ckpt
*.onnx
*.engine
*.plan
*.bin
```

## 3. 发布前检查

```bash
find . -name '__pycache__' -o -name '*.pyc' -o -name '*.log'
find . -path './dataset/rosbag2/*' -o -name '*.bag' -o -name '*.db3'
find . -path './evaluation_tools/results/*'
bash scripts/build_ros2.sh
python3 evaluation_tools/summarize_self_collected_methods.py
```

如果没有本地数据，`summarize_self_collected_methods.py` 可能缺少结果文件，这是正常的。GitHub 版本的核心要求是源码可构建、文档能说明如何恢复数据和重跑实验。

## 4. 推荐发布节奏

1. 先发布代码和文档，不发布数据。
2. 用 README 写清楚数据不在 GitHub，需要从内部盘或云盘恢复。
3. 论文接收或投稿前，再整理一个 frozen release，包括：

```text
source code tag
small annotations
result summary CSV/MD
dataset download instructions
environment notes
```

4. 大 bag、模型权重、完整 PCD 地图放 Zenodo/百度云/实验室服务器，不放 Git。

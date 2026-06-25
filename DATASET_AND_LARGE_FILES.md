# Dataset And Large File Policy

This repository is meant to be cloneable and buildable without the local
experiment cache.  It intentionally excludes datasets, generated maps, model
checkpoints, third-party reconstruction repositories, and large baseline
workspaces.

Excluded from GitHub:

```text
dataset/rosbag/
dataset/rosbag2/
dataset/public/
dataset/reconstruction/
evaluation_tools/results/
third_party/
baselines/
Sentinal-LIO_ws/models/
*.bag
*.pcd
*.ckpt
*.pth
*.onnx
*.engine
```

The core ROS2 MirrorSentinel system builds without these files:

```bash
scripts/build_ros2.sh
```

To reproduce the current self-collected experiments, restore a ROS2 bag under:

```text
dataset/rosbag2/
```

The local prototype bag used in the current notes is:

```text
dataset/rosbag2/2026-03-30-21-31-03_rescued
```

For auxiliary perception and reconstruction experiments, follow:

```text
dataset/README_DATASETS_CN.md
evaluation_tools/reconstruction/README.md
```

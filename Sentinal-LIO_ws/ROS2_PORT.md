# MirrorSentinel ROS2 Workspace

This workspace is organized for ROS2 Humble.  The old ROS1 catkin build/devel
artifacts and ROS1 launch entry points are not part of the maintained path.

Current usable packages:

```text
sentinel_lio
  ROS2 C++ LiDAR-inertial mapping node, executable `fastlio_mapping`.

mirrorsentinel_visual_prior
  ROS2 Python visual-prior nodes for DA3 depth, reflection mask inference, and
  replayed mask priors.
```

## Build

```bash
cd <MirrorSentinel>/Sentinal-LIO_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select sentinel_lio mirrorsentinel_visual_prior
```

From the repository root, prefer:

```bash
bash scripts/build_ros2.sh
```

## Run Mapping

```bash
cd <MirrorSentinel>/Sentinal-LIO_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ROS_LOG_DIR=/tmp/ros2_launch_log ros2 launch sentinel_lio run_sentinel_ouster.launch.py
```

## Enable DA3 Depth Prior

```bash
export SENTINEL_LIO_CONDA_SH=~/anaconda3/etc/profile.d/conda.sh
export SENTINEL_LIO_CONDA_ENV=MirrSLAM_rtdepth
ROS_LOG_DIR=/tmp/ros2_launch_log ros2 launch sentinel_lio run_sentinel_ouster.launch.py \
  enable_da3_depth:=true \
  da3_backend:=module \
  da3_model:=depth_anything_3.api:DepthAnything3 \
  da3_checkpoint:=depth-anything/DA3Metric-Large
```

The DA3 node publishes:

```text
/vfm/depth_image
```

The `/vfm` prefix is kept as a generic visual-foundation-model prior namespace;
it does not imply any stereo-depth dependency.

## Optional Reflection Mask Prior

Directory replay / zero-mask ablation:

```bash
ros2 launch sentinel_lio run_sentinel_ouster.launch.py \
  enable_mask_prior:=true \
  mask_mode:=directory \
  mask_image_topic:=/zed2/zed_node/left/image_rect_color \
  mask_dir:=<MirrorSentinel>/dataset/public/3DRef/reflection/rgb/alllabel/test/mask
```

Learned lightweight mask head:

```bash
ros2 launch sentinel_lio run_sentinel_ouster.launch.py \
  enable_reflection_mask:=true \
  reflection_mask_checkpoint:=<checkpoint.pth>
```

## Default Topics

Configured in `src/sentinel_lio/config/sentinel_ouster.yaml`:

```text
/ouster/points
/ouster/imu
/zed2/zed_node/left/image_rect_color
/vfm/depth_image
/vfm/mirror_mask
/cloud_registered
/Odometry
/path
/mirror_sentinel/frame_stats
```

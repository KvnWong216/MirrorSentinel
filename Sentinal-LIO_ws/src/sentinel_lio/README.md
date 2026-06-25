# MirrorSentinel / Sentinel-LIO ROS2

This package is the ROS2 Humble mapping front end for MirrorSentinel: a
Fast-LIO based LiDAR-inertial mapper with visual depth and
non-Lambertian/mirror confidence gating for the YuGong SLAM research pipeline.

The old ROS1 `catkin_make` and `roslaunch` entry points have been removed from this workspace. Use `colcon` and `ros2 launch` only.

## Package Layout

- Mapping executable: `sentinel_lio/fastlio_mapping`
- Main launch file: `launch/run_sentinel_ouster.launch.py`
- Main parameter file: `config/sentinel_ouster.yaml`
- Custom message: `msg/Pose6D.msg`
- Visual gating logic: `include/MirrorSentinel.hpp`

## Build

Build from the workspace root:

```bash
cd <MirrorSentinel>
scripts/build_ros2.sh
```

If Anaconda is active, use a clean environment so CMake links against system Boost/OpenCV/PCL:

```bash
cd <MirrorSentinel>/Sentinal-LIO_ws
env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_EXE -u CONDA_PYTHON_EXE \
  -u CONDA_PROMPT_MODIFIER -u CONDA_SHLVL \
  PATH=/opt/ros/humble/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages \
  AMENT_PREFIX_PATH=/opt/ros/humble \
  CMAKE_PREFIX_PATH=/opt/ros/humble \
  LD_LIBRARY_PATH=/opt/ros/humble/opt/rviz_ogre_vendor/lib:/opt/ros/humble/lib/x86_64-linux-gnu:/opt/ros/humble/lib \
  /usr/bin/bash -lc 'source /opt/ros/humble/setup.bash && colcon build \
    --symlink-install --packages-select sentinel_lio fast_foundation_stereo_ros \
    --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    -DBoost_DIR=/usr/lib/x86_64-linux-gnu/cmake/Boost-1.74.0 \
    -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4 \
    -DPCL_DIR=/usr/lib/x86_64-linux-gnu/cmake/pcl'
```

## Run

```bash
cd <MirrorSentinel>/Sentinal-LIO_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ROS_LOG_DIR=/tmp/ros2_launch_log ros2 launch sentinel_lio run_sentinel_ouster.launch.py
```

Launch with the MirrorSentinel RViz panel:

```bash
cd <MirrorSentinel>
scripts/mirrorsentinel_rviz.sh
```

Or directly from the workspace:

```bash
cd <MirrorSentinel>/Sentinal-LIO_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch sentinel_lio run_sentinel_ouster.launch.py rviz:=true
```

Then play a bag from another shell:

```bash
cd <MirrorSentinel>
scripts/mirrorsentinel_play_bag.sh
```

Enable the Fast-FoundationStereo TensorRT depth node after engine files exist:

```bash
export SENTINEL_LIO_CONDA_SH=~/anaconda3/etc/profile.d/conda.sh
export SENTINEL_LIO_CONDA_ENV=Sentinel-LIO
ROS_LOG_DIR=/tmp/ros2_launch_log ros2 launch sentinel_lio run_sentinel_ouster.launch.py enable_vfm:=true
```

## 3DRef Mask Prior Smoke Test

Use this when you want to run our Sentinel-LIO mask-consumption path before deploying RGB reflection baselines. It publishes 3DRef mask files as `/vfm/mirror_mask`, synchronized by image timestamp.

```bash
cd <MirrorSentinel>/Sentinal-LIO_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch sentinel_lio run_sentinel_ouster.launch.py \
  enable_mask_prior:=true \
  mask_mode:=directory \
  mask_image_topic:=/zed2/zed_node/left/image_rect_color \
  mask_dir:=<MirrorSentinel>/dataset/public/3DRef/reflection/rgb/alllabel/test/mask \
  mask_match_tolerance:=0.001
```

For ablation, use `mask_mode:=zeros` to publish an all-zero mask on the same topic. Do not enable both `enable_vfm:=true` and `enable_mask_prior:=true` with the same `mask_topic` unless you intentionally want two mask publishers.

If `rclpy._rclpy_pybind11` fails to import, you are probably running ROS2 with Anaconda's `python3`. Start from a clean shell or use the clean environment shown in this README's build section.

## Default Topics

Configured in `config/sentinel_ouster.yaml`:

- LiDAR: `/ouster/points`
- IMU: `/ouster/imu`
- RGB: `/zed2/zed_node/left/image_rect_color`
- VFM depth: `/vfm/depth_image`
- mirror/non-Lambertian mask: `/vfm/mirror_mask`
- registered map cloud: `/cloud_registered`
- odometry: `/Odometry`
- path: `/path`
- frame statistics: `/mirror_sentinel/frame_stats`

## Current Scope

- Standard `sensor_msgs/msg/PointCloud2` LiDAR input is supported.
- Ouster is the maintained default configuration for the current benchmark work.
- Livox `CustomMsg` support is intentionally disabled until `livox_ros_driver2` is added and tested.

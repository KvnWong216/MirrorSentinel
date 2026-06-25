# Sentinel-LIO ROS2 Workspace

This workspace is now organized as a ROS2 Humble workspace. The old ROS1 catkin build/devel artifacts and ROS1 launch files have been removed. Current usable packages are:

- `sentinel_lio`: ROS2 C++ mapping node, executable `fastlio_mapping`.
- `fast_foundation_stereo_ros`: ROS2 Python Fast-FoundationStereo/TensorRT wrapper, executables `vfm_stereo_node.py`, `vfm_stereo_node_wrapper.sh`, and `build_trt.py`.

## Build

Use standard ROS2 build directories: `build/`, `install/`, and `log/`.

```bash
cd /home/wangyg/yugong_research/yugong_slam/Sentinal-LIO_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select sentinel_lio fast_foundation_stereo_ros
```

If Anaconda contaminates CMake/Boost/OpenCV/PCL, build in a clean environment:

```bash
cd /home/wangyg/yugong_research/yugong_slam/Sentinal-LIO_ws
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

## Run Mapping

```bash
cd /home/wangyg/yugong_research/yugong_slam/Sentinal-LIO_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ROS_LOG_DIR=/tmp/ros2_launch_log ros2 launch sentinel_lio run_sentinel_ouster.launch.py
```

Enable Fast-FoundationStereo TensorRT depth publishing only after the engine files exist:

```bash
export SENTINEL_LIO_CONDA_SH=~/anaconda3/etc/profile.d/conda.sh
export SENTINEL_LIO_CONDA_ENV=Sentinel-LIO
ROS_LOG_DIR=/tmp/ros2_launch_log ros2 launch sentinel_lio run_sentinel_ouster.launch.py enable_vfm:=true
```

Override model location when needed:

```bash
ROS_LOG_DIR=/tmp/ros2_launch_log ros2 launch sentinel_lio run_sentinel_ouster.launch.py \
  enable_vfm:=true \
  vfm_model_root:=/path/to/Fast-FoundationStereo
```

The VFM node also honors `SENTINEL_LIO_MODEL_ROOT`. If neither launch argument nor environment variable is set, it searches upward from the installed/source script path for `models/Fast-FoundationStereo`.

## TensorRT Engines

Default ONNX/engine directory:

```text
/home/wangyg/yugong_research/yugong_slam/Sentinal-LIO_ws/models/Fast-FoundationStereo/output
```

Build engines from ONNX files:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run fast_foundation_stereo_ros build_trt.py
```

Optional overrides:

```bash
ros2 run fast_foundation_stereo_ros build_trt.py \
  --model-root /path/to/Fast-FoundationStereo \
  --onnx-dir /path/to/onnx_dir \
  --engine-dir /path/to/engine_dir
```

## Default Topics

Configured in `src/sentinel_lio/config/sentinel_ouster.yaml`:

- LiDAR: `/ouster/points`
- IMU: `/ouster/imu`
- RGB: `/zed2/zed_node/left/image_rect_color`
- VFM depth: `/vfm/depth_image`
- mirror/non-Lambertian mask: `/vfm/mirror_mask`
- mapping cloud: `/cloud_registered`
- odometry: `/Odometry`
- path: `/path`
- Sentinel frame stats: `/mirror_sentinel/frame_stats`

## Current Limitations

- Standard `sensor_msgs/msg/PointCloud2` LiDAR input is supported.
- Ouster is the maintained default configuration for the current benchmark work.
- Livox `CustomMsg` support is disabled until `livox_ros_driver2` is added and tested.
- Runtime validation still needs a ROS2 bag with matching LiDAR, IMU, image, depth, and mask topics.

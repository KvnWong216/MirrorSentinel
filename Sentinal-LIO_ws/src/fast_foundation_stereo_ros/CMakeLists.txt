cmake_minimum_required(VERSION 3.8)
project(fast_foundation_stereo_ros)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)

install(PROGRAMS
  scripts/build_trt.py
  scripts/da3_depth_node.py
  scripts/export_reflection_seg_onnx.py
  scripts/mask_prior_node.py
  scripts/reflection_mask_node.py
  scripts/run_reflection_seg_infer.py
  scripts/train_reflection_seg_3dref.py
  scripts/vfm_stereo_node.py
  scripts/vfm_stereo_node_wrapper.sh
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  scripts/da3_adapter.py
  scripts/mask_prior_utils.py
  scripts/reflection_seg_model.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()

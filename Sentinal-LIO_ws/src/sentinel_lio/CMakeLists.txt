cmake_minimum_required(VERSION 3.8)
project(sentinel_lio)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-std=c++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_definitions(-DROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/")
add_definitions(-DMP_EN)

include(ProcessorCount)
ProcessorCount(N)
if(N GREATER 4)
  add_definitions(-DMP_PROC_NUM=3)
elseif(N GREATER 3)
  add_definitions(-DMP_PROC_NUM=2)
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters kdtree)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP QUIET)
find_package(cv_bridge REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Pose6D.msg"
  DEPENDENCIES geometry_msgs
)

include_directories(
  include
  src
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(fastlio_mapping
  src/laserMapping.cpp
  include/ikd-Tree/ikd_Tree.cpp
  src/preprocess.cpp
)

ament_target_dependencies(fastlio_mapping
  rclcpp
  geometry_msgs
  nav_msgs
  sensor_msgs
  std_msgs
  pcl_conversions
  cv_bridge
  tf2
  tf2_ros
  tf2_geometry_msgs
)

target_link_libraries(fastlio_mapping
  ${PCL_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

if(OpenMP_CXX_FOUND)
  target_link_libraries(fastlio_mapping OpenMP::OpenMP_CXX)
endif()

rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} rosidl_typesupport_cpp)
target_link_libraries(fastlio_mapping "${cpp_typesupport_target}")

install(TARGETS fastlio_mapping
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config rviz_cfg
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
  FILES_MATCHING PATTERN "*.launch.py"
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()

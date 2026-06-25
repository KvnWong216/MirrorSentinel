cmake_minimum_required(VERSION 3.16)
project(pcl_room_annotator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PCL 1.10 REQUIRED COMPONENTS common io visualization filters)
find_package(X11 REQUIRED)

add_executable(pcl_room_annotator src/pcl_room_annotator.cpp)
target_include_directories(pcl_room_annotator PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(pcl_room_annotator PRIVATE ${PCL_LIBRARIES} ${X11_LIBRARIES})
target_compile_definitions(pcl_room_annotator PRIVATE ${PCL_DEFINITIONS})

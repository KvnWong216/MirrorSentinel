from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    repo_root = Path(__file__).resolve().parents[2]
    default_params = repo_root / "evaluation_tools" / "configs" / "lio_sam_self_collected.yaml"

    params_file = LaunchConfiguration("params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")

    common_parameters = [params_file, {"use_sim_time": use_sim_time}]

    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=str(default_params)),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="lio_sam_map_to_odom_static",
                arguments=["0", "0", "0", "0", "0", "0", "map", "odom"],
                parameters=[{"use_sim_time": use_sim_time}],
                output="screen",
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="lio_sam_base_to_os_sensor_static",
                arguments=["0", "0", "0", "0", "0", "0", "base_link", "os_sensor"],
                parameters=[{"use_sim_time": use_sim_time}],
                output="screen",
            ),
            Node(
                package="lio_sam",
                executable="lio_sam_imuPreintegration",
                name="lio_sam_imuPreintegration",
                parameters=common_parameters,
                output="screen",
            ),
            Node(
                package="lio_sam",
                executable="lio_sam_imageProjection",
                name="lio_sam_imageProjection",
                parameters=common_parameters,
                output="screen",
            ),
            Node(
                package="lio_sam",
                executable="lio_sam_featureExtraction",
                name="lio_sam_featureExtraction",
                parameters=common_parameters,
                output="screen",
            ),
            Node(
                package="lio_sam",
                executable="lio_sam_mapOptimization",
                name="lio_sam_mapOptimization",
                parameters=common_parameters,
                output="screen",
            ),
        ]
    )

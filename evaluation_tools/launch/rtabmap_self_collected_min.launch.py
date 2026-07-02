from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():
    repo_root = Path(__file__).resolve().parents[2]
    dense_flag_script = repo_root / "evaluation_tools" / "ros2_pointcloud_dense_flag.py"

    use_sim_time = LaunchConfiguration("use_sim_time")
    frame_id = LaunchConfiguration("frame_id")
    map_frame_id = LaunchConfiguration("map_frame_id")
    lidar_topic = LaunchConfiguration("lidar_topic")
    imu_topic = LaunchConfiguration("imu_topic")
    database_path = LaunchConfiguration("database_path")
    expected_update_rate = LaunchConfiguration("expected_update_rate")
    assembling_time = LaunchConfiguration("assembling_time")
    qos = LaunchConfiguration("qos")

    icp_params = {
        "use_sim_time": use_sim_time,
        "frame_id": frame_id,
        "odom_frame_id": "icp_odom",
        "publish_tf": True,
        "wait_for_transform": 0.2,
        "expected_update_rate": expected_update_rate,
        "guess_frame_id": "",
        "deskewing": False,
        "qos": qos,
        "scan_cloud_max_points": 30000,
        "scan_downsampling_step": 1,
        "scan_voxel_size": 0.20,
        "scan_normal_k": 0,
        "scan_normal_radius": 0.0,
        "Icp/PointToPlane": "false",
        "Icp/Iterations": "10",
        "Icp/VoxelSize": "0",
        "Icp/Epsilon": "0.001",
        "Icp/PointToPlaneK": "0",
        "Icp/PointToPlaneRadius": "0",
        "Icp/MaxTranslation": "3",
        "Icp/MaxCorrespondenceDistance": "1.0",
        "Icp/Strategy": "1",
        "Icp/OutlierRatio": "0.7",
        "Icp/CorrespondenceRatio": "0.01",
        "Odom/ScanKeyFrameThr": "0.4",
        "OdomF2M/ScanSubtractRadius": "0.20",
        "OdomF2M/ScanMaxSize": "8000",
        "OdomF2M/BundleAdjustment": "false",
    }

    slam_params = {
        "use_sim_time": use_sim_time,
        "frame_id": frame_id,
        "map_frame_id": map_frame_id,
        "publish_tf": True,
        "subscribe_depth": False,
        "subscribe_rgb": False,
        "subscribe_rgbd": False,
        "subscribe_stereo": False,
        "subscribe_scan": False,
        "subscribe_scan_cloud": True,
        "subscribe_odom_info": True,
        "odom_sensor_sync": True,
        "wait_for_transform": 0.2,
        "database_path": database_path,
        "qos_scan": 2,
        "qos_odom": 2,
        "topic_queue_size": 40,
        "sync_queue_size": 40,
        "RGBD/CreateOccupancyGrid": "false",
        "Mem/NotLinkedNodesKept": "false",
        "Mem/STMSize": "30",
        "Rtabmap/DetectionRate": "0",
        "RGBD/AngularUpdate": "0.05",
        "RGBD/LinearUpdate": "0.05",
        "RGBD/ProximityMaxGraphDepth": "0",
        "RGBD/ProximityPathMaxNeighbors": "1",
        "Reg/Strategy": "1",
        "Icp/PointToPlane": "false",
        "Icp/Iterations": "10",
        "Icp/VoxelSize": "0.20",
        "Icp/Epsilon": "0.001",
        "Icp/PointToPlaneK": "0",
        "Icp/PointToPlaneRadius": "0",
        "Icp/MaxTranslation": "3",
        "Icp/MaxCorrespondenceDistance": "1.0",
        "Icp/CorrespondenceRatio": "0.2",
        "delete_db_on_start": True,
        "latch": False,
    }

    assembler_params = {
        "use_sim_time": use_sim_time,
        "assembling_time": assembling_time,
        "fixed_frame_id": "",
        "qos": qos,
        "qos_odom": 2,
        "topic_queue_size": 30,
        "sync_queue_size": 30,
        "range_min": 1.0,
        "voxel_size": 0.0,
    }

    map_assembler_params = {
        "use_sim_time": use_sim_time,
        "rtabmap": "rtabmap",
        "Grid/FromDepth": "false",
        "Grid/RangeMin": "1.0",
        "Grid/NormalsSegmentation": "false",
    }

    container = ComposableNodeContainer(
        name="rtabmap_self_collected_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        output="screen",
        composable_node_descriptions=[
            ComposableNode(
                package="rtabmap_odom",
                plugin="rtabmap_odom::ICPOdometry",
                name="icp_odometry",
                parameters=[icp_params],
                remappings=[
                    ("odom", "icp_odom"),
                    ("odom_info", "odom_info"),
                    ("scan_cloud", "/ouster/points/sanitized"),
                    ("imu", imu_topic),
                ],
            ),
            ComposableNode(
                package="rtabmap_util",
                plugin="rtabmap_util::PointCloudAssembler",
                name="point_cloud_assembler",
                parameters=[assembler_params],
                remappings=[
                    ("cloud", "/ouster/points/sanitized"),
                    ("odom", "icp_odom"),
                    ("odom_info", "odom_info"),
                ],
            ),
            ComposableNode(
                package="rtabmap_slam",
                plugin="rtabmap_slam::CoreWrapper",
                name="rtabmap",
                parameters=[slam_params],
                remappings=[
                    ("scan_cloud", "assembled_cloud"),
                    ("odom", "icp_odom"),
                    ("odom_info", "odom_info"),
                    ("imu", imu_topic),
                ],
            ),
            ComposableNode(
                package="rtabmap_util",
                plugin="rtabmap_util::MapAssembler",
                name="map_assembler",
                parameters=[map_assembler_params],
            ),
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("frame_id", default_value="os_sensor"),
            DeclareLaunchArgument("map_frame_id", default_value="map"),
            DeclareLaunchArgument("lidar_topic", default_value="/ouster/points"),
            DeclareLaunchArgument("imu_topic", default_value="/ouster/imu"),
            DeclareLaunchArgument("database_path", default_value="/tmp/rtabmap_self_collected.db"),
            DeclareLaunchArgument("expected_update_rate", default_value="15.0"),
            DeclareLaunchArgument("assembling_time", default_value="1.0"),
            DeclareLaunchArgument("qos", default_value="2"),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="rtabmap_identity_os_sensor_to_os_imu",
                output="screen",
                arguments=["0", "0", "0", "0", "0", "0", "os_sensor", "os_imu"],
            ),
            ExecuteProcess(
                cmd=[
                    "/usr/bin/python3",
                    str(dense_flag_script),
                    "--input-topic",
                    "/ouster/points",
                    "--output-topic",
                    "/ouster/points/sanitized",
                    "--compact-xyz",
                    "--stride",
                    "3",
                    "--range-min",
                    "1.0",
                    "--range-max",
                    "25.0",
                    "--max-points",
                    "30000",
                    "--use-sim-time",
                ],
                output="screen",
            ),
            container,
        ]
    )

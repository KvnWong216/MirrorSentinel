// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <limits>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>
#include <so3_math.h>
#include <rclcpp/rclcpp.hpp>
#include "ros2_compat.hpp"
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/vector3.hpp>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <cv_bridge/cv_bridge.h>
#include "MirrorSentinel.hpp"
#include <std_msgs/msg/float32_multi_array.hpp>

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic, depth_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;
bool   current_explicit_mask_enabled = true;
bool   map_depth_gate_enable = true;
bool   map_mask_gate_enable = true;
double map_depth_ghost_margin_abs = 0.80;
double map_depth_ghost_margin_rel = 0.03;
double map_mask_foreground_keep_margin = 0.20;
bool   map_mask_require_depth_confirmation = true;
bool   map_mask_invalid_depth_reject = false;
bool   map_depth_require_calibration = true;
bool   map_gate_apply_to_ikdtree = false;
bool   map_history_cleanup_enable = false;
bool   map_accumulated_cleanup_enable = false;
bool   map_export_ikdtree = false;
bool   map_export_apply_history_cleanup = false;
int    map_history_cleanup_period = 10;
double map_history_cleanup_range = 8.0;
double map_history_cleanup_export_radius = 0.35;
bool   global_depth_prior_enable = false;
bool   frontend_confidence_enable = false;
bool   rgb_subscribe_enable = false;
bool   depth_calibration_enable = true;
bool   depth_calibration_use_non_mask = true;
int    depth_calibration_min_points = 50;
double depth_calibration_scale_min = 0.50;
double depth_calibration_scale_max = 2.00;
double depth_calibration_max_raw_residual = 1.50;
int lidar_type;

vector<vector<int>>  pointSearchInd_surf; 
vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
PointCloudXYZI::Ptr map_history_deleted_markers(new PointCloudXYZI());
PointCloudXYZI::Ptr map_history_anchor_markers(new PointCloudXYZI());
PointCloudXYZI::Ptr map_history_surface_anchors(new PointCloudXYZI());

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;
std::shared_ptr<MirrorSentinel> sentinel_ptr;

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

using PointCloudPublisher = rclcpp::Publisher<sensor_msgs::msg::PointCloud2>;
using OdometryPublisher = rclcpp::Publisher<nav_msgs::msg::Odometry>;
using PathPublisher = rclcpp::Publisher<nav_msgs::msg::Path>;
using StatsPublisher = rclcpp::Publisher<std_msgs::msg::Float32MultiArray>;

std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

bool ensure_directory(const std::string& dir_path)
{
    struct stat info;
    if (stat(dir_path.c_str(), &info) == 0 && S_ISDIR(info.st_mode))
    {
        return true;
    }

    if (mkdir(dir_path.c_str(), 0777) == 0 || errno == EEXIST)
    {
        return true;
    }

    ROS_WARN("Failed to create directory: %s", dir_path.c_str());
    return false;
}

PointCloudXYZI::Ptr export_map_cloud_for_save()
{
    if (map_export_ikdtree && ikdtree.Root_Node != nullptr)
    {
        PointVector storage;
        ikdtree.flatten(ikdtree.Root_Node, storage, NOT_RECORD);
        PointCloudXYZI::Ptr cloud(new PointCloudXYZI());
        cloud->reserve(storage.size());
        for (const auto& pt : storage)
        {
            cloud->push_back(pt);
        }
        return cloud;
    }

    if (map_export_apply_history_cleanup && map_history_deleted_markers && !map_history_deleted_markers->empty() &&
        pcl_wait_save && !pcl_wait_save->empty())
    {
        pcl::KdTreeFLANN<PointType> deleted_tree;
        deleted_tree.setInputCloud(map_history_deleted_markers);
        PointCloudXYZI::Ptr filtered(new PointCloudXYZI());
        filtered->reserve(pcl_wait_save->size());
        std::vector<int> nearest_indices;
        std::vector<float> nearest_distances;
        const double radius = std::max(0.01, map_history_cleanup_export_radius);
        for (const auto& pt : pcl_wait_save->points)
        {
            nearest_indices.clear();
            nearest_distances.clear();
            if (deleted_tree.radiusSearch(pt, radius, nearest_indices, nearest_distances, 1) <= 0)
            {
                filtered->push_back(pt);
            }
        }
        ROS_INFO("[Sentinel] export history cleanup kept %zu/%zu accumulated map points using %zu markers radius=%.3f",
                 filtered->size(), pcl_wait_save->size(), map_history_deleted_markers->size(), radius);
        return filtered;
    }

    return pcl_wait_save;
}

bool write_pcd_if_nonempty(const string& file_name, PointCloudXYZI::Ptr cloud)
{
    if (!cloud || cloud->empty())
    {
        ROS_INFO("PCD save skipped for %s: size=%zu", file_name.c_str(), cloud ? cloud->size() : 0);
        return false;
    }
    string pcd_dir( string( ROOT_DIR ) + "PCD" );
    if (!ensure_directory(pcd_dir))
    {
        return false;
    }
    string all_points_dir( pcd_dir + string("/") + file_name );
    pcl::PCDWriter pcd_writer;
    cout << "current scan saved to /PCD/" << file_name << endl;
    const int save_result = pcd_writer.writeBinary( all_points_dir, *cloud );
    if (save_result != 0)
    {
        ROS_ERROR("Failed to save PCD to %s (code=%d)", all_points_dir.c_str(), save_result);
        return false;
    }
    ROS_INFO("Saved PCD to %s with %zu points", all_points_dir.c_str(), cloud->size());
    return true;
}

void save_debug_pcds()
{
    if (!pcd_save_en)
    {
        return;
    }
    write_pcd_if_nonempty("scans_raw_accumulated.pcd", pcl_wait_save);
    write_pcd_if_nonempty("history_deleted_markers.pcd", map_history_deleted_markers);
    write_pcd_if_nonempty("history_anchor_markers.pcd", map_history_anchor_markers);
    write_pcd_if_nonempty("history_surface_anchors.pcd", map_history_surface_anchors);
}

void save_pcd_on_shutdown()
{
    PointCloudXYZI::Ptr save_cloud = export_map_cloud_for_save();
    if ( !pcd_save_en || !save_cloud || save_cloud->empty() )
    {
        ROS_INFO("PCD save skipped: enabled=%d size=%zu", pcd_save_en ? 1 : 0, save_cloud ? save_cloud->size() : 0);
        return;
    }
    save_debug_pcds();
    write_pcd_if_nonempty("scans.pcd", save_cloud);
}

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
vect3 pos_lid;

nav_msgs::msg::Path path;
nav_msgs::msg::Odometry odomAftMapped;
geometry_msgs::msg::Quaternion geoQuat;
geometry_msgs::msg::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

PointType bodyPointToWorldPoint(const PointType& point_body)
{
    PointType point_world;
    pointBodyToWorld(&point_body, &point_world);
    return point_world;
}

void trim_cloud_to_last(PointCloudXYZI::Ptr& cloud, size_t max_points)
{
    if (!cloud || cloud->size() <= max_points)
    {
        return;
    }
    PointCloudXYZI::Ptr trimmed(new PointCloudXYZI());
    trimmed->reserve(max_points);
    const size_t start = cloud->size() - max_points;
    for (size_t i = start; i < cloud->size(); ++i)
    {
        trimmed->push_back(cloud->points[i]);
    }
    cloud = trimmed;
}

void trim_paired_clouds_to_last(PointCloudXYZI::Ptr& markers,
                                PointCloudXYZI::Ptr& anchors,
                                size_t max_pairs)
{
    if (!markers || !anchors)
    {
        return;
    }
    const size_t pair_count = std::min(markers->size(), anchors->size());
    if (pair_count <= max_pairs && markers->size() == anchors->size())
    {
        return;
    }
    const size_t keep = std::min(pair_count, max_pairs);
    PointCloudXYZI::Ptr trimmed_markers(new PointCloudXYZI());
    PointCloudXYZI::Ptr trimmed_anchors(new PointCloudXYZI());
    trimmed_markers->reserve(keep);
    trimmed_anchors->reserve(keep);
    const size_t start = pair_count - keep;
    for (size_t i = start; i < pair_count; ++i)
    {
        trimmed_markers->push_back(markers->points[i]);
        trimmed_anchors->push_back(anchors->points[i]);
    }
    markers = trimmed_markers;
    anchors = trimmed_anchors;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
void lasermap_fov_segment()
{
    cub_needrm.clear();
    kdtree_delete_counter = 0;
    kdtree_delete_time = 0.0;    
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid;
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) 
{
    mtx_buffer.lock();
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    if (sentinel_lio_ros2::stampToSec(msg->header.stamp) < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(sentinel_lio_ros2::stampToSec(msg->header.stamp));
    last_timestamp_lidar = sentinel_lio_ros2::stampToSec(msg->header.stamp);
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
void imu_cbk(const sensor_msgs::msg::Imu::ConstSharedPtr msg_in) 
{
    publish_count ++;
    // cout<<"IMU got at: "<<sentinel_lio_ros2::stampToSec(msg_in->header.stamp)<<endl;
    auto msg = std::make_shared<sensor_msgs::msg::Imu>(*msg_in);

    msg->header.stamp = sentinel_lio_ros2::stampFromSec(sentinel_lio_ros2::stampToSec(msg_in->header.stamp) - time_diff_lidar_to_imu);
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = \
        sentinel_lio_ros2::stampFromSec(timediff_lidar_wrt_imu + sentinel_lio_ros2::stampToSec(msg_in->header.stamp));
    }

    double timestamp = sentinel_lio_ros2::stampToSec(msg->header.stamp);

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();


        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }
        if(lidar_type == MARSIM)
            lidar_end_time = meas.lidar_beg_time;

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = sentinel_lio_ros2::stampToSec(imu_buffer.front()->header.stamp);
    meas.imu.clear();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = sentinel_lio_ros2::stampToSec(imu_buffer.front()->header.stamp);
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int process_increments = 0;
void cleanup_history_map_with_prior(int current_frame_num)
{
    if (!map_history_cleanup_enable || !sentinel_ptr || ikdtree.Root_Node == nullptr)
    {
        return;
    }

    if (map_history_cleanup_period <= 0 || current_frame_num % map_history_cleanup_period != 0)
    {
        return;
    }

    BoxPointType cleanup_box;
    const float range = static_cast<float>(std::max(0.5, map_history_cleanup_range));
    cleanup_box.vertex_min[0] = static_cast<float>(pos_lid(0)) - range;
    cleanup_box.vertex_max[0] = static_cast<float>(pos_lid(0)) + range;
    cleanup_box.vertex_min[1] = static_cast<float>(pos_lid(1)) - range;
    cleanup_box.vertex_max[1] = static_cast<float>(pos_lid(1)) + range;
    cleanup_box.vertex_min[2] = static_cast<float>(pos_lid(2)) - range;
    cleanup_box.vertex_max[2] = static_cast<float>(pos_lid(2)) + range;

    size_t accumulated_removed = 0;
    size_t accumulated_checked = 0;
    if (map_accumulated_cleanup_enable && pcl_wait_save && !pcl_wait_save->empty())
    {
        PointCloudXYZI::Ptr kept(new PointCloudXYZI());
        kept->reserve(pcl_wait_save->size());
        for (const auto& point_world : pcl_wait_save->points)
        {
            const bool in_cleanup_box =
                point_world.x >= cleanup_box.vertex_min[0] && point_world.x <= cleanup_box.vertex_max[0] &&
                point_world.y >= cleanup_box.vertex_min[1] && point_world.y <= cleanup_box.vertex_max[1] &&
                point_world.z >= cleanup_box.vertex_min[2] && point_world.z <= cleanup_box.vertex_max[2];
            if (!in_cleanup_box)
            {
                kept->push_back(point_world);
                continue;
            }

            accumulated_checked++;
            PointType point_body;
            const V3D p_world(point_world.x, point_world.y, point_world.z);
            const V3D p_body = state_point.offset_R_L_I.conjugate() *
                (state_point.rot.conjugate() * (p_world - state_point.pos) - state_point.offset_T_L_I);
            point_body.x = p_body(0);
            point_body.y = p_body(1);
            point_body.z = p_body(2);
            point_body.intensity = point_world.intensity;
            PointType anchor_body;
            anchor_body.x = std::numeric_limits<float>::quiet_NaN();
            anchor_body.y = std::numeric_limits<float>::quiet_NaN();
            anchor_body.z = std::numeric_limits<float>::quiet_NaN();
            const bool keep_point = sentinel_ptr->shouldKeepMapPoint(point_body, current_explicit_mask_enabled, &anchor_body);
            if (keep_point)
            {
                kept->push_back(point_world);
            }
            else
            {
                accumulated_removed++;
                if (map_history_deleted_markers)
                {
                    map_history_deleted_markers->push_back(point_world);
                }
                if (map_history_anchor_markers && map_history_surface_anchors && std::isfinite(anchor_body.x) &&
                    std::isfinite(anchor_body.y) && std::isfinite(anchor_body.z))
                {
                    map_history_anchor_markers->push_back(point_world);
                    map_history_surface_anchors->push_back(bodyPointToWorldPoint(anchor_body));
                }
            }
        }
        pcl_wait_save = kept;
    }

    PointVector nearby_points;
    ikdtree.Box_Search(cleanup_box, nearby_points);
    if (nearby_points.empty())
    {
        return;
    }

    PointVector points_to_delete;
    PointVector anchor_markers_for_delete;
    PointVector anchors_for_delete;
    points_to_delete.reserve(nearby_points.size() / 8 + 1);
    anchor_markers_for_delete.reserve(nearby_points.size() / 8 + 1);
    anchors_for_delete.reserve(nearby_points.size() / 8 + 1);
    for (const auto& point_world : nearby_points)
    {
        PointType point_body;
        const V3D p_world(point_world.x, point_world.y, point_world.z);
        const V3D p_body = state_point.offset_R_L_I.conjugate() *
            (state_point.rot.conjugate() * (p_world - state_point.pos) - state_point.offset_T_L_I);
        point_body.x = p_body(0);
        point_body.y = p_body(1);
        point_body.z = p_body(2);
        point_body.intensity = point_world.intensity;
        PointType anchor_body;
        anchor_body.x = std::numeric_limits<float>::quiet_NaN();
        anchor_body.y = std::numeric_limits<float>::quiet_NaN();
        anchor_body.z = std::numeric_limits<float>::quiet_NaN();
        if (!sentinel_ptr->shouldKeepMapPoint(point_body, current_explicit_mask_enabled, &anchor_body))
        {
            points_to_delete.push_back(point_world);
            if (std::isfinite(anchor_body.x) && std::isfinite(anchor_body.y) && std::isfinite(anchor_body.z))
            {
                anchor_markers_for_delete.push_back(point_world);
                anchors_for_delete.push_back(bodyPointToWorldPoint(anchor_body));
            }
        }
    }

    if (!points_to_delete.empty())
    {
        ikdtree.Delete_Points(points_to_delete);
        if (map_history_deleted_markers)
        {
            map_history_deleted_markers->reserve(map_history_deleted_markers->size() + points_to_delete.size());
            for (const auto& pt : points_to_delete)
            {
                map_history_deleted_markers->push_back(pt);
            }
            if (map_history_anchor_markers && map_history_surface_anchors && !anchors_for_delete.empty())
            {
                map_history_anchor_markers->reserve(map_history_anchor_markers->size() + anchor_markers_for_delete.size());
                map_history_surface_anchors->reserve(map_history_surface_anchors->size() + anchors_for_delete.size());
                for (size_t i = 0; i < anchor_markers_for_delete.size() && i < anchors_for_delete.size(); ++i)
                {
                    map_history_anchor_markers->push_back(anchor_markers_for_delete[i]);
                    map_history_surface_anchors->push_back(anchors_for_delete[i]);
                }
                trim_paired_clouds_to_last(map_history_anchor_markers, map_history_surface_anchors, 200000);
            }
            if (map_history_deleted_markers->size() > 200000)
            {
                trim_cloud_to_last(map_history_deleted_markers, 200000);
            }
        }
        ROS_INFO("[Sentinel] history cleanup removed %zu/%zu local map points",
                 points_to_delete.size(), nearby_points.size());
    }
    if (accumulated_removed > 0)
    {
        ROS_INFO("[Sentinel] accumulated history cleanup removed %zu/%zu saved map points",
                 accumulated_removed, accumulated_checked);
    }
}

void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    std::vector<int> map_keep_indices;
    if (sentinel_ptr && map_gate_apply_to_ikdtree) {
        map_keep_indices = sentinel_ptr->mapKeepIndices(feats_down_body, current_explicit_mask_enabled);
    } else {
        map_keep_indices.reserve(feats_down_size);
        for (int i = 0; i < feats_down_size; ++i) {
            map_keep_indices.push_back(i);
        }
    }
    for (const int i : map_keep_indices)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
void publish_frame_world(const PointCloudPublisher::SharedPtr & pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = sentinel_lio_ros2::stampFromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull->publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        PointCloudXYZI::Ptr map_save_body(dense_pub_en ? feats_undistort : feats_down_body);
        if (sentinel_ptr) {
            map_save_body = sentinel_ptr->filterMapCloud(feats_undistort, current_explicit_mask_enabled);
            if (!dense_pub_en) {
                map_save_body = sentinel_ptr->filterMapCloud(feats_down_body, current_explicit_mask_enabled);
            }
        }
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI());
        laserCloudWorld->reserve(map_save_body->points.size());

        for (const auto& point_body : map_save_body->points)
        {
            PointType point_world;
            RGBpointBodyToWorld(&point_body, &point_world);
            laserCloudWorld->push_back(point_world);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string pcd_dir( string( ROOT_DIR ) + "PCD" );
            if (!ensure_directory(pcd_dir))
            {
                pcl_wait_save->clear();
                scan_wait_num = 0;
                return;
            }
            string all_points_dir(pcd_dir + string("/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            const int save_result = pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            if (save_result != 0)
            {
                ROS_ERROR("Failed to save intermediate PCD to %s (code=%d)", all_points_dir.c_str(), save_result);
            }
            else
            {
                ROS_INFO("Saved intermediate PCD to %s with %zu points", all_points_dir.c_str(), pcl_wait_save->size());
            }
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void publish_frame_body(const PointCloudPublisher::SharedPtr & pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = sentinel_lio_ros2::stampFromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body->publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(const PointCloudPublisher::SharedPtr & pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::msg::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = sentinel_lio_ros2::stampFromSec(lidar_end_time);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect->publish(laserCloudFullRes3);
}

void publish_map(const PointCloudPublisher::SharedPtr & pubLaserCloudMap)
{
    sensor_msgs::msg::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = sentinel_lio_ros2::stampFromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap->publish(laserCloudMap);
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    
}

void publish_odometry(const OdometryPublisher::SharedPtr & pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = sentinel_lio_ros2::stampFromSec(lidar_end_time);// sentinel_lio_ros2::stampFromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped->publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    if (tf_broadcaster) {
        geometry_msgs::msg::TransformStamped transform;
        transform.header.stamp = odomAftMapped.header.stamp;
        transform.header.frame_id = "camera_init";
        transform.child_frame_id = "body";
        transform.transform.translation.x = odomAftMapped.pose.pose.position.x;
        transform.transform.translation.y = odomAftMapped.pose.pose.position.y;
        transform.transform.translation.z = odomAftMapped.pose.pose.position.z;
        transform.transform.rotation = odomAftMapped.pose.pose.orientation;
        tf_broadcaster->sendTransform(transform);
    }
}

void publish_path(const PathPublisher::SharedPtr & pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = sentinel_lio_ros2::stampFromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        pubPath->publish(path);
    }
}

void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime();
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 

    /** closest surface search and residual computation **/
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++)
    {
        PointType &point_body  = feats_down_body->points[i]; 
        PointType &point_world = feats_down_world->points[i]; 

        /* transform to world frame */
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);

        auto &points_near = Nearest_Points[i];

        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        if (!point_selected_surf[i]) continue;

        VF(4) pabcd;
        point_selected_surf[i] = false;
        if (esti_plane(pabcd, points_near, 0.1f))
        {
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());

            if (s > 0.9)
            {
                point_selected_surf[i] = true;
                normvec->points[i].x = pabcd(0);
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2;
                res_last[i] = abs(pd2);
            }
        }
    }
    
    effct_feat_num = 0;

    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i];
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num;
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        ekfom_data.h(i) = -norm_p.intensity;
        float confidence = laser_p.intensity; 
        if (confidence < 0.01f) confidence = 0.01f;
        if (confidence > 1.0f)  confidence = 1.0f;
        float weight = std::sqrt(confidence);
        ekfom_data.h_x.block<1, 12>(i, 0) *= weight;
        ekfom_data.h(i) *= weight;
    }
    solve_time += omp_get_wtime() - solve_start_;
}

void stereoDepthCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
        if (sentinel_ptr) {
            sentinel_ptr->updateDepthMap(cv_ptr->image);
        }
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}

void stereoMaskCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
        if (sentinel_ptr) {
            sentinel_ptr->updateMirrorMask(cv_ptr->image);
        }
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}

void stereoRgbCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        if (sentinel_ptr) {
            sentinel_ptr->updateRgbImage(cv_ptr->image);
        }
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto nh = std::make_shared<rclcpp::Node>("laserMapping");
    sentinel_lio_ros2::NodeContext::set(nh);
    tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(nh);

    sentinel_lio_ros2::param(nh, "publish.path_en", path_en, true);
    sentinel_lio_ros2::param(nh, "publish.scan_publish_en", scan_pub_en, true);
    sentinel_lio_ros2::param(nh, "publish.dense_publish_en", dense_pub_en, true);
    sentinel_lio_ros2::param(nh, "publish.scan_bodyframe_pub_en", scan_body_pub_en, true);
    sentinel_lio_ros2::param(nh, "max_iteration", NUM_MAX_ITERATIONS, 4);
    sentinel_lio_ros2::param(nh, "map_file_path", map_file_path, std::string(""));
    sentinel_lio_ros2::param(nh, "common.lid_topic", lid_topic, std::string("/ouster/points"));
    sentinel_lio_ros2::param(nh, "common.imu_topic", imu_topic, std::string("/ouster/imu"));
    sentinel_lio_ros2::param(nh, "common.time_sync_en", time_sync_en, false);
    sentinel_lio_ros2::param(nh, "common.time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    sentinel_lio_ros2::param(nh, "filter_size_corner", filter_size_corner_min, 0.5);
    sentinel_lio_ros2::param(nh, "filter_size_surf", filter_size_surf_min, 0.5);
    sentinel_lio_ros2::param(nh, "filter_size_map", filter_size_map_min, 0.5);
    sentinel_lio_ros2::param(nh, "cube_side_length", cube_len, 200.0);
    sentinel_lio_ros2::param(nh, "mapping.det_range", DET_RANGE, 300.0f);
    sentinel_lio_ros2::param(nh, "mapping.fov_degree", fov_deg, 180.0);
    sentinel_lio_ros2::param(nh, "mapping.gyr_cov", gyr_cov, 0.1);
    sentinel_lio_ros2::param(nh, "mapping.acc_cov", acc_cov, 0.1);
    sentinel_lio_ros2::param(nh, "mapping.b_gyr_cov", b_gyr_cov, 0.0001);
    sentinel_lio_ros2::param(nh, "mapping.b_acc_cov", b_acc_cov, 0.0001);
    sentinel_lio_ros2::param(nh, "preprocess.blind", p_pre->blind, 0.01);
    sentinel_lio_ros2::param(nh, "preprocess.lidar_type", lidar_type, static_cast<int>(OUST64));
    sentinel_lio_ros2::param(nh, "preprocess.scan_line", p_pre->N_SCANS, 16);
    sentinel_lio_ros2::param(nh, "preprocess.timestamp_unit", p_pre->time_unit, static_cast<int>(US));
    sentinel_lio_ros2::param(nh, "preprocess.scan_rate", p_pre->SCAN_RATE, 10);
    sentinel_lio_ros2::param(nh, "point_filter_num", p_pre->point_filter_num, 2);
    sentinel_lio_ros2::param(nh, "feature_extract_enable", p_pre->feature_enabled, false);
    sentinel_lio_ros2::param(nh, "runtime_pos_log_enable", runtime_pos_log, false);
    sentinel_lio_ros2::param(nh, "mapping.extrinsic_est_en", extrinsic_est_en, true);
    sentinel_lio_ros2::param(nh, "pcd_save.pcd_save_en", pcd_save_en, false);
    sentinel_lio_ros2::param(nh, "pcd_save.interval", pcd_save_interval, -1);
    sentinel_lio_ros2::param(nh, "mapping.extrinsic_T", extrinT, std::vector<double>{0.0, 0.0, 0.0});
    sentinel_lio_ros2::param(nh, "mapping.extrinsic_R", extrinR, std::vector<double>{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
    std::atexit(save_pcd_on_shutdown);
    double depth_thresh = 1.0, vel_thresh = 0.1;
    double confidence_floor = 0.3;
    double fx = 541.56, fy = 541.56, cx = 496.91, cy = 264.66;
    int img_w = 960, img_h = 540;
    int mask_erode_kernel = 5;
    int mask_boundary_band = 8;
    double mask_threshold = 127.0;
    double depth_sigma_abs = 0.15;
    double depth_sigma_rel = 0.04;
    double ghost_margin_abs = 0.15;
    double ghost_margin_rel = 0.02;
    double mirror_surface_confidence = 0.35;
    double mask_boundary_confidence = 0.50;
    double invalid_depth_confidence = 0.80;
    double reflective_invalid_depth_confidence = 0.15;
    bool use_depth_consistency = true;
    bool map_depth_require_calibration_param = true;
    bool map_gate_apply_to_ikdtree_param = false;
    bool map_history_cleanup_enable_param = false;
    bool map_accumulated_cleanup_enable_param = false;
    bool map_export_ikdtree_param = false;
    bool map_export_apply_history_cleanup_param = false;
    bool global_depth_prior_enable_param = false;
    bool frontend_confidence_enable_param = false;
    bool rgb_subscribe_enable_param = false;
    bool depth_calibration_enable_param = true;
    bool depth_calibration_use_non_mask_param = true;
    bool stats_enable = true;
    bool explicit_mask_enable = true;
    int mask_ab_mode = 0;
    int mask_ab_period = 30;
    std::string stats_topic = "/mirror_sentinel/frame_stats";
    std::string mask_topic = "/vfm/mirror_mask";
    std::string mask_viz_topic = "/mirror_sentinel/mask_viz";
    std::string mask_viz_frame_id = "camera_init";
    std::string rgb_topic = "/zed2/zed_node/left/image_rect_color";
    std::string mask_overlay_topic = "/mirror_sentinel/mask_overlay";
    vector<double> ext_R_vec, ext_t_vec;
    Eigen::Matrix3d sentinel_R_ext = Eigen::Matrix3d::Identity(); 
    Eigen::Vector3d sentinel_t_ext = Eigen::Vector3d::Zero();

    sentinel_lio_ros2::param(nh, "sentinel.depth_diff_threshold", depth_thresh, 1.0);
    sentinel_lio_ros2::param(nh, "sentinel.ego_vel_threshold", vel_thresh, 0.1);
    sentinel_lio_ros2::param(nh, "sentinel.confidence_floor", confidence_floor, 0.3);
    sentinel_lio_ros2::param(nh, "sentinel.camera_matrix.fx", fx, 541.56);
    sentinel_lio_ros2::param(nh, "sentinel.camera_matrix.fy", fy, 541.56);
    sentinel_lio_ros2::param(nh, "sentinel.camera_matrix.cx", cx, 496.91);
    sentinel_lio_ros2::param(nh, "sentinel.camera_matrix.cy", cy, 264.66);
    sentinel_lio_ros2::param(nh, "sentinel.image_width", img_w, 960);
    sentinel_lio_ros2::param(nh, "sentinel.image_height", img_h, 540);
    sentinel_lio_ros2::param(nh, "sentinel.mask_erode_kernel", mask_erode_kernel, 5);
    sentinel_lio_ros2::param(nh, "sentinel.mask_boundary_band", mask_boundary_band, 8);
    sentinel_lio_ros2::param(nh, "sentinel.mask_threshold", mask_threshold, 127.0);
    sentinel_lio_ros2::param(nh, "sentinel.depth_sigma_abs", depth_sigma_abs, 0.15);
    sentinel_lio_ros2::param(nh, "sentinel.depth_sigma_rel", depth_sigma_rel, 0.04);
    sentinel_lio_ros2::param(nh, "sentinel.ghost_margin_abs", ghost_margin_abs, 0.15);
    sentinel_lio_ros2::param(nh, "sentinel.ghost_margin_rel", ghost_margin_rel, 0.02);
    sentinel_lio_ros2::param(nh, "sentinel.mirror_surface_confidence", mirror_surface_confidence, 0.35);
    sentinel_lio_ros2::param(nh, "sentinel.mask_boundary_confidence", mask_boundary_confidence, 0.50);
    sentinel_lio_ros2::param(nh, "sentinel.invalid_depth_confidence", invalid_depth_confidence, 0.80);
    sentinel_lio_ros2::param(nh, "sentinel.reflective_invalid_depth_confidence", reflective_invalid_depth_confidence, 0.15);
    sentinel_lio_ros2::param(nh, "sentinel.map_depth_gate_enable", map_depth_gate_enable, true);
    sentinel_lio_ros2::param(nh, "sentinel.map_mask_gate_enable", map_mask_gate_enable, true);
    sentinel_lio_ros2::param(nh, "sentinel.map_depth_ghost_margin_abs", map_depth_ghost_margin_abs, 0.80);
    sentinel_lio_ros2::param(nh, "sentinel.map_depth_ghost_margin_rel", map_depth_ghost_margin_rel, 0.03);
    sentinel_lio_ros2::param(nh, "sentinel.map_mask_foreground_keep_margin", map_mask_foreground_keep_margin, 0.20);
    sentinel_lio_ros2::param(nh, "sentinel.map_mask_require_depth_confirmation", map_mask_require_depth_confirmation, true);
    sentinel_lio_ros2::param(nh, "sentinel.map_mask_invalid_depth_reject", map_mask_invalid_depth_reject, false);
    sentinel_lio_ros2::param(nh, "sentinel.map_depth_require_calibration", map_depth_require_calibration_param, true);
    map_depth_require_calibration = map_depth_require_calibration_param;
    sentinel_lio_ros2::param(nh, "sentinel.map_gate_apply_to_ikdtree", map_gate_apply_to_ikdtree_param, false);
    map_gate_apply_to_ikdtree = map_gate_apply_to_ikdtree_param;
    sentinel_lio_ros2::param(nh, "sentinel.map_history_cleanup_enable", map_history_cleanup_enable_param, false);
    map_history_cleanup_enable = map_history_cleanup_enable_param;
    sentinel_lio_ros2::param(nh, "sentinel.map_accumulated_cleanup_enable", map_accumulated_cleanup_enable_param, false);
    map_accumulated_cleanup_enable = map_accumulated_cleanup_enable_param;
    sentinel_lio_ros2::param(nh, "sentinel.map_history_cleanup_period", map_history_cleanup_period, 10);
    sentinel_lio_ros2::param(nh, "sentinel.map_history_cleanup_range", map_history_cleanup_range, 8.0);
    sentinel_lio_ros2::param(nh, "sentinel.map_history_cleanup_export_radius", map_history_cleanup_export_radius, 0.35);
    sentinel_lio_ros2::param(nh, "sentinel.map_export_ikdtree", map_export_ikdtree_param, false);
    map_export_ikdtree = map_export_ikdtree_param;
    sentinel_lio_ros2::param(nh, "sentinel.map_export_apply_history_cleanup", map_export_apply_history_cleanup_param, false);
    map_export_apply_history_cleanup = map_export_apply_history_cleanup_param;
    sentinel_lio_ros2::param(nh, "sentinel.global_depth_prior_enable", global_depth_prior_enable_param, false);
    global_depth_prior_enable = global_depth_prior_enable_param;
    sentinel_lio_ros2::param(nh, "sentinel.frontend_confidence_enable", frontend_confidence_enable_param, false);
    frontend_confidence_enable = frontend_confidence_enable_param;
    sentinel_lio_ros2::param(nh, "sentinel.rgb_subscribe_enable", rgb_subscribe_enable_param, false);
    rgb_subscribe_enable = rgb_subscribe_enable_param;
    sentinel_lio_ros2::param(nh, "sentinel.depth_calibration_enable", depth_calibration_enable_param, true);
    depth_calibration_enable = depth_calibration_enable_param;
    sentinel_lio_ros2::param(nh, "sentinel.depth_calibration_use_non_mask", depth_calibration_use_non_mask_param, true);
    depth_calibration_use_non_mask = depth_calibration_use_non_mask_param;
    sentinel_lio_ros2::param(nh, "sentinel.depth_calibration_min_points", depth_calibration_min_points, 50);
    sentinel_lio_ros2::param(nh, "sentinel.depth_calibration_scale_min", depth_calibration_scale_min, 0.50);
    sentinel_lio_ros2::param(nh, "sentinel.depth_calibration_scale_max", depth_calibration_scale_max, 2.00);
    sentinel_lio_ros2::param(nh, "sentinel.depth_calibration_max_raw_residual", depth_calibration_max_raw_residual, 1.50);
    sentinel_lio_ros2::param(nh, "sentinel.use_depth_consistency", use_depth_consistency, true);
    sentinel_lio_ros2::param(nh, "sentinel.stats_enable", stats_enable, true);
    sentinel_lio_ros2::param(nh, "sentinel.explicit_mask_enable", explicit_mask_enable, true);
    sentinel_lio_ros2::param(nh, "sentinel.mask_ab_mode", mask_ab_mode, 0);
    sentinel_lio_ros2::param(nh, "sentinel.mask_ab_period", mask_ab_period, 30);
    sentinel_lio_ros2::param(nh, "sentinel.stats_topic", stats_topic, std::string("/mirror_sentinel/frame_stats"));
    sentinel_lio_ros2::param(nh, "sentinel.mask_topic", mask_topic, std::string("/vfm/mirror_mask"));
    sentinel_lio_ros2::param(nh, "sentinel.mask_viz_topic", mask_viz_topic, std::string("/mirror_sentinel/mask_viz"));
    sentinel_lio_ros2::param(nh, "sentinel.mask_viz_frame_id", mask_viz_frame_id, std::string("camera_init"));
    sentinel_lio_ros2::param(nh, "sentinel.rgb_topic", rgb_topic, std::string("/zed2/zed_node/left/image_rect_color"));
    sentinel_lio_ros2::param(nh, "sentinel.depth_topic", depth_topic, std::string("/vfm/depth_image"));
    sentinel_lio_ros2::param(nh, "sentinel.mask_overlay_topic", mask_overlay_topic, std::string("/mirror_sentinel/mask_overlay"));

    if (sentinel_lio_ros2::getParam(nh, "sentinel.extrinsic_R", ext_R_vec) && ext_R_vec.size() == 9) {
        sentinel_R_ext << ext_R_vec[0], ext_R_vec[1], ext_R_vec[2],
                          ext_R_vec[3], ext_R_vec[4], ext_R_vec[5],
                          ext_R_vec[6], ext_R_vec[7], ext_R_vec[8];
    } else {
        ROS_WARN("[Sentinel] Failed to load extrinsic_R, using Identity!");
    }

    if (sentinel_lio_ros2::getParam(nh, "sentinel.extrinsic_t", ext_t_vec) && ext_t_vec.size() == 3) {
        sentinel_t_ext << ext_t_vec[0], ext_t_vec[1], ext_t_vec[2];
    } else {
        ROS_WARN("[Sentinel] Failed to load extrinsic_t, using Zero!");
    }

    sentinel_ptr = std::make_shared<MirrorSentinel>(
        fx, fy, cx, cy, img_w, img_h, sentinel_R_ext, sentinel_t_ext,
        static_cast<float>(confidence_floor), mask_erode_kernel, mask_boundary_band, static_cast<float>(mask_threshold),
        mask_viz_topic, mask_viz_frame_id,
        mask_overlay_topic,
        static_cast<float>(depth_sigma_abs), static_cast<float>(depth_sigma_rel),
        static_cast<float>(ghost_margin_abs), static_cast<float>(ghost_margin_rel),
        static_cast<float>(mirror_surface_confidence), static_cast<float>(mask_boundary_confidence),
        static_cast<float>(invalid_depth_confidence), static_cast<float>(reflective_invalid_depth_confidence),
        map_depth_gate_enable, map_mask_gate_enable,
        static_cast<float>(map_depth_ghost_margin_abs), static_cast<float>(map_depth_ghost_margin_rel),
        static_cast<float>(map_mask_foreground_keep_margin),
        map_mask_require_depth_confirmation,
        map_mask_invalid_depth_reject,
        map_depth_require_calibration,
        global_depth_prior_enable,
        depth_calibration_enable,
        depth_calibration_use_non_mask,
        depth_calibration_min_points,
        static_cast<float>(depth_calibration_scale_min),
        static_cast<float>(depth_calibration_scale_max),
        static_cast<float>(depth_calibration_max_raw_residual),
        use_depth_consistency);

    p_pre->lidar_type = lidar_type;
    cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;
    
    path.header.stamp    = sentinel_lio_ros2::NodeContext::now();
    path.header.frame_id ="camera_init";

    /*** variables definition ***/
    int effect_feat_num = 0, frame_num = 0;
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;
    
    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
    p_imu->lidar_type = lidar_type;
    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(),"w");

    ofstream fout_pre, fout_out, fout_dbg;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
    if (fout_pre && fout_out)
        cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
    else
        cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

    if (p_pre->lidar_type == AVIA) {
        ROS_WARN("[Sentinel-LIO ROS2] Livox CustomMsg input is not enabled in this ROS2 port. Use PointCloud2 input or add livox_ros_driver2 support.");
    }

    auto sub_pcl = nh->create_subscription<sensor_msgs::msg::PointCloud2>(lid_topic, rclcpp::SensorDataQoS(), standard_pcl_cbk);
    auto sub_imu = nh->create_subscription<sensor_msgs::msg::Imu>(imu_topic, rclcpp::SensorDataQoS(), imu_cbk);
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_rgb;
    if (rgb_subscribe_enable) {
        sub_rgb = nh->create_subscription<sensor_msgs::msg::Image>(rgb_topic, 10, stereoRgbCallback);
    }
    auto image_qos = rclcpp::SensorDataQoS().keep_last(1);
    auto sub_depth = nh->create_subscription<sensor_msgs::msg::Image>(depth_topic, image_qos, stereoDepthCallback);
    auto sub_mask = nh->create_subscription<sensor_msgs::msg::Image>(mask_topic, image_qos, stereoMaskCallback);
    auto pubLaserCloudFull = nh->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered", 100000);
    auto pubLaserCloudFull_body = nh->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_registered_body", 100000);
    auto pubLaserCloudEffect = nh->create_publisher<sensor_msgs::msg::PointCloud2>("/cloud_effected", 100000);
    auto pubLaserCloudMap = nh->create_publisher<sensor_msgs::msg::PointCloud2>("/Laser_map", 100000);
    auto pubOdomAftMapped = nh->create_publisher<nav_msgs::msg::Odometry>("/Odometry", 100000);
    auto pubPath = nh->create_publisher<nav_msgs::msg::Path>("/path", 100000);
    auto pubSentinelStats = stats_enable ? nh->create_publisher<std_msgs::msg::Float32MultiArray>(stats_topic, 10) : nullptr;

    signal(SIGINT, SigHandle);
    rclcpp::Rate rate(5000);
    bool status = rclcpp::ok();
    while (status)
    {
        if (flg_exit) break;
        rclcpp::spin_some(nh);
        if(sync_packages(Measures)) 
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                continue;
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();

            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            if (!feats_undistort || feats_undistort->empty())
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            bool use_explicit_mask = explicit_mask_enable;
            if (mask_ab_mode == 1 && mask_ab_period > 0) {
                use_explicit_mask = ((frame_num / mask_ab_period) % 2 == 0) ? explicit_mask_enable : !explicit_mask_enable;
            }
            current_explicit_mask_enabled = use_explicit_mask;

            if (sentinel_ptr && frontend_confidence_enable) {
                feats_undistort = sentinel_ptr->assignConfidence(
                    feats_undistort,
                    use_explicit_mask,
                    true,
                    nullptr);
            }

            frame_num++;

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size();

            MirrorFrameStats mirror_stats;
            if (sentinel_ptr) {
                sentinel_ptr->updatePriorState(
                    feats_down_body,
                    use_explicit_mask,
                    stats_enable ? &mirror_stats : nullptr);
                mirror_stats.ab_mode = mask_ab_mode;
            }

            if (stats_enable && pubSentinelStats) {
                std_msgs::msg::Float32MultiArray stats_msg;
                stats_msg.data.reserve(21);
                stats_msg.data.push_back(static_cast<float>(frame_num - 1));
                stats_msg.data.push_back(static_cast<float>(mirror_stats.input_points));
                stats_msg.data.push_back(static_cast<float>(mirror_stats.output_points));
                stats_msg.data.push_back(static_cast<float>(mirror_stats.masked_points));
                stats_msg.data.push_back(mirror_stats.mean_confidence);
                stats_msg.data.push_back(mirror_stats.mask_coverage);
                stats_msg.data.push_back(mirror_stats.depth_valid_ratio);
                stats_msg.data.push_back(mirror_stats.explicit_mask_enabled ? 1.0f : 0.0f);
                stats_msg.data.push_back(static_cast<float>(mirror_stats.ab_mode));
                stats_msg.data.push_back(static_cast<float>(mirror_stats.depth_checked_points));
                stats_msg.data.push_back(static_cast<float>(mirror_stats.depth_inconsistent_points));
                stats_msg.data.push_back(static_cast<float>(mirror_stats.ghost_candidate_points));
                stats_msg.data.push_back(static_cast<float>(mirror_stats.mask_core_points));
                stats_msg.data.push_back(static_cast<float>(mirror_stats.mask_boundary_points));
                stats_msg.data.push_back(mirror_stats.mean_depth_residual);
                stats_msg.data.push_back(mirror_stats.depth_calibration_valid ? 1.0f : 0.0f);
                stats_msg.data.push_back(static_cast<float>(mirror_stats.depth_calibration_points));
                stats_msg.data.push_back(mirror_stats.depth_scale);
                stats_msg.data.push_back(mirror_stats.depth_shift);
                stats_msg.data.push_back(mirror_stats.calibration_mean_raw_residual);
                stats_msg.data.push_back(mirror_stats.calibration_mean_calibrated_residual);
                pubSentinelStats->publish(stats_msg);
            }

            /*** initialize the map kdtree ***/
            if(ikdtree.Root_Node == nullptr)
            {
                if(feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                    std::vector<int> map_keep_indices;
                    if (sentinel_ptr && map_gate_apply_to_ikdtree) {
                        map_keep_indices = sentinel_ptr->mapKeepIndices(feats_down_body, current_explicit_mask_enabled);
                    } else {
                        map_keep_indices.reserve(feats_down_size);
                        for (int i = 0; i < feats_down_size; ++i) {
                            map_keep_indices.push_back(i);
                        }
                    }
                    PointVector initial_map_points;
                    initial_map_points.reserve(map_keep_indices.size());
                    for(const int i : map_keep_indices)
                    {
                        PointType point_world;
                        pointBodyToWorld(&(feats_down_body->points[i]), &point_world);
                        initial_map_points.push_back(point_world);
                    }
                    if (initial_map_points.size() > 5) {
                        ikdtree.Build(initial_map_points);
                    }
                }
                continue;
            }
            int featsFromMapNum = ikdtree.validnum();
            kdtree_size_st = ikdtree.size();
            
            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }
            
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I);
            fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;

            if(0) // If you need to see map point, change to "if(1)"
            {
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int  rematch_num = 0;
            bool nearest_search_en = true; //

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            state_point = kf.get_x();
            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();
            map_incremental();
            cleanup_history_map_with_prior(frame_num);
            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath);
            if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
            // publish_effect_world(pubLaserCloudEffect);
            // publish_map(pubLaserCloudMap);

            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter ++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
                dump_lio_state_to_log(fp);
            }
        }

        status = rclcpp::ok();
        rate.sleep();
    }

    PointCloudXYZI::Ptr final_save_cloud = export_map_cloud_for_save();
    if (final_save_cloud && final_save_cloud->size() > 0 && pcd_save_en)
    {
        save_debug_pcds();
        write_pcd_if_nonempty("scans.pcd", final_save_cloud);
    }

    fout_out.close();
    fout_pre.close();
    rclcpp::shutdown();

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;    
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(),"w");
        fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0;i<time_log_counter; i++){
            fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}

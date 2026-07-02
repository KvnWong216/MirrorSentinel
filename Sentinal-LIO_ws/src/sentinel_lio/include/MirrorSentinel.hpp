#ifndef MIRROR_SENTINEL_HPP
#define MIRROR_SENTINEL_HPP

#include <rclcpp/rclcpp.hpp>
#include "ros2_compat.hpp"
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <Eigen/Dense>
#include <mutex>
#include <random>
#include <vector>

#include "ikd-Tree/ikd_Tree.h"

// 确保点云类型带有 intensity，用于提取门框/结构边缘
typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

struct MirrorFrameStats {
    size_t input_points = 0;
    size_t output_points = 0;
    size_t masked_points = 0;
    size_t depth_checked_points = 0;
    size_t depth_inconsistent_points = 0;
    size_t ghost_candidate_points = 0;
    size_t mask_core_points = 0;
    size_t mask_boundary_points = 0;
    float mean_confidence = 0.0f;
    float mask_coverage = 0.0f;
    float depth_valid_ratio = 0.0f;
    float mean_depth_residual = 0.0f;
    bool depth_calibration_valid = false;
    size_t depth_calibration_points = 0;
    float depth_scale = 1.0f;
    float depth_shift = 0.0f;
    float calibration_mean_raw_residual = 0.0f;
    float calibration_mean_calibrated_residual = 0.0f;
    bool explicit_mask_enabled = false;
    int ab_mode = 0;
};

struct DepthCalibrationResult {
    bool valid = false;
    size_t points = 0;
    float scale = 1.0f;
    float shift = 0.0f;
    float mean_abs_raw_residual = 0.0f;
    float mean_abs_calibrated_residual = 0.0f;
};

class MirrorSentinel {
public:
    MirrorSentinel(double fx, double fy, double cx, double cy, 
                   int img_width, int img_height, 
                                     Eigen::Matrix3d R_cl, Eigen::Vector3d t_cl,
                                     float confidence_floor = 0.3f,
                                     int mask_erode_kernel = 5,
                                                                         int mask_boundary_band = 8,
                                                                         float mirror_mask_threshold = 127.0f,
                                                                         const std::string& mask_viz_topic = "/mirror_sentinel/mask_viz",
                                                                                                                                                 const std::string& mask_viz_frame_id = "camera_init",
                                                                                                                                                 const std::string& mask_overlay_topic = "/mirror_sentinel/mask_overlay",
                                                                                                                                                 float depth_sigma_abs = 0.15f,
                                                                                                                                                 float depth_sigma_rel = 0.04f,
                                                                                                                                                 float ghost_margin_abs = 0.15f,
                                                                                                                                                 float ghost_margin_rel = 0.02f,
	                                                                                                                                                 float mirror_surface_confidence = 0.35f,
	                                                                                                                                                 float mask_boundary_confidence = 0.50f,
	                                                                                                                                                 float invalid_depth_confidence = 0.80f,
	                                                                                                                                                 float reflective_invalid_depth_confidence = 0.15f,
	                                                                                                                                                 bool map_depth_gate_enable = true,
	                                                                                                                                                 bool map_mask_gate_enable = true,
	                                                                                                                                                 float map_depth_ghost_margin_abs = 0.80f,
	                                                                                                                                                 float map_depth_ghost_margin_rel = 0.03f,
	                                                                                                                                                 float map_mask_foreground_keep_margin = 0.20f,
	                                                                                                                                                 bool map_mask_require_depth_confirmation = true,
	                                                                                                                                                 bool map_mask_invalid_depth_reject = false,
	                                                                                                                                                 bool map_depth_require_calibration = true,
	                                                                                                                                                 bool global_depth_prior_enable = false,
	                                                                                                                                                 bool depth_calibration_enable = true,
	                                                                                                                                                 bool depth_calibration_use_non_mask = true,
	                                                                                                                                                 int depth_calibration_min_points = 50,
	                                                                                                                                                 float depth_calibration_scale_min = 0.50f,
	                                                                                                                                                 float depth_calibration_scale_max = 2.00f,
	                                                                                                                                                 float depth_calibration_max_raw_residual = 1.50f,
	                                                                                                                                                 bool use_depth_consistency = true)
        : fx_(fx), fy_(fy), cx_(cx), cy_(cy), 
          img_width_(img_width), img_height_(img_height), 
                    R_cl_(R_cl), t_cl_(t_cl),
                    confidence_floor_(confidence_floor),
                    mask_erode_kernel_(mask_erode_kernel),
                                        mask_boundary_band_(mask_boundary_band),
                                        mirror_mask_threshold_(mirror_mask_threshold),
                                        mask_viz_topic_(mask_viz_topic),
                                                                                mask_viz_frame_id_(mask_viz_frame_id),
                                                                                mask_overlay_topic_(mask_overlay_topic),
                                                                                depth_sigma_abs_(depth_sigma_abs),
                                                                                depth_sigma_rel_(depth_sigma_rel),
                                                                                ghost_margin_abs_(ghost_margin_abs),
                                                                                ghost_margin_rel_(ghost_margin_rel),
	                                                                                mirror_surface_confidence_(mirror_surface_confidence),
	                                                                                mask_boundary_confidence_(mask_boundary_confidence),
	                                                                                invalid_depth_confidence_(invalid_depth_confidence),
	                                                                                reflective_invalid_depth_confidence_(reflective_invalid_depth_confidence),
	                                                                                map_depth_gate_enable_(map_depth_gate_enable),
	                                                                                map_mask_gate_enable_(map_mask_gate_enable),
	                                                                                map_depth_ghost_margin_abs_(map_depth_ghost_margin_abs),
	                                                                                map_depth_ghost_margin_rel_(map_depth_ghost_margin_rel),
	                                                                                map_mask_foreground_keep_margin_(map_mask_foreground_keep_margin),
	                                                                                map_mask_require_depth_confirmation_(map_mask_require_depth_confirmation),
	                                                                                map_mask_invalid_depth_reject_(map_mask_invalid_depth_reject),
	                                                                                map_depth_require_calibration_(map_depth_require_calibration),
	                                                                                global_depth_prior_enable_(global_depth_prior_enable),
	                                                                                depth_calibration_enable_(depth_calibration_enable),
	                                                                                depth_calibration_use_non_mask_(depth_calibration_use_non_mask),
	                                                                                depth_calibration_min_points_(depth_calibration_min_points),
	                                                                                depth_calibration_scale_min_(depth_calibration_scale_min),
	                                                                                depth_calibration_scale_max_(depth_calibration_scale_max),
	                                                                                depth_calibration_max_raw_residual_(depth_calibration_max_raw_residual),
	                                                                                use_depth_consistency_(use_depth_consistency)
    {
        auto node = sentinel_lio_ros2::NodeContext::node();
        if (node) {
            pub_ghost_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("/mirror_sentinel/ghost_points", 1);
            pub_mask_viz_ = node->create_publisher<sensor_msgs::msg::Image>(mask_viz_topic_, 1);
            pub_mask_overlay_ = node->create_publisher<sensor_msgs::msg::Image>(mask_overlay_topic_, 1);
        }
        
        std::random_device rd;
        rng_ = std::mt19937(rd());
        dist_ = std::uniform_real_distribution<float>(0.0, 1.0);

        ROS_INFO("\033[1;32m[MirrorSentinel] Initialized with Decoupled Logic: Soft Weighting Front-end & Strict Clean Back-end.\033[0m");
    }

    void updateDepthMap(const cv::Mat& depth_image) {
        std::lock_guard<std::mutex> lock(depth_mutex);
        if (depth_image.empty()) {
            current_visual_depth_.release();
            depth_valid_ratio_ = 0.0f;
            return;
        }

        current_visual_depth_ = depth_image.clone();
        const int total_pixels = current_visual_depth_.rows * current_visual_depth_.cols;
        if (total_pixels <= 0) {
            depth_valid_ratio_ = 0.0f;
            return;
        }

        const int valid_pixels = cv::countNonZero(current_visual_depth_ > 0.1f);
        depth_valid_ratio_ = static_cast<float>(valid_pixels) / static_cast<float>(total_pixels);
    }

    void updateRgbImage(const cv::Mat& rgb_image) {
        std::lock_guard<std::mutex> lock(depth_mutex);
        if (rgb_image.empty()) {
            current_rgb_image_.release();
            return;
        }

        cv::Mat bgr_image;
        if (rgb_image.channels() == 3) {
            bgr_image = rgb_image.clone();
        } else if (rgb_image.channels() == 4) {
            cv::cvtColor(rgb_image, bgr_image, cv::COLOR_BGRA2BGR);
        } else {
            cv::cvtColor(rgb_image, bgr_image, cv::COLOR_GRAY2BGR);
        }

        if (bgr_image.rows != img_height_ || bgr_image.cols != img_width_) {
            cv::resize(bgr_image, bgr_image, cv::Size(img_width_, img_height_), 0.0, 0.0, cv::INTER_LINEAR);
        }

        current_rgb_image_ = bgr_image;
        publishMaskOverlay();
    }

    void updateMirrorMask(const cv::Mat& mask_image) {
        std::lock_guard<std::mutex> lock(depth_mutex);
        if (mask_image.empty()) {
            current_mirror_mask_raw_.release();
            current_mirror_mask_.release();
            current_mirror_distance_.release();
            mirror_mask_valid_ratio_ = 0.0f;
            has_mirror_mask_ = false;
            return;
        }

        cv::Mat mask_gray;
        if (mask_image.channels() == 1) {
            mask_gray = mask_image.clone();
        } else {
            cv::cvtColor(mask_image, mask_gray, cv::COLOR_BGR2GRAY);
        }

        if (mask_gray.rows != img_height_ || mask_gray.cols != img_width_) {
            cv::resize(mask_gray, mask_gray, cv::Size(img_width_, img_height_), 0.0, 0.0, cv::INTER_LINEAR);
        }

        if (mask_gray.type() != CV_8UC1) {
            cv::Mat mask_u8;
            double min_val = 0.0;
            double max_val = 0.0;
            cv::minMaxLoc(mask_gray, &min_val, &max_val);
            const double scale = (max_val <= 1.5) ? 255.0 : 1.0;
            mask_gray.convertTo(mask_u8, CV_8UC1, scale);
            mask_gray = mask_u8;
        } else {
            double min_val = 0.0;
            double max_val = 0.0;
            cv::minMaxLoc(mask_gray, &min_val, &max_val);
            if (max_val <= 1.5) {
                mask_gray = mask_gray * 255;
            }
        }

        cv::threshold(mask_gray, current_mirror_mask_raw_, mirror_mask_threshold_, 255, cv::THRESH_BINARY);
        current_mirror_mask_ = current_mirror_mask_raw_.clone();

        if (mask_erode_kernel_ > 1) {
            const int kernel_size = mask_erode_kernel_ % 2 == 0 ? mask_erode_kernel_ + 1 : mask_erode_kernel_;
            const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
            cv::erode(current_mirror_mask_, current_mirror_mask_, kernel);
        }

        if (cv::countNonZero(current_mirror_mask_) == 0) {
            current_mirror_mask_ = current_mirror_mask_raw_.clone();
        }

        current_mirror_distance_.release();
        if (mask_boundary_band_ > 0) {
            cv::Mat distance_source = 255 - current_mirror_mask_;
            cv::distanceTransform(distance_source, current_mirror_distance_, cv::DIST_L2, 3);
        }

        const int total_pixels = current_mirror_mask_raw_.rows * current_mirror_mask_raw_.cols;
        if (total_pixels <= 0) {
            current_mirror_mask_raw_.release();
            current_mirror_mask_.release();
            current_mirror_distance_.release();
            mirror_mask_valid_ratio_ = 0.0f;
            has_mirror_mask_ = false;
            return;
        }

        mirror_mask_valid_ratio_ = static_cast<float>(cv::countNonZero(current_mirror_mask_raw_)) /
                                   static_cast<float>(total_pixels);
        has_mirror_mask_ = mirror_mask_valid_ratio_ > 0.0f;

        publishMaskViz();
        publishMaskOverlay();
    }

    PointCloudXYZI::Ptr assignConfidence(PointCloudXYZI::Ptr cloud_in,
                                         bool explicit_mask_enabled,
                                         bool apply_frontend_confidence,
                                         MirrorFrameStats* stats) {
        if (!cloud_in) {
            return cloud_in;
        }

        std::lock_guard<std::mutex> lock(depth_mutex);
        PointCloudXYZI::Ptr cloud_out = apply_frontend_confidence ? PointCloudXYZI::Ptr(new PointCloudXYZI(*cloud_in)) : cloud_in;
        updatePriorStateUnlocked(cloud_in, explicit_mask_enabled, apply_frontend_confidence ? cloud_out.get() : nullptr, stats);

        return cloud_out;
    }

    void updatePriorState(PointCloudXYZI::Ptr cloud_in,
                          bool explicit_mask_enabled,
                          MirrorFrameStats* stats) {
        if (!cloud_in) {
            return;
        }

        std::lock_guard<std::mutex> lock(depth_mutex);
        updatePriorStateUnlocked(cloud_in, explicit_mask_enabled, nullptr, stats);
    }

    PointCloudXYZI::Ptr assignConfidence(PointCloudXYZI::Ptr cloud_in,
                                         bool explicit_mask_enabled,
                                         MirrorFrameStats* stats) {
        return assignConfidence(cloud_in, explicit_mask_enabled, true, stats);
    }

    PointCloudXYZI::Ptr assignConfidence(PointCloudXYZI::Ptr cloud_in) {
        return assignConfidence(cloud_in, true, nullptr);
    }

	    bool hasMirrorMask() const {
	        return has_mirror_mask_ && !current_mirror_mask_raw_.empty();
	    }

	    bool hasActivePrior(bool explicit_mask_enabled) const {
	        std::lock_guard<std::mutex> lock(depth_mutex);
	        return hasActivePriorUnlocked(explicit_mask_enabled);
	    }

	    std::vector<int> mapKeepIndices(PointCloudXYZI::Ptr cloud_in,
	                                    bool explicit_mask_enabled) const {
	        std::vector<int> keep_indices;
	        if (!cloud_in) {
	            return keep_indices;
	        }

	        keep_indices.reserve(cloud_in->size());
	        std::lock_guard<std::mutex> lock(depth_mutex);
	        if (!hasActivePriorUnlocked(explicit_mask_enabled)) {
	            for (int i = 0; i < static_cast<int>(cloud_in->size()); ++i) {
	                keep_indices.push_back(i);
	            }
	            return keep_indices;
	        }

	        for (int i = 0; i < static_cast<int>(cloud_in->size()); ++i) {
	            if (shouldKeepPointForMapUnlocked(cloud_in->points[i], explicit_mask_enabled)) {
	                keep_indices.push_back(i);
	            }
	        }
	        return keep_indices;
	    }

	    bool shouldKeepMapPoint(const PointType& pt,
	                            bool explicit_mask_enabled,
	                            PointType* surface_anchor_body = nullptr) const {
	        std::lock_guard<std::mutex> lock(depth_mutex);
	        return shouldKeepPointForMapUnlocked(pt, explicit_mask_enabled, surface_anchor_body);
	    }

	    PointCloudXYZI::Ptr filterMapCloud(PointCloudXYZI::Ptr cloud_in,
	                                       bool explicit_mask_enabled) const {
	        if (!cloud_in || cloud_in->empty()) {
	            return cloud_in;
	        }

	        std::lock_guard<std::mutex> lock(depth_mutex);
	        if (!hasActivePriorUnlocked(explicit_mask_enabled)) {
	            return cloud_in;
	        }

	        PointCloudXYZI::Ptr cloud_out(new PointCloudXYZI());
	        cloud_out->reserve(cloud_in->size());
	        for (const auto& pt : cloud_in->points) {
	            if (shouldKeepPointForMapUnlocked(pt, explicit_mask_enabled)) {
	                cloud_out->push_back(pt);
	            }
	        }
	        return cloud_out;
	    }

    // =======================================================
    // [模块 1] 前端：密度降权 (等效 Soft Weighting) + 边缘保底
    // 目标：给 LIO 提供足够且不被带偏的约束
    // =======================================================
    PointCloudXYZI::Ptr filterCurrentScan(PointCloudXYZI::Ptr cloud_in) {
        std::lock_guard<std::mutex> lock(depth_mutex);
        PointCloudXYZI::Ptr cloud_out(new PointCloudXYZI());
        PointCloudXYZI::Ptr discarded_cloud(new PointCloudXYZI());

        if (!cloud_in || cloud_in->empty()) {
            return cloud_in;
        }

        if (!hasMirrorMask() && !isDepthReliable()) {
            return cloud_in;
        }

        const float safe_dist = 1.5f;
        const float transition_dist = 5.0f;
        const float high_intensity_threshold = 100.0f;
        const float ghost_depth_margin = 0.8f;

        for (const auto& pt : cloud_in->points) {
            Eigen::Vector3d p_l(pt.x, pt.y, pt.z);
            const float dist_xy_lio = std::sqrt(p_l.x() * p_l.x() + p_l.y() * p_l.y());

            if (dist_xy_lio < safe_dist || pt.intensity > high_intensity_threshold) {
                cloud_out->push_back(pt);
                continue;
            }

            if (hasMirrorMask() && isPointInMirrorMask(pt)) {
                if (dist_(rng_) < confidence_floor_) {
                    cloud_out->push_back(pt);
                } else {
                    discarded_cloud->push_back(pt);
                }
                continue;
            }

            Eigen::Vector3d p_c = R_cl_ * p_l + t_cl_;
            if (p_c.z() > 0.1f) {
                const int u = std::round(fx_ * p_c.x() / p_c.z() + cx_);
                const int v = std::round(fy_ * p_c.y() / p_c.z() + cy_);
                if (u >= 0 && u < img_width_ && v >= 0 && v < img_height_) {
                    const float z_vfm = current_visual_depth_.at<float>(v, u);
                    if (z_vfm > 0.1f && (p_c.z() - z_vfm) >= ghost_depth_margin) {
                        if (dist_(rng_) < confidence_floor_) {
                            cloud_out->push_back(pt);
                        } else {
                            discarded_cloud->push_back(pt);
                        }
                        continue;
                    }
                    cloud_out->push_back(pt);
                    continue;
                }
            }

            if (dist_xy_lio < transition_dist) {
                const float keep_probability = 1.0f - (dist_xy_lio - safe_dist) / (transition_dist - safe_dist);
                if (dist_(rng_) < keep_probability) {
                    cloud_out->push_back(pt);
                } else {
                    discarded_cloud->push_back(pt);
                }
            } else {
                discarded_cloud->push_back(pt);
            }
        }

        if (pub_ghost_ && pub_ghost_->get_subscription_count() > 0 && !discarded_cloud->empty()) {
            sensor_msgs::msg::PointCloud2 msg_ghost;
            pcl::toROSMsg(*discarded_cloud, msg_ghost);
            msg_ghost.header.frame_id = "body"; 
            msg_ghost.header.stamp = sentinel_lio_ros2::NodeContext::now();
            pub_ghost_->publish(msg_ghost);
        }

        return cloud_out;
    }

    // =======================================================
    // [模块 2] 后端：地图渲染不需要约束，对鬼影执行 100% 斩杀
    // 目标：保证输出的点云地图里绝对干净
    // =======================================================
    void detectMapGhosts(const Eigen::Matrix4d& T_w_l, 
                         const KD_TREE<PointType>::PointVector& nearby_points,
                         KD_TREE<PointType>::PointVector& points_to_delete) {
        std::lock_guard<std::mutex> lock(depth_mutex);
        if (!hasMirrorMask() && !isDepthReliable()) return;

        Eigen::Matrix4d T_l_w = T_w_l.inverse();
        points_to_delete.clear();
        
        const float safe_dist = 1.5f;   
        const float transition_dist = 5.0f; 

        for (auto &pt_w : nearby_points) {
            Eigen::Vector3d p_w(pt_w.x, pt_w.y, pt_w.z);
            Eigen::Vector3d p_l = (T_l_w.block<3,3>(0,0) * p_w + T_l_w.block<3,1>(0,3));
            float dist_xy_lio = std::sqrt(p_l.x() * p_l.x() + p_l.y() * p_l.y());

            // 保命区：地图中也保留地板等近处结构
            if (dist_xy_lio < safe_dist) continue;

            Eigen::Vector3d p_c = R_cl_ * p_l + t_cl_;
            bool in_fov_frustum = false;
            int u = 0, v = 0;
            if (p_c.z() > 0.1f) {
                u = std::round(fx_ * p_c.x() / p_c.z() + cx_);
                v = std::round(fy_ * p_c.y() / p_c.z() + cy_);
                if (u >= 0 && u < img_width_ && v >= 0 && v < img_height_) {
                    in_fov_frustum = true;
                }
            }

            if (hasMirrorMask()) {
                if (in_fov_frustum && current_mirror_mask_.at<uchar>(v, u) > 0) {
                    points_to_delete.push_back(pt_w);
                }
                continue;
            }

            // 【兜底逻辑】：没有显式镜面 mask 时，保留旧的深度差判定
            if (in_fov_frustum) {
                float z_vfm = current_visual_depth_.at<float>(v, u);
                if (z_vfm > 0.1f && (p_c.z() - z_vfm) >= 0.8f) {
                    points_to_delete.push_back(pt_w); 
                }
                continue;
            }

            // 视野外盲区处理
            if (dist_xy_lio < transition_dist) {
                float keep_probability = 1.0f - (dist_xy_lio - safe_dist) / (transition_dist - safe_dist);
                // 建图时可以稍微比前端严格一点，概率没命中就删
                if (dist_(rng_) >= keep_probability) {
                    points_to_delete.push_back(pt_w); 
                }
            } else {
                points_to_delete.push_back(pt_w); 
            }
        }
    }

private:
	    void resetStatsUnlocked(PointCloudXYZI::Ptr cloud_in,
	                            bool explicit_mask_enabled,
	                            MirrorFrameStats* stats) const {
	        if (!stats) {
	            return;
	        }
	        const size_t point_count = cloud_in ? cloud_in->size() : 0;
	        stats->input_points = point_count;
	        stats->output_points = point_count;
	        stats->masked_points = 0;
	        stats->depth_checked_points = 0;
	        stats->depth_inconsistent_points = 0;
	        stats->ghost_candidate_points = 0;
	        stats->mask_core_points = 0;
	        stats->mask_boundary_points = 0;
	        stats->mean_confidence = 0.0f;
	        stats->mask_coverage = has_mirror_mask_ ? mirror_mask_valid_ratio_ : 0.0f;
	        stats->depth_valid_ratio = depth_valid_ratio_;
	        stats->mean_depth_residual = 0.0f;
	        stats->depth_calibration_valid = false;
	        stats->depth_calibration_points = 0;
	        stats->depth_scale = 1.0f;
	        stats->depth_shift = 0.0f;
	        stats->calibration_mean_raw_residual = 0.0f;
	        stats->calibration_mean_calibrated_residual = 0.0f;
	        stats->explicit_mask_enabled = explicit_mask_enabled;
	    }

	    void updatePriorStateUnlocked(PointCloudXYZI::Ptr cloud_in,
	                                  bool explicit_mask_enabled,
	                                  PointCloudXYZI* frontend_cloud_out,
	                                  MirrorFrameStats* stats) {
	        resetStatsUnlocked(cloud_in, explicit_mask_enabled, stats);
	        if (!cloud_in || cloud_in->empty()) {
	            return;
	        }

	        if (!hasActivePriorUnlocked(explicit_mask_enabled)) {
	            if (stats) {
	                stats->mean_confidence = 1.0f;
	            }
	            return;
	        }

	        const DepthCalibrationResult calibration = estimateDepthCalibrationUnlocked(cloud_in, explicit_mask_enabled);
	        current_depth_calibration_ = calibration;
	        if (stats) {
	            stats->depth_calibration_valid = calibration.valid;
	            stats->depth_calibration_points = calibration.points;
	            stats->depth_scale = calibration.scale;
	            stats->depth_shift = calibration.shift;
	            stats->calibration_mean_raw_residual = calibration.mean_abs_raw_residual;
	            stats->calibration_mean_calibrated_residual = calibration.mean_abs_calibrated_residual;
	        }

	        if (!stats && !frontend_cloud_out) {
	            return;
	        }

	        size_t masked_points = 0;
	        double confidence_sum = 0.0;
	        double depth_residual_sum = 0.0;
	        for (size_t i = 0; i < cloud_in->points.size(); ++i) {
	            float depth_residual = 0.0f;
	            const float confidence = computePointConfidence(
	                cloud_in->points[i],
	                explicit_mask_enabled,
	                calibration,
	                stats,
	                &depth_residual);
	            if (confidence < 0.999f) {
	                masked_points++;
	            }
	            if (frontend_cloud_out && i < frontend_cloud_out->points.size()) {
	                frontend_cloud_out->points[i].intensity = confidence;
	            }
	            confidence_sum += confidence;
	            depth_residual_sum += depth_residual;
	        }

	        if (stats) {
	            stats->masked_points = masked_points;
	            stats->output_points = frontend_cloud_out ? frontend_cloud_out->size() : cloud_in->size();
	            stats->mean_confidence = cloud_in->empty() ? 0.0f :
	                static_cast<float>(confidence_sum / static_cast<double>(cloud_in->size()));
	            stats->mean_depth_residual = stats->depth_checked_points == 0 ? 0.0f :
	                static_cast<float>(depth_residual_sum / static_cast<double>(stats->depth_checked_points));
	        }
	    }

	    bool isDepthReliable() const {
	        return !current_visual_depth_.empty() && depth_valid_ratio_ >= min_depth_valid_ratio_;
	    }

	    bool hasActivePriorUnlocked(bool explicit_mask_enabled) const {
	        const bool mask_prior_active = explicit_mask_enabled && hasMirrorMask();
	        const bool depth_prior_active = use_depth_consistency_ && isDepthReliable() &&
	                                        (global_depth_prior_enable_ || mask_prior_active);
	        return depth_prior_active || mask_prior_active;
	    }

	    float mapDepthGhostMargin(float z) const {
	        return map_depth_ghost_margin_abs_ + map_depth_ghost_margin_rel_ * std::max(0.0f, z);
	    }

	    float calibratedDepth(float z_vfm, const DepthCalibrationResult& calibration) const {
	        if (!calibration.valid) {
	            return z_vfm;
	        }
	        return calibration.scale * z_vfm + calibration.shift;
	    }

	    bool isPixelInAnyMaskUnlocked(int u, int v) const {
	        if (u < 0 || u >= img_width_ || v < 0 || v >= img_height_) {
	            return false;
	        }
	        if (!current_mirror_mask_.empty() && current_mirror_mask_.at<uchar>(v, u) > 0) {
	            return true;
	        }
	        if (!current_mirror_mask_raw_.empty() && current_mirror_mask_raw_.at<uchar>(v, u) > 0) {
	            return true;
	        }
	        return false;
	    }

	    DepthCalibrationResult estimateDepthCalibrationUnlocked(PointCloudXYZI::Ptr cloud_in,
	                                                            bool explicit_mask_enabled) const {
	        DepthCalibrationResult result;
	        if (!depth_calibration_enable_ || !use_depth_consistency_ || !isDepthReliable() || !cloud_in) {
	            return result;
	        }

	        std::vector<float> ratios;
	        ratios.reserve(cloud_in->size());
	        double raw_abs_sum = 0.0;
	        size_t raw_count = 0;

	        const bool use_mask_exclusion = depth_calibration_use_non_mask_ && explicit_mask_enabled && hasMirrorMask();
	        for (const auto& pt : cloud_in->points) {
	            Eigen::Vector3d p_c;
	            int u = 0;
	            int v = 0;
	            if (!projectPointToCamera(pt, p_c, u, v)) {
	                continue;
	            }
	            if (use_mask_exclusion && isPixelInAnyMaskUnlocked(u, v)) {
	                continue;
	            }
	            const float z_lidar = static_cast<float>(p_c.z());
	            const float z_vfm = current_visual_depth_.at<float>(v, u);
	            if (z_lidar <= 0.1f || z_vfm <= 0.1f) {
	                continue;
	            }
	            const float raw_residual = std::fabs(z_lidar - z_vfm);
	            if (raw_residual > depth_calibration_max_raw_residual_) {
	                continue;
	            }
	            ratios.push_back(z_lidar / z_vfm);
	            raw_abs_sum += raw_residual;
	            raw_count++;
	        }

	        result.points = ratios.size();
	        if (ratios.size() < static_cast<size_t>(std::max(1, depth_calibration_min_points_))) {
	            return result;
	        }

	        const auto mid = ratios.begin() + static_cast<long>(ratios.size() / 2);
	        std::nth_element(ratios.begin(), mid, ratios.end());
	        float scale = *mid;
	        if (ratios.size() % 2 == 0 && ratios.size() > 1) {
	            const auto mid_low = ratios.begin() + static_cast<long>(ratios.size() / 2 - 1);
	            std::nth_element(ratios.begin(), mid_low, ratios.end());
	            scale = 0.5f * (scale + *mid_low);
	        }
	        if (!std::isfinite(scale)) {
	            return result;
	        }
	        scale = std::max(depth_calibration_scale_min_, std::min(depth_calibration_scale_max_, scale));

	        double calibrated_abs_sum = 0.0;
	        size_t calibrated_count = 0;
	        for (const auto& pt : cloud_in->points) {
	            Eigen::Vector3d p_c;
	            int u = 0;
	            int v = 0;
	            if (!projectPointToCamera(pt, p_c, u, v)) {
	                continue;
	            }
	            if (use_mask_exclusion && isPixelInAnyMaskUnlocked(u, v)) {
	                continue;
	            }
	            const float z_lidar = static_cast<float>(p_c.z());
	            const float z_vfm = current_visual_depth_.at<float>(v, u);
	            if (z_lidar <= 0.1f || z_vfm <= 0.1f) {
	                continue;
	            }
	            const float raw_residual = std::fabs(z_lidar - z_vfm);
	            if (raw_residual > depth_calibration_max_raw_residual_) {
	                continue;
	            }
	            calibrated_abs_sum += std::fabs(z_lidar - scale * z_vfm);
	            calibrated_count++;
	        }

	        result.valid = true;
	        result.scale = scale;
	        result.shift = 0.0f;
	        result.mean_abs_raw_residual = raw_count == 0 ? 0.0f :
	            static_cast<float>(raw_abs_sum / static_cast<double>(raw_count));
	        result.mean_abs_calibrated_residual = calibrated_count == 0 ? 0.0f :
	            static_cast<float>(calibrated_abs_sum / static_cast<double>(calibrated_count));
	        return result;
	    }

	    bool depthSurfaceAnchorUnlocked(const Eigen::Vector3d& p_c,
	                                    int u,
	                                    int v,
	                                    PointType& anchor_body) const {
	        if (current_visual_depth_.empty() || u < 0 || u >= img_width_ || v < 0 || v >= img_height_) {
	            return false;
	        }
	        const float z_vfm = current_visual_depth_.at<float>(v, u);
	        if (!std::isfinite(z_vfm) || z_vfm <= 0.1f || p_c.z() <= 0.1) {
	            return false;
	        }
	        const float z_calibrated = calibratedDepth(z_vfm, current_depth_calibration_);
	        if (!std::isfinite(z_calibrated) || z_calibrated <= 0.1f) {
	            return false;
	        }

	        const Eigen::Vector3d p_anchor_c = p_c * (static_cast<double>(z_calibrated) / p_c.z());
	        const Eigen::Vector3d p_anchor_l = R_cl_.transpose() * (p_anchor_c - t_cl_);
	        if (!std::isfinite(p_anchor_l.x()) || !std::isfinite(p_anchor_l.y()) || !std::isfinite(p_anchor_l.z())) {
	            return false;
	        }

	        anchor_body = PointType();
	        anchor_body.x = static_cast<float>(p_anchor_l.x());
	        anchor_body.y = static_cast<float>(p_anchor_l.y());
	        anchor_body.z = static_cast<float>(p_anchor_l.z());
	        anchor_body.intensity = z_calibrated;
	        return true;
	    }

	    bool setDepthSurfaceAnchorUnlocked(const Eigen::Vector3d& p_c,
	                                       int u,
	                                       int v,
	                                       PointType* surface_anchor_body) const {
	        if (!surface_anchor_body) {
	            return false;
	        }
	        return depthSurfaceAnchorUnlocked(p_c, u, v, *surface_anchor_body);
	    }

	    bool shouldKeepPointForMapUnlocked(const PointType& pt,
	                                       bool explicit_mask_enabled,
	                                       PointType* surface_anchor_body = nullptr) const {
	        if (!hasActivePriorUnlocked(explicit_mask_enabled)) {
	            return true;
	        }

	        Eigen::Vector3d p_c;
	        int u = 0;
	        int v = 0;
	        const bool in_fov = projectPointToCamera(pt, p_c, u, v);
	        if (!in_fov) {
	            return true;
	        }

	        const bool mask_prior_active = explicit_mask_enabled && hasMirrorMask();
	        const bool depth_ready = use_depth_consistency_ && isDepthReliable() &&
	                                 (global_depth_prior_enable_ || mask_prior_active) &&
	                                 (!map_depth_require_calibration_ || current_depth_calibration_.valid);
	        float residual = 0.0f;
	        bool has_valid_depth = false;
	        if (depth_ready) {
	            const float z_vfm = current_visual_depth_.at<float>(v, u);
	            if (z_vfm > 0.1f) {
	                residual = static_cast<float>(p_c.z()) - calibratedDepth(z_vfm, current_depth_calibration_);
	                has_valid_depth = true;
	            }
	        }

	        if (explicit_mask_enabled && hasMirrorMask()) {
	            const bool in_mask_core = !current_mirror_mask_.empty() && current_mirror_mask_.at<uchar>(v, u) > 0;
	            const bool in_mask_raw = !current_mirror_mask_raw_.empty() && current_mirror_mask_raw_.at<uchar>(v, u) > 0;

	            if (map_mask_gate_enable_ && in_mask_core) {
	                if (!has_valid_depth) {
	                    return !map_mask_invalid_depth_reject_;
	                }
	                if (residual < -map_mask_foreground_keep_margin_) {
	                    return true;
	                }
	                if (residual > mapDepthGhostMargin(static_cast<float>(p_c.z()))) {
	                    setDepthSurfaceAnchorUnlocked(p_c, u, v, surface_anchor_body);
	                    return false;
	                }
	                return map_mask_require_depth_confirmation_;
	            }

	            if (map_mask_gate_enable_ && in_mask_raw && has_valid_depth &&
	                residual > mapDepthGhostMargin(static_cast<float>(p_c.z()))) {
	                setDepthSurfaceAnchorUnlocked(p_c, u, v, surface_anchor_body);
	                return false;
	            }
	        }

	        if (global_depth_prior_enable_ && map_depth_gate_enable_ && has_valid_depth &&
	            residual > mapDepthGhostMargin(static_cast<float>(p_c.z()))) {
	            setDepthSurfaceAnchorUnlocked(p_c, u, v, surface_anchor_body);
	            return false;
	        }

	        return true;
	    }

    bool isPointInMirrorMask(const PointType& pt) const {
        if (!hasMirrorMask()) {
            return false;
        }

        int u = 0;
        int v = 0;
        if (!projectPointToImage(pt, u, v)) {
            return false;
        }

        return current_mirror_mask_.at<uchar>(v, u) > 0;
    }

    bool projectPointToImage(const PointType& pt, int& u, int& v) const {

        Eigen::Vector3d p_l(pt.x, pt.y, pt.z);
        Eigen::Vector3d p_c = R_cl_ * p_l + t_cl_;
        if (p_c.z() <= 0.1f) {
            return false;
        }

        u = std::round(fx_ * p_c.x() / p_c.z() + cx_);
        v = std::round(fy_ * p_c.y() / p_c.z() + cy_);
        return (u >= 0 && u < img_width_ && v >= 0 && v < img_height_);
    }

    float clampConfidence(float value) const {
        return std::max(confidence_floor_, std::min(1.0f, value));
    }

    bool projectPointToCamera(const PointType& pt, Eigen::Vector3d& p_c, int& u, int& v) const {
        Eigen::Vector3d p_l(pt.x, pt.y, pt.z);
        p_c = R_cl_ * p_l + t_cl_;
        if (p_c.z() <= 0.1f) {
            return false;
        }

        u = std::round(fx_ * p_c.x() / p_c.z() + cx_);
        v = std::round(fy_ * p_c.y() / p_c.z() + cy_);
        return (u >= 0 && u < img_width_ && v >= 0 && v < img_height_);
    }

    float computePointConfidence(const PointType& pt,
                                 bool explicit_mask_enabled,
                                 const DepthCalibrationResult& calibration,
                                 MirrorFrameStats* stats,
                                 float* depth_residual_out) const {
        Eigen::Vector3d p_c;
        int u = 0;
        int v = 0;
        const bool in_fov = projectPointToCamera(pt, p_c, u, v);
        const bool use_mask = explicit_mask_enabled && hasMirrorMask() && in_fov;
        bool in_mask_core = false;
        bool in_mask_raw = false;
        bool in_mask_boundary = false;

        float mask_confidence = 1.0f;
        if (use_mask) {
            in_mask_core = !current_mirror_mask_.empty() && current_mirror_mask_.at<uchar>(v, u) > 0;
            in_mask_raw = !current_mirror_mask_raw_.empty() && current_mirror_mask_raw_.at<uchar>(v, u) > 0;
            in_mask_boundary = in_mask_raw && !in_mask_core;

            if (in_mask_core) {
                mask_confidence = mirror_surface_confidence_;
                if (stats) stats->mask_core_points++;
            } else if (in_mask_boundary) {
                if (current_mirror_distance_.empty()) {
                    mask_confidence = mask_boundary_confidence_;
                } else {
                    const float band = std::max(1, mask_boundary_band_);
                    const float distance_to_core = current_mirror_distance_.at<float>(v, u);
                    float normalized = std::max(0.0f, std::min(1.0f, distance_to_core / band));
                    normalized = normalized * normalized * (3.0f - 2.0f * normalized);
                    mask_confidence = mask_boundary_confidence_ + (1.0f - mask_boundary_confidence_) * normalized;
                }
                if (stats) stats->mask_boundary_points++;
            }
        }

        const bool depth_allowed_here = global_depth_prior_enable_ ||
            (use_mask && (in_mask_core || in_mask_boundary));
        if (!use_depth_consistency_ || !isDepthReliable() || !in_fov || !depth_allowed_here) {
            return clampConfidence(mask_confidence);
        }

        const float z_vfm = current_visual_depth_.at<float>(v, u);
        if (z_vfm <= 0.1f) {
            if (use_mask && (in_mask_core || in_mask_boundary)) {
                return clampConfidence(mask_confidence * reflective_invalid_depth_confidence_);
            }
            return clampConfidence(mask_confidence * invalid_depth_confidence_);
        }

        const float z_calibrated = calibratedDepth(z_vfm, calibration);
        const float residual = static_cast<float>(p_c.z()) - z_calibrated;
        const float abs_residual = std::fabs(residual);
        const float sigma = std::max(0.05f, depth_sigma_abs_ + depth_sigma_rel_ * static_cast<float>(p_c.z()));
        const float ghost_margin = ghost_margin_abs_ + ghost_margin_rel_ * static_cast<float>(p_c.z());
        float depth_confidence = 1.0f;

        if (stats) {
            stats->depth_checked_points++;
        }
        if (depth_residual_out) {
            *depth_residual_out = abs_residual;
        }

        if (residual > ghost_margin) {
            depth_confidence = confidence_floor_;
            if (stats) {
                stats->depth_inconsistent_points++;
                stats->ghost_candidate_points++;
            }
        } else if (residual > 0.0f) {
            const float normalized = std::max(0.0f, std::min(1.0f, residual / std::max(ghost_margin, sigma)));
            depth_confidence = 1.0f - (1.0f - mirror_surface_confidence_) * normalized;
            if (stats && abs_residual > sigma) {
                stats->depth_inconsistent_points++;
            }
        } else if (abs_residual > 2.0f * sigma) {
            const float normalized = std::max(0.0f, std::min(1.0f, (abs_residual - 2.0f * sigma) / (3.0f * sigma)));
            depth_confidence = 1.0f - 0.3f * normalized;
            if (stats) {
                stats->depth_inconsistent_points++;
            }
        }

        return clampConfidence(mask_confidence * depth_confidence);
    }

    float computeMirrorConfidence(const PointType& pt) const {
        if (!hasMirrorMask()) {
            return 1.0f;
        }

        int u = 0;
        int v = 0;
        if (!projectPointToImage(pt, u, v)) {
            return 1.0f;
        }

        if (!current_mirror_mask_.empty() && current_mirror_mask_.at<uchar>(v, u) > 0) {
            return confidence_floor_;
        }

        if (!current_mirror_mask_raw_.empty() && current_mirror_mask_raw_.at<uchar>(v, u) > 0) {
            if (current_mirror_distance_.empty()) {
                return confidence_floor_;
            }

            const float band = std::max(1, mask_boundary_band_);
            const float distance_to_core = current_mirror_distance_.at<float>(v, u);
            float normalized = std::max(0.0f, std::min(1.0f, distance_to_core / band));
            normalized = normalized * normalized * (3.0f - 2.0f * normalized);
            return confidence_floor_ + (1.0f - confidence_floor_) * normalized;
        }

        return 1.0f;
    }

    float computeMaskConfidence(const PointType& pt) const {
        if (isPointInMirrorMask(pt)) {
            return confidence_floor_;
        }
        return 1.0f;
    }

    float computeLegacyDepthConfidence(const PointType& pt) const {
        const float safe_dist = 1.5f;
        const float transition_dist = 5.0f;
        const float high_intensity_threshold = 100.0f;
        const float ghost_depth_margin = 0.8f;

        Eigen::Vector3d p_l(pt.x, pt.y, pt.z);
        const float dist_xy_lio = std::sqrt(p_l.x() * p_l.x() + p_l.y() * p_l.y());

        if (dist_xy_lio < safe_dist) {
            return 1.0f;
        }

        if (pt.intensity > high_intensity_threshold) {
            return 1.0f;
        }

        Eigen::Vector3d p_c = R_cl_ * p_l + t_cl_;
        if (p_c.z() <= 0.1f) {
            if (dist_xy_lio < transition_dist) {
                const float keep_probability = 1.0f - (dist_xy_lio - safe_dist) / (transition_dist - safe_dist);
                return confidence_floor_ + (1.0f - confidence_floor_) * std::max(0.0f, std::min(1.0f, keep_probability));
            }
            return confidence_floor_;
        }

        const int u = std::round(fx_ * p_c.x() / p_c.z() + cx_);
        const int v = std::round(fy_ * p_c.y() / p_c.z() + cy_);
        const bool in_fov_frustum = (u >= 0 && u < img_width_ && v >= 0 && v < img_height_);

        if (in_fov_frustum) {
            const float z_vfm = current_visual_depth_.at<float>(v, u);
            if (z_vfm > 0.1f) {
                const float depth_delta = p_c.z() - z_vfm;
                if (depth_delta >= ghost_depth_margin) {
                    return confidence_floor_;
                }
                if (depth_delta <= 0.0f) {
                    return 1.0f;
                }

                const float normalized = depth_delta / ghost_depth_margin;
                return 1.0f - (1.0f - confidence_floor_) * std::max(0.0f, std::min(1.0f, normalized));
            }
        }

        if (dist_xy_lio < transition_dist) {
            const float keep_probability = 1.0f - (dist_xy_lio - safe_dist) / (transition_dist - safe_dist);
            return confidence_floor_ + (1.0f - confidence_floor_) * std::max(0.0f, std::min(1.0f, keep_probability));
        }

        return confidence_floor_;
    }

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_ghost_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_mask_viz_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_mask_overlay_;
    cv::Mat current_visual_depth_;
    cv::Mat current_rgb_image_;
    cv::Mat current_mirror_mask_raw_;
    cv::Mat current_mirror_mask_;
    cv::Mat current_mirror_distance_;
	    mutable std::mutex depth_mutex;
    float depth_valid_ratio_ = 0.0f;
    float mirror_mask_valid_ratio_ = 0.0f;
    const float min_depth_valid_ratio_ = 0.02f;
    bool has_mirror_mask_ = false;
    mutable DepthCalibrationResult current_depth_calibration_;

    float fx_, fy_, cx_, cy_;
    int img_width_, img_height_;
    Eigen::Matrix3d R_cl_;
    Eigen::Vector3d t_cl_;
    float confidence_floor_;
    int mask_erode_kernel_;
    int mask_boundary_band_;
    float mirror_mask_threshold_;
    float depth_sigma_abs_;
    float depth_sigma_rel_;
    float ghost_margin_abs_;
    float ghost_margin_rel_;
    float mirror_surface_confidence_;
	    float mask_boundary_confidence_;
	    float invalid_depth_confidence_;
	    float reflective_invalid_depth_confidence_;
	    bool map_depth_gate_enable_;
	    bool map_mask_gate_enable_;
	    float map_depth_ghost_margin_abs_;
	    float map_depth_ghost_margin_rel_;
	    float map_mask_foreground_keep_margin_;
	    bool map_mask_require_depth_confirmation_;
	    bool map_mask_invalid_depth_reject_;
	    bool map_depth_require_calibration_;
	    bool global_depth_prior_enable_;
	    bool depth_calibration_enable_;
	    bool depth_calibration_use_non_mask_;
	    int depth_calibration_min_points_;
	    float depth_calibration_scale_min_;
	    float depth_calibration_scale_max_;
	    float depth_calibration_max_raw_residual_;
	    bool use_depth_consistency_;
    std::string mask_viz_topic_;
    std::string mask_viz_frame_id_;
    std::string mask_overlay_topic_;

    void publishMaskViz() {
        if (!pub_mask_viz_ || pub_mask_viz_->get_subscription_count() == 0 || current_mirror_mask_.empty()) {
            return;
        }

        cv_bridge::CvImage viz_msg;
        viz_msg.header.stamp = sentinel_lio_ros2::NodeContext::now();
        viz_msg.header.frame_id = mask_viz_frame_id_;
        viz_msg.encoding = "mono8";
        viz_msg.image = current_mirror_mask_;
        pub_mask_viz_->publish(*viz_msg.toImageMsg());
    }

    void publishMaskOverlay() {
        if (!pub_mask_overlay_ || pub_mask_overlay_->get_subscription_count() == 0 || current_rgb_image_.empty()) {
            return;
        }

        cv::Mat overlay = current_rgb_image_.clone();
        if (!current_mirror_mask_.empty() && current_mirror_mask_.rows == overlay.rows && current_mirror_mask_.cols == overlay.cols) {
            cv::Mat red_layer(overlay.size(), overlay.type(), cv::Scalar(0, 0, 255));
            cv::Mat tinted;
            cv::addWeighted(overlay, 0.65, red_layer, 0.35, 0.0, tinted);
            tinted.copyTo(overlay, current_mirror_mask_);
        }

        cv_bridge::CvImage overlay_msg;
        overlay_msg.header.stamp = sentinel_lio_ros2::NodeContext::now();
        overlay_msg.header.frame_id = mask_viz_frame_id_;
        overlay_msg.encoding = "bgr8";
        overlay_msg.image = overlay;
        pub_mask_overlay_->publish(*overlay_msg.toImageMsg());
    }

    std::mt19937 rng_;
    std::uniform_real_distribution<float> dist_;
};

#endif // MIRROR_SENTINEL_HPP

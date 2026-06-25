#include <pcl/common/common.h>
#include <pcl/filters/random_sample.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <X11/Xlib.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

using PointT = pcl::PointXYZ;
using CloudT = pcl::PointCloud<PointT>;

struct Corner2 {
  double x = 0.0;
  double y = 0.0;
};

struct Options {
  std::string map_path;
  std::string output_yaml;
  std::string sequence = "2026-03-30-21-31-03_rescued";
  std::string bag = "dataset/rosbag2/2026-03-30-21-31-03_rescued";
  std::string map_frame = "camera_init";
  double z_min = std::numeric_limits<double>::quiet_NaN();
  double z_max = std::numeric_limits<double>::quiet_NaN();
  double z_step = 0.05;
  double interior_margin = 0.25;
  double exterior_margin = 1.20;
  int max_points = 60000;
  int point_size = 1;
  bool color_by_z = false;
  bool show_text = false;
  bool coordinate_system = false;
  std::vector<Corner2> corners;
};

struct State {
  std::vector<Corner2> corners;
  double z_min = -0.3;
  double z_max = 2.2;
  bool save_requested = false;
  bool quit_requested = false;
  bool overlay_dirty = true;
};

struct CallbackContext {
  State* state = nullptr;
  double z_step = 0.05;
};

static void usage(const char* argv0) {
  std::cerr
      << "Usage:\n"
      << "  " << argv0 << " --map MAP.pcd --output-yaml OUT.yaml [options]\n\n"
      << "Options:\n"
      << "  --sequence NAME              YAML sequence field\n"
      << "  --bag PATH                   YAML bag field\n"
      << "  --map-frame FRAME            YAML map_frame field\n"
      << "  --z-min Z --z-max Z          Initial cuboid vertical bounds\n"
      << "  --z-step Z                   Keyboard adjustment step, default 0.05\n"
      << "  --interior-margin M          Evaluation margin inside room, default 0.25\n"
      << "  --exterior-margin M          Evaluation margin outside mirror wall, default 1.20\n"
      << "  --max-points N               Downsample display cloud, default 60000\n"
      << "  --point-size N               Viewer point size, default 1\n\n"
      << "  --color-by-z                 Color points by z. Off by default for VTK stability\n"
      << "  --show-text                  Draw in-window labels. Off by default for VTK stability\n"
      << "  --coordinate-system          Draw PCL coordinate axis. Off by default for VTK stability\n"
      << "  --corners x1 y1 x2 y2 x3 y3 x4 y4\n"
      << "                               Non-interactive mode: write YAML directly\n\n"
      << "Interaction:\n"
      << "  Shift + left click point cloud: pick one floor corner in XY\n"
      << "  u: undo last corner\n"
      << "  r: reset corners\n"
      << "  [: lower z_min, ]: raise z_min\n"
      << "  ;: lower z_max, ': raise z_max\n"
      << "  s: save YAML after four corners are selected\n"
      << "  q or Esc: quit\n";
}

static bool parse_args(int argc, char** argv, Options& opt) {
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto need = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + name);
      }
      return argv[++i];
    };
    if (key == "--map") {
      opt.map_path = need(key);
    } else if (key == "--output-yaml") {
      opt.output_yaml = need(key);
    } else if (key == "--sequence") {
      opt.sequence = need(key);
    } else if (key == "--bag") {
      opt.bag = need(key);
    } else if (key == "--map-frame") {
      opt.map_frame = need(key);
    } else if (key == "--z-min") {
      opt.z_min = std::stod(need(key));
    } else if (key == "--z-max") {
      opt.z_max = std::stod(need(key));
    } else if (key == "--z-step") {
      opt.z_step = std::stod(need(key));
    } else if (key == "--interior-margin") {
      opt.interior_margin = std::stod(need(key));
    } else if (key == "--exterior-margin") {
      opt.exterior_margin = std::stod(need(key));
    } else if (key == "--max-points") {
      opt.max_points = std::stoi(need(key));
    } else if (key == "--point-size") {
      opt.point_size = std::stoi(need(key));
    } else if (key == "--color-by-z") {
      opt.color_by_z = true;
    } else if (key == "--show-text") {
      opt.show_text = true;
    } else if (key == "--coordinate-system") {
      opt.coordinate_system = true;
    } else if (key == "--corners") {
      if (i + 8 >= argc) {
        throw std::runtime_error("--corners expects eight numbers");
      }
      opt.corners.clear();
      for (int k = 0; k < 4; ++k) {
        const double x = std::stod(argv[++i]);
        const double y = std::stod(argv[++i]);
        opt.corners.push_back(Corner2{x, y});
      }
    } else if (key == "-h" || key == "--help") {
      usage(argv[0]);
      return false;
    } else {
      throw std::runtime_error("unknown argument: " + key);
    }
  }
  if (opt.map_path.empty() || opt.output_yaml.empty()) {
    usage(argv[0]);
    return false;
  }
  return true;
}

static double signed_area(const std::vector<Corner2>& pts) {
  double area = 0.0;
  for (std::size_t i = 0; i < pts.size(); ++i) {
    const auto& a = pts[i];
    const auto& b = pts[(i + 1) % pts.size()];
    area += a.x * b.y - a.y * b.x;
  }
  return 0.5 * area;
}

static std::vector<Corner2> order_corners_ccw(const std::vector<Corner2>& corners) {
  if (corners.size() != 4) {
    throw std::runtime_error("need exactly four corners");
  }
  Corner2 center;
  for (const auto& p : corners) {
    center.x += p.x;
    center.y += p.y;
  }
  center.x /= 4.0;
  center.y /= 4.0;
  std::vector<Corner2> out = corners;
  std::sort(out.begin(), out.end(), [&](const Corner2& a, const Corner2& b) {
    return std::atan2(a.y - center.y, a.x - center.x) <
           std::atan2(b.y - center.y, b.x - center.x);
  });
  if (std::abs(signed_area(out)) < 1e-8) {
    throw std::runtime_error("selected corners are nearly collinear");
  }
  if (signed_area(out) < 0.0) {
    std::reverse(out.begin(), out.end());
  }
  return out;
}

static CloudT::Ptr display_cloud(const CloudT::Ptr& cloud, int max_points) {
  if (max_points <= 0 || static_cast<int>(cloud->size()) <= max_points) {
    return cloud;
  }
  pcl::RandomSample<PointT> sampler;
  sampler.setInputCloud(cloud);
  sampler.setSample(max_points);
  auto sampled = CloudT::Ptr(new CloudT);
  sampler.filter(*sampled);
  return sampled;
}

static void add_line(
    pcl::visualization::PCLVisualizer& viewer,
    const PointT& a,
    const PointT& b,
    const std::string& id,
    double r,
    double g,
    double bl,
    int width = 2) {
  viewer.removeShape(id);
  viewer.addLine<PointT>(a, b, r, g, bl, id);
  viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, width, id);
}

static void clear_overlay(pcl::visualization::PCLVisualizer& viewer) {
  for (int i = 0; i < 16; ++i) {
    viewer.removeShape("room_line_floor_" + std::to_string(i));
    viewer.removeShape("room_line_ceil_" + std::to_string(i));
    viewer.removeShape("room_line_vertical_" + std::to_string(i));
    viewer.removeShape("corner_sphere_" + std::to_string(i));
    viewer.removeShape("corner_label_" + std::to_string(i));
  }
  viewer.removeShape("status_text");
}

static void update_overlay(
    pcl::visualization::PCLVisualizer& viewer,
    const State& state,
    bool show_text) {
  clear_overlay(viewer);

  if (show_text) {
    std::ostringstream status;
    status << "corners=" << state.corners.size() << "/4"
           << "  z=[" << std::fixed << std::setprecision(2) << state.z_min << ", " << state.z_max << "]"
           << "  Shift+LeftClick pick; s save; u undo; r reset";
    viewer.addText(status.str(), 12, 18, 14, 1.0, 1.0, 1.0, "status_text");
  }

  for (std::size_t i = 0; i < state.corners.size(); ++i) {
    PointT p;
    p.x = static_cast<float>(state.corners[i].x);
    p.y = static_cast<float>(state.corners[i].y);
    p.z = static_cast<float>(state.z_min);
    viewer.addSphere(p, 0.06, 1.0, 0.1, 0.1, "corner_sphere_" + std::to_string(i));
    if (show_text) {
      viewer.addText3D(std::to_string(i + 1), p, 0.18, 1.0, 0.1, 0.1, "corner_label_" + std::to_string(i));
    }
  }

  if (state.corners.size() < 2) {
    return;
  }
  std::vector<Corner2> draw = state.corners;
  if (draw.size() == 4) {
    try {
      draw = order_corners_ccw(draw);
    } catch (const std::exception&) {
    }
  }
  const bool closed = draw.size() == 4;
  const std::size_t edge_count = closed ? 4 : draw.size() - 1;
  for (std::size_t i = 0; i < edge_count; ++i) {
    const auto& a2 = draw[i];
    const auto& b2 = draw[(i + 1) % draw.size()];
    PointT a0{static_cast<float>(a2.x), static_cast<float>(a2.y), static_cast<float>(state.z_min)};
    PointT b0{static_cast<float>(b2.x), static_cast<float>(b2.y), static_cast<float>(state.z_min)};
    PointT a1{static_cast<float>(a2.x), static_cast<float>(a2.y), static_cast<float>(state.z_max)};
    PointT b1{static_cast<float>(b2.x), static_cast<float>(b2.y), static_cast<float>(state.z_max)};
    add_line(viewer, a0, b0, "room_line_floor_" + std::to_string(i), 1.0, 0.0, 0.0, 3);
    if (closed) {
      add_line(viewer, a1, b1, "room_line_ceil_" + std::to_string(i), 1.0, 0.4, 0.0, 2);
      add_line(viewer, a0, a1, "room_line_vertical_" + std::to_string(i), 1.0, 0.0, 0.0, 2);
    }
  }
}

static void write_yaml(const Options& opt, const State& state) {
  auto corners = order_corners_ccw(state.corners);
  fs::path out_path(opt.output_yaml);
  if (out_path.has_parent_path()) {
    fs::create_directories(out_path.parent_path());
  }
  std::ofstream out(out_path);
  if (!out) {
    throw std::runtime_error("failed to open output YAML: " + out_path.string());
  }
  out << std::fixed << std::setprecision(9);
  out << "sequence: " << opt.sequence << "\n";
  out << "bag: " << opt.bag << "\n";
  out << "map_frame: " << opt.map_frame << "\n";
  out << "thresholds_m: [0.05, 0.10, 0.20]\n\n";
  out << "room_bounds:\n";
  out << "  type: cuboid\n";
  out << "  floor_corners_xy:\n";
  for (const auto& p : corners) {
    out << "    - [" << p.x << ", " << p.y << "]\n";
  }
  out << "  z: [" << state.z_min << ", " << state.z_max << "]\n";
  out << "  wall_types: [mirror, mirror, mirror, mirror]\n";
  out << "  floor_type: floor\n";
  out << "  ceiling_type: ceiling\n";
  out << "  interior_margin_m: " << opt.interior_margin << "\n";
  out << "  exterior_margin_m: " << opt.exterior_margin << "\n";
  out << "  thresholds_m: [0.05, 0.10, 0.20]\n";
  out << "  bottom_corners:\n";
  for (const auto& p : corners) {
    out << "    - [" << p.x << ", " << p.y << ", " << state.z_min << "]\n";
  }
  out << "  top_corners:\n";
  for (const auto& p : corners) {
    out << "    - [" << p.x << ", " << p.y << ", " << state.z_max << "]\n";
  }
  out << "  faces:\n";
  for (int i = 0; i < 4; ++i) {
    const int j = (i + 1) % 4;
    out << "    - id: wall_" << i << "\n";
    out << "      type: mirror\n";
    out << "      corner_indices: [" << i << ", " << j << ", " << j + 4 << ", " << i + 4 << "]\n";
  }
  out << "    - id: floor\n";
  out << "      type: floor\n";
  out << "      corner_indices: [0, 1, 2, 3]\n";
  out << "    - id: ceiling\n";
  out << "      type: ceiling\n";
  out << "      corner_indices: [4, 5, 6, 7]\n";
  out.close();
  std::cout << "wrote room cuboid annotation: " << out_path << "\n";
}

static void point_pick_callback(const pcl::visualization::PointPickingEvent& event, void* cookie) {
  auto* ctx = static_cast<CallbackContext*>(cookie);
  auto* state = ctx ? ctx->state : nullptr;
  if (!state || state->corners.size() >= 4) {
    return;
  }
  float x = 0.0F;
  float y = 0.0F;
  float z = 0.0F;
  event.getPoint(x, y, z);
  if (!std::isfinite(x) || !std::isfinite(y)) {
    return;
  }
  state->corners.push_back(Corner2{static_cast<double>(x), static_cast<double>(y)});
  state->overlay_dirty = true;
  std::cout << "corner_" << state->corners.size() << ": [" << x << ", " << y << "]"
            << " picked from z=" << z << "\n";
}

static void keyboard_callback(const pcl::visualization::KeyboardEvent& event, void* cookie) {
  auto* ctx = static_cast<CallbackContext*>(cookie);
  auto* state = ctx ? ctx->state : nullptr;
  if (!state || !event.keyDown()) {
    return;
  }
  const double z_step = ctx ? ctx->z_step : 0.05;
  const std::string key = event.getKeySym();
  if (key == "u" && !state->corners.empty()) {
    state->corners.pop_back();
    state->overlay_dirty = true;
    std::cout << "undo corner, remaining=" << state->corners.size() << "\n";
  } else if (key == "r") {
    state->corners.clear();
    state->overlay_dirty = true;
    std::cout << "reset corners\n";
  } else if (key == "bracketleft") {
    state->z_min -= z_step;
    state->overlay_dirty = true;
    std::cout << "z_min=" << state->z_min << "\n";
  } else if (key == "bracketright") {
    state->z_min += z_step;
    state->overlay_dirty = true;
    std::cout << "z_min=" << state->z_min << "\n";
  } else if (key == "semicolon") {
    state->z_max -= z_step;
    state->overlay_dirty = true;
    std::cout << "z_max=" << state->z_max << "\n";
  } else if (key == "apostrophe") {
    state->z_max += z_step;
    state->overlay_dirty = true;
    std::cout << "z_max=" << state->z_max << "\n";
  } else if (key == "s") {
    state->save_requested = true;
  } else if (key == "q" || key == "Escape") {
    state->quit_requested = true;
  }
}

int main(int argc, char** argv) {
  try {
    XInitThreads();

    Options opt;
    if (!parse_args(argc, argv, opt)) {
      return 1;
    }

    auto cloud = CloudT::Ptr(new CloudT);
    if (pcl::io::loadPCDFile<PointT>(opt.map_path, *cloud) != 0 || cloud->empty()) {
      throw std::runtime_error("failed to load non-empty PCD: " + opt.map_path);
    }
    PointT min_pt;
    PointT max_pt;
    pcl::getMinMax3D(*cloud, min_pt, max_pt);

    State state;
    state.z_min = std::isfinite(opt.z_min) ? opt.z_min : static_cast<double>(min_pt.z);
    state.z_max = std::isfinite(opt.z_max) ? opt.z_max : static_cast<double>(max_pt.z);
    state.corners = opt.corners;
    if (!state.corners.empty()) {
      if (state.corners.size() != 4) {
        throw std::runtime_error("non-interactive mode needs exactly four corners");
      }
      if (state.z_max <= state.z_min) {
        throw std::runtime_error("non-interactive mode needs valid --z-min/--z-max");
      }
      write_yaml(opt, state);
      return 0;
    }

    auto shown = display_cloud(cloud, opt.max_points);
    std::cout << "loaded " << cloud->size() << " points, showing " << shown->size() << "\n";
    std::cout << "map bounds: [" << min_pt.x << ", " << min_pt.y << ", " << min_pt.z << "] -> ["
              << max_pt.x << ", " << max_pt.y << ", " << max_pt.z << "]\n";
    usage(argv[0]);

    pcl::visualization::PCLVisualizer viewer("Yugong elevator cuboid annotator");
    viewer.setBackgroundColor(0.02, 0.02, 0.02);
    if (opt.color_by_z) {
      pcl::visualization::PointCloudColorHandlerGenericField<PointT> color(shown, "z");
      if (color.isCapable()) {
        viewer.addPointCloud<PointT>(shown, color, "map");
      } else {
        pcl::visualization::PointCloudColorHandlerCustom<PointT> mono(shown, 180, 180, 180);
        viewer.addPointCloud<PointT>(shown, mono, "map");
      }
    } else {
      pcl::visualization::PointCloudColorHandlerCustom<PointT> mono(shown, 180, 180, 180);
      viewer.addPointCloud<PointT>(shown, mono, "map");
    }
    viewer.setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, opt.point_size, "map");
    if (opt.coordinate_system) {
      viewer.addCoordinateSystem(0.5);
    }
    viewer.initCameraParameters();
    CallbackContext callback_context{&state, opt.z_step};
    viewer.registerPointPickingCallback(point_pick_callback, static_cast<void*>(&callback_context));
    viewer.registerKeyboardCallback(keyboard_callback, static_cast<void*>(&callback_context));

    while (!viewer.wasStopped() && !state.quit_requested) {
      if (state.overlay_dirty) {
        update_overlay(viewer, state, opt.show_text);
        state.overlay_dirty = false;
      }
      viewer.spinOnce(80);
      if (state.save_requested) {
        state.save_requested = false;
        if (state.corners.size() != 4) {
          std::cerr << "need four floor corners before saving, got " << state.corners.size() << "\n";
        } else if (state.z_max <= state.z_min) {
          std::cerr << "invalid z range: z_max <= z_min\n";
        } else {
          write_yaml(opt, state);
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 2;
  }
}

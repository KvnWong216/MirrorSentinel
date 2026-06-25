#ifndef SENTINEL_LIO_ROS2_COMPAT_HPP
#define SENTINEL_LIO_ROS2_COMPAT_HPP

#include <cassert>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <builtin_interfaces/msg/time.hpp>

namespace sentinel_lio_ros2
{

inline double stampToSec(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
}

inline builtin_interfaces::msg::Time stampFromSec(double sec)
{
  builtin_interfaces::msg::Time stamp;
  const double sec_floor = std::floor(sec);
  stamp.sec = static_cast<int32_t>(sec_floor);
  stamp.nanosec = static_cast<uint32_t>((sec - sec_floor) * 1e9);
  return stamp;
}

class NodeContext
{
public:
  static void set(const rclcpp::Node::SharedPtr & node)
  {
    node_ = node;
  }

  static rclcpp::Node::SharedPtr node()
  {
    return node_;
  }

  static rclcpp::Logger logger()
  {
    if (node_) {
      return node_->get_logger();
    }
    return rclcpp::get_logger("sentinel_lio");
  }

  static rclcpp::Time now()
  {
    if (node_) {
      return node_->now();
    }
    return rclcpp::Clock().now();
  }

private:
  inline static rclcpp::Node::SharedPtr node_ = nullptr;
};

template<typename T>
inline void declareIfUnset(const rclcpp::Node::SharedPtr & node, const std::string & name, const T & default_value)
{
  if (!node->has_parameter(name)) {
    node->declare_parameter<T>(name, default_value);
  }
}

template<typename T>
inline void param(const rclcpp::Node::SharedPtr & node, const std::string & name, T & value, const T & default_value)
{
  declareIfUnset(node, name, default_value);
  value = node->get_parameter(name).get_value<T>();
}

inline void param(const rclcpp::Node::SharedPtr & node, const std::string & name, float & value, const float default_value)
{
  if (!node->has_parameter(name)) {
    node->declare_parameter<double>(name, static_cast<double>(default_value));
  }
  value = static_cast<float>(node->get_parameter(name).as_double());
}

template<>
inline void param<std::vector<double>>(const rclcpp::Node::SharedPtr & node,
                                       const std::string & name,
                                       std::vector<double> & value,
                                       const std::vector<double> & default_value)
{
  declareIfUnset(node, name, default_value);
  value = node->get_parameter(name).as_double_array();
}

inline bool getParam(const rclcpp::Node::SharedPtr & node, const std::string & name, std::vector<double> & value)
{
  if (!node->has_parameter(name)) {
    node->declare_parameter<std::vector<double>>(name, std::vector<double>());
  }
  value = node->get_parameter(name).as_double_array();
  return !value.empty();
}

}  // namespace sentinel_lio_ros2

#define ROS2_LOGGER sentinel_lio_ros2::NodeContext::logger()
#define ROS_INFO(...) RCLCPP_INFO(ROS2_LOGGER, __VA_ARGS__)
#define ROS_WARN(...) RCLCPP_WARN(ROS2_LOGGER, __VA_ARGS__)
#define ROS_ERROR(...) RCLCPP_ERROR(ROS2_LOGGER, __VA_ARGS__)
#define ROS_INFO_ONCE(...) RCLCPP_INFO_ONCE(ROS2_LOGGER, __VA_ARGS__)
#define ROS_WARN_ONCE(...) RCLCPP_WARN_ONCE(ROS2_LOGGER, __VA_ARGS__)
#define ROS_ASSERT(cond) assert(cond)

#endif  // SENTINEL_LIO_ROS2_COMPAT_HPP

#!/usr/bin/env python3
"""Monocular DA3-style depth publisher for `/vfm/depth_image`."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from da3_adapter import DA3DepthRunner, make_da3_config_from_ros


def to_rgb(cv_image: np.ndarray) -> np.ndarray:
    if cv_image.ndim == 2:
        return cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
    if cv_image.shape[2] == 4:
        return cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)


class DA3DepthNode(Node):
    def __init__(self):
        super().__init__("da3_depth_node")
        self.declare_parameter("image_topic", "/zed2/zed_node/left/image_rect_color")
        self.declare_parameter("depth_topic", "/vfm/depth_image")
        self.declare_parameter("target_fps", 10.0)
        self.declare_parameter("stats_log_interval", 5.0)
        self.declare_parameter("da3_backend", "none")
        self.declare_parameter("da3_model", "")
        self.declare_parameter("da3_checkpoint", "")
        self.declare_parameter("da3_device", "cuda")
        self.declare_parameter("da3_input_width", 518)
        self.declare_parameter("da3_input_height", 518)
        self.declare_parameter("da3_metric_scale", 1.0)
        self.declare_parameter("da3_metric_shift", 0.0)
        self.declare_parameter("da3_min_depth", 0.05)
        self.declare_parameter("da3_max_depth", 80.0)

        image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        self.target_fps = self.get_parameter("target_fps").get_parameter_value().double_value
        self.stats_log_interval = self.get_parameter("stats_log_interval").get_parameter_value().double_value
        self.min_period = 0.0 if self.target_fps <= 0.0 else 1.0 / self.target_fps
        self.last_stamp = None
        self.bridge = CvBridge()
        self.da3 = DA3DepthRunner(make_da3_config_from_ros(self))
        if not self.da3.enabled():
            raise ValueError("da3_depth_node requires da3_backend != none")

        self.pub = self.create_publisher(Image, depth_topic, 1)
        self.sub = self.create_subscription(Image, image_topic, self.image_callback, 3)
        self.stats_start = time.monotonic()
        self.last_stats = self.stats_start
        self.recv = 0
        self.published = 0
        self.skipped = 0
        self.failed = 0
        self.total_ms = 0.0
        self.max_ms = 0.0
        self.get_logger().info(
            f"DA3 depth node ready: backend={self.da3.backend}, target_fps={self.target_fps:.1f}, "
            f"topic={image_topic} -> {depth_topic}"
        )

    def stamp_seconds(self, stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def should_process(self, stamp) -> bool:
        if self.min_period <= 0.0:
            return True
        stamp_s = self.stamp_seconds(stamp)
        if self.last_stamp is None or stamp_s < self.last_stamp:
            self.last_stamp = stamp_s
            return True
        if stamp_s - self.last_stamp + 1e-9 < self.min_period:
            self.skipped += 1
            self.log_stats()
            return False
        self.last_stamp = stamp_s
        return True

    def log_stats(self, force: bool = False) -> None:
        if self.stats_log_interval <= 0.0:
            return
        now = time.monotonic()
        if not force and now - self.last_stats < self.stats_log_interval:
            return
        fps = self.published / max(now - self.stats_start, 1e-6)
        avg = self.total_ms / max(self.published, 1)
        self.get_logger().info(
            f"DA3 depth stats: recv={self.recv}, pub={self.published}, skip={self.skipped}, "
            f"fail={self.failed}, fps={fps:.2f}, avg={avg:.1f} ms, max={self.max_ms:.1f} ms"
        )
        self.last_stats = now

    def image_callback(self, msg: Image) -> None:
        self.recv += 1
        if not self.should_process(msg.header.stamp):
            return
        start = time.perf_counter()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            depth = self.da3.infer(to_rgb(cv_image)).astype(np.float32)
            out = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
            out.header = msg.header
            self.pub.publish(out)
            elapsed = (time.perf_counter() - start) * 1000.0
            self.published += 1
            self.total_ms += elapsed
            self.max_ms = max(self.max_ms, elapsed)
            self.log_stats()
        except Exception as exc:
            self.failed += 1
            self.get_logger().error(f"DA3 depth callback failed: {exc}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = DA3DepthNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

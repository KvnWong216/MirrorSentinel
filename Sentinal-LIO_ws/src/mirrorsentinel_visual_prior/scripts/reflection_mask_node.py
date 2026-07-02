#!/usr/bin/env python3
"""Real-time RGB/DA3-guided reflection mask publisher."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from da3_adapter import DA3DepthRunner, make_da3_config_from_ros, normalize_depth_prior
from reflection_seg_model import build_model_from_checkpoint


def to_rgb(cv_image: np.ndarray) -> np.ndarray:
    if cv_image.ndim == 2:
        return cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
    if cv_image.shape[2] == 4:
        return cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)


def heuristic_reflection_mask(rgb: np.ndarray, threshold: float) -> np.ndarray:
    """Cheap RGB-only fallback for ROS plumbing tests, not a learned method."""

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    sat = hsv[..., 1] / 255.0
    val = hsv[..., 2] / 255.0
    edges = cv2.Canny(gray.astype(np.uint8), 40, 120).astype(np.float32) / 255.0
    low_texture = 1.0 - cv2.GaussianBlur(edges, (0, 0), 2.0)
    score = 0.55 * val + 0.25 * (1.0 - sat) + 0.20 * low_texture
    score = cv2.GaussianBlur(score, (0, 0), 2.0)
    mask = (score > threshold).astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


class ReflectionMaskNode(Node):
    def __init__(self):
        super().__init__("reflection_mask_node")

        self.declare_parameter("image_topic", "/zed2/zed_node/left/image_rect_color")
        self.declare_parameter("mask_topic", "/vfm/mirror_mask")
        self.declare_parameter("checkpoint", "")
        self.declare_parameter("backend", "torch")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("target_fps", 10.0)
        self.declare_parameter("image_width", 512)
        self.declare_parameter("image_height", 288)
        self.declare_parameter("prob_threshold", 0.5)
        self.declare_parameter("publish_prob", False)
        self.declare_parameter("prob_topic", "/vfm/mirror_prob")
        self.declare_parameter("stats_log_interval", 5.0)
        self.declare_parameter("use_da3_prior", False)
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
        mask_topic = self.get_parameter("mask_topic").get_parameter_value().string_value
        prob_topic = self.get_parameter("prob_topic").get_parameter_value().string_value
        self.backend = self.get_parameter("backend").get_parameter_value().string_value.lower()
        self.checkpoint = self.get_parameter("checkpoint").get_parameter_value().string_value
        device_param = self.get_parameter("device").get_parameter_value().string_value
        self.device = device_param if torch.cuda.is_available() and device_param.startswith("cuda") else "cpu"
        self.target_fps = self.get_parameter("target_fps").get_parameter_value().double_value
        self.input_width = self.get_parameter("image_width").get_parameter_value().integer_value
        self.input_height = self.get_parameter("image_height").get_parameter_value().integer_value
        self.prob_threshold = self.get_parameter("prob_threshold").get_parameter_value().double_value
        self.publish_prob = self.get_parameter("publish_prob").get_parameter_value().bool_value
        self.stats_log_interval = self.get_parameter("stats_log_interval").get_parameter_value().double_value
        self.use_da3_prior = self.get_parameter("use_da3_prior").get_parameter_value().bool_value
        self.min_period = 0.0 if self.target_fps <= 0.0 else 1.0 / self.target_fps
        self.last_stamp = None

        self.bridge = CvBridge()
        self.model = None
        self.da3 = None
        if self.use_da3_prior:
            self.da3 = DA3DepthRunner(make_da3_config_from_ros(self))
            self.get_logger().info(f"DA3 prior backend: {self.da3.backend}")

        if self.backend in ("torch", "model"):
            if not self.checkpoint:
                raise ValueError("reflection mask backend=torch requires checkpoint")
            self.model = build_model_from_checkpoint(self.checkpoint, self.device)
            self.get_logger().info(f"Loaded reflection mask checkpoint: {self.checkpoint}")
        elif self.backend == "heuristic":
            self.get_logger().warning("Using heuristic reflection mask backend; do not report as learned/DA3 result")
        else:
            raise ValueError(f"Unsupported reflection mask backend: {self.backend}")

        self.mask_pub = self.create_publisher(Image, mask_topic, 1)
        self.prob_pub = self.create_publisher(Image, prob_topic, 1) if self.publish_prob else None
        self.sub = self.create_subscription(Image, image_topic, self.image_callback, 3)

        self.stats_start = time.monotonic()
        self.last_stats_log = self.stats_start
        self.recv = 0
        self.pub = 0
        self.skip = 0
        self.fail = 0
        self.total_ms = 0.0
        self.max_ms = 0.0
        self.get_logger().info(
            f"reflection mask node ready: backend={self.backend}, target_fps={self.target_fps:.1f}, "
            f"topic={image_topic} -> {mask_topic}"
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
            self.skip += 1
            self.log_stats()
            return False
        self.last_stamp = stamp_s
        return True

    def make_input(self, rgb: np.ndarray) -> torch.Tensor:
        orig_rgb = rgb
        rgb = cv2.resize(rgb, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        x = rgb.astype(np.float32) / 255.0
        x = (x - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        x = x.transpose(2, 0, 1)
        if self.use_da3_prior:
            if self.da3 is None:
                raise RuntimeError("use_da3_prior is true but DA3 runner is not initialized")
            depth = self.da3.infer(orig_rgb)
            depth = cv2.resize(normalize_depth_prior(depth), (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
            x = np.concatenate([x, depth[None]], axis=0)
        return torch.from_numpy(x[None]).to(self.device)

    @torch.inference_mode()
    def predict_mask(self, rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = rgb.shape[:2]
        if self.backend == "heuristic":
            mask = heuristic_reflection_mask(rgb, self.prob_threshold)
            prob = mask.astype(np.float32) / 255.0
            return mask, prob

        x = self.make_input(rgb)
        logits = self.model(x)
        prob = torch.sigmoid(logits)[0, 0].detach().float().cpu().numpy()
        if prob.shape != (h, w):
            prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = (prob > self.prob_threshold).astype(np.uint8) * 255
        return mask, prob.astype(np.float32)

    def log_stats(self, force: bool = False) -> None:
        if self.stats_log_interval <= 0.0:
            return
        now = time.monotonic()
        if not force and now - self.last_stats_log < self.stats_log_interval:
            return
        fps = self.pub / max(now - self.stats_start, 1e-6)
        avg = self.total_ms / max(self.pub, 1)
        self.get_logger().info(
            f"reflection mask stats: recv={self.recv}, pub={self.pub}, skip={self.skip}, "
            f"fail={self.fail}, fps={fps:.2f}, avg={avg:.1f} ms, max={self.max_ms:.1f} ms"
        )
        self.last_stats_log = now

    def image_callback(self, msg: Image) -> None:
        self.recv += 1
        if not self.should_process(msg.header.stamp):
            return
        start = time.perf_counter()
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            rgb = to_rgb(cv_image)
            mask, prob = self.predict_mask(rgb)
            out = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
            out.header = msg.header
            self.mask_pub.publish(out)
            if self.prob_pub is not None:
                prob_msg = self.bridge.cv2_to_imgmsg(prob.astype(np.float32), encoding="32FC1")
                prob_msg.header = msg.header
                self.prob_pub.publish(prob_msg)
            elapsed = (time.perf_counter() - start) * 1000.0
            self.pub += 1
            self.total_ms += elapsed
            self.max_ms = max(self.max_ms, elapsed)
            self.log_stats()
        except Exception as exc:
            self.fail += 1
            self.get_logger().error(f"reflection mask callback failed: {exc}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = ReflectionMaskNode()
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

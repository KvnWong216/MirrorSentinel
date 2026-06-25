#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from cv_bridge import CvBridge
from omegaconf import OmegaConf
from sensor_msgs.msg import Image

import message_filters
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from mask_prior_utils import DirectoryMaskPrior, empty_mask


class VFMStereoNode(Node):
    def __init__(self):
        super().__init__('vfm_stereo_trt_node')

        self.declare_parameter('model_root', str(self.default_model_root()))
        self.declare_parameter('onnx_dir', '')
        self.declare_parameter('left_topic', '/zed2/zed_node/left/image_rect_color')
        self.declare_parameter('right_topic', '/zed2/zed_node/right/image_rect_color')
        self.declare_parameter('depth_topic', '/vfm/depth_image')
        self.declare_parameter('mask_topic', '/vfm/mirror_mask')
        self.declare_parameter('mask_mode', 'none')
        self.declare_parameter('mask_dir', '')
        self.declare_parameter('mask_threshold', 0.0)
        self.declare_parameter('mask_match_tolerance', 0.05)
        self.declare_parameter('publish_empty_mask', True)
        self.declare_parameter('camera_fx', 541.56)
        self.declare_parameter('baseline', 0.12)
        self.declare_parameter('sync_queue_size', 3)
        self.declare_parameter('target_fps', 10.0)
        self.declare_parameter('stats_log_interval', 5.0)

        self.model_root = self.get_parameter('model_root').get_parameter_value().string_value
        if not self.model_root:
            self.model_root = str(self.default_model_root())
        self.onnx_dir = self.get_parameter('onnx_dir').get_parameter_value().string_value
        if not self.onnx_dir:
            self.onnx_dir = os.path.join(self.model_root, 'output')
        left_topic = self.get_parameter('left_topic').get_parameter_value().string_value
        right_topic = self.get_parameter('right_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        mask_topic = self.get_parameter('mask_topic').get_parameter_value().string_value
        self.mask_mode = self.get_parameter('mask_mode').get_parameter_value().string_value.lower()
        mask_dir = self.get_parameter('mask_dir').get_parameter_value().string_value
        mask_threshold = self.get_parameter('mask_threshold').get_parameter_value().double_value
        mask_match_tolerance = self.get_parameter('mask_match_tolerance').get_parameter_value().double_value
        self.publish_empty_mask = self.get_parameter('publish_empty_mask').get_parameter_value().bool_value
        self.camera_fx = self.get_parameter('camera_fx').get_parameter_value().double_value
        self.baseline = self.get_parameter('baseline').get_parameter_value().double_value
        sync_queue_size = self.get_parameter('sync_queue_size').get_parameter_value().integer_value
        self.target_fps = self.get_parameter('target_fps').get_parameter_value().double_value
        self.stats_log_interval = self.get_parameter('stats_log_interval').get_parameter_value().double_value
        self.min_depth_period = 0.0 if self.target_fps <= 0.0 else 1.0 / self.target_fps
        self.last_depth_stamp = None
        self.stats_start_time = time.monotonic()
        self.last_stats_log_time = self.stats_start_time
        self.received_pairs = 0
        self.skipped_pairs = 0
        self.published_depths = 0
        self.failed_callbacks = 0
        self.stats_active = False
        self.total_callback_ms = 0.0
        self.total_model_ms = 0.0
        self.max_callback_ms = 0.0
        self.max_model_ms = 0.0

        if self.model_root and self.model_root not in sys.path:
            sys.path.append(self.model_root)

        from core.foundation_stereo import TrtRunner

        self.bridge = CvBridge()
        onnx_cfg_path = self.resolve_onnx_cfg_path(self.onnx_dir)
        with open(onnx_cfg_path, 'r') as f:
            cfg = yaml.safe_load(f)
        self.args = OmegaConf.create(cfg)

        self.get_logger().info('[*] Loading TensorRT engines...')
        feature_engine = os.path.join(self.onnx_dir, 'feature_runner.engine')
        post_engine = os.path.join(self.onnx_dir, 'post_runner.engine')

        if not os.path.exists(feature_engine) or not os.path.exists(post_engine):
            raise FileNotFoundError(
                f'Missing TensorRT engine files: {feature_engine}, {post_engine}')

        torch.autograd.set_grad_enabled(False)
        self.model = TrtRunner(self.args, feature_engine, post_engine)
        self.get_logger().info('[*] TensorRT model loaded successfully')

        self.depth_pub = self.create_publisher(Image, depth_topic, 1)
        self.mask_pub = None
        self.mask_prior = None
        self.missing_mask_count = 0
        if self.mask_mode in ('directory', 'dir', 'oracle'):
            self.mask_prior = DirectoryMaskPrior(Path(mask_dir), mask_threshold, mask_match_tolerance)
            self.mask_pub = self.create_publisher(Image, mask_topic, 1)
            self.get_logger().info(
                f'[*] Publishing mask prior from {mask_dir} to {mask_topic} '
                f'({len(self.mask_prior)} files, tolerance={mask_match_tolerance:.3f}s)'
            )
        elif self.mask_mode in ('zeros', 'empty'):
            self.mask_pub = self.create_publisher(Image, mask_topic, 1)
            self.get_logger().info(f'[*] Publishing empty masks to {mask_topic}')
        elif self.mask_mode in ('none', ''):
            self.mask_mode = 'none'
        else:
            raise ValueError(f'Unsupported mask_mode: {self.mask_mode}')

        sub_left = message_filters.Subscriber(self, Image, left_topic)
        sub_right = message_filters.Subscriber(self, Image, right_topic)
        self.ts = message_filters.TimeSynchronizer([sub_left, sub_right], sync_queue_size)
        self.ts.registerCallback(self.image_callback)

        fps_text = 'all synced frames' if self.target_fps <= 0.0 else f'{self.target_fps:.1f} Hz'
        self.get_logger().info(
            f'[*] VFM realtime mode: target_fps={fps_text}, sync_queue_size={sync_queue_size}'
        )

    def default_model_root(self) -> Path:
        env_root = os.environ.get('SENTINEL_LIO_MODEL_ROOT')
        if env_root:
            return Path(env_root).expanduser()

        for base in Path(__file__).resolve().parents:
            candidate = base / 'models' / 'Fast-FoundationStereo'
            if candidate.exists():
                return candidate

        return Path(__file__).resolve().parents[3] / 'models' / 'Fast-FoundationStereo'

    def resolve_onnx_cfg_path(self, onnx_dir: str) -> str:
        candidates = [
            os.path.join(onnx_dir, 'onnx.yaml'),
            os.path.join(os.path.dirname(onnx_dir), 'onnx.yaml'),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f'onnx.yaml not found near {onnx_dir}')

    def stamp_seconds(self, stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def should_process(self, stamp) -> bool:
        if self.min_depth_period <= 0.0:
            return True

        stamp_s = self.stamp_seconds(stamp)
        if self.last_depth_stamp is None or stamp_s < self.last_depth_stamp:
            self.last_depth_stamp = stamp_s
            return True

        if stamp_s - self.last_depth_stamp + 1e-9 < self.min_depth_period:
            self.skipped_pairs += 1
            self.maybe_log_stats()
            return False

        self.last_depth_stamp = stamp_s
        return True

    def maybe_log_stats(self, force: bool = False) -> None:
        if self.stats_log_interval <= 0.0:
            return
        now = time.monotonic()
        if not force and now - self.last_stats_log_time < self.stats_log_interval:
            return

        elapsed = max(now - self.stats_start_time, 1e-6)
        avg_cb = self.total_callback_ms / max(self.published_depths, 1)
        avg_model = self.total_model_ms / max(self.published_depths, 1)
        effective_fps = self.published_depths / elapsed
        self.get_logger().info(
            'VFM depth stats: '
            f'recv={self.received_pairs}, pub={self.published_depths}, '
            f'skip={self.skipped_pairs}, fail={self.failed_callbacks}, '
            f'effective_fps={effective_fps:.2f}, '
            f'avg_cb={avg_cb:.1f} ms, avg_model={avg_model:.1f} ms, '
            f'max_cb={self.max_callback_ms:.1f} ms, max_model={self.max_model_ms:.1f} ms'
        )
        self.last_stats_log_time = now

    def publish_mask_for_image(self, msg_left, width, height):
        if self.mask_pub is None or self.mask_mode == 'none':
            return

        lookup = None
        if self.mask_mode in ('directory', 'dir', 'oracle'):
            lookup = self.mask_prior.load(msg_left.header.stamp, width, height)
            if lookup is None:
                self.missing_mask_count += 1
                if self.publish_empty_mask:
                    mask = empty_mask(width, height)
                else:
                    if self.missing_mask_count == 1 or self.missing_mask_count % 100 == 0:
                        self.get_logger().warning(
                            f'No mask matched stamp {msg_left.header.stamp.sec}.'
                            f'{msg_left.header.stamp.nanosec:09d}; skipped '
                            f'{self.missing_mask_count} masks'
                        )
                    return
            else:
                mask = lookup.mask
        else:
            mask = empty_mask(width, height)

        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        mask_msg.header = msg_left.header
        self.mask_pub.publish(mask_msg)

    def image_callback(self, msg_left, msg_right):
        if not self.stats_active:
            now = time.monotonic()
            self.stats_start_time = now
            self.last_stats_log_time = now
            self.stats_active = True

        self.received_pairs += 1
        if not self.should_process(msg_left.header.stamp):
            return

        callback_start = time.perf_counter()
        try:
            cv_left = self.bridge.imgmsg_to_cv2(msg_left, 'passthrough')
            cv_right = self.bridge.imgmsg_to_cv2(msg_right, 'passthrough')

            if len(cv_left.shape) == 2:
                cv_left = cv2.cvtColor(cv_left, cv2.COLOR_GRAY2RGB)
                cv_right = cv2.cvtColor(cv_right, cv2.COLOR_GRAY2RGB)
            elif cv_left.shape[2] == 4:
                cv_left = cv2.cvtColor(cv_left, cv2.COLOR_BGRA2RGB)
                cv_right = cv2.cvtColor(cv_right, cv2.COLOR_BGRA2RGB)
            else:
                cv_left = cv2.cvtColor(cv_left, cv2.COLOR_BGR2RGB)
                cv_right = cv2.cvtColor(cv_right, cv2.COLOR_BGR2RGB)

            orig_h, orig_w = cv_left.shape[:2]
            self.publish_mask_for_image(msg_left, orig_w, orig_h)

            target_h, target_w = self.args.image_size[0], self.args.image_size[1]
            fx_scale = target_w / orig_w
            fy_scale = target_h / orig_h

            if fx_scale != 1.0 or fy_scale != 1.0:
                cv_left = cv2.resize(cv_left, (target_w, target_h))
                cv_right = cv2.resize(cv_right, (target_w, target_h))

            img0 = torch.as_tensor(cv_left).cuda().float()[None].permute(0, 3, 1, 2)
            img1 = torch.as_tensor(cv_right).cuda().float()[None].permute(0, 3, 1, 2)

            model_start = time.perf_counter()
            disp = self.model.forward(img0, img1)
            torch.cuda.synchronize()
            model_ms = (time.perf_counter() - model_start) * 1000.0
            disp_np = disp.data.cpu().numpy().reshape(target_h, target_w).clip(0.1, None) * (1.0 / fx_scale)

            if fx_scale != 1.0 or fy_scale != 1.0:
                disp_np = cv2.resize(disp_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            depth_np = (self.camera_fx * self.baseline) / disp_np
            depth_np[depth_np > 100.0] = 0.0
            depth_np[depth_np < 0.0] = 0.0

            depth_msg = self.bridge.cv2_to_imgmsg(depth_np.astype(np.float32), encoding='32FC1')
            depth_msg.header = msg_left.header
            self.depth_pub.publish(depth_msg)
            callback_ms = (time.perf_counter() - callback_start) * 1000.0
            self.published_depths += 1
            self.total_callback_ms += callback_ms
            self.total_model_ms += model_ms
            self.max_callback_ms = max(self.max_callback_ms, callback_ms)
            self.max_model_ms = max(self.max_model_ms, model_ms)
            self.maybe_log_stats()

        except Exception as exc:
            self.failed_callbacks += 1
            self.get_logger().error(f'VFM stereo callback failed: {exc}')


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = VFMStereoNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/python3
import sys
from pathlib import Path

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import rclpy
from rclpy.node import Node

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from mask_prior_utils import DirectoryMaskPrior, empty_mask


class ReflectionMaskPriorNode(Node):
    def __init__(self):
        super().__init__('reflection_mask_prior_node')

        self.declare_parameter('image_topic', '/zed2/zed_node/left/image_rect_color')
        self.declare_parameter('mask_topic', '/vfm/mirror_mask')
        self.declare_parameter('mask_mode', 'directory')
        self.declare_parameter('mask_dir', '')
        self.declare_parameter('mask_threshold', 0.0)
        self.declare_parameter('mask_match_tolerance', 0.05)
        self.declare_parameter('publish_empty_mask', True)
        self.declare_parameter('target_fps', 10.0)

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        mask_topic = self.get_parameter('mask_topic').get_parameter_value().string_value
        self.mask_mode = self.get_parameter('mask_mode').get_parameter_value().string_value.lower()
        mask_dir = self.get_parameter('mask_dir').get_parameter_value().string_value
        mask_threshold = self.get_parameter('mask_threshold').get_parameter_value().double_value
        mask_match_tolerance = self.get_parameter('mask_match_tolerance').get_parameter_value().double_value
        self.publish_empty_mask = self.get_parameter('publish_empty_mask').get_parameter_value().bool_value
        self.target_fps = self.get_parameter('target_fps').get_parameter_value().double_value
        self.min_period = 0.0 if self.target_fps <= 0.0 else 1.0 / self.target_fps
        self.last_stamp = None

        self.bridge = CvBridge()
        self.mask_prior = None
        self.missing_mask_count = 0

        if self.mask_mode in ('directory', 'dir', 'oracle'):
            self.mask_prior = DirectoryMaskPrior(Path(mask_dir), mask_threshold, mask_match_tolerance)
            self.get_logger().info(
                f'Publishing mask prior from {mask_dir} to {mask_topic} '
                f'({len(self.mask_prior)} files, tolerance={mask_match_tolerance:.3f}s)'
            )
        elif self.mask_mode in ('zeros', 'empty'):
            self.get_logger().info(f'Publishing empty masks to {mask_topic}')
        else:
            raise ValueError(f'Unsupported mask_mode: {self.mask_mode}')

        self.mask_pub = self.create_publisher(Image, mask_topic, 1)
        self.image_sub = self.create_subscription(Image, image_topic, self.image_callback, 10)

    def stamp_seconds(self, stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def should_publish(self, stamp):
        if self.min_period <= 0.0:
            return True
        stamp_s = self.stamp_seconds(stamp)
        if self.last_stamp is None or stamp_s < self.last_stamp:
            self.last_stamp = stamp_s
            return True
        if stamp_s - self.last_stamp + 1e-9 < self.min_period:
            return False
        self.last_stamp = stamp_s
        return True

    def image_callback(self, msg):
        if not self.should_publish(msg.header.stamp):
            return
        mask = None
        if self.mask_mode in ('directory', 'dir', 'oracle'):
            lookup = self.mask_prior.load(msg.header.stamp, msg.width, msg.height)
            if lookup is None:
                self.missing_mask_count += 1
                if self.publish_empty_mask:
                    mask = empty_mask(msg.width, msg.height)
                else:
                    if self.missing_mask_count == 1 or self.missing_mask_count % 100 == 0:
                        self.get_logger().warning(
                            f'No mask matched stamp {msg.header.stamp.sec}.'
                            f'{msg.header.stamp.nanosec:09d}; skipped '
                            f'{self.missing_mask_count} masks'
                        )
                    return
            else:
                mask = lookup.mask
        else:
            mask = empty_mask(msg.width, msg.height)

        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        mask_msg.header = msg.header
        self.mask_pub.publish(mask_msg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ReflectionMaskPriorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

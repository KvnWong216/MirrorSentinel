#!/usr/bin/env python3
"""Republish PointCloud2 with corrected dense metadata for downstream consumers.

Some recorded Ouster clouds contain invalid xyz samples but still advertise
`is_dense=true`. RTAB-Map's ICP path trusts this flag and skips NaN removal,
which later triggers PCL assertions. This relay only flips the metadata to
`false` by default, preserving the payload and timestamps so downstream C++
code can run its normal invalid-point cleanup.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import numpy as np
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField


@dataclass
class RelayConfig:
    input_topic: str
    output_topic: str
    use_sim_time: bool
    compact_xyz: bool
    stride: int
    range_min: float
    range_max: float
    max_points: int


class DenseFlagRelay(Node):
    def __init__(self, cfg: RelayConfig) -> None:
        super().__init__("pointcloud_dense_flag_relay")
        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", cfg.use_sim_time)
        self.set_parameters(
            [rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, cfg.use_sim_time)]
        )
        self._pub = self.create_publisher(PointCloud2, cfg.output_topic, qos_profile_sensor_data)
        self._sub = self.create_subscription(
            PointCloud2, cfg.input_topic, self._callback, qos_profile_sensor_data
        )
        self._input_topic = cfg.input_topic
        self._output_topic = cfg.output_topic
        self._cfg = cfg
        self._frames = 0
        self._flipped = 0
        self._in_points = 0
        self._out_points = 0
        self._last_log_ns = 0
        self.get_logger().info(
            f"relay {self._input_topic} -> {self._output_topic} "
            f"(compact_xyz={cfg.compact_xyz}, stride={cfg.stride}, "
            f"range=[{cfg.range_min}, {cfg.range_max}], max_points={cfg.max_points})"
        )

    def _callback(self, msg: PointCloud2) -> None:
        self._frames += 1
        if self._cfg.compact_xyz:
            out = self._compact_xyz(msg)
        else:
            out = PointCloud2()
            out.header = msg.header
            out.height = msg.height
            out.width = msg.width
            out.fields = list(msg.fields)
            out.is_bigendian = msg.is_bigendian
            out.point_step = msg.point_step
            out.row_step = msg.row_step
            out.data = bytes(msg.data)
            out.is_dense = False

        if msg.is_dense:
            self._flipped += 1
        try:
            self._pub.publish(out)
        except RCLError as exc:
            if "publisher's context is invalid" in str(exc):
                return
            raise

        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._last_log_ns > 5_000_000_000:
            self._last_log_ns = now_ns
            self.get_logger().info(
                f"frames={self._frames} flipped_dense_flag={self._flipped} "
                f"points_in={self._in_points} points_out={self._out_points}"
            )

    def _compact_xyz(self, msg: PointCloud2) -> PointCloud2:
        xyz = self._xyz_array(msg)
        self._in_points += int(xyz.shape[0])

        if self._cfg.stride > 1:
            xyz = xyz[:: self._cfg.stride]

        finite = np.isfinite(xyz).all(axis=1)
        if self._cfg.range_min > 0.0 or self._cfg.range_max > 0.0:
            squared = np.einsum("ij,ij->i", xyz, xyz)
            if self._cfg.range_min > 0.0:
                finite &= squared >= self._cfg.range_min * self._cfg.range_min
            if self._cfg.range_max > 0.0:
                finite &= squared <= self._cfg.range_max * self._cfg.range_max
        xyz = xyz[finite]

        if self._cfg.max_points > 0 and xyz.shape[0] > self._cfg.max_points:
            step = int(math.ceil(xyz.shape[0] / self._cfg.max_points))
            xyz = xyz[::step][: self._cfg.max_points]

        xyz = np.ascontiguousarray(xyz, dtype="<f4")
        self._out_points += int(xyz.shape[0])

        out = PointCloud2()
        out.header = msg.header
        out.height = 1
        out.width = int(xyz.shape[0])
        out.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        out.is_bigendian = False
        out.point_step = 12
        out.row_step = out.point_step * out.width
        out.data = xyz.tobytes()
        out.is_dense = True
        return out

    @staticmethod
    def _xyz_array(msg: PointCloud2) -> np.ndarray:
        offsets = {}
        for field in msg.fields:
            if field.name in ("x", "y", "z"):
                if field.datatype != PointField.FLOAT32:
                    raise ValueError(f"field {field.name} must be FLOAT32, got {field.datatype}")
                offsets[field.name] = field.offset
        missing = {"x", "y", "z"} - set(offsets)
        if missing:
            raise ValueError(f"missing PointCloud2 xyz fields: {sorted(missing)}")

        endian = ">" if msg.is_bigendian else "<"
        dtype = np.dtype(
            {
                "names": ["x", "y", "z"],
                "formats": [endian + "f4", endian + "f4", endian + "f4"],
                "offsets": [offsets["x"], offsets["y"], offsets["z"]],
                "itemsize": msg.point_step,
            }
        )
        count = msg.width * msg.height
        structured = np.frombuffer(msg.data, dtype=dtype, count=count)
        return np.column_stack((structured["x"], structured["y"], structured["z"]))


def parse_args() -> RelayConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-topic", default="/ouster/points/deskewed")
    parser.add_argument("--output-topic", default="/ouster/points/deskewed_sanitized")
    parser.add_argument("--use-sim-time", action="store_true")
    parser.add_argument(
        "--compact-xyz",
        action="store_true",
        help="filter invalid points and republish an unorganized xyz-only cloud",
    )
    parser.add_argument("--stride", type=int, default=1, help="keep every Nth input point")
    parser.add_argument("--range-min", type=float, default=0.0)
    parser.add_argument("--range-max", type=float, default=0.0)
    parser.add_argument("--max-points", type=int, default=0)
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be >= 1")
    return RelayConfig(
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        use_sim_time=args.use_sim_time,
        compact_xyz=args.compact_xyz,
        stride=args.stride,
        range_min=args.range_min,
        range_max=args.range_max,
        max_points=args.max_points,
    )


def main() -> int:
    cfg = parse_args()
    rclpy.init()
    node = DenseFlagRelay(cfg)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

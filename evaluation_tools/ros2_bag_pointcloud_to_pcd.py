#!/usr/bin/env python3
"""Export the latest PointCloud2 message from a ROS2 bag to ASCII PCD."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosidl_runtime_py.utilities import get_message
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


def pcd_header(point_count: int) -> str:
    return "\n".join(
        [
            "# .PCD v0.7 - Point Cloud Data file format",
            "VERSION 0.7",
            "FIELDS x y z",
            "SIZE 4 4 4",
            "TYPE F F F",
            "COUNT 1 1 1",
            f"WIDTH {point_count}",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {point_count}",
            "DATA ascii",
        ]
    ) + "\n"


def write_pcd_ascii(path: Path, points: Sequence[Sequence[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(pcd_header(len(points)))
        for x, y, z in points:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def read_latest_pointcloud(bag_dir: Path, topic: str) -> PointCloud2:
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )

    topic_type = None
    for meta in reader.get_all_topics_and_types():
        if meta.name == topic:
            topic_type = meta.type
            break
    if topic_type is None:
        raise ValueError(f"topic not found in bag: {topic}")
    if topic_type != "sensor_msgs/msg/PointCloud2":
        raise ValueError(f"topic {topic} is not PointCloud2: {topic_type}")

    msg_type = get_message(topic_type)
    latest = None
    while reader.has_next():
        name, data, _ = reader.read_next()
        if name != topic:
            continue
        latest = deserialize_message(data, msg_type)
    if latest is None:
        raise ValueError(f"no messages found on topic: {topic}")
    return latest


def cloud_to_xyz(msg: PointCloud2) -> List[List[float]]:
    points_iter: Iterable = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
    return [[float(x), float(y), float(z)] for x, y, z in points_iter]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--topic", default="/cloud_map")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    msg = read_latest_pointcloud(args.bag, args.topic)
    points = cloud_to_xyz(msg)
    if not points:
        raise SystemExit(f"no valid xyz points found in {args.topic}")
    write_pcd_ascii(args.out, points)
    print(f"wrote {args.out} ({len(points)} points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

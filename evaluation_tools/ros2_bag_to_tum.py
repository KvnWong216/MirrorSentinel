#!/usr/bin/env python3
"""Export ROS2 bag trajectory topics to TUM text format.

Supported message types:
  - nav_msgs/msg/Odometry
  - geometry_msgs/msg/PoseStamped
  - nav_msgs/msg/Path

This uses the pure-Python `rosbags` reader instead of rosbag2_py, so it works
in the current ROS2 Humble workspace even when rosbag2_py bindings are missing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


Row = Tuple[float, float, float, float, float, float, float, float]


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def ns_to_sec(stamp_ns: int) -> float:
    return float(stamp_ns) * 1e-9


def pose_to_row(stamp: float, pose) -> Row:
    p = pose.position
    q = pose.orientation
    return (stamp, p.x, p.y, p.z, q.x, q.y, q.z, q.w)


def iter_pose_rows(msg, msgtype: str, bag_time_ns: int, prefer_header_stamp: bool) -> Iterable[Row]:
    if msgtype == "nav_msgs/msg/Odometry":
        stamp = stamp_to_sec(msg.header.stamp) if prefer_header_stamp else ns_to_sec(bag_time_ns)
        yield pose_to_row(stamp, msg.pose.pose)
    elif msgtype == "geometry_msgs/msg/PoseStamped":
        stamp = stamp_to_sec(msg.header.stamp) if prefer_header_stamp else ns_to_sec(bag_time_ns)
        yield pose_to_row(stamp, msg.pose)
    elif msgtype == "nav_msgs/msg/Path":
        for pose_stamped in msg.poses:
            stamp = stamp_to_sec(pose_stamped.header.stamp) if prefer_header_stamp else ns_to_sec(bag_time_ns)
            yield pose_to_row(stamp, pose_stamped.pose)
    else:
        raise TypeError(f"unsupported message type: {msgtype}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="ROS2 bag directory")
    parser.add_argument("--topic", default="/Odometry")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bag-time", action="store_true", help="use bag receive time instead of message header time")
    parser.add_argument("--start", type=float, default=None, help="optional bag start time in seconds")
    parser.add_argument("--end", type=float, default=None, help="optional bag end time in seconds")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    count = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with AnyReader([args.bag], default_typestore=typestore) as reader, args.out.open("w", encoding="utf-8") as f:
        connections = [conn for conn in reader.connections if conn.topic == args.topic]
        if not connections:
            topics = ", ".join(sorted({conn.topic for conn in reader.connections}))
            raise SystemExit(f"topic {args.topic!r} not found. Available topics: {topics}")
        supported = {"nav_msgs/msg/Odometry", "geometry_msgs/msg/PoseStamped", "nav_msgs/msg/Path"}
        bad = [conn.msgtype for conn in connections if conn.msgtype not in supported]
        if bad:
            raise SystemExit(f"topic {args.topic!r} has unsupported message type(s): {bad}")

        start_ns = int(args.start * 1e9) if args.start is not None else None
        end_ns = int(args.end * 1e9) if args.end is not None else None
        for conn, timestamp, rawdata in reader.messages(connections=connections, start=start_ns, stop=end_ns):
            msg = reader.deserialize(rawdata, conn.msgtype)
            for row in iter_pose_rows(msg, conn.msgtype, timestamp, prefer_header_stamp=not args.bag_time):
                f.write(
                    f"{row[0]:.9f} {row[1]:.9f} {row[2]:.9f} {row[3]:.9f} "
                    f"{row[4]:.9f} {row[5]:.9f} {row[6]:.9f} {row[7]:.9f}\n"
                )
                count += 1

    print(f"wrote {count} poses to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Export ROS trajectory topics to the TUM trajectory text format.

Supported message types:
  - nav_msgs/Odometry
  - geometry_msgs/PoseStamped
  - nav_msgs/Path

Run this inside a ROS1 environment that has the rosbag Python module.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple


def stamp_to_sec(stamp) -> float:
    if hasattr(stamp, "to_sec"):
        return float(stamp.to_sec())
    return float(stamp.secs) + float(stamp.nsecs) * 1e-9


def pose_to_tuple(stamp: float, pose) -> Tuple[float, float, float, float, float, float, float, float]:
    p = pose.position
    q = pose.orientation
    return (stamp, p.x, p.y, p.z, q.x, q.y, q.z, q.w)


def iter_pose_rows(msg, bag_time, prefer_header_stamp: bool) -> Iterable[Tuple[float, float, float, float, float, float, float, float]]:
    typename = getattr(msg, "_type", "")
    if typename == "nav_msgs/Odometry":
        stamp = stamp_to_sec(msg.header.stamp) if prefer_header_stamp else stamp_to_sec(bag_time)
        yield pose_to_tuple(stamp, msg.pose.pose)
    elif typename == "geometry_msgs/PoseStamped":
        stamp = stamp_to_sec(msg.header.stamp) if prefer_header_stamp else stamp_to_sec(bag_time)
        yield pose_to_tuple(stamp, msg.pose)
    elif typename == "nav_msgs/Path":
        for pose_stamped in msg.poses:
            stamp = stamp_to_sec(pose_stamped.header.stamp) if prefer_header_stamp else stamp_to_sec(bag_time)
            yield pose_to_tuple(stamp, pose_stamped.pose)
    else:
        raise TypeError(f"unsupported message type: {typename}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--topic", default="/Odometry")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bag-time", action="store_true", help="use bag receive time instead of message header time")
    parser.add_argument("--start", type=float, default=None, help="optional bag start time in seconds")
    parser.add_argument("--end", type=float, default=None, help="optional bag end time in seconds")
    return parser


def optional_ros_time(seconds):
    if seconds is None:
        return None
    try:
        import rospy  # type: ignore

        return rospy.Time.from_sec(float(seconds))
    except ImportError:
        import genpy  # type: ignore

        return genpy.Time.from_sec(float(seconds))


def main() -> int:
    args = build_parser().parse_args()
    try:
        import rosbag  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "rosbag Python module not found. Source ROS1 first, e.g. "
            "`source /opt/ros/noetic/setup.bash`, then rerun."
        ) from exc

    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    start_time = optional_ros_time(args.start)
    end_time = optional_ros_time(args.end)
    with rosbag.Bag(str(args.bag), "r") as bag, args.out.open("w", encoding="utf-8") as f:
        for topic, msg, t in bag.read_messages(topics=[args.topic], start_time=start_time, end_time=end_time):
            del topic
            for row in iter_pose_rows(msg, t, prefer_header_stamp=not args.bag_time):
                f.write(
                    f"{row[0]:.9f} {row[1]:.9f} {row[2]:.9f} {row[3]:.9f} "
                    f"{row[4]:.9f} {row[5]:.9f} {row[6]:.9f} {row[7]:.9f}\n"
                )
                count += 1
    print(f"wrote {count} poses to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

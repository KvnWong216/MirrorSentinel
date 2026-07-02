#!/usr/bin/env python3
"""Summarize a ROS2 bag topic rate into the runtime metric schema.

This is useful for external baselines such as RTAB-Map, where we record the
published map/cloud topics but do not have MirrorSentinel's internal runtime
statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict

import yaml


def read_metadata(bag_dir: Path) -> Dict[str, Any]:
    metadata_path = bag_dir / "metadata.yaml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"ROS2 bag metadata not found: {metadata_path}")
    data = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    info = data.get("rosbag2_bagfile_information", {}) if isinstance(data, dict) else {}
    if not isinstance(info, dict):
        raise ValueError(f"invalid ROS2 bag metadata: {metadata_path}")
    return info


def topic_count(info: Dict[str, Any], topic: str) -> int:
    topics = info.get("topics_with_message_count", [])
    for item in topics:
        metadata = item.get("topic_metadata", {}) if isinstance(item, dict) else {}
        if metadata.get("name") == topic:
            return int(item.get("message_count", 0))
    available = sorted(
        item.get("topic_metadata", {}).get("name", "")
        for item in topics
        if isinstance(item, dict) and item.get("topic_metadata", {}).get("name")
    )
    raise ValueError(f"topic {topic!r} not found. Available topics: {', '.join(available)}")


def duration_seconds(info: Dict[str, Any]) -> float:
    duration = info.get("duration", {})
    if isinstance(duration, dict):
        ns = int(duration.get("nanoseconds", 0))
    else:
        ns = 0
    if ns <= 0:
        raise ValueError("bag duration is missing or non-positive")
    return float(ns) * 1e-9


def write_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "topic",
        "cloud_registered_frames",
        "pipeline_cloud_fps",
        "play_wall_time_s",
        "requested_play_rate",
        "depth_prior_fps",
        "reflection_prior_fps",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({name: row.get(name) for name in fields})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="ROS2 output bag directory")
    parser.add_argument("--topic", default="/cloud_map", help="topic used as pipeline output rate")
    parser.add_argument("--requested-play-rate", type=float, default=1.0)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    info = read_metadata(args.bag)
    count = topic_count(info, args.topic)
    duration_s = duration_seconds(info)
    fps = float(count) / duration_s if duration_s > 0.0 else 0.0
    result = {
        "topic": args.topic,
        "cloud_registered_frames": count,
        "odometry_frames": 0,
        "frame_stats_frames": 0,
        "pipeline_cloud_fps": fps,
        "odometry_fps": 0.0,
        "frame_stats_fps": 0.0,
        "depth_prior_fps": 0.0,
        "reflection_prior_fps": 0.0,
        "play_wall_time_s": duration_s,
        "requested_play_rate": float(args.requested_play_rate),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_csv is not None:
        write_csv(args.output_csv, result)
    print(f"{args.topic}: {count} messages / {duration_s:.3f}s = {fps:.3f} Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

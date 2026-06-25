#!/usr/bin/env python3
"""Summarize `/mirror_sentinel/frame_stats` from a ROS2 bag.

The topic is published as `std_msgs/msg/Float32MultiArray` by the Sentinel-LIO
node.  This script keeps the bag-level SLAM evaluation honest by reporting
whether an explicit mask was actually present and how many points were affected
by the mirror/depth priors.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


FIELDS = [
    "frame",
    "input_points",
    "output_points",
    "masked_points",
    "mean_confidence",
    "mask_coverage",
    "depth_valid_ratio",
    "explicit_mask_enabled",
    "ab_mode",
    "depth_checked_points",
    "depth_inconsistent_points",
    "ghost_candidate_points",
    "mask_core_points",
    "mask_boundary_points",
    "mean_depth_residual",
    "depth_calibration_valid",
    "depth_calibration_points",
    "depth_scale",
    "depth_shift",
    "calibration_mean_raw_residual",
    "calibration_mean_calibrated_residual",
]


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def summarize(values: Iterable[float]) -> Dict[str, float]:
    items = [float(v) for v in values]
    if not items:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(mean(items)),
        "p50": percentile(items, 0.50),
        "p95": percentile(items, 0.95),
        "max": float(max(items)),
    }


def read_rows(bag: Path, topic: str) -> List[Dict[str, float]]:
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    rows: List[Dict[str, float]] = []

    with AnyReader([bag], default_typestore=typestore) as reader:
        connections = [conn for conn in reader.connections if conn.topic == topic]
        if not connections:
            return rows
        bad = [conn.msgtype for conn in connections if conn.msgtype != "std_msgs/msg/Float32MultiArray"]
        if bad:
            raise SystemExit(f"topic {topic!r} has unsupported message type(s): {bad}")

        for conn, _timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, conn.msgtype)
            data = [float(x) for x in msg.data]
            row = {name: data[idx] if idx < len(data) else 0.0 for idx, name in enumerate(FIELDS)}
            if row["input_points"] > 0:
                row["masked_ratio"] = row["masked_points"] / row["input_points"]
                row["depth_checked_ratio"] = row["depth_checked_points"] / row["input_points"]
                row["ghost_candidate_ratio"] = row["ghost_candidate_points"] / row["input_points"]
                row["mask_core_ratio"] = row["mask_core_points"] / row["input_points"]
            else:
                row["masked_ratio"] = 0.0
                row["depth_checked_ratio"] = 0.0
                row["ghost_candidate_ratio"] = 0.0
                row["mask_core_ratio"] = 0.0
            if row["depth_checked_points"] > 0:
                row["depth_inconsistent_ratio"] = row["depth_inconsistent_points"] / row["depth_checked_points"]
            else:
                row["depth_inconsistent_ratio"] = 0.0
            rows.append(row)

    return rows


def aggregate(rows: List[Dict[str, float]]) -> Dict[str, object]:
    if not rows:
        return {
            "num_frames": 0,
            "warnings": ["missing /mirror_sentinel/frame_stats"],
        }

    warnings: List[str] = []
    mask_active_frames = sum(1 for row in rows if row["mask_coverage"] > 1e-6)
    depth_active_frames = sum(1 for row in rows if row["depth_valid_ratio"] > 1e-6)
    depth_checked_frames = sum(1 for row in rows if row["depth_checked_points"] > 0)
    mask_core_frames = sum(1 for row in rows if row["mask_core_points"] > 0)
    ghost_candidate_frames = sum(1 for row in rows if row["ghost_candidate_points"] > 0)
    explicit_enabled_frames = sum(1 for row in rows if row["explicit_mask_enabled"] > 0.5)
    calibration_valid_frames = sum(1 for row in rows if row["depth_calibration_valid"] > 0.5)

    if explicit_enabled_frames > 0 and mask_active_frames == 0:
        warnings.append("explicit mask was enabled, but mask_coverage stayed zero")
    if depth_active_frames > 0 and depth_checked_frames == 0:
        warnings.append("depth images were valid, but no LiDAR points were depth-checked")
    if mask_active_frames > 0 and mask_core_frames == 0:
        warnings.append("mask_coverage was nonzero, but no LiDAR points projected into mask core")
    if depth_active_frames > 0 and calibration_valid_frames == 0:
        warnings.append("depth was active, but non-ROI depth calibration never became valid")

    return {
        "num_frames": len(rows),
        "input_points_total": float(sum(row["input_points"] for row in rows)),
        "masked_points_total": float(sum(row["masked_points"] for row in rows)),
        "depth_checked_points_total": float(sum(row["depth_checked_points"] for row in rows)),
        "depth_inconsistent_points_total": float(sum(row["depth_inconsistent_points"] for row in rows)),
        "ghost_candidate_points_total": float(sum(row["ghost_candidate_points"] for row in rows)),
        "mask_core_points_total": float(sum(row["mask_core_points"] for row in rows)),
        "mask_boundary_points_total": float(sum(row["mask_boundary_points"] for row in rows)),
        "explicit_enabled_frame_ratio": explicit_enabled_frames / len(rows),
        "mask_active_frame_ratio": mask_active_frames / len(rows),
        "depth_active_frame_ratio": depth_active_frames / len(rows),
        "depth_checked_frame_ratio": depth_checked_frames / len(rows),
        "mask_core_frame_ratio": mask_core_frames / len(rows),
        "ghost_candidate_frame_ratio": ghost_candidate_frames / len(rows),
        "depth_calibration_valid_frame_ratio": calibration_valid_frames / len(rows),
        "depth_calibration_points": summarize(row["depth_calibration_points"] for row in rows),
        "depth_scale": summarize(row["depth_scale"] for row in rows if row["depth_calibration_valid"] > 0.5),
        "depth_shift": summarize(row["depth_shift"] for row in rows if row["depth_calibration_valid"] > 0.5),
        "calibration_mean_raw_residual": summarize(
            row["calibration_mean_raw_residual"] for row in rows if row["depth_calibration_valid"] > 0.5
        ),
        "calibration_mean_calibrated_residual": summarize(
            row["calibration_mean_calibrated_residual"] for row in rows if row["depth_calibration_valid"] > 0.5
        ),
        "mean_confidence": summarize(row["mean_confidence"] for row in rows),
        "masked_ratio": summarize(row["masked_ratio"] for row in rows),
        "mask_coverage": summarize(row["mask_coverage"] for row in rows),
        "depth_valid_ratio": summarize(row["depth_valid_ratio"] for row in rows),
        "depth_checked_ratio": summarize(row["depth_checked_ratio"] for row in rows),
        "depth_inconsistent_ratio": summarize(row["depth_inconsistent_ratio"] for row in rows),
        "ghost_candidate_ratio": summarize(row["ghost_candidate_ratio"] for row in rows),
        "mask_core_ratio": summarize(row["mask_core_ratio"] for row in rows),
        "mean_depth_residual": summarize(row["mean_depth_residual"] for row in rows),
        "warnings": warnings,
    }


def write_csv(path: Path, rows: List[Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = FIELDS + [
        "masked_ratio",
        "depth_checked_ratio",
        "depth_inconsistent_ratio",
        "ghost_candidate_ratio",
        "mask_core_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, 0.0) for name in fieldnames})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path, help="ROS2 bag directory")
    parser.add_argument("--topic", default="/mirror_sentinel/frame_stats")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = read_rows(args.bag, args.topic)
    summary = aggregate(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_csv is not None:
        write_csv(args.output_csv, rows)
    print(
        "Frame stats: "
        f"frames={summary.get('num_frames', 0)}, "
        f"mask_active={summary.get('mask_active_frame_ratio', 0.0):.3f}, "
        f"depth_active={summary.get('depth_active_frame_ratio', 0.0):.3f}, "
        f"ghost_candidates={summary.get('ghost_candidate_points_total', 0.0):.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

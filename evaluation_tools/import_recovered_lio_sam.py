#!/usr/bin/env python3
"""Import a recovered LIO-SAM result folder into the unified eval layout.

The historical ROS1-era LIO-SAM exports usually contain separate CornerMap.pcd
and SurfMap.pcd files rather than a single global map.  This importer preserves
those originals, writes a combined xyz map.pcd for the current evaluators, and
records runtime metadata derived from odometry.txt/run_summary.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from map_ghost_eval import read_point_cloud
from reconstruction.apply_neuralrecon_prior_to_map import write_pcd_ascii


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path("/home/wangyg/Downloads/lio_sam/newseq_001")
DEFAULT_OUT = (
    REPO_ROOT
    / "evaluation_tools/results/slam/self_collected"
    / "2026-03-30-21-31-03_lio_sam_ros1_recovered/lio_sam"
)


def read_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def count_nonempty_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return sum(1 for line in f if line.strip())


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    return max(0, len(rows) - 1)


def odometry_duration(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    first: Optional[float] = None
    last: Optional[float] = None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            try:
                stamp = float(parts[0])
            except ValueError:
                continue
            if first is None:
                first = stamp
            last = stamp
    if first is None or last is None:
        return None
    return max(0.0, last - first)


def write_runtime_json(path: Path, runtime: Dict[str, Any]) -> None:
    path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_runtime_csv(path: Path, runtime: Dict[str, Any]) -> None:
    fields = [
        "source",
        "map_point_count",
        "corner_map_points",
        "surf_map_points",
        "trajectory_keyframes",
        "odometry_frames",
        "odometry_duration_sec",
        "odometry_fps",
        "pipeline_cloud_fps",
        "cloud_registered_frames",
        "frame_stats_frames",
        "depth_prior_fps",
        "reflection_prior_fps",
        "runtime_note",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: runtime.get(key) for key in fields})


def copy_if_exists(source: Path, out_dir: Path) -> None:
    if source.exists():
        shutil.copy2(source, out_dir / source.name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source = args.source.expanduser().resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    corner_path = source / "CornerMap.pcd"
    surf_path = source / "SurfMap.pcd"
    if not corner_path.exists():
        raise SystemExit(f"missing recovered LIO-SAM CornerMap: {corner_path}")
    if not surf_path.exists():
        raise SystemExit(f"missing recovered LIO-SAM SurfMap: {surf_path}")

    corner = read_point_cloud(corner_path)
    surf = read_point_cloud(surf_path)
    combined = np.vstack([corner, surf])
    write_pcd_ascii(out_dir / "map.pcd", combined)

    for name in [
        "CornerMap.pcd",
        "SurfMap.pcd",
        "trajectory.pcd",
        "transformations.pcd",
        "odometry.txt",
        "frame_stats.csv",
        "run_summary.json",
    ]:
        copy_if_exists(source / name, out_dir)

    summary = read_json_if_exists(source / "run_summary.json")
    odom_count = int(summary.get("odom_count") or count_nonempty_lines(source / "odometry.txt"))
    duration = summary.get("odom_duration_sec")
    if duration is None:
        duration = odometry_duration(source / "odometry.txt")
    duration_value = float(duration) if duration is not None else 0.0
    fps = float(summary.get("odom_fps") or (odom_count / duration_value if duration_value > 0.0 else 0.0))
    trajectory_points = 0
    if (source / "trajectory.pcd").exists():
        trajectory_points = int(read_point_cloud(source / "trajectory.pcd").shape[0])
    frame_stats_count = int(summary.get("stats_count") or count_csv_rows(source / "frame_stats.csv"))

    runtime: Dict[str, Any] = {
        "source": str(source),
        "map_point_count": int(combined.shape[0]),
        "corner_map_points": int(corner.shape[0]),
        "surf_map_points": int(surf.shape[0]),
        "trajectory_keyframes": trajectory_points,
        "odometry_frames": odom_count,
        "odometry_duration_sec": duration_value,
        "odometry_fps": fps,
        "pipeline_cloud_fps": fps,
        "cloud_registered_frames": trajectory_points,
        "frame_stats_frames": frame_stats_count,
        "depth_prior_fps": 0.0,
        "reflection_prior_fps": 0.0,
        "runtime_note": (
            "recovered historical ROS1-era LIO-SAM output; FPS is derived "
            "from odometry.txt/run_summary.json, not a current ROS2 replay"
        ),
    }
    write_runtime_json(out_dir / "metrics_runtime.json", runtime)
    write_runtime_csv(out_dir / "metrics_runtime.csv", runtime)

    metadata = {
        "source": str(source),
        "out_dir": str(out_dir),
        "combined_from": ["CornerMap.pcd", "SurfMap.pcd"],
        "map": str(out_dir / "map.pcd"),
        "runtime": runtime,
    }
    (out_dir / "import_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"wrote {out_dir / 'map.pcd'} ({combined.shape[0]} points)")
    print(f"runtime fps: {fps:.3f}, odom frames: {odom_count}, keyframes: {trajectory_points}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

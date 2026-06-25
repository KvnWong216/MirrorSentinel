#!/usr/bin/env python3
"""No-GT trajectory consistency metrics for self-collected scenes.

Use this for real reflective-scene experiments when no motion-capture or
survey-grade ground truth is available. It does not replace ATE/RPE; it reports
loop closure drift, trajectory length, and optional revisit-pair consistency.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from tum_trajectory_eval import Pose, pose_matrix, read_tum_trajectory, rotation_angle, stats


def sorted_poses(path: Path) -> List[Pose]:
    return [p for _, p in sorted(read_tum_trajectory(path).items())]


def trajectory_length(poses: Sequence[Pose]) -> float:
    if len(poses) < 2:
        return 0.0
    return float(sum(np.linalg.norm(poses[i].t - poses[i - 1].t) for i in range(1, len(poses))))


def nearest_index_by_time(poses: Sequence[Pose], stamp: float) -> int:
    times = np.asarray([p.stamp for p in poses], dtype=np.float64)
    return int(np.argmin(np.abs(times - stamp)))


def loop_metrics(
    poses: Sequence[Pose],
    start_time: Optional[float],
    end_time: Optional[float],
    start_index: Optional[int],
    end_index: Optional[int],
) -> Dict[str, float]:
    if len(poses) < 2:
        raise ValueError("need at least two poses")
    i = start_index if start_index is not None else 0
    j = end_index if end_index is not None else len(poses) - 1
    if start_time is not None:
        i = nearest_index_by_time(poses, start_time)
    if end_time is not None:
        j = nearest_index_by_time(poses, end_time)
    if i == j:
        raise ValueError("loop start and end resolve to the same pose")
    T_i = pose_matrix(poses[i])
    T_j = pose_matrix(poses[j])
    rel = np.linalg.inv(T_i) @ T_j
    length = trajectory_length(poses[min(i, j) : max(i, j) + 1])
    trans = float(np.linalg.norm(rel[:3, 3]))
    rot_rad = float(rotation_angle(rel[:3, :3]))
    return {
        "start_index": float(i),
        "end_index": float(j),
        "start_time": poses[i].stamp,
        "end_time": poses[j].stamp,
        "trajectory_length_m": length,
        "loop_translation_error_m": trans,
        "loop_rotation_error_rad": rot_rad,
        "loop_rotation_error_deg": math.degrees(rot_rad),
        "loop_drift_ratio": trans / length if length > 0.0 else float("nan"),
    }


def read_revisit_pairs(path: Path) -> List[Tuple[float, float]]:
    pairs: List[Tuple[float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                raise ValueError(f"{path}:{line_no}: expected two timestamps")
            pairs.append((float(parts[0]), float(parts[1])))
    return pairs


def revisit_metrics(poses: Sequence[Pose], pairs: Sequence[Tuple[float, float]]) -> Dict[str, object]:
    trans_errors: List[float] = []
    rot_errors: List[float] = []
    rows: List[Dict[str, float]] = []
    for a, b in pairs:
        i = nearest_index_by_time(poses, a)
        j = nearest_index_by_time(poses, b)
        T_i = pose_matrix(poses[i])
        T_j = pose_matrix(poses[j])
        rel = np.linalg.inv(T_i) @ T_j
        trans = float(np.linalg.norm(rel[:3, 3]))
        rot = float(rotation_angle(rel[:3, :3]))
        trans_errors.append(trans)
        rot_errors.append(rot)
        rows.append(
            {
                "requested_time_a": a,
                "requested_time_b": b,
                "matched_time_a": poses[i].stamp,
                "matched_time_b": poses[j].stamp,
                "translation_error_m": trans,
                "rotation_error_deg": math.degrees(rot),
            }
        )
    return {
        "num_pairs": len(rows),
        "translation_error_m": stats(trans_errors),
        "rotation_error_deg": stats([math.degrees(x) for x in rot_errors]),
        "pairs": rows,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path, help="TUM-format estimated trajectory")
    parser.add_argument("--output-json", type=Path, default=Path("evaluation_tools/results/self_collected/no_gt_metrics.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("evaluation_tools/results/self_collected/no_gt_metrics.csv"))
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--revisit-pairs", type=Path, default=None, help="CSV/txt with timestamp_a,timestamp_b pairs")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    poses = sorted_poses(args.trajectory)
    result: Dict[str, object] = {
        "trajectory": str(args.trajectory),
        "num_poses": len(poses),
        "duration_s": poses[-1].stamp - poses[0].stamp if len(poses) >= 2 else 0.0,
        "trajectory_length_m": trajectory_length(poses),
        "loop": loop_metrics(poses, args.start_time, args.end_time, args.start_index, args.end_index),
    }
    if args.revisit_pairs:
        result["revisit"] = revisit_metrics(poses, read_revisit_pairs(args.revisit_pairs))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    loop = result["loop"]  # type: ignore[index]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in loop.items():  # type: ignore[union-attr]
            writer.writerow({"metric": f"loop.{key}", "value": value})
        writer.writerow({"metric": "trajectory_length_m", "value": result["trajectory_length_m"]})
        writer.writerow({"metric": "num_poses", "value": result["num_poses"]})

    print(
        "Loop trans/rot/drift: "
        f"{loop['loop_translation_error_m']:.6f} m, "  # type: ignore[index]
        f"{loop['loop_rotation_error_deg']:.6f} deg, "  # type: ignore[index]
        f"{loop['loop_drift_ratio']:.6f}"  # type: ignore[index]
    )
    print(f"wrote: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

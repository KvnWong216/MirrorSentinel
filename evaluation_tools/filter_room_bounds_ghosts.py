#!/usr/bin/env python3
"""Filter behind-wall points using the annotated room-boundary oracle.

This is a diagnostic/oracle tool, not an online SLAM method.  It uses the same
`room_bounds`/reflective-plane annotation and behind-plane definition as
`map_ghost_eval.py`, then exports a map with those evaluated ghost points
removed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "evaluation_tools") not in sys.path:
    sys.path.append(str(REPO_ROOT / "evaluation_tools"))

from map_ghost_eval import (  # noqa: E402
    behind_mask,
    load_annotations,
    points_in_roi,
    read_point_cloud,
    signed_distances,
)


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
            "",
        ]
    )


def write_pcd_ascii(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float64)
    with path.open("w", encoding="utf-8") as f:
        f.write(pcd_header(points.shape[0]))
        if points.shape[0] > 0:
            np.savetxt(f, points[:, :3], fmt="%.6f %.6f %.6f")


def finite_stats(values: np.ndarray) -> Dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": None, "median": None, "p90": None, "p95": None, "max": None}
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, type=Path, help="input map point cloud")
    parser.add_argument("--annotation", required=True, type=Path, help="room_bounds or reflective plane YAML")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="behind-plane margin in meters; default matches primary map_ghost_eval threshold",
    )
    parser.add_argument(
        "--thresholds-m",
        default="0.05,0.10,0.20",
        help="fallback thresholds if annotation does not specify thresholds_m",
    )
    parser.add_argument("--skip-eval", action="store_true", help="only write filtered/rejected clouds")
    return parser


def run_eval(map_path: Path, annotation: Path, output_json: Path, output_csv: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "evaluation_tools/map_ghost_eval.py"),
            "--map",
            str(map_path),
            "--annotation",
            str(annotation),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )


def main() -> int:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fallback_thresholds = [float(x) for x in args.thresholds_m.split(",") if x.strip()]
    points = read_point_cloud(args.map)
    _, reflective, _ = load_annotations(args.annotation, fallback_thresholds)
    if not reflective:
        raise ValueError(f"{args.annotation}: no reflective room/plane annotations")

    reject_mask = np.zeros(points.shape[0], dtype=bool)
    candidate_mask = np.zeros(points.shape[0], dtype=bool)
    min_behind_distance = np.full(points.shape[0], np.inf, dtype=np.float64)
    per_region = []

    for plane in reflective:
        roi_mask = points_in_roi(points, plane.roi)
        candidate_mask |= roi_mask
        idx = np.flatnonzero(roi_mask)
        if idx.size == 0:
            per_region.append(
                {
                    "id": plane.region_id,
                    "roi_point_count": 0,
                    "rejected_point_count": 0,
                    "margin_m": args.margin,
                }
            )
            continue

        dist = signed_distances(points[idx], plane)
        behind = behind_mask(dist, plane.front_side, args.margin)
        rejected_idx = idx[behind]
        reject_mask[rejected_idx] = True
        min_behind_distance[rejected_idx] = np.minimum(min_behind_distance[rejected_idx], np.abs(dist[behind]))
        per_region.append(
            {
                "id": plane.region_id,
                "type": plane.plane_type,
                "roi_point_count": int(idx.size),
                "rejected_point_count": int(rejected_idx.size),
                "margin_m": args.margin,
                "rejected_distance_m": finite_stats(np.abs(dist[behind])),
            }
        )

    filtered = points[~reject_mask]
    rejected = points[reject_mask]
    candidates = points[candidate_mask]

    filtered_map = args.out_dir / "filtered_map.pcd"
    rejected_points = args.out_dir / "rejected_behind_points.pcd"
    candidate_points = args.out_dir / "room_candidate_points.pcd"
    write_pcd_ascii(filtered_map, filtered)
    write_pcd_ascii(rejected_points, rejected)
    write_pcd_ascii(candidate_points, candidates)

    summary: Dict[str, Any] = {
        "map": str(args.map),
        "annotation": str(args.annotation),
        "output_dir": str(args.out_dir),
        "margin_m": args.margin,
        "counts": {
            "input_points": int(points.shape[0]),
            "candidate_points": int(np.count_nonzero(candidate_mask)),
            "rejected_points": int(np.count_nonzero(reject_mask)),
            "kept_points": int(filtered.shape[0]),
        },
        "ratios": {
            "candidate_ratio": float(np.count_nonzero(candidate_mask) / points.shape[0]) if points.shape[0] else 0.0,
            "reject_ratio_all_points": float(np.count_nonzero(reject_mask) / points.shape[0]) if points.shape[0] else 0.0,
            "reject_ratio_candidates": (
                float(np.count_nonzero(reject_mask) / np.count_nonzero(candidate_mask))
                if np.count_nonzero(candidate_mask)
                else 0.0
            ),
        },
        "rejected_distance_m": finite_stats(min_behind_distance[reject_mask]),
        "per_region": per_region,
        "outputs": {
            "filtered_map": str(filtered_map),
            "rejected_behind_points": str(rejected_points),
            "room_candidate_points": str(candidate_points),
            "summary": str(args.out_dir / "summary.json"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not args.skip_eval:
        run_eval(args.map, args.annotation, args.out_dir / "metrics_before.json", args.out_dir / "metrics_before.csv")
        run_eval(filtered_map, args.annotation, args.out_dir / "metrics_after.json", args.out_dir / "metrics_after.csv")

    print(
        "Room-boundary oracle filter: "
        f"input={points.shape[0]} candidate={np.count_nonzero(candidate_mask)} "
        f"rejected={np.count_nonzero(reject_mask)} kept={filtered.shape[0]}"
    )
    print(f"wrote: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

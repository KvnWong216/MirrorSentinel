#!/usr/bin/env python3
"""Filter a dense accumulated map by deleted-history marker points."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "evaluation_tools") not in sys.path:
    sys.path.append(str(REPO_ROOT / "evaluation_tools"))

from map_ghost_eval import read_point_cloud  # noqa: E402
from reconstruction.apply_neuralrecon_prior_to_map import write_pcd_ascii  # noqa: E402


def finite_stats(values: np.ndarray) -> Dict[str, Optional[float]]:
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
    parser.add_argument("--map", required=True, type=Path, help="dense accumulated map")
    parser.add_argument("--markers", required=True, type=Path, help="deleted-history marker point cloud")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--radius", required=True, type=float, help="reject points within this radius of any marker")
    parser.add_argument("--annotation", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=200000)
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
    points = read_point_cloud(args.map)
    markers = read_point_cloud(args.markers)
    if markers.shape[0] == 0:
        raise ValueError(f"{args.markers}: empty marker cloud")
    if args.radius <= 0.0:
        raise ValueError("--radius must be positive")

    tree = cKDTree(markers[:, :3])
    reject = np.zeros(points.shape[0], dtype=bool)
    nearest = np.full(points.shape[0], np.inf, dtype=np.float32)
    all_idx = np.arange(points.shape[0])
    for start in range(0, points.shape[0], args.batch_size):
        idx = all_idx[start : start + args.batch_size]
        dist = tree.query(points[idx, :3], k=1, workers=-1)[0].astype(np.float32)
        nearest[idx] = dist
        reject[idx] = dist <= args.radius

    filtered = points[~reject]
    rejected = points[reject]
    filtered_map = args.out_dir / "filtered_map.pcd"
    rejected_points = args.out_dir / "rejected_by_markers.pcd"
    write_pcd_ascii(filtered_map, filtered)
    write_pcd_ascii(rejected_points, rejected)

    summary: Dict[str, Any] = {
        "map": str(args.map),
        "markers": str(args.markers),
        "radius": args.radius,
        "counts": {
            "input_points": int(points.shape[0]),
            "marker_points": int(markers.shape[0]),
            "rejected_points": int(np.count_nonzero(reject)),
            "kept_points": int(filtered.shape[0]),
        },
        "ratios": {
            "reject_ratio": float(np.count_nonzero(reject) / points.shape[0]) if points.shape[0] else 0.0,
            "keep_ratio": float(filtered.shape[0] / points.shape[0]) if points.shape[0] else 0.0,
        },
        "nearest_marker_distance_m": finite_stats(nearest),
        "outputs": {
            "filtered_map": str(filtered_map),
            "rejected_by_markers": str(rejected_points),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.annotation:
        run_eval(filtered_map, args.annotation, args.out_dir / "metrics_mapping.json", args.out_dir / "metrics_mapping.csv")

    print(
        "Marker filter: "
        f"input={points.shape[0]} markers={markers.shape[0]} "
        f"rejected={np.count_nonzero(reject)} kept={filtered.shape[0]} radius={args.radius:.3f}"
    )
    print(f"wrote: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

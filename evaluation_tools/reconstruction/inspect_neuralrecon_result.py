#!/usr/bin/env python3
"""Inspect NeuralRecon outputs and optional overlap with a SLAM map."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "evaluation_tools") not in sys.path:
    sys.path.append(str(REPO_ROOT / "evaluation_tools"))

from map_ghost_eval import read_point_cloud  # noqa: E402


def stats(values: np.ndarray) -> Dict[str, Optional[float]]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None, "p50": None, "p95": None}
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def point_cloud_summary(path: Path) -> Dict[str, Any]:
    points = read_point_cloud(path)
    if points.shape[0] == 0:
        return {"path": str(path), "point_count": 0}
    return {
        "path": str(path),
        "point_count": int(points.shape[0]),
        "min": points.min(axis=0).tolist(),
        "max": points.max(axis=0).tolist(),
        "range": np.ptp(points, axis=0).tolist(),
    }


def tsdf_summary(path: Path) -> Dict[str, Any]:
    data = np.load(path)
    tsdf = np.asarray(data["tsdf"], dtype=np.float32)
    origin = np.asarray(data["origin"], dtype=np.float64)
    voxel_size = float(data["voxel_size"])
    shape = np.asarray(tsdf.shape, dtype=np.int64)
    bounds_min = origin
    bounds_max = origin + shape.astype(np.float64) * voxel_size
    valid = np.abs(tsdf) < 0.999
    near_surface = np.abs(tsdf) < 0.05
    return {
        "path": str(path),
        "shape": [int(x) for x in tsdf.shape],
        "origin": origin.tolist(),
        "voxel_size": voxel_size,
        "bounds_min": bounds_min.tolist(),
        "bounds_max": bounds_max.tolist(),
        "tsdf_min": float(tsdf.min()),
        "tsdf_max": float(tsdf.max()),
        "valid_voxel_count_abs_lt_0_999": int(np.count_nonzero(valid)),
        "negative_voxel_count": int(np.count_nonzero(tsdf < 0)),
        "near_surface_voxel_count_abs_lt_0_05": int(np.count_nonzero(near_surface)),
    }


def overlap_summary(map_path: Path, bounds_min: np.ndarray, bounds_max: np.ndarray) -> Dict[str, Any]:
    points = read_point_cloud(map_path)
    inside = np.logical_and(points >= bounds_min, points <= bounds_max).all(axis=1)
    inside_points = points[inside]
    return {
        "map": str(map_path),
        "map_point_count": int(points.shape[0]),
        "points_inside_neuralrecon_volume": int(inside.sum()),
        "inside_ratio": float(inside.sum() / points.shape[0]) if points.shape[0] else 0.0,
        "inside_bounds_min": inside_points.min(axis=0).tolist() if inside_points.size else None,
        "inside_bounds_max": inside_points.max(axis=0).tolist() if inside_points.size else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neuralrecon-npz", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--map", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tsdf = tsdf_summary(args.neuralrecon_npz)
    mesh = point_cloud_summary(args.mesh)
    result: Dict[str, Any] = {"tsdf": tsdf, "mesh": mesh}
    if args.map is not None:
        result["map_overlap"] = overlap_summary(
            args.map,
            np.asarray(tsdf["bounds_min"], dtype=np.float64),
            np.asarray(tsdf["bounds_max"], dtype=np.float64),
        )

    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
        print(f"wrote: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

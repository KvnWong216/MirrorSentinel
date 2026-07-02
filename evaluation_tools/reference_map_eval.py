#!/usr/bin/env python3
"""Reference-cloud map quality metrics.

This is the lightweight ETH3D/Tanks-and-Temples style evaluator for a saved
SLAM map against a reference point cloud.  It reports precision/accuracy,
recall/completeness, F-score, and nearest-neighbor distance summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from map_ghost_eval import read_point_cloud


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def parse_thresholds(text: str) -> List[float]:
    values = [float(x) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("thresholds must not be empty")
    return values


def crop_points(
    points: np.ndarray,
    crop_min: Optional[Sequence[float]],
    crop_max: Optional[Sequence[float]],
) -> np.ndarray:
    if crop_min is None or crop_max is None:
        return points
    a = np.asarray(crop_min, dtype=np.float64)
    b = np.asarray(crop_max, dtype=np.float64)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return points[np.logical_and(points >= lo, points <= hi).all(axis=1)]


def sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def nearest_distances(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    if src.shape[0] == 0 or dst.shape[0] == 0:
        return np.full(src.shape[0], np.inf, dtype=np.float64)
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(dst)
        dist, _ = tree.query(src, k=1, workers=-1)
        return np.asarray(dist, dtype=np.float64)
    except Exception:
        chunk = 8192
        out = np.empty(src.shape[0], dtype=np.float64)
        for start in range(0, src.shape[0], chunk):
            part = src[start : start + chunk]
            diff = part[:, None, :] - dst[None, :, :]
            out[start : start + chunk] = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
        return out


def distance_stats(values: np.ndarray) -> Dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0.0, "mean": None, "median": None, "p90": None, "p95": None, "rmse": None}
    return {
        "count": float(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "rmse": float(np.sqrt(np.mean(finite * finite))),
    }


def evaluate(map_points: np.ndarray, ref_points: np.ndarray, thresholds: Sequence[float]) -> Dict[str, Any]:
    map_to_ref = nearest_distances(map_points, ref_points)
    ref_to_map = nearest_distances(ref_points, map_points)
    rows: Dict[str, Dict[str, float]] = {}
    for tau in thresholds:
        precision = safe_div(float(np.count_nonzero(map_to_ref <= tau)), float(map_points.shape[0]))
        recall = safe_div(float(np.count_nonzero(ref_to_map <= tau)), float(ref_points.shape[0]))
        rows[f"{tau:.3f}"] = {
            "tau_m": float(tau),
            "accuracy_precision": precision,
            "completeness_recall": recall,
            "f_score": safe_div(2.0 * precision * recall, precision + recall),
        }
    return {
        "map_point_count": float(map_points.shape[0]),
        "reference_point_count": float(ref_points.shape[0]),
        "map_to_reference_distance_m": distance_stats(map_to_ref),
        "reference_to_map_distance_m": distance_stats(ref_to_map),
        "thresholds": rows,
    }


def write_outputs(output_json: Path, output_csv: Optional[Path], result: Dict[str, Any]) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if output_csv is None:
        return
    rows = []
    for key, value in result["aggregate"]["thresholds"].items():
        row = {"threshold": key}
        row.update(value)
        rows.append(row)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "tau_m", "accuracy_precision", "completeness_recall", "f_score"])
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--thresholds-m", default="0.05,0.10,0.20")
    parser.add_argument("--crop-min", type=float, nargs=3, default=None)
    parser.add_argument("--crop-max", type=float, nargs=3, default=None)
    parser.add_argument("--max-map-points", type=int, default=300000)
    parser.add_argument("--max-reference-points", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    thresholds = parse_thresholds(args.thresholds_m)
    map_points = crop_points(read_point_cloud(args.map), args.crop_min, args.crop_max)
    ref_points = crop_points(read_point_cloud(args.reference), args.crop_min, args.crop_max)
    map_points = sample_points(map_points, args.max_map_points, args.seed)
    ref_points = sample_points(ref_points, args.max_reference_points, args.seed + 1)
    if map_points.shape[0] == 0:
        raise SystemExit("map has no points after crop")
    if ref_points.shape[0] == 0:
        raise SystemExit("reference has no points after crop")
    result = {
        "map": str(args.map),
        "reference": str(args.reference),
        "thresholds_m": thresholds,
        "aggregate": evaluate(map_points, ref_points, thresholds),
    }
    write_outputs(args.output_json, args.output_csv, result)
    primary = result["aggregate"]["thresholds"].get(f"{thresholds[0]:.3f}", {})
    print(
        "Reference map eval: "
        f"P={primary.get('accuracy_precision', 0.0):.6f}, "
        f"R={primary.get('completeness_recall', 0.0):.6f}, "
        f"F={primary.get('f_score', 0.0):.6f} @ {thresholds[0]:.3f}m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

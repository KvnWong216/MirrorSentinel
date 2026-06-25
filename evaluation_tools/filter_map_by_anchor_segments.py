#!/usr/bin/env python3
"""Filter a dense map with paired surface-anchor and deleted-marker segments.

Each anchor-marker pair represents one historical map point rejected by the
depth/reconstruction prior.  The anchor is the prior surface point on the same
camera ray, and the marker is the rejected ghost point behind it.  This tool
rejects dense accumulated map points close to these anchor->marker capsules,
which covers the reflective ghost band better than a marker-only radius filter.
"""

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
    parser.add_argument("--markers", required=True, type=Path, help="paired rejected marker cloud")
    parser.add_argument("--anchors", required=True, type=Path, help="paired prior surface anchor cloud")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--radius", type=float, default=0.12, help="capsule radius in meters")
    parser.add_argument(
        "--search-radius",
        type=float,
        default=None,
        help="KD-tree marker search radius; default max-segment-length + max-extension + radius",
    )
    parser.add_argument("--max-extension", type=float, default=0.75, help="allow points this far beyond marker")
    parser.add_argument(
        "--min-along-ratio",
        type=float,
        default=0.05,
        help="start rejecting after this fraction of the anchor->marker segment",
    )
    parser.add_argument(
        "--max-segment-length",
        type=float,
        default=8.0,
        help="discard implausibly long anchor-marker pairs",
    )
    parser.add_argument(
        "--min-segment-length",
        type=float,
        default=0.15,
        help="discard tiny anchor-marker pairs",
    )
    parser.add_argument("--max-neighbors", type=int, default=64, help="cap candidate segments per point")
    parser.add_argument("--annotation", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=50000)
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


def reject_batch(
    points: np.ndarray,
    marker_tree: cKDTree,
    anchors: np.ndarray,
    markers: np.ndarray,
    directions: np.ndarray,
    lengths: np.ndarray,
    radius: float,
    search_radius: float,
    max_extension: float,
    min_along_ratio: float,
    max_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reject = np.zeros(points.shape[0], dtype=bool)
    best_distance = np.full(points.shape[0], np.inf, dtype=np.float32)
    best_along = np.full(points.shape[0], np.nan, dtype=np.float32)
    best_segment = np.full(points.shape[0], -1, dtype=np.int64)

    candidates = marker_tree.query_ball_point(points[:, :3], search_radius, workers=-1)
    radius_sq = radius * radius
    for i, cand in enumerate(candidates):
        if not cand:
            continue
        if len(cand) > max_neighbors:
            marker_dists = np.linalg.norm(markers[cand] - points[i, :3], axis=1)
            cand = [cand[j] for j in np.argsort(marker_dists)[:max_neighbors]]

        idx = np.asarray(cand, dtype=np.int64)
        vec = points[i, :3][None, :] - anchors[idx]
        along = np.einsum("ij,ij->i", vec, directions[idx])
        lower = min_along_ratio * lengths[idx]
        upper = lengths[idx] + max_extension
        valid = (along >= lower) & (along <= upper)
        if not np.any(valid):
            continue

        idx = idx[valid]
        along = along[valid]
        closest = anchors[idx] + directions[idx] * along[:, None]
        dist_sq = np.sum((points[i, :3][None, :] - closest) ** 2, axis=1)
        best_local = int(np.argmin(dist_sq))
        if dist_sq[best_local] < best_distance[i] * best_distance[i]:
            best_distance[i] = float(np.sqrt(dist_sq[best_local]))
            best_along[i] = float(along[best_local])
            best_segment[i] = int(idx[best_local])
        if dist_sq[best_local] <= radius_sq:
            reject[i] = True

    return reject, best_distance, best_along, best_segment


def main() -> int:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.radius <= 0.0:
        raise ValueError("--radius must be positive")
    if args.max_neighbors <= 0:
        raise ValueError("--max-neighbors must be positive")

    points = read_point_cloud(args.map)
    markers_all = read_point_cloud(args.markers)
    anchors_all = read_point_cloud(args.anchors)
    pair_count = min(markers_all.shape[0], anchors_all.shape[0])
    if pair_count == 0:
        raise ValueError("empty paired marker/anchor clouds")
    markers = markers_all[:pair_count, :3]
    anchors = anchors_all[:pair_count, :3]
    segments = markers - anchors
    lengths = np.linalg.norm(segments, axis=1)
    valid_pairs = (
        np.isfinite(markers).all(axis=1)
        & np.isfinite(anchors).all(axis=1)
        & (lengths >= args.min_segment_length)
        & (lengths <= args.max_segment_length)
    )
    markers = markers[valid_pairs]
    anchors = anchors[valid_pairs]
    lengths = lengths[valid_pairs]
    if markers.shape[0] == 0:
        raise ValueError("no valid anchor-marker pairs after length filtering")
    directions = (markers - anchors) / lengths[:, None]
    search_radius = args.search_radius
    if search_radius is None:
        search_radius = args.max_segment_length + max(0.0, args.max_extension) + args.radius
    search_radius = max(search_radius, args.radius)

    marker_tree = cKDTree(markers)
    reject = np.zeros(points.shape[0], dtype=bool)
    best_distance = np.full(points.shape[0], np.inf, dtype=np.float32)
    best_along = np.full(points.shape[0], np.nan, dtype=np.float32)
    best_segment = np.full(points.shape[0], -1, dtype=np.int64)
    for start in range(0, points.shape[0], args.batch_size):
        end = min(start + args.batch_size, points.shape[0])
        batch_reject, batch_dist, batch_along, batch_segment = reject_batch(
            points[start:end],
            marker_tree,
            anchors,
            markers,
            directions,
            lengths,
            args.radius,
            search_radius,
            args.max_extension,
            args.min_along_ratio,
            args.max_neighbors,
        )
        reject[start:end] = batch_reject
        best_distance[start:end] = batch_dist
        best_along[start:end] = batch_along
        best_segment[start:end] = batch_segment

    filtered = points[~reject]
    rejected = points[reject]
    filtered_map = args.out_dir / "filtered_map.pcd"
    rejected_points = args.out_dir / "rejected_by_anchor_segments.pcd"
    valid_marker_path = args.out_dir / "valid_segment_markers.pcd"
    valid_anchor_path = args.out_dir / "valid_segment_anchors.pcd"
    write_pcd_ascii(filtered_map, filtered)
    write_pcd_ascii(rejected_points, rejected)
    write_pcd_ascii(valid_marker_path, markers)
    write_pcd_ascii(valid_anchor_path, anchors)

    summary: Dict[str, Any] = {
        "map": str(args.map),
        "markers": str(args.markers),
        "anchors": str(args.anchors),
        "parameters": {
            "radius": args.radius,
            "search_radius": search_radius,
            "max_extension": args.max_extension,
            "min_along_ratio": args.min_along_ratio,
            "min_segment_length": args.min_segment_length,
            "max_segment_length": args.max_segment_length,
            "max_neighbors": args.max_neighbors,
        },
        "counts": {
            "input_points": int(points.shape[0]),
            "marker_points": int(markers_all.shape[0]),
            "anchor_points": int(anchors_all.shape[0]),
            "paired_points": int(pair_count),
            "valid_pairs": int(markers.shape[0]),
            "rejected_points": int(np.count_nonzero(reject)),
            "kept_points": int(filtered.shape[0]),
        },
        "ratios": {
            "reject_ratio": float(np.count_nonzero(reject) / points.shape[0]) if points.shape[0] else 0.0,
            "keep_ratio": float(filtered.shape[0] / points.shape[0]) if points.shape[0] else 0.0,
        },
        "segment_length_m": finite_stats(lengths),
        "nearest_segment_distance_m": finite_stats(best_distance),
        "nearest_segment_along_m": finite_stats(best_along),
        "matched_segment_count": int(np.count_nonzero(best_segment >= 0)),
        "outputs": {
            "filtered_map": str(filtered_map),
            "rejected_by_anchor_segments": str(rejected_points),
            "valid_segment_markers": str(valid_marker_path),
            "valid_segment_anchors": str(valid_anchor_path),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.annotation:
        run_eval(filtered_map, args.annotation, args.out_dir / "metrics_mapping.json", args.out_dir / "metrics_mapping.csv")

    print(
        "Anchor segment filter: "
        f"input={points.shape[0]} valid_pairs={markers.shape[0]} "
        f"rejected={np.count_nonzero(reject)} kept={filtered.shape[0]} radius={args.radius:.3f}"
    )
    print(f"wrote: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Apply a NeuralRecon mesh/TSDF prior to a saved SLAM map.

This is an offline sanity-check stage: points inside NeuralRecon's reconstructed
volume that are far from the reconstructed surface are marked as prior
inconsistent. Points outside prior coverage are preserved.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "evaluation_tools") not in sys.path:
    sys.path.append(str(REPO_ROOT / "evaluation_tools"))

from map_ghost_eval import load_annotations, points_in_roi, read_point_cloud  # noqa: E402


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


def finite_stats(values: np.ndarray) -> Dict[str, Optional[float]]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def batched_indices(indices: np.ndarray, batch_size: int) -> Iterable[np.ndarray]:
    for start in range(0, indices.size, batch_size):
        yield indices[start : start + batch_size]


class MeshDistance:
    def __init__(self, mesh_path: Path):
        self.mesh_path = mesh_path
        self.backend = "vertices_kdtree"
        self.scene = None
        self.kdtree: Optional[cKDTree] = None
        self._load(mesh_path)

    def _load(self, mesh_path: Path) -> None:
        try:
            import open3d as o3d

            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            if len(mesh.vertices) == 0:
                raise ValueError(f"{mesh_path}: no vertices")
            if len(mesh.triangles) > 0:
                legacy_mesh = mesh
                tmesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy_mesh)
                scene = o3d.t.geometry.RaycastingScene()
                scene.add_triangles(tmesh)
                self.scene = scene
                self.backend = "open3d_raycasting_scene"
                return
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
        except Exception:
            vertices = read_point_cloud(mesh_path)

        if vertices.shape[0] == 0:
            raise ValueError(f"{mesh_path}: empty mesh/point cloud")
        self.kdtree = cKDTree(vertices[:, :3])

    def compute(self, points: np.ndarray, batch_size: int) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        distances = np.full(points.shape[0], np.inf, dtype=np.float32)
        all_idx = np.arange(points.shape[0], dtype=np.int64)

        if self.scene is not None:
            import open3d as o3d

            for idx in batched_indices(all_idx, batch_size):
                tensor = o3d.core.Tensor(points[idx].astype(np.float32), dtype=o3d.core.Dtype.Float32)
                distances[idx] = self.scene.compute_distance(tensor).numpy().astype(np.float32)
            return distances

        if self.kdtree is None:
            raise RuntimeError("mesh distance backend was not initialized")
        for idx in batched_indices(all_idx, batch_size):
            distances[idx] = self.kdtree.query(points[idx], k=1, workers=-1)[0].astype(np.float32)
        return distances


def load_tsdf_prior(path: Path, valid_abs: float) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    tsdf = np.asarray(data["tsdf"], dtype=np.float32)
    origin = np.asarray(data["origin"], dtype=np.float64)
    voxel_size = float(data["voxel_size"])
    shape = np.asarray(tsdf.shape, dtype=np.int64)
    bounds_min = origin
    bounds_max = origin + shape.astype(np.float64) * voxel_size
    valid_ijk = np.argwhere(np.abs(tsdf) < valid_abs)
    valid_centers = origin + (valid_ijk.astype(np.float64) + 0.5) * voxel_size
    return tsdf, voxel_size, bounds_min, bounds_max, valid_centers


def annotation_mask(points: np.ndarray, annotation: Optional[Path], thresholds: Sequence[float]) -> np.ndarray:
    if annotation is None:
        return np.ones(points.shape[0], dtype=bool)
    _, reflective, _ = load_annotations(annotation, thresholds)
    if not reflective:
        return np.zeros(points.shape[0], dtype=bool)
    mask = np.zeros(points.shape[0], dtype=bool)
    for plane in reflective:
        mask |= points_in_roi(points, plane.roi)
    return mask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument("--neuralrecon-npz", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--annotation", type=Path, default=None, help="optional reflective ROI YAML")
    parser.add_argument("--restrict-to-annotation", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reject-distance", type=float, default=0.35)
    parser.add_argument("--soft-distance", type=float, default=0.15)
    parser.add_argument("--aabb-padding", type=float, default=0.05)
    parser.add_argument(
        "--coverage-mode",
        choices=("aabb", "tsdf", "aabb_and_tsdf"),
        default="aabb",
        help="which prior coverage test permits filtering",
    )
    parser.add_argument("--tsdf-valid-abs", type=float, default=0.999)
    parser.add_argument("--tsdf-coverage-radius", type=float, default=0.45)
    parser.add_argument("--batch-size", type=int, default=200000)
    parser.add_argument("--thresholds-m", default="0.05,0.10,0.20")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    points = read_point_cloud(args.map)
    tsdf, voxel_size, bounds_min, bounds_max, valid_centers = load_tsdf_prior(
        args.neuralrecon_npz, args.tsdf_valid_abs
    )
    padded_min = bounds_min - args.aabb_padding
    padded_max = bounds_max + args.aabb_padding
    inside_aabb = np.logical_and(points >= padded_min, points <= padded_max).all(axis=1)

    fallback_thresholds = [float(x) for x in args.thresholds_m.split(",") if x.strip()]
    ann_mask = annotation_mask(points, args.annotation, fallback_thresholds)
    if not args.restrict_to_annotation:
        ann_mask[:] = True

    tsdf_covered = np.zeros(points.shape[0], dtype=bool)
    if args.coverage_mode in {"tsdf", "aabb_and_tsdf"}:
        if valid_centers.shape[0] == 0:
            raise ValueError(f"{args.neuralrecon_npz}: no valid TSDF voxels with abs < {args.tsdf_valid_abs}")
        tree = cKDTree(valid_centers)
        candidate_idx = np.flatnonzero(inside_aabb)
        nearest = np.full(candidate_idx.shape[0], np.inf, dtype=np.float32)
        for out_start in range(0, candidate_idx.size, args.batch_size):
            idx = candidate_idx[out_start : out_start + args.batch_size]
            nearest[out_start : out_start + idx.size] = tree.query(points[idx], k=1, workers=-1)[0].astype(np.float32)
        tsdf_covered[candidate_idx] = nearest <= args.tsdf_coverage_radius
    else:
        tsdf_covered[:] = True

    if args.coverage_mode == "aabb":
        prior_covered = inside_aabb
    elif args.coverage_mode == "tsdf":
        prior_covered = tsdf_covered
    else:
        prior_covered = inside_aabb & tsdf_covered
    prior_candidate = prior_covered & ann_mask

    distance_backend = MeshDistance(args.mesh)
    distances = np.full(points.shape[0], np.inf, dtype=np.float32)
    candidate_idx = np.flatnonzero(prior_candidate)
    if candidate_idx.size:
        distances[candidate_idx] = distance_backend.compute(points[candidate_idx], args.batch_size)

    denom = max(args.reject_distance - args.soft_distance, 1e-6)
    ghost_score = np.zeros(points.shape[0], dtype=np.float32)
    ghost_score[candidate_idx] = np.clip((distances[candidate_idx] - args.soft_distance) / denom, 0.0, 1.0)
    reject_mask = prior_candidate & (distances > args.reject_distance)
    keep_mask = ~reject_mask

    filtered = points[keep_mask]
    rejected = points[reject_mask]
    candidates = points[prior_candidate]

    write_pcd_ascii(args.out_dir / "filtered_map.pcd", filtered)
    write_pcd_ascii(args.out_dir / "rejected_points.pcd", rejected)
    write_pcd_ascii(args.out_dir / "prior_candidate_points.pcd", candidates)

    np.savez_compressed(
        args.out_dir / "point_scores.npz",
        points=points.astype(np.float32),
        distance_to_prior=distances,
        ghost_score=ghost_score,
        keep_mask=keep_mask,
        reject_mask=reject_mask,
        prior_candidate_mask=prior_candidate,
        inside_aabb_mask=inside_aabb,
        tsdf_covered_mask=tsdf_covered,
    )

    result: Dict[str, Any] = {
        "map": str(args.map),
        "neuralrecon_npz": str(args.neuralrecon_npz),
        "mesh": str(args.mesh),
        "mesh_distance_backend": distance_backend.backend,
        "output_dir": str(args.out_dir),
        "params": {
            "reject_distance": args.reject_distance,
            "soft_distance": args.soft_distance,
            "aabb_padding": args.aabb_padding,
            "coverage_mode": args.coverage_mode,
            "tsdf_valid_abs": args.tsdf_valid_abs,
            "tsdf_coverage_radius": args.tsdf_coverage_radius,
            "restrict_to_annotation": args.restrict_to_annotation,
            "annotation": str(args.annotation) if args.annotation else None,
        },
        "prior": {
            "voxel_size": voxel_size,
            "tsdf_shape": [int(x) for x in tsdf.shape],
            "bounds_min": bounds_min.tolist(),
            "bounds_max": bounds_max.tolist(),
            "valid_tsdf_voxel_count": int(valid_centers.shape[0]),
        },
        "counts": {
            "input_points": int(points.shape[0]),
            "inside_aabb_points": int(np.count_nonzero(inside_aabb)),
            "tsdf_covered_points": int(np.count_nonzero(tsdf_covered)),
            "prior_candidate_points": int(np.count_nonzero(prior_candidate)),
            "rejected_points": int(np.count_nonzero(reject_mask)),
            "kept_points": int(np.count_nonzero(keep_mask)),
        },
        "ratios": {
            "inside_aabb_ratio": float(np.count_nonzero(inside_aabb) / points.shape[0]) if points.shape[0] else 0.0,
            "candidate_ratio": float(np.count_nonzero(prior_candidate) / points.shape[0]) if points.shape[0] else 0.0,
            "reject_ratio_all_points": float(np.count_nonzero(reject_mask) / points.shape[0]) if points.shape[0] else 0.0,
            "reject_ratio_candidates": (
                float(np.count_nonzero(reject_mask) / np.count_nonzero(prior_candidate))
                if np.count_nonzero(prior_candidate)
                else 0.0
            ),
        },
        "distance_to_prior_m": finite_stats(distances[candidate_idx]),
        "outputs": {
            "filtered_map": str(args.out_dir / "filtered_map.pcd"),
            "rejected_points": str(args.out_dir / "rejected_points.pcd"),
            "prior_candidate_points": str(args.out_dir / "prior_candidate_points.pcd"),
            "point_scores": str(args.out_dir / "point_scores.npz"),
            "summary": str(args.out_dir / "summary.json"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "NeuralRecon prior filter: "
        f"input={points.shape[0]} candidate={np.count_nonzero(prior_candidate)} "
        f"rejected={np.count_nonzero(reject_mask)} kept={np.count_nonzero(keep_mask)}"
    )
    print(f"distance stats: {json.dumps(result['distance_to_prior_m'], sort_keys=True)}")
    print(f"wrote: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

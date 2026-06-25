#!/usr/bin/env python3
"""Automatically detect reflective room planes from a SLAM map.

This first version targets elevator / small-room scenes where the dominant
reflective structure is a vertical cuboid.  It detects vertical wall lines in
the XY projection, fits a rotated rectangle, and writes the same room_bounds
annotation schema used by map_ghost_eval.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

try:
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
except ImportError:  # pragma: no cover - optional for map-only detection
    AnyReader = None
    Stores = None
    get_typestore = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "evaluation_tools") not in sys.path:
    sys.path.append(str(REPO_ROOT / "evaluation_tools"))

from annotation_helper import write_room_bounds_yaml, write_pcd_ascii  # noqa: E402
from map_ghost_eval import read_point_cloud  # noqa: E402


@dataclass
class LineCandidate:
    normal: List[float]
    d: float
    tangent: List[float]
    support_count: int
    support_length_m: float
    support_z_span_m: float
    residual_median_m: float
    residual_p90_m: float
    score: float


@dataclass
class AxisPair:
    low_index: int
    high_index: int
    low_position: float
    high_position: float
    separation_m: float
    center_penalty_m: float
    score: float


@dataclass
class CuboidFit:
    corners_xy: List[List[float]]
    center_xy: List[float]
    axis_u: List[float]
    axis_v: List[float]
    side_lengths_m: List[float]
    score: float
    u_pair: AxisPair
    v_pair: AxisPair


@dataclass
class CuboidSearchResult:
    fit: CuboidFit
    lines: List[LineCandidate]
    crop_points: np.ndarray
    crop_info: Dict[str, Any]
    center_xy: np.ndarray
    search_score: float
    odometry_inside_ratio: Optional[float] = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, type=Path, help="input map_raw.pcd")
    parser.add_argument("--bag", type=Path, default=None, help="optional output bag with /Odometry for local crop center")
    parser.add_argument("--odom-topic", default="/Odometry")
    parser.add_argument("--out-yaml", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--bag-name", default=None)
    parser.add_argument("--map-frame", default="camera_init")
    parser.add_argument("--mode", choices=["cuboid"], default="cuboid")
    parser.add_argument("--z-min", type=float, default=-0.3)
    parser.add_argument("--z-max", type=float, default=2.2)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    parser.add_argument("--interior-margin", type=float, default=0.25)
    parser.add_argument("--exterior-margin", type=float, default=1.20)
    parser.add_argument("--crop-min", type=float, nargs=3, default=None)
    parser.add_argument("--crop-max", type=float, nargs=3, default=None)
    parser.add_argument("--auto-crop-radius", type=float, default=3.0)
    parser.add_argument(
        "--search-crop-centers",
        choices=["single", "trajectory"],
        default="trajectory",
        help="single uses one crop center; trajectory scans odometry centers and chooses the best cuboid",
    )
    parser.add_argument("--trajectory-center-stride", type=int, default=10)
    parser.add_argument("--max-trajectory-centers", type=int, default=24)
    parser.add_argument("--trajectory-tail-fraction", type=float, default=0.25)
    parser.add_argument("--line-threshold", type=float, default=0.065)
    parser.add_argument("--line-remove-threshold", type=float, default=0.13)
    parser.add_argument("--ransac-iterations", type=int, default=1200)
    parser.add_argument("--max-lines", type=int, default=18)
    parser.add_argument("--min-line-support", type=int, default=180)
    parser.add_argument("--min-line-length", type=float, default=0.55)
    parser.add_argument("--min-line-z-span", type=float, default=0.50)
    parser.add_argument("--nms-angle-deg", type=float, default=7.5)
    parser.add_argument("--nms-distance", type=float, default=0.13)
    parser.add_argument("--orthogonal-tolerance-deg", type=float, default=18.0)
    parser.add_argument("--parallel-tolerance-deg", type=float, default=14.0)
    parser.add_argument("--max-axis-pairs", type=int, default=12)
    parser.add_argument("--min-side-length", type=float, default=0.75)
    parser.add_argument(
        "--max-side-length",
        type=float,
        default=0.0,
        help="optional absolute upper bound for cuboid side length; <=0 disables this scale prior",
    )
    parser.add_argument("--max-center-outside", type=float, default=0.05)
    parser.add_argument("--odometry-inside-weight", type=float, default=220000.0)
    parser.add_argument("--odometry-inside-saturation", type=float, default=0.85)
    parser.add_argument(
        "--compactness-weight",
        type=float,
        default=45000.0,
        help="scale-free penalty for unnecessarily large cuboids after odometry coverage is satisfied",
    )
    parser.add_argument("--sample-points", type=int, default=180000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--write-crop-pcd", action="store_true")
    return parser


def sample_points(points: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or points.shape[0] <= limit:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=limit, replace=False)
    return points[idx]


def point_line_dist(xy: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    return np.abs(xy @ normal + d)


def line_angle(normal: np.ndarray) -> float:
    angle = math.atan2(float(normal[1]), float(normal[0]))
    angle = angle % math.pi
    return angle


def angle_diff_mod_pi(a: float, b: float) -> float:
    diff = abs((a - b + math.pi / 2.0) % math.pi - math.pi / 2.0)
    return diff


def perpendicular_angle_error(a: float, b: float) -> float:
    return abs(angle_diff_mod_pi(a, b) - math.pi / 2.0)


def canonical_line(normal: np.ndarray, d: float) -> Tuple[np.ndarray, float]:
    normal = np.asarray(normal, dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        raise ValueError("line normal must be non-zero")
    normal = normal / norm
    d = float(d) / norm
    # Deterministic sign for NMS/debug output.
    if normal[0] < 0.0 or (abs(normal[0]) < 1e-12 and normal[1] < 0.0):
        normal = -normal
        d = -d
    return normal, d


def fit_line_pca(xy: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    centroid = np.mean(xy, axis=0)
    centered = xy - centroid[None, :]
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    tangent = vh[0]
    tangent = tangent / np.linalg.norm(tangent)
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    normal, d = canonical_line(normal, -float(normal @ centroid))
    tangent = np.asarray([-normal[1], normal[0]], dtype=np.float64)
    return normal, d, tangent


def line_metrics(points: np.ndarray, normal: np.ndarray, d: float, threshold: float) -> Optional[LineCandidate]:
    xy = points[:, :2]
    dist = point_line_dist(xy, normal, d)
    mask = dist <= threshold
    count = int(np.count_nonzero(mask))
    if count < 2:
        return None
    inliers = points[mask]
    normal_refined, d_refined, tangent = fit_line_pca(inliers[:, :2])
    dist = point_line_dist(xy, normal_refined, d_refined)
    mask = dist <= threshold
    inliers = points[mask]
    count = int(inliers.shape[0])
    if count < 2:
        return None
    coord = inliers[:, :2] @ tangent
    lo, hi = np.percentile(coord, [2.0, 98.0])
    z_lo, z_hi = np.percentile(inliers[:, 2], [5.0, 95.0])
    residuals = dist[mask]
    length = float(max(0.0, hi - lo))
    z_span = float(max(0.0, z_hi - z_lo))
    median = float(np.median(residuals))
    p90 = float(np.percentile(residuals, 90))
    score = float(count * math.sqrt(max(length, 1e-6)) * math.sqrt(max(z_span, 1e-6)) / (median + 0.01))
    return LineCandidate(
        normal=normal_refined.tolist(),
        d=float(d_refined),
        tangent=tangent.tolist(),
        support_count=count,
        support_length_m=length,
        support_z_span_m=z_span,
        residual_median_m=median,
        residual_p90_m=p90,
        score=score,
    )


def ransac_line(points: np.ndarray, *, threshold: float, iterations: int, seed: int) -> Optional[LineCandidate]:
    if points.shape[0] < 2:
        return None
    xy = points[:, :2]
    rng = np.random.default_rng(seed)
    best: Optional[Tuple[np.ndarray, float, int, float]] = None
    for _ in range(max(1, iterations)):
        ids = rng.choice(points.shape[0], size=2, replace=False)
        p0, p1 = xy[ids[0]], xy[ids[1]]
        tangent = p1 - p0
        length = float(np.linalg.norm(tangent))
        if length < 1e-5:
            continue
        tangent /= length
        normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
        normal, d = canonical_line(normal, -float(normal @ p0))
        dist = point_line_dist(xy, normal, d)
        mask = dist <= threshold
        count = int(np.count_nonzero(mask))
        if count <= 0:
            continue
        median = float(np.median(dist[mask]))
        if best is None or count > best[2] or (count == best[2] and median < best[3]):
            best = (normal, d, count, median)
    if best is None:
        return None
    return line_metrics(points, best[0], best[1], threshold)


def remove_line_support(points: np.ndarray, candidate: LineCandidate, threshold: float) -> np.ndarray:
    normal = np.asarray(candidate.normal, dtype=np.float64)
    dist = point_line_dist(points[:, :2], normal, candidate.d)
    return dist > threshold


def detect_lines(points: np.ndarray, args: argparse.Namespace) -> List[LineCandidate]:
    remaining = points.copy()
    candidates: List[LineCandidate] = []
    for line_idx in range(args.max_lines):
        if remaining.shape[0] < args.min_line_support:
            break
        candidate = ransac_line(
            remaining,
            threshold=args.line_threshold,
            iterations=args.ransac_iterations,
            seed=args.seed + line_idx * 9973,
        )
        if candidate is None:
            break
        if (
            candidate.support_count >= args.min_line_support
            and candidate.support_length_m >= args.min_line_length
            and candidate.support_z_span_m >= args.min_line_z_span
        ):
            candidates.append(candidate)
        keep = remove_line_support(remaining, candidate, args.line_remove_threshold)
        if int(np.count_nonzero(keep)) == remaining.shape[0]:
            break
        remaining = remaining[keep]
    return nms_lines(candidates, angle_tol=math.radians(args.nms_angle_deg), distance_tol=args.nms_distance)


def nms_lines(
    candidates: Sequence[LineCandidate],
    *,
    angle_tol: float,
    distance_tol: float,
) -> List[LineCandidate]:
    kept: List[LineCandidate] = []
    for cand in sorted(candidates, key=lambda x: x.score, reverse=True):
        n = np.asarray(cand.normal, dtype=np.float64)
        duplicate = False
        for prev in kept:
            p = np.asarray(prev.normal, dtype=np.float64)
            if angle_diff_mod_pi(line_angle(n), line_angle(p)) > angle_tol:
                continue
            if float(n @ p) < 0.0:
                d = -cand.d
            else:
                d = cand.d
            if abs(d - prev.d) <= distance_tol:
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
    return kept


def orient_line_to_axis(candidate: LineCandidate, axis: np.ndarray) -> Tuple[np.ndarray, float, float]:
    normal = np.asarray(candidate.normal, dtype=np.float64)
    d = float(candidate.d)
    if float(normal @ axis) < 0.0:
        normal = -normal
        d = -d
    position = -d
    return normal, d, position


def select_axis_pairs(
    candidates: Sequence[LineCandidate],
    axis: np.ndarray,
    center_xy: np.ndarray,
    *,
    angle_tol: float,
    min_side: float,
    max_side: float,
    max_center_outside: float,
    max_pairs: int,
) -> List[AxisPair]:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    center_s = float(center_xy @ axis)
    aligned: List[Tuple[int, float, float]] = []
    axis_angle = line_angle(axis)
    for idx, cand in enumerate(candidates):
        n = np.asarray(cand.normal, dtype=np.float64)
        if angle_diff_mod_pi(line_angle(n), axis_angle) > angle_tol:
            continue
        _, _, pos = orient_line_to_axis(cand, axis)
        aligned.append((idx, pos, cand.score))
    if len(aligned) < 2:
        return []
    pairs: List[AxisPair] = []
    for i in range(len(aligned)):
        for j in range(i + 1, len(aligned)):
            idx_a, pos_a, score_a = aligned[i]
            idx_b, pos_b, score_b = aligned[j]
            lo, hi = sorted([pos_a, pos_b])
            sep = hi - lo
            if sep < min_side:
                continue
            if max_side > 0.0 and sep > max_side:
                continue
            if center_s < lo:
                center_penalty = lo - center_s
            elif center_s > hi:
                center_penalty = center_s - hi
            else:
                center_penalty = 0.0
            if center_penalty > max_center_outside:
                continue
            score = float(score_a + score_b - 250.0 * center_penalty)
            pair = AxisPair(
                low_index=idx_a if pos_a <= pos_b else idx_b,
                high_index=idx_b if pos_a <= pos_b else idx_a,
                low_position=float(lo),
                high_position=float(hi),
                separation_m=float(sep),
                center_penalty_m=float(center_penalty),
                score=score,
            )
            pairs.append(pair)
    pairs.sort(key=lambda x: x.score, reverse=True)
    return pairs[: max(1, int(max_pairs))]


def order_corners_ccw(corners_xy: np.ndarray) -> np.ndarray:
    center = corners_xy.mean(axis=0)
    angles = np.arctan2(corners_xy[:, 1] - center[1], corners_xy[:, 0] - center[0])
    ordered = corners_xy[np.argsort(angles)]
    area = 0.5 * float(
        np.dot(ordered[:, 0], np.roll(ordered[:, 1], -1))
        - np.dot(ordered[:, 1], np.roll(ordered[:, 0], -1))
    )
    if area < 0.0:
        ordered = ordered[::-1]
    return ordered


def points_in_convex_polygon(points_xy: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    points_xy = np.asarray(points_xy, dtype=np.float64)
    polygon_xy = order_corners_ccw(np.asarray(polygon_xy, dtype=np.float64))
    inside = np.ones(points_xy.shape[0], dtype=bool)
    for i in range(polygon_xy.shape[0]):
        p0 = polygon_xy[i]
        p1 = polygon_xy[(i + 1) % polygon_xy.shape[0]]
        edge = p1 - p0
        rel = points_xy - p0[None, :]
        cross = edge[0] * rel[:, 1] - edge[1] * rel[:, 0]
        inside &= cross >= -1e-9
    return inside


def fit_cuboid(candidates: Sequence[LineCandidate], center_xy: np.ndarray, args: argparse.Namespace) -> CuboidFit:
    if len(candidates) < 4:
        raise ValueError(f"need at least four line candidates for cuboid fitting, got {len(candidates)}")
    parallel_tol = math.radians(args.parallel_tolerance_deg)
    orth_tol = math.radians(args.orthogonal_tolerance_deg)
    best: Optional[CuboidFit] = None
    for i, cand_u in enumerate(candidates):
        u = np.asarray(cand_u.normal, dtype=np.float64)
        u = u / np.linalg.norm(u)
        angle_u = line_angle(u)
        for j, cand_v in enumerate(candidates):
            if i == j:
                continue
            v = np.asarray(cand_v.normal, dtype=np.float64)
            v = v / np.linalg.norm(v)
            if perpendicular_angle_error(angle_u, line_angle(v)) > orth_tol:
                continue
            # Re-orthogonalize v for clean rectangle corners.
            v = np.asarray([-u[1], u[0]], dtype=np.float64)
            if float(v @ np.asarray(cand_v.normal, dtype=np.float64)) < 0.0:
                v = -v
            u_pairs = select_axis_pairs(
                candidates,
                u,
                center_xy,
                angle_tol=parallel_tol,
                min_side=args.min_side_length,
                max_side=args.max_side_length,
                max_center_outside=args.max_center_outside,
                max_pairs=args.max_axis_pairs,
            )
            v_pairs = select_axis_pairs(
                candidates,
                v,
                center_xy,
                angle_tol=parallel_tol,
                min_side=args.min_side_length,
                max_side=args.max_side_length,
                max_center_outside=args.max_center_outside,
                max_pairs=args.max_axis_pairs,
            )
            if not u_pairs or not v_pairs:
                continue
            for u_pair in u_pairs:
                for v_pair in v_pairs:
                    corners = np.asarray(
                        [
                            u * u_pair.low_position + v * v_pair.low_position,
                            u * u_pair.high_position + v * v_pair.low_position,
                            u * u_pair.high_position + v * v_pair.high_position,
                            u * u_pair.low_position + v * v_pair.high_position,
                        ],
                        dtype=np.float64,
                    )
                    corners = order_corners_ccw(corners)
                    score = float(
                        u_pair.score
                        + v_pair.score
                        - 1000.0 * perpendicular_angle_error(angle_u, line_angle(cand_v.normal))
                    )
                    fit = CuboidFit(
                        corners_xy=corners.tolist(),
                        center_xy=(0.5 * (corners[0] + corners[2])).tolist(),
                        axis_u=u.tolist(),
                        axis_v=v.tolist(),
                        side_lengths_m=[u_pair.separation_m, v_pair.separation_m],
                        score=score,
                        u_pair=u_pair,
                        v_pair=v_pair,
                    )
                    if best is None or fit.score > best.score:
                        best = fit
    if best is None:
        raise ValueError("failed to fit a cuboid from detected line candidates")
    return best


def crop_points_around_center(
    points: np.ndarray,
    center_xy: np.ndarray,
    *,
    radius: float,
    z_min: float,
    z_max: float,
    mode: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    z_lo, z_hi = sorted([float(z_min), float(z_max)])
    center_xy = np.asarray(center_xy, dtype=np.float64)
    mask = (
        (points[:, 2] >= z_lo)
        & (points[:, 2] <= z_hi)
        & (points[:, 0] >= center_xy[0] - radius)
        & (points[:, 0] <= center_xy[0] + radius)
        & (points[:, 1] >= center_xy[1] - radius)
        & (points[:, 1] <= center_xy[1] + radius)
    )
    info: Dict[str, Any] = {
        "mode": mode,
        "center_xy": center_xy.tolist(),
        "radius_m": float(radius),
        "z": [z_lo, z_hi],
    }
    if extra:
        info.update(extra)
    return points[mask], info


def trajectory_crop_centers(odom_xy: np.ndarray, args: argparse.Namespace) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
    if odom_xy.size == 0:
        return []
    stride = max(1, int(args.trajectory_center_stride))
    ids = list(range(0, odom_xy.shape[0], stride))
    if ids[-1] != odom_xy.shape[0] - 1:
        ids.append(odom_xy.shape[0] - 1)
    if len(ids) > args.max_trajectory_centers:
        keep = np.linspace(0, len(ids) - 1, int(args.max_trajectory_centers), dtype=np.int64)
        ids = [ids[int(i)] for i in keep]
    centers: List[Tuple[np.ndarray, Dict[str, Any]]] = []
    for idx in ids:
        centers.append((odom_xy[idx], {"odom_index": int(idx), "odom_count": int(odom_xy.shape[0])}))
    # The median of the full trajectory often lands inside a revisited small room.
    centers.append((np.median(odom_xy, axis=0), {"odom_index": "median", "odom_count": int(odom_xy.shape[0])}))
    return centers


def cuboid_search_score(fit: CuboidFit, args: argparse.Namespace, odom_xy: Optional[np.ndarray]) -> Tuple[float, Optional[float]]:
    area = float(fit.side_lengths_m[0] * fit.side_lengths_m[1])
    odom_inside_ratio: Optional[float] = None
    odom_bonus = 0.0
    if odom_xy is not None and odom_xy.size:
        corners = np.asarray(fit.corners_xy, dtype=np.float64)
        # Use a small buffer around the fitted room because odometry can skim
        # the threshold when the robot is close to a reflective wall.
        center = corners.mean(axis=0)
        buffered = center[None, :] + 1.08 * (corners - center[None, :])
        inside = points_in_convex_polygon(odom_xy[:, :2], buffered)
        odom_inside_ratio = float(np.count_nonzero(inside) / odom_xy.shape[0]) if odom_xy.shape[0] else 0.0
        saturation = max(float(args.odometry_inside_saturation), 1e-6)
        odom_bonus = float(args.odometry_inside_weight) * min(odom_inside_ratio, saturation) / saturation
    compactness_penalty = float(args.compactness_weight) * math.log1p(max(area, 0.0))
    return float(fit.score + odom_bonus - compactness_penalty), odom_inside_ratio


def search_best_cuboid(points: np.ndarray, args: argparse.Namespace) -> CuboidSearchResult:
    base_crop, base_info, base_center = crop_points(points, args)
    centers: List[Tuple[np.ndarray, Dict[str, Any]]] = [(base_center, {"source": "base"})]
    all_odom_xy = read_odometry_xy(args.bag, args.odom_topic) if args.bag else None
    if args.search_crop_centers == "trajectory" and all_odom_xy is not None and all_odom_xy.size:
        centers = trajectory_crop_centers(all_odom_xy, args)

    best: Optional[CuboidSearchResult] = None
    failures: List[Dict[str, Any]] = []
    for center_idx, (center_xy, meta) in enumerate(centers):
        if args.crop_min is not None and args.crop_max is not None:
            crop = base_crop
            crop_info = dict(base_info)
            center = base_center
        else:
            crop, crop_info = crop_points_around_center(
                points,
                center_xy,
                radius=float(args.auto_crop_radius),
                z_min=float(args.z_min),
                z_max=float(args.z_max),
                mode="trajectory_center_radius" if args.search_crop_centers == "trajectory" else "single_center_radius",
                extra={"center_search_index": int(center_idx), **meta},
            )
            center = np.asarray(center_xy, dtype=np.float64)
        local_odom_xy = None
        if all_odom_xy is not None and all_odom_xy.size:
            r = float(args.auto_crop_radius)
            odom_mask = (
                (all_odom_xy[:, 0] >= center[0] - r)
                & (all_odom_xy[:, 0] <= center[0] + r)
                & (all_odom_xy[:, 1] >= center[1] - r)
                & (all_odom_xy[:, 1] <= center[1] + r)
            )
            local_odom_xy = all_odom_xy[odom_mask]
        if crop.shape[0] < args.min_line_support * 2:
            failures.append({"center": center.tolist(), "cropped_points": int(crop.shape[0]), "reason": "too_few_points"})
            continue
        try:
            ransac_points = sample_points(crop, args.sample_points, args.seed + center_idx * 31)
            lines = detect_lines(ransac_points, args)
            fit = fit_cuboid(lines, center, args)
            score, odom_inside_ratio = cuboid_search_score(fit, args, local_odom_xy)
        except Exception as exc:
            failures.append({"center": center.tolist(), "cropped_points": int(crop.shape[0]), "reason": str(exc)})
            continue
        result = CuboidSearchResult(
            fit=fit,
            lines=lines,
            crop_points=crop,
            crop_info={
                **crop_info,
                "local_odom_count": int(local_odom_xy.shape[0]) if local_odom_xy is not None else 0,
                "search_failures": failures[-12:],
            },
            center_xy=center,
            search_score=score,
            odometry_inside_ratio=odom_inside_ratio,
        )
        if best is None or result.search_score > best.search_score:
            best = result
    if best is None:
        raise ValueError(f"failed to detect cuboid from {len(centers)} crop centers; failures={failures[:5]}")
    best.crop_info["evaluated_centers"] = len(centers)
    return best


def read_odometry_xy(bag: Path, odom_topic: str) -> Optional[np.ndarray]:
    if AnyReader is None or get_typestore is None or Stores is None:
        return None
    if not bag.exists():
        return None
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    poses: List[Tuple[float, float]] = []
    with AnyReader([bag], default_typestore=typestore) as reader:
        connections = [conn for conn in reader.connections if conn.topic == odom_topic]
        if not connections:
            return None
        for conn, _, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, conn.msgtype)
            p = msg.pose.pose.position
            poses.append((float(p.x), float(p.y)))
    return np.asarray(poses, dtype=np.float64) if poses else None


def crop_points(points: np.ndarray, args: argparse.Namespace) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
    z_lo, z_hi = sorted([float(args.z_min), float(args.z_max)])
    z_mask = (points[:, 2] >= z_lo) & (points[:, 2] <= z_hi)
    center_xy: Optional[np.ndarray] = None
    crop_info: Dict[str, Any] = {"z": [z_lo, z_hi]}

    if args.crop_min is not None and args.crop_max is not None:
        lo = np.minimum(np.asarray(args.crop_min, dtype=np.float64), np.asarray(args.crop_max, dtype=np.float64))
        hi = np.maximum(np.asarray(args.crop_min, dtype=np.float64), np.asarray(args.crop_max, dtype=np.float64))
        mask = z_mask & np.logical_and(points >= lo, points <= hi).all(axis=1)
        center_xy = 0.5 * (lo[:2] + hi[:2])
        crop_info.update({"mode": "manual_box", "min": lo.tolist(), "max": hi.tolist()})
        return points[mask], crop_info, center_xy

    odom_xy = read_odometry_xy(args.bag, args.odom_topic) if args.bag else None
    if odom_xy is not None and odom_xy.size:
        tail_fraction = min(max(float(args.trajectory_tail_fraction), 0.01), 1.0)
        start = max(0, int(round((1.0 - tail_fraction) * odom_xy.shape[0])))
        center_xy = np.median(odom_xy[start:], axis=0)
        radius = float(args.auto_crop_radius)
        mask = (
            z_mask
            & (points[:, 0] >= center_xy[0] - radius)
            & (points[:, 0] <= center_xy[0] + radius)
            & (points[:, 1] >= center_xy[1] - radius)
            & (points[:, 1] <= center_xy[1] + radius)
        )
        crop_info.update(
            {
                "mode": "trajectory_tail_radius",
                "center_xy": center_xy.tolist(),
                "radius_m": radius,
                "odom_count": int(odom_xy.shape[0]),
                "tail_fraction": tail_fraction,
            }
        )
        return points[mask], crop_info, center_xy

    mask = z_mask
    filtered = points[mask]
    if filtered.size:
        center_xy = np.median(filtered[:, :2], axis=0)
    else:
        center_xy = np.zeros(2, dtype=np.float64)
    crop_info.update({"mode": "z_only", "center_xy": center_xy.tolist()})
    return filtered, crop_info, center_xy


def draw_line(ax: Any, normal: np.ndarray, d: float, xy_extent: np.ndarray, **kwargs: Any) -> None:
    normal = np.asarray(normal, dtype=np.float64)
    tangent = np.asarray([-normal[1], normal[0]], dtype=np.float64)
    center = -d * normal
    coords = xy_extent @ tangent
    lo, hi = float(np.min(coords)) - 0.5, float(np.max(coords)) + 0.5
    pts = np.vstack([center + tangent * lo, center + tangent * hi])
    ax.plot(pts[:, 0], pts[:, 1], **kwargs)


def render_preview(
    out_path: Path,
    points: np.ndarray,
    crop_points_: np.ndarray,
    candidates: Sequence[LineCandidate],
    fit: CuboidFit,
    center_xy: np.ndarray,
    *,
    sample_limit: int,
    seed: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 7.2), constrained_layout=True)
    sampled_all = sample_points(points, sample_limit, seed)
    sampled_crop = sample_points(crop_points_, sample_limit, seed + 1)
    ax.scatter(sampled_all[:, 0], sampled_all[:, 1], s=0.12, c="#9ca3af", alpha=0.18, linewidths=0, label="map")
    ax.scatter(sampled_crop[:, 0], sampled_crop[:, 1], s=0.2, c="#1f2937", alpha=0.35, linewidths=0, label="crop")
    if sampled_crop.size:
        extent = sampled_crop[:, :2]
    else:
        extent = sampled_all[:, :2]
    for idx, cand in enumerate(candidates):
        color = "#2563eb" if idx in {fit.u_pair.low_index, fit.u_pair.high_index, fit.v_pair.low_index, fit.v_pair.high_index} else "#93c5fd"
        alpha = 0.85 if color == "#2563eb" else 0.30
        draw_line(ax, np.asarray(cand.normal), cand.d, extent, color=color, linewidth=1.0, alpha=alpha)
    corners = np.asarray(fit.corners_xy, dtype=np.float64)
    closed = np.vstack([corners, corners[0]])
    ax.plot(closed[:, 0], closed[:, 1], color="#dc2626", linewidth=2.2, label="auto cuboid")
    ax.scatter([center_xy[0]], [center_xy[1]], s=35, c="#f59e0b", marker="x", label="crop center")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Automatic reflective room-bound detection")
    ax.grid(True, linewidth=0.25, alpha=0.30)
    ax.legend(frameon=False, loc="best")
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> int:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    points = read_point_cloud(args.map)
    search = search_best_cuboid(points, args)
    crop = search.crop_points
    crop_info = search.crop_info
    center_xy = search.center_xy
    lines = search.lines
    fit = search.fit
    detected_input_count = min(crop.shape[0], args.sample_points) if args.sample_points > 0 else crop.shape[0]

    sequence = args.sequence
    if sequence is None:
        sequence = args.map.parent.parent.name if args.map.parent.name else args.map.stem
    bag_name = args.bag_name if args.bag_name is not None else str(args.bag) if args.bag else None
    data = write_room_bounds_yaml(
        args.out_yaml,
        corners_xy=np.asarray(fit.corners_xy, dtype=np.float64),
        z_min=args.z_min,
        z_max=args.z_max,
        thresholds=args.thresholds,
        sequence=sequence,
        bag=bag_name,
        map_frame=args.map_frame,
        wall_types=["mirror", "mirror", "mirror", "mirror"],
        floor_type="floor",
        ceiling_type="ceiling",
        interior_margin=args.interior_margin,
        exterior_margin=args.exterior_margin,
    )

    debug: Dict[str, Any] = {
        "map": str(args.map),
        "bag": str(args.bag) if args.bag else None,
        "out_yaml": str(args.out_yaml),
        "crop": crop_info,
        "counts": {
            "input_points": int(points.shape[0]),
            "cropped_points": int(crop.shape[0]),
            "ransac_points": int(detected_input_count),
            "line_candidates": int(len(lines)),
        },
        "parameters": {
            "z_min": args.z_min,
            "z_max": args.z_max,
            "line_threshold": args.line_threshold,
            "line_remove_threshold": args.line_remove_threshold,
            "min_line_support": args.min_line_support,
            "min_line_length": args.min_line_length,
            "min_line_z_span": args.min_line_z_span,
            "parallel_tolerance_deg": args.parallel_tolerance_deg,
            "orthogonal_tolerance_deg": args.orthogonal_tolerance_deg,
            "min_side_length": args.min_side_length,
            "max_side_length": args.max_side_length,
            "odometry_inside_weight": args.odometry_inside_weight,
            "compactness_weight": args.compactness_weight,
        },
        "lines": [asdict(x) for x in lines],
        "cuboid": {
            "corners_xy": fit.corners_xy,
            "center_xy": fit.center_xy,
            "axis_u": fit.axis_u,
            "axis_v": fit.axis_v,
            "side_lengths_m": fit.side_lengths_m,
            "score": fit.score,
            "search_score": search.search_score,
            "odometry_inside_ratio": search.odometry_inside_ratio,
            "u_pair": asdict(fit.u_pair),
            "v_pair": asdict(fit.v_pair),
        },
        "annotation": data,
    }
    (args.out_dir / "plane_detection_debug.json").write_text(
        json.dumps(debug, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out_dir / "plane_detection_debug.yaml").write_text(
        yaml.safe_dump(debug, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    render_preview(
        args.out_dir / "plane_detection_preview.png",
        points,
        crop,
        lines,
        fit,
        center_xy,
        sample_limit=min(args.sample_points, 120000),
        seed=args.seed,
    )
    if args.write_crop_pcd:
        write_pcd_ascii(args.out_dir / "plane_detection_crop.pcd", crop)
    print(f"detected cuboid corners_xy={np.asarray(fit.corners_xy)}")
    print(f"side_lengths_m={fit.side_lengths_m} line_candidates={len(lines)} cropped_points={crop.shape[0]}")
    print(f"wrote annotation: {args.out_yaml}")
    print(f"wrote debug artifacts: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

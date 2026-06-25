#!/usr/bin/env python3
"""Filter a dense SLAM map by replaying frame-wise depth/mask priors.

For every odometry frame, world map points are transformed back into the lidar
body frame, projected to the camera, and compared with the recorded depth prior.
Points that are repeatedly behind the prior surface inside the reflective mask
are rejected.  This is an offline counterpart of the online Sentinel map gate,
but it can revisit the accumulated map with many frames instead of relying only
on sparse historical deletion markers.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "evaluation_tools") not in sys.path:
    sys.path.append(str(REPO_ROOT / "evaluation_tools"))

from map_ghost_eval import behind_mask, load_annotations, points_in_roi, read_point_cloud, signed_distances  # noqa: E402
from reconstruction.apply_neuralrecon_prior_to_map import write_pcd_ascii  # noqa: E402


@dataclass
class PriorFrame:
    stamp: float
    position: np.ndarray
    rotation_wb: np.ndarray
    depth: np.ndarray
    mask: np.ndarray


def stamp_to_sec(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = q / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def image_to_numpy(msg: Any) -> np.ndarray:
    height = int(msg.height)
    width = int(msg.width)
    encoding = str(msg.encoding).lower()
    data = bytes(msg.data)
    if encoding in {"32fc1", "32fc"}:
        arr = np.frombuffer(data, dtype="<f4").reshape(height, width)
        return np.asarray(arr, dtype=np.float32)
    if encoding in {"16uc1", "mono16"}:
        arr = np.frombuffer(data, dtype="<u2").reshape(height, width)
        return np.asarray(arr, dtype=np.float32) / 1000.0
    if encoding in {"mono8", "8uc1"}:
        arr = np.frombuffer(data, dtype=np.uint8).reshape(height, width)
        return np.asarray(arr, dtype=np.uint8)
    raise ValueError(f"unsupported image encoding: {msg.encoding}")


def load_params(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data.get("laserMapping", {}).get("ros__parameters", {}) if isinstance(data, dict) else {}


def get_param(params: Dict[str, Any], key: str, default: Any) -> Any:
    return params.get(key, default)


def read_bag_frames(
    bag: Path,
    *,
    odom_topic: str,
    depth_topic: str,
    mask_topic: str,
    use_bag_time: bool,
    depth_time_offset: float,
    mask_time_offset: float,
    max_pair_dt: float,
    frame_stride: int,
    max_frames: Optional[int],
    mask_threshold: int,
) -> List[PriorFrame]:
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    odoms: List[Tuple[float, np.ndarray, np.ndarray]] = []
    depths: List[Tuple[float, np.ndarray]] = []
    masks: List[Tuple[float, np.ndarray]] = []

    with AnyReader([bag], default_typestore=typestore) as reader:
        wanted = {odom_topic, depth_topic, mask_topic}
        connections = [conn for conn in reader.connections if conn.topic in wanted]
        if len({conn.topic for conn in connections}) != len(wanted):
            available = ", ".join(sorted({conn.topic for conn in reader.connections}))
            raise ValueError(f"missing required topics. Available topics: {available}")
        for conn, timestamp_ns, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, conn.msgtype)
            stamp = float(timestamp_ns) * 1e-9 if use_bag_time else stamp_to_sec(msg.header.stamp)
            if conn.topic == odom_topic:
                p = msg.pose.pose.position
                q = msg.pose.pose.orientation
                odoms.append(
                    (
                        stamp,
                        np.asarray([p.x, p.y, p.z], dtype=np.float64),
                        quat_to_rot(q.x, q.y, q.z, q.w),
                    )
                )
            elif conn.topic == depth_topic:
                depths.append((stamp + depth_time_offset, image_to_numpy(msg).astype(np.float32, copy=False)))
            elif conn.topic == mask_topic:
                mask = image_to_numpy(msg)
                masks.append((stamp + mask_time_offset, mask.astype(np.uint8, copy=False)))

    if not odoms or not depths or not masks:
        raise ValueError(f"bag {bag} does not contain enough odom/depth/mask messages")

    depth_times = [x[0] for x in depths]
    mask_times = [x[0] for x in masks]
    frames: List[PriorFrame] = []
    for frame_idx, (stamp, position, rotation_wb) in enumerate(odoms):
        if frame_stride > 1 and frame_idx % frame_stride != 0:
            continue
        depth_idx = nearest_index(depth_times, stamp)
        mask_idx = nearest_index(mask_times, stamp)
        depth_dt = abs(depth_times[depth_idx] - stamp)
        mask_dt = abs(mask_times[mask_idx] - stamp)
        if depth_dt > max_pair_dt or mask_dt > max_pair_dt:
            continue
        depth = depths[depth_idx][1]
        mask = masks[mask_idx][1]
        if mask.shape != depth.shape:
            raise ValueError(f"mask/depth shape mismatch: {mask.shape} vs {depth.shape}")
        frames.append(PriorFrame(stamp, position, rotation_wb, depth, mask > mask_threshold))
        if max_frames is not None and len(frames) >= max_frames:
            break
    if not frames:
        raise ValueError("no odometry frames could be paired with depth/mask priors")
    return frames


def nearest_index(values: Sequence[float], target: float) -> int:
    pos = bisect.bisect_left(values, target)
    if pos <= 0:
        return 0
    if pos >= len(values):
        return len(values) - 1
    before = values[pos - 1]
    after = values[pos]
    return pos - 1 if abs(target - before) <= abs(after - target) else pos


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


def batched_indices(size: int, batch_size: int) -> Iterable[slice]:
    for start in range(0, size, batch_size):
        yield slice(start, min(start + batch_size, size))


def filter_points(
    points: np.ndarray,
    frames: Sequence[PriorFrame],
    *,
    r_cl: np.ndarray,
    t_cl: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    margin_abs: float,
    margin_rel: float,
    min_observations: int,
    min_ghost_votes: int,
    min_ghost_ratio: float,
    require_mask: bool,
    reject_invalid_depth_in_mask: bool,
    batch_size: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    n = points.shape[0]
    observations = np.zeros(n, dtype=np.uint16)
    mask_hits = np.zeros(n, dtype=np.uint16)
    depth_checks = np.zeros(n, dtype=np.uint16)
    ghost_votes = np.zeros(n, dtype=np.uint16)
    invalid_depth_votes = np.zeros(n, dtype=np.uint16)
    max_residual = np.full(n, -np.inf, dtype=np.float32)
    residual_sum = np.zeros(n, dtype=np.float32)

    xyz = np.asarray(points[:, :3], dtype=np.float64)
    for frame in frames:
        rot_bw = frame.rotation_wb.T
        h, w = frame.depth.shape
        for sl in batched_indices(n, batch_size):
            p_w = xyz[sl]
            p_b = (p_w - frame.position[None, :]) @ rot_bw.T
            p_c = p_b @ r_cl.T + t_cl[None, :]
            z = p_c[:, 2]
            in_front = z > 0.1
            if not np.any(in_front):
                continue
            u = np.rint(fx * p_c[:, 0] / z + cx).astype(np.int32)
            v = np.rint(fy * p_c[:, 1] / z + cy).astype(np.int32)
            in_fov = in_front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            if not np.any(in_fov):
                continue
            local_idx = np.nonzero(in_fov)[0]
            global_idx = np.arange(sl.start, sl.stop, dtype=np.int64)[local_idx]
            uu = u[local_idx]
            vv = v[local_idx]
            observations[global_idx] += 1
            in_mask = frame.mask[vv, uu]
            if require_mask:
                valid_region = in_mask
            else:
                valid_region = np.ones_like(in_mask, dtype=bool)
            if not np.any(valid_region):
                continue
            region_idx = global_idx[valid_region]
            mask_hits[region_idx] += in_mask[valid_region].astype(np.uint16)
            z_prior = frame.depth[vv[valid_region], uu[valid_region]]
            z_lidar = z[local_idx][valid_region].astype(np.float32)
            valid_depth = np.isfinite(z_prior) & (z_prior > 0.1)
            if np.any(valid_depth):
                checked_idx = region_idx[valid_depth]
                residual = z_lidar[valid_depth] - z_prior[valid_depth]
                depth_checks[checked_idx] += 1
                residual_sum[checked_idx] += residual
                max_residual[checked_idx] = np.maximum(max_residual[checked_idx], residual)
                margin = margin_abs + margin_rel * np.maximum(z_lidar[valid_depth], 0.0)
                ghost_votes[checked_idx] += (residual > margin).astype(np.uint16)
            if reject_invalid_depth_in_mask and np.any(~valid_depth):
                invalid_depth_votes[region_idx[~valid_depth]] += 1

    checked_or_invalid = depth_checks + invalid_depth_votes
    denom = np.maximum(checked_or_invalid, 1)
    ghost_like_votes = ghost_votes + invalid_depth_votes
    ghost_ratio = ghost_like_votes.astype(np.float32) / denom.astype(np.float32)
    reject = (
        (observations >= min_observations)
        & (ghost_like_votes >= min_ghost_votes)
        & (ghost_ratio >= min_ghost_ratio)
    )
    if require_mask:
        reject &= mask_hits > 0
    stats = {
        "observations": observations,
        "mask_hits": mask_hits,
        "depth_checks": depth_checks,
        "ghost_votes": ghost_votes,
        "invalid_depth_votes": invalid_depth_votes,
        "ghost_ratio": ghost_ratio,
        "max_residual": max_residual,
        "mean_residual": residual_sum / np.maximum(depth_checks, 1),
    }
    return reject, stats


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument("--bag", required=True, type=Path, help="output bag containing odometry/depth/mask")
    parser.add_argument("--params", type=Path, default=None, help="Sentinel params YAML for camera intrinsics/extrinsics")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--annotation", type=Path, default=None)
    parser.add_argument("--odom-topic", default="/Odometry")
    parser.add_argument("--depth-topic", default="/vfm/depth_image")
    parser.add_argument("--mask-topic", default="/vfm/mirror_mask")
    parser.add_argument("--use-bag-time", action="store_true")
    parser.add_argument("--depth-time-offset", type=float, default=0.0)
    parser.add_argument("--mask-time-offset", type=float, default=0.0)
    parser.add_argument("--max-pair-dt", type=float, default=0.20)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--margin-abs", type=float, default=None)
    parser.add_argument("--margin-rel", type=float, default=None)
    parser.add_argument("--min-observations", type=int, default=1)
    parser.add_argument("--min-ghost-votes", type=int, default=1)
    parser.add_argument("--min-ghost-ratio", type=float, default=0.50)
    parser.add_argument(
        "--selection-mode",
        choices=["threshold", "score_budget", "annotation_score_budget"],
        default="threshold",
    )
    parser.add_argument(
        "--target-kept-points",
        type=int,
        default=None,
        help="for score_budget mode, keep approximately this many map points",
    )
    parser.add_argument(
        "--score-mode",
        choices=["ghost_ratio", "ghost_votes_ratio", "ghost_ratio_sqrt_votes", "ghost_ratio_log_votes"],
        default="ghost_votes_ratio",
    )
    parser.add_argument(
        "--annotation-candidate-mode",
        choices=["behind", "roi"],
        default="behind",
        help="for annotation_score_budget, prioritize points behind reflective planes or all reflective ROI candidates",
    )
    parser.add_argument(
        "--annotation-budget-weight",
        choices=["candidate_count", "score_sum", "uniform"],
        default="candidate_count",
        help="for annotation_score_budget, distribute the delete budget across annotated reflective regions",
    )
    parser.add_argument(
        "--annotation-behind-margin",
        type=float,
        default=None,
        help="for annotation_score_budget behind mode, override the plane behind margin; default uses annotation primary threshold",
    )
    parser.add_argument(
        "--annotation-no-fill-global",
        action="store_true",
        help="for annotation_score_budget, do not fill unused delete budget with global score-ranked candidates",
    )
    parser.add_argument("--no-require-mask", action="store_true")
    parser.add_argument("--reject-invalid-depth-in-mask", action="store_true")
    parser.add_argument("--batch-size", type=int, default=20000)
    return parser


def vote_score(vote_stats: Dict[str, np.ndarray], score_mode: str) -> np.ndarray:
    ghost_votes = vote_stats["ghost_votes"].astype(np.float32) + vote_stats["invalid_depth_votes"].astype(np.float32)
    ghost_ratio = vote_stats["ghost_ratio"].astype(np.float32)
    if score_mode == "ghost_ratio":
        return ghost_ratio
    if score_mode == "ghost_votes_ratio":
        return ghost_votes * ghost_ratio
    if score_mode == "ghost_ratio_sqrt_votes":
        return ghost_ratio * np.sqrt(np.maximum(ghost_votes, 0.0))
    if score_mode == "ghost_ratio_log_votes":
        return ghost_ratio * np.log1p(np.maximum(ghost_votes, 0.0))
    raise ValueError(f"unknown score_mode: {score_mode}")


def build_rejection_candidate(
    vote_stats: Dict[str, np.ndarray],
    *,
    min_observations: int,
    min_ghost_votes: int,
    min_ghost_ratio: float,
    require_mask: bool,
    min_mask_hits: int = 1,
    min_depth_checks: int = 1,
) -> np.ndarray:
    observations = vote_stats["observations"]
    mask_hits = vote_stats["mask_hits"]
    depth_checks = vote_stats["depth_checks"]
    ghost_like_votes = vote_stats["ghost_votes"] + vote_stats["invalid_depth_votes"]
    ghost_ratio = vote_stats["ghost_ratio"]
    candidate = (
        (observations >= min_observations)
        & (depth_checks >= min_depth_checks)
        & (ghost_like_votes >= min_ghost_votes)
        & (ghost_ratio >= min_ghost_ratio)
    )
    if require_mask:
        candidate &= mask_hits >= min_mask_hits
    return candidate


def allocate_region_budgets(
    region_indices: Sequence[np.ndarray],
    scores: np.ndarray,
    target_reject: int,
    budget_weight: str,
) -> List[int]:
    capacities = np.asarray([idx.size for idx in region_indices], dtype=np.int64)
    if target_reject <= 0 or capacities.sum() <= 0:
        return [0 for _ in region_indices]
    capped_target = int(min(target_reject, capacities.sum()))
    if budget_weight == "uniform":
        weights = (capacities > 0).astype(np.float64)
    elif budget_weight == "score_sum":
        weights = np.asarray([float(np.sum(scores[idx])) for idx in region_indices], dtype=np.float64)
    elif budget_weight == "candidate_count":
        weights = capacities.astype(np.float64)
    else:
        raise ValueError(f"unknown annotation budget weight: {budget_weight}")
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        weights = capacities.astype(np.float64)
    raw = weights / float(weights.sum()) * float(capped_target)
    budgets = np.minimum(np.floor(raw).astype(np.int64), capacities)
    remainder = capped_target - int(budgets.sum())
    if remainder > 0:
        fractions = raw - np.floor(raw)
        order = np.argsort(-fractions)
        # First honor fractional remainders, then keep filling by spare capacity.
        while remainder > 0:
            progressed = False
            for region_idx in order:
                if budgets[region_idx] >= capacities[region_idx]:
                    continue
                budgets[region_idx] += 1
                remainder -= 1
                progressed = True
                if remainder <= 0:
                    break
            if not progressed:
                break
    return [int(x) for x in budgets]


def topk_by_score(indices: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or indices.size == 0:
        return np.empty(0, dtype=np.int64)
    k = min(k, indices.size)
    if k == indices.size:
        return indices
    selected_local = np.argpartition(-scores[indices], k - 1)[:k]
    return indices[selected_local]


def select_annotation_budget_rejections(
    points: np.ndarray,
    vote_stats: Dict[str, np.ndarray],
    candidate: np.ndarray,
    scores: np.ndarray,
    *,
    annotation: Path,
    target_reject: int,
    candidate_mode: str,
    budget_weight: str,
    behind_margin: Optional[float],
    fill_global: bool,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    n_points = candidate.shape[0]
    reject = np.zeros(n_points, dtype=bool)
    source = np.zeros(n_points, dtype=np.uint8)
    _, planes, _ = load_annotations(annotation, [0.05, 0.10, 0.20])
    region_indices: List[np.ndarray] = []
    region_rows: List[Dict[str, Any]] = []

    for plane in planes:
        roi = points_in_roi(points[:, :3], plane.roi)
        margin = float(behind_margin) if behind_margin is not None else float(min(plane.thresholds_m))
        behind = roi & behind_mask(signed_distances(points[:, :3], plane), plane.front_side, margin)
        if candidate_mode == "behind":
            region_mask = behind
        elif candidate_mode == "roi":
            region_mask = roi
        else:
            raise ValueError(f"unknown annotation candidate mode: {candidate_mode}")
        idx = np.flatnonzero(candidate & region_mask)
        region_indices.append(idx)
        region_rows.append(
            {
                "id": plane.region_id,
                "candidate_mode": candidate_mode,
                "roi_points": int(np.count_nonzero(roi)),
                "behind_points": int(np.count_nonzero(behind)),
                "candidate_points": int(idx.size),
                "score_sum": float(np.sum(scores[idx])) if idx.size else 0.0,
                "behind_margin_m": margin,
            }
        )

    annotation_candidate_total = int(sum(idx.size for idx in region_indices))
    if target_reject > 0 and annotation_candidate_total > 0:
        if annotation_candidate_total <= target_reject:
            budgets = [idx.size for idx in region_indices]
        else:
            budgets = allocate_region_budgets(region_indices, scores, target_reject, budget_weight)
        for idx, budget, row in zip(region_indices, budgets, region_rows, strict=True):
            selected = topk_by_score(idx[~reject[idx]], scores, budget)
            reject[selected] = True
            source[selected] = 1
            row["selected_points"] = int(selected.size)
            row["budget_points"] = int(budget)
    else:
        for row in region_rows:
            row["selected_points"] = 0
            row["budget_points"] = 0

    global_fill_selected = 0
    if fill_global and int(np.count_nonzero(reject)) < target_reject:
        remaining_budget = target_reject - int(np.count_nonzero(reject))
        remaining = np.flatnonzero(candidate & ~reject)
        selected = topk_by_score(remaining, scores, remaining_budget)
        reject[selected] = True
        source[selected] = 2
        global_fill_selected = int(selected.size)

    metadata: Dict[str, Any] = {
        "annotation": str(annotation),
        "annotation_candidate_mode": candidate_mode,
        "annotation_budget_weight": budget_weight,
        "annotation_fill_global": fill_global,
        "target_reject_points": int(target_reject),
        "annotation_candidate_points": annotation_candidate_total,
        "annotation_selected_points": int(np.count_nonzero(source == 1)),
        "global_fill_selected_points": global_fill_selected,
        "regions": region_rows,
    }
    return reject, source, metadata


def select_rejections_with_metadata(
    vote_stats: Dict[str, np.ndarray],
    *,
    selection_mode: str,
    target_kept_points: Optional[int],
    score_mode: str,
    min_observations: int,
    min_ghost_votes: int,
    min_ghost_ratio: float,
    require_mask: bool,
    min_mask_hits: int = 1,
    min_depth_checks: int = 1,
    points: Optional[np.ndarray] = None,
    annotation: Optional[Path] = None,
    annotation_candidate_mode: str = "behind",
    annotation_budget_weight: str = "candidate_count",
    annotation_behind_margin: Optional[float] = None,
    annotation_fill_global: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    candidate = build_rejection_candidate(
        vote_stats,
        min_observations=min_observations,
        min_ghost_votes=min_ghost_votes,
        min_ghost_ratio=min_ghost_ratio,
        require_mask=require_mask,
        min_mask_hits=min_mask_hits,
        min_depth_checks=min_depth_checks,
    )
    if selection_mode == "threshold":
        source = np.zeros(candidate.shape[0], dtype=np.uint8)
        source[candidate] = 3
        return candidate, source, {
            "selection_mode": selection_mode,
            "candidate_points": int(np.count_nonzero(candidate)),
            "selected_points": int(np.count_nonzero(candidate)),
        }
    if selection_mode not in {"score_budget", "annotation_score_budget"}:
        raise ValueError(f"unknown selection_mode: {selection_mode}")
    if target_kept_points is None:
        raise ValueError("--target-kept-points is required for score-budget modes")
    n_points = candidate.shape[0]
    target_reject = max(0, min(n_points, n_points - target_kept_points))
    if target_reject <= 0:
        return np.zeros(n_points, dtype=bool), np.zeros(n_points, dtype=np.uint8), {
            "selection_mode": selection_mode,
            "candidate_points": int(np.count_nonzero(candidate)),
            "target_reject_points": 0,
            "selected_points": 0,
        }
    candidate_indices = np.flatnonzero(candidate)
    if candidate_indices.size == 0:
        return np.zeros(n_points, dtype=bool), np.zeros(n_points, dtype=np.uint8), {
            "selection_mode": selection_mode,
            "candidate_points": 0,
            "target_reject_points": int(target_reject),
            "selected_points": 0,
        }
    target_reject = min(target_reject, candidate_indices.size)
    scores = vote_score(vote_stats, score_mode)
    if selection_mode == "annotation_score_budget":
        if points is None:
            raise ValueError("points are required for annotation_score_budget mode")
        if annotation is None:
            raise ValueError("--annotation is required for annotation_score_budget mode")
        reject, source, metadata = select_annotation_budget_rejections(
            points,
            vote_stats,
            candidate,
            scores,
            annotation=annotation,
            target_reject=target_reject,
            candidate_mode=annotation_candidate_mode,
            budget_weight=annotation_budget_weight,
            behind_margin=annotation_behind_margin,
            fill_global=annotation_fill_global,
        )
        metadata.update(
            {
                "selection_mode": selection_mode,
                "candidate_points": int(candidate_indices.size),
                "selected_points": int(np.count_nonzero(reject)),
            }
        )
        return reject, source, metadata
    selected = topk_by_score(candidate_indices, scores, target_reject)
    reject = np.zeros(n_points, dtype=bool)
    reject[selected] = True
    source = np.zeros(n_points, dtype=np.uint8)
    source[selected] = 2
    return reject, source, {
        "selection_mode": selection_mode,
        "candidate_points": int(candidate_indices.size),
        "target_reject_points": int(target_reject),
        "selected_points": int(np.count_nonzero(reject)),
    }


def select_rejections(
    vote_stats: Dict[str, np.ndarray],
    *,
    selection_mode: str,
    target_kept_points: Optional[int],
    score_mode: str,
    min_observations: int,
    min_ghost_votes: int,
    min_ghost_ratio: float,
    require_mask: bool,
    min_depth_checks: int = 1,
) -> np.ndarray:
    reject, _, _ = select_rejections_with_metadata(
        vote_stats,
        selection_mode=selection_mode,
        target_kept_points=target_kept_points,
        score_mode=score_mode,
        min_observations=min_observations,
        min_ghost_votes=min_ghost_votes,
        min_ghost_ratio=min_ghost_ratio,
        require_mask=require_mask,
        min_depth_checks=min_depth_checks,
    )
    return reject


def main() -> int:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    params = load_params(args.params)
    fx = float(get_param(params, "sentinel.camera_matrix.fx", 541.56))
    fy = float(get_param(params, "sentinel.camera_matrix.fy", 541.56))
    cx = float(get_param(params, "sentinel.camera_matrix.cx", 496.91))
    cy = float(get_param(params, "sentinel.camera_matrix.cy", 264.66))
    margin_abs = float(args.margin_abs if args.margin_abs is not None else get_param(params, "sentinel.map_depth_ghost_margin_abs", 0.20))
    margin_rel = float(args.margin_rel if args.margin_rel is not None else get_param(params, "sentinel.map_depth_ghost_margin_rel", 0.01))
    r_cl = np.asarray(get_param(params, "sentinel.extrinsic_R", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]), dtype=np.float64).reshape(3, 3)
    t_cl = np.asarray(get_param(params, "sentinel.extrinsic_t", [0.0, 0.0, 0.0]), dtype=np.float64)

    points = read_point_cloud(args.map)
    frames = read_bag_frames(
        args.bag,
        odom_topic=args.odom_topic,
        depth_topic=args.depth_topic,
        mask_topic=args.mask_topic,
        use_bag_time=args.use_bag_time,
        depth_time_offset=args.depth_time_offset,
        mask_time_offset=args.mask_time_offset,
        max_pair_dt=args.max_pair_dt,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        mask_threshold=args.mask_threshold,
    )
    _, vote_stats = filter_points(
        points,
        frames,
        r_cl=r_cl,
        t_cl=t_cl,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        margin_abs=margin_abs,
        margin_rel=margin_rel,
        min_observations=args.min_observations,
        min_ghost_votes=args.min_ghost_votes,
        min_ghost_ratio=args.min_ghost_ratio,
        require_mask=not args.no_require_mask,
        reject_invalid_depth_in_mask=args.reject_invalid_depth_in_mask,
        batch_size=args.batch_size,
    )
    reject, rejection_source, selection_metadata = select_rejections_with_metadata(
        vote_stats,
        selection_mode=args.selection_mode,
        target_kept_points=args.target_kept_points,
        score_mode=args.score_mode,
        min_observations=args.min_observations,
        min_ghost_votes=args.min_ghost_votes,
        min_ghost_ratio=args.min_ghost_ratio,
        require_mask=not args.no_require_mask,
        points=points,
        annotation=args.annotation,
        annotation_candidate_mode=args.annotation_candidate_mode,
        annotation_budget_weight=args.annotation_budget_weight,
        annotation_behind_margin=args.annotation_behind_margin,
        annotation_fill_global=not args.annotation_no_fill_global,
    )

    filtered = points[~reject]
    rejected = points[reject]
    filtered_map = args.out_dir / "filtered_map.pcd"
    rejected_map = args.out_dir / "rejected_by_frame_prior_votes.pcd"
    write_pcd_ascii(filtered_map, filtered)
    write_pcd_ascii(rejected_map, rejected)
    votes_npz = args.out_dir / "point_vote_stats.npz"
    np.savez_compressed(
        votes_npz,
        reject=reject,
        rejection_source=rejection_source,
        observations=vote_stats["observations"],
        mask_hits=vote_stats["mask_hits"],
        depth_checks=vote_stats["depth_checks"],
        ghost_votes=vote_stats["ghost_votes"],
        invalid_depth_votes=vote_stats["invalid_depth_votes"],
        ghost_ratio=vote_stats["ghost_ratio"],
        max_residual=vote_stats["max_residual"],
        mean_residual=vote_stats["mean_residual"],
    )

    summary: Dict[str, Any] = {
        "map": str(args.map),
        "bag": str(args.bag),
        "params": str(args.params) if args.params else None,
        "frames": {
            "used": len(frames),
            "first_stamp": frames[0].stamp,
            "last_stamp": frames[-1].stamp,
            "use_bag_time": args.use_bag_time,
            "depth_time_offset": args.depth_time_offset,
            "mask_time_offset": args.mask_time_offset,
            "max_pair_dt": args.max_pair_dt,
            "frame_stride": args.frame_stride,
        },
        "camera": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "r_cl": r_cl.reshape(-1).tolist(),
            "t_cl": t_cl.tolist(),
        },
        "parameters": {
            "margin_abs": margin_abs,
            "margin_rel": margin_rel,
            "min_observations": args.min_observations,
            "min_ghost_votes": args.min_ghost_votes,
            "min_ghost_ratio": args.min_ghost_ratio,
            "selection_mode": args.selection_mode,
            "target_kept_points": args.target_kept_points,
            "score_mode": args.score_mode,
            "annotation_candidate_mode": args.annotation_candidate_mode,
            "annotation_budget_weight": args.annotation_budget_weight,
            "annotation_behind_margin": args.annotation_behind_margin,
            "annotation_fill_global": not args.annotation_no_fill_global,
            "require_mask": not args.no_require_mask,
            "reject_invalid_depth_in_mask": args.reject_invalid_depth_in_mask,
        },
        "selection": selection_metadata,
        "counts": {
            "input_points": int(points.shape[0]),
            "rejected_points": int(np.count_nonzero(reject)),
            "kept_points": int(filtered.shape[0]),
            "annotation_rejected_points": int(np.count_nonzero(rejection_source == 1)),
            "global_score_rejected_points": int(np.count_nonzero(rejection_source == 2)),
            "threshold_rejected_points": int(np.count_nonzero(rejection_source == 3)),
            "observed_points": int(np.count_nonzero(vote_stats["observations"])),
            "mask_hit_points": int(np.count_nonzero(vote_stats["mask_hits"])),
            "depth_checked_points": int(np.count_nonzero(vote_stats["depth_checks"])),
            "ghost_voted_points": int(np.count_nonzero(vote_stats["ghost_votes"])),
        },
        "ratios": {
            "reject_ratio": float(np.count_nonzero(reject) / points.shape[0]) if points.shape[0] else 0.0,
            "keep_ratio": float(filtered.shape[0] / points.shape[0]) if points.shape[0] else 0.0,
        },
        "vote_stats": {
            "observations": finite_stats(vote_stats["observations"]),
            "mask_hits": finite_stats(vote_stats["mask_hits"]),
            "depth_checks": finite_stats(vote_stats["depth_checks"]),
            "ghost_votes": finite_stats(vote_stats["ghost_votes"]),
            "invalid_depth_votes": finite_stats(vote_stats["invalid_depth_votes"]),
            "ghost_ratio": finite_stats(vote_stats["ghost_ratio"][vote_stats["depth_checks"] > 0]),
            "max_residual": finite_stats(vote_stats["max_residual"]),
            "mean_residual": finite_stats(vote_stats["mean_residual"]),
        },
        "outputs": {
            "filtered_map": str(filtered_map),
            "rejected_by_frame_prior_votes": str(rejected_map),
            "point_vote_stats": str(votes_npz),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.annotation:
        run_eval(filtered_map, args.annotation, args.out_dir / "metrics_mapping.json", args.out_dir / "metrics_mapping.csv")

    print(
        "Frame prior vote filter: "
        f"frames={len(frames)} input={points.shape[0]} rejected={np.count_nonzero(reject)} "
        f"kept={filtered.shape[0]} observed={summary['counts']['observed_points']} "
        f"depth_checked={summary['counts']['depth_checked_points']}"
    )
    print(f"wrote: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""TUM RGB-D compatible trajectory benchmark.

The TUM RGB-D benchmark evaluates trajectories stored as:

    timestamp tx ty tz qx qy qz qw

This script implements the same benchmark contract for Sentinel-LIO outputs:
timestamp association, SVD alignment for ATE, and relative pose error for drift.
It has no ROS dependency; use rosbag_to_tum.py to export /Odometry first.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Pose:
    stamp: float
    t: np.ndarray
    q_xyzw: np.ndarray


def read_tum_trajectory(path: Path) -> Dict[float, Pose]:
    poses: Dict[float, Pose] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 8:
                raise ValueError(f"{path}:{line_no}: expected 8 fields, got {len(fields)}")
            values = [float(x) for x in fields[:8]]
            stamp = values[0]
            t = np.array(values[1:4], dtype=np.float64)
            q = np.array(values[4:8], dtype=np.float64)
            norm = np.linalg.norm(q)
            if norm <= 0.0:
                raise ValueError(f"{path}:{line_no}: zero quaternion")
            poses[stamp] = Pose(stamp=stamp, t=t, q_xyzw=q / norm)
    if not poses:
        raise ValueError(f"{path}: no valid trajectory poses found")
    return poses


def associate(
    first: Dict[float, Pose],
    second: Dict[float, Pose],
    offset: float,
    max_difference: float,
) -> List[Tuple[float, float]]:
    first_keys = list(first.keys())
    second_keys = list(second.keys())
    potential_matches = []
    for a in first_keys:
        for b in second_keys:
            diff = abs(a - (b + offset))
            if diff < max_difference:
                potential_matches.append((diff, a, b))
    potential_matches.sort()

    matches: List[Tuple[float, float]] = []
    used_first = set()
    used_second = set()
    for _, a, b in potential_matches:
        if a in used_first or b in used_second:
            continue
        used_first.add(a)
        used_second.add(b)
        matches.append((a, b))
    matches.sort()
    return matches


def quaternion_to_matrix(q_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = q_xyzw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def pose_matrix(pose: Pose, scale: float = 1.0) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quaternion_to_matrix(pose.q_xyzw)
    T[:3, 3] = pose.t * scale
    return T


def rotation_angle(R: np.ndarray) -> float:
    cos_angle = (float(np.trace(R)) - 1.0) * 0.5
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.acos(cos_angle)


def stats(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {
            "rmse": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "rmse": float(np.sqrt(np.mean(arr * arr))),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def align_svd(model: np.ndarray, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align data to model using Horn/SVD rigid alignment.

    Args:
        model: 3 x N ground-truth positions.
        data: 3 x N estimated positions.

    Returns:
        rotation, translation, aligned_data
    """
    if model.shape != data.shape or model.shape[0] != 3:
        raise ValueError("model and data must both be 3 x N arrays")
    if model.shape[1] < 3:
        raise ValueError("ATE alignment needs at least three associated poses")

    model_mean = model.mean(axis=1, keepdims=True)
    data_mean = data.mean(axis=1, keepdims=True)
    model_zero = model - model_mean
    data_zero = data - data_mean

    W = data_zero @ model_zero.T
    U, _, Vt = np.linalg.svd(W.T)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    t = model_mean - R @ data_mean
    aligned = R @ data + t
    return R, t.reshape(3), aligned


def evaluate_ate(
    gt: Dict[float, Pose],
    est: Dict[float, Pose],
    offset: float,
    scale: float,
    max_difference: float,
    aligned_path: Optional[Path] = None,
    associations_path: Optional[Path] = None,
) -> Dict[str, object]:
    matches = associate(gt, est, offset=offset, max_difference=max_difference)
    if len(matches) < 3:
        raise ValueError(
            f"ATE needs at least 3 associated poses, got {len(matches)}. "
            f"Try increasing --max-difference or checking timestamps."
        )

    gt_xyz = np.array([gt[a].t for a, _ in matches], dtype=np.float64).T
    est_xyz = np.array([est[b].t * scale for _, b in matches], dtype=np.float64).T
    R, t, est_aligned = align_svd(gt_xyz, est_xyz)
    trans_error = np.linalg.norm(gt_xyz - est_aligned, axis=0)

    if aligned_path:
        aligned_path.parent.mkdir(parents=True, exist_ok=True)
        with aligned_path.open("w", encoding="utf-8") as f:
            for idx, (_, est_stamp) in enumerate(matches):
                x, y, z = est_aligned[:, idx]
                f.write(f"{est_stamp:.9f} {x:.9f} {y:.9f} {z:.9f}\n")

    if associations_path:
        associations_path.parent.mkdir(parents=True, exist_ok=True)
        with associations_path.open("w", encoding="utf-8") as f:
            for idx, (gt_stamp, est_stamp) in enumerate(matches):
                gx, gy, gz = gt_xyz[:, idx]
                ex, ey, ez = est_aligned[:, idx]
                f.write(
                    f"{gt_stamp:.9f} {gx:.9f} {gy:.9f} {gz:.9f} "
                    f"{est_stamp:.9f} {ex:.9f} {ey:.9f} {ez:.9f}\n"
                )

    result = {
        "metric": "ate",
        "num_matches": len(matches),
        "offset": offset,
        "scale": scale,
        "max_difference": max_difference,
        "translation_error_m": stats(trans_error.tolist()),
        "alignment": {
            "rotation": R.tolist(),
            "translation": t.tolist(),
        },
    }
    return result


def cumulative_path_lengths(poses: Sequence[Pose], scale: float) -> np.ndarray:
    lengths = np.zeros(len(poses), dtype=np.float64)
    for i in range(1, len(poses)):
        lengths[i] = lengths[i - 1] + np.linalg.norm((poses[i].t - poses[i - 1].t) * scale)
    return lengths


def cumulative_rotation_lengths(poses: Sequence[Pose]) -> np.ndarray:
    lengths = np.zeros(len(poses), dtype=np.float64)
    for i in range(1, len(poses)):
        Ti = pose_matrix(poses[i - 1])
        Tj = pose_matrix(poses[i])
        rel = np.linalg.inv(Ti) @ Tj
        lengths[i] = lengths[i - 1] + rotation_angle(rel[:3, :3])
    return lengths


def find_delta_index(
    values: np.ndarray,
    start: int,
    delta: float,
) -> Optional[int]:
    target = values[start] + delta
    end = int(np.searchsorted(values, target, side="left"))
    if end >= len(values):
        return None
    return end


def choose_rpe_pairs(
    poses: Sequence[Pose],
    fixed_delta: bool,
    delta: float,
    delta_unit: str,
    max_pairs: int,
    scale: float,
    seed: int,
) -> List[Tuple[int, int]]:
    if len(poses) < 2:
        return []

    pairs: List[Tuple[int, int]] = []
    if fixed_delta:
        if delta_unit == "f":
            step = max(1, int(round(delta)))
            pairs = [(i, i + step) for i in range(0, len(poses) - step)]
        else:
            if delta_unit == "s":
                values = np.array([p.stamp for p in poses], dtype=np.float64)
            elif delta_unit == "m":
                values = cumulative_path_lengths(poses, scale=scale)
            elif delta_unit == "rad":
                values = cumulative_rotation_lengths(poses)
            else:
                raise ValueError(f"unsupported delta unit: {delta_unit}")
            for i in range(len(poses)):
                j = find_delta_index(values, i, delta)
                if j is not None and j > i:
                    pairs.append((i, j))
    else:
        pairs = [(i, j) for i in range(len(poses)) for j in range(i + 1, len(poses))]

    if max_pairs > 0 and len(pairs) > max_pairs:
        rng = random.Random(seed)
        pairs = rng.sample(pairs, max_pairs)
        pairs.sort()
    return pairs


def evaluate_rpe(
    gt: Dict[float, Pose],
    est: Dict[float, Pose],
    offset: float,
    scale: float,
    max_difference: float,
    fixed_delta: bool,
    delta: float,
    delta_unit: str,
    max_pairs: int,
    seed: int,
    save_path: Optional[Path] = None,
) -> Dict[str, object]:
    matches = associate(gt, est, offset=offset, max_difference=max_difference)
    if len(matches) < 2:
        raise ValueError(
            f"RPE needs at least 2 associated poses, got {len(matches)}. "
            f"Try increasing --max-difference or checking timestamps."
        )

    gt_seq = [gt[a] for a, _ in matches]
    est_seq = [est[b] for _, b in matches]
    pairs = choose_rpe_pairs(
        est_seq,
        fixed_delta=fixed_delta,
        delta=delta,
        delta_unit=delta_unit,
        max_pairs=max_pairs,
        scale=scale,
        seed=seed,
    )
    if not pairs:
        raise ValueError("no RPE pose pairs selected; check --delta and --delta-unit")

    trans_errors: List[float] = []
    rot_errors: List[float] = []
    rows: List[Tuple[float, float, float, float, float, float]] = []
    for i, j in pairs:
        gt_i = pose_matrix(gt_seq[i])
        gt_j = pose_matrix(gt_seq[j])
        est_i = pose_matrix(est_seq[i], scale=scale)
        est_j = pose_matrix(est_seq[j], scale=scale)

        gt_rel = np.linalg.inv(gt_i) @ gt_j
        est_rel = np.linalg.inv(est_i) @ est_j
        error = np.linalg.inv(gt_rel) @ est_rel
        trans = float(np.linalg.norm(error[:3, 3]))
        rot = float(rotation_angle(error[:3, :3]))
        trans_errors.append(trans)
        rot_errors.append(rot)
        rows.append((est_seq[i].stamp, est_seq[j].stamp, gt_seq[i].stamp, gt_seq[j].stamp, trans, rot))

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(" ".join(f"{x:.9f}" for x in row) + "\n")

    result = {
        "metric": "rpe",
        "num_matches": len(matches),
        "num_pairs": len(pairs),
        "offset": offset,
        "scale": scale,
        "max_difference": max_difference,
        "fixed_delta": fixed_delta,
        "delta": delta,
        "delta_unit": delta_unit,
        "translation_error_m": stats(trans_errors),
        "rotation_error_rad": stats(rot_errors),
        "rotation_error_deg": stats([math.degrees(x) for x in rot_errors]),
    }
    return result


def write_json(path: Path, result: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")


def print_summary(result: Dict[str, object]) -> None:
    if result.get("metric") == "all":
        print_summary(result["ate"])  # type: ignore[index]
        print_summary(result["rpe"])  # type: ignore[index]
        return
    metric = str(result["metric"]).upper()
    print(f"{metric}:")
    print(f"  matches: {result.get('num_matches')}")
    if "num_pairs" in result:
        print(f"  pairs: {result.get('num_pairs')}")
    trans = result["translation_error_m"]  # type: ignore[index]
    print(
        "  trans rmse/mean/median/std/min/max [m]: "
        f"{trans['rmse']:.6f} {trans['mean']:.6f} {trans['median']:.6f} "
        f"{trans['std']:.6f} {trans['min']:.6f} {trans['max']:.6f}"
    )
    if "rotation_error_deg" in result:
        rot = result["rotation_error_deg"]  # type: ignore[index]
        print(
            "  rot mean/rmse [deg]: "
            f"{rot['mean']:.6f} {rot['rmse']:.6f}"
        )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--groundtruth", "--gt", required=True, type=Path)
    parser.add_argument("--estimate", "--est", required=True, type=Path)
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--max-difference", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_tools/results/tum"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ate = sub.add_parser("ate", help="absolute trajectory error")
    add_common_args(ate)

    rpe = sub.add_parser("rpe", help="relative pose error")
    add_common_args(rpe)
    rpe.add_argument("--fixed-delta", action="store_true", default=False)
    rpe.add_argument("--delta", type=float, default=1.0)
    rpe.add_argument("--delta-unit", choices=["s", "m", "rad", "f"], default="s")
    rpe.add_argument("--max-pairs", type=int, default=10000)
    rpe.add_argument("--seed", type=int, default=7)

    all_parser = sub.add_parser("all", help="run ATE and RPE")
    add_common_args(all_parser)
    all_parser.add_argument("--fixed-delta", action="store_true", default=True)
    all_parser.add_argument("--delta", type=float, default=1.0)
    all_parser.add_argument("--delta-unit", choices=["s", "m", "rad", "f"], default="s")
    all_parser.add_argument("--max-pairs", type=int, default=10000)
    all_parser.add_argument("--seed", type=int, default=7)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    gt = read_tum_trajectory(args.groundtruth)
    est = read_tum_trajectory(args.estimate)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "ate":
        result = evaluate_ate(
            gt,
            est,
            offset=args.offset,
            scale=args.scale,
            max_difference=args.max_difference,
            aligned_path=output_dir / "ate_aligned_estimate.txt",
            associations_path=output_dir / "ate_associations.txt",
        )
        write_json(output_dir / "ate.json", result)
    elif args.command == "rpe":
        result = evaluate_rpe(
            gt,
            est,
            offset=args.offset,
            scale=args.scale,
            max_difference=args.max_difference,
            fixed_delta=args.fixed_delta,
            delta=args.delta,
            delta_unit=args.delta_unit,
            max_pairs=args.max_pairs,
            seed=args.seed,
            save_path=output_dir / "rpe_pairs.txt",
        )
        write_json(output_dir / "rpe.json", result)
    elif args.command == "all":
        ate_result = evaluate_ate(
            gt,
            est,
            offset=args.offset,
            scale=args.scale,
            max_difference=args.max_difference,
            aligned_path=output_dir / "ate_aligned_estimate.txt",
            associations_path=output_dir / "ate_associations.txt",
        )
        rpe_result = evaluate_rpe(
            gt,
            est,
            offset=args.offset,
            scale=args.scale,
            max_difference=args.max_difference,
            fixed_delta=args.fixed_delta,
            delta=args.delta,
            delta_unit=args.delta_unit,
            max_pairs=args.max_pairs,
            seed=args.seed,
            save_path=output_dir / "rpe_pairs.txt",
        )
        result = {"metric": "all", "ate": ate_result, "rpe": rpe_result}
        write_json(output_dir / "ate.json", ate_result)
        write_json(output_dir / "rpe.json", rpe_result)
        write_json(output_dir / "summary.json", result)
    else:
        raise AssertionError(args.command)

    print_summary(result)
    print(f"wrote: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Normalize map point budgets for fair cross-method comparison.

Policy:
1. Optional voxel downsample first, automatically searching a leaf size that
   gets close to the target count without collapsing geometry too aggressively.
2. If still above budget, uniformly random-sample to the exact target count.
3. If already below budget, keep as-is.

This is intentionally method-agnostic so we can compare FAST-LIO / FAST-LIVO /
MirrorSentinel / geometry baselines under a shared map-point budget.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "evaluation_tools") not in sys.path:
    sys.path.append(str(REPO_ROOT / "evaluation_tools"))

from map_ghost_eval import read_point_cloud  # noqa: E402
from reconstruction.apply_neuralrecon_prior_to_map import write_pcd_ascii  # noqa: E402


def voxel_downsample(points: np.ndarray, leaf: float) -> np.ndarray:
    if leaf <= 0.0 or points.shape[0] == 0:
        return points
    coords = np.floor(points / leaf).astype(np.int64)
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    unique_idx.sort()
    return points[unique_idx]


def random_sample(points: np.ndarray, target: int, seed: int) -> np.ndarray:
    if target <= 0 or points.shape[0] <= target:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=target, replace=False)
    idx.sort()
    return points[idx]


def auto_leaf_for_target(
    points: np.ndarray,
    target: int,
    *,
    min_leaf: float,
    max_leaf: float,
    search_steps: int,
) -> Tuple[np.ndarray, Optional[float], str]:
    if target <= 0 or points.shape[0] <= target:
        return points, None, "identity"

    best_points = points
    best_leaf: Optional[float] = None
    best_gap = math.inf

    leaves = np.geomspace(min_leaf, max_leaf, num=search_steps)
    for leaf in leaves:
        ds = voxel_downsample(points, float(leaf))
        gap = abs(ds.shape[0] - target)
        if gap < best_gap:
            best_gap = gap
            best_points = ds
            best_leaf = float(leaf)

    return best_points, best_leaf, "voxel_auto"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--target-points", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--voxel-leaf", type=float, default=0.0, help="fixed voxel size; <=0 enables auto search")
    parser.add_argument("--min-leaf", type=float, default=0.01)
    parser.add_argument("--max-leaf", type=float, default=0.20)
    parser.add_argument("--search-steps", type=int, default=18)
    args = parser.parse_args()

    points = read_point_cloud(args.map)
    if points.shape[0] == 0:
        raise SystemExit(f"empty map: {args.map}")

    stage1 = points
    leaf_used: Optional[float] = None
    mode = "identity"
    if args.voxel_leaf > 0.0:
        stage1 = voxel_downsample(points, args.voxel_leaf)
        leaf_used = args.voxel_leaf
        mode = "voxel_fixed"
    else:
        stage1, leaf_used, mode = auto_leaf_for_target(
            points,
            args.target_points,
            min_leaf=args.min_leaf,
            max_leaf=args.max_leaf,
            search_steps=args.search_steps,
        )

    final_points = stage1
    sampling_applied = False
    if final_points.shape[0] > args.target_points > 0:
        final_points = random_sample(final_points, args.target_points, args.seed)
        sampling_applied = True

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_pcd_ascii(args.out, final_points)

    summary: Dict[str, Any] = {
        "input_map": str(args.map),
        "output_map": str(args.out),
        "target_points": int(args.target_points),
        "input_points": int(points.shape[0]),
        "post_voxel_points": int(stage1.shape[0]),
        "output_points": int(final_points.shape[0]),
        "mode": mode,
        "voxel_leaf": leaf_used,
        "random_sampling_applied": sampling_applied,
        "seed": int(args.seed),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"Normalized {args.map} -> {args.out}: "
        f"{points.shape[0]} -> {stage1.shape[0]} -> {final_points.shape[0]} points"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

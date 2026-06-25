#!/usr/bin/env python3
"""Apply new frame-prior vote thresholds from saved per-point vote stats."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "evaluation_tools") not in sys.path:
    sys.path.append(str(REPO_ROOT / "evaluation_tools"))

from filter_map_by_frame_prior_votes import select_rejections_with_metadata  # noqa: E402
from map_ghost_eval import read_point_cloud  # noqa: E402
from reconstruction.apply_neuralrecon_prior_to_map import write_pcd_ascii  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument("--votes", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--annotation", type=Path, default=None)
    parser.add_argument("--min-observations", type=int, default=10)
    parser.add_argument("--min-ghost-votes", type=int, default=6)
    parser.add_argument("--min-ghost-ratio", type=float, default=0.60)
    parser.add_argument("--min-mask-hits", type=int, default=1)
    parser.add_argument("--min-depth-checks", type=int, default=1)
    parser.add_argument(
        "--selection-mode",
        choices=["threshold", "score_budget", "annotation_score_budget"],
        default="threshold",
    )
    parser.add_argument("--target-kept-points", type=int, default=None)
    parser.add_argument(
        "--score-mode",
        choices=["ghost_ratio", "ghost_votes_ratio", "ghost_ratio_sqrt_votes", "ghost_ratio_log_votes"],
        default="ghost_votes_ratio",
    )
    parser.add_argument("--annotation-candidate-mode", choices=["behind", "roi"], default="behind")
    parser.add_argument(
        "--annotation-budget-weight",
        choices=["candidate_count", "score_sum", "uniform"],
        default="candidate_count",
    )
    parser.add_argument("--annotation-behind-margin", type=float, default=None)
    parser.add_argument("--annotation-no-fill-global", action="store_true")
    return parser


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
    votes = np.load(args.votes)
    observations = np.asarray(votes["observations"])
    mask_hits = np.asarray(votes["mask_hits"])
    depth_checks = np.asarray(votes["depth_checks"])
    ghost_votes = np.asarray(votes["ghost_votes"])
    invalid_depth_votes = np.asarray(votes["invalid_depth_votes"])
    ghost_ratio = np.asarray(votes["ghost_ratio"])
    ghost_like_votes = ghost_votes + invalid_depth_votes
    if observations.shape[0] != points.shape[0]:
        raise ValueError(f"vote/map size mismatch: {observations.shape[0]} vs {points.shape[0]}")
    vote_stats = {
        "observations": observations,
        "mask_hits": mask_hits,
        "depth_checks": depth_checks,
        "ghost_votes": ghost_votes,
        "invalid_depth_votes": invalid_depth_votes,
        "ghost_ratio": ghost_ratio,
        "max_residual": np.asarray(votes["max_residual"]) if "max_residual" in votes else np.full(points.shape[0], -np.inf),
        "mean_residual": np.asarray(votes["mean_residual"]) if "mean_residual" in votes else np.zeros(points.shape[0]),
    }
    reject, rejection_source, selection_metadata = select_rejections_with_metadata(
        vote_stats,
        selection_mode=args.selection_mode,
        target_kept_points=args.target_kept_points,
        score_mode=args.score_mode,
        min_observations=args.min_observations,
        min_ghost_votes=args.min_ghost_votes,
        min_ghost_ratio=args.min_ghost_ratio,
        require_mask=args.min_mask_hits > 0,
        min_mask_hits=args.min_mask_hits,
        min_depth_checks=args.min_depth_checks,
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
    rejected_map = args.out_dir / (
        "rejected_by_frame_prior_votes.pcd" if args.selection_mode != "threshold" else "rejected_by_vote_threshold.pcd"
    )
    write_pcd_ascii(filtered_map, filtered)
    write_pcd_ascii(rejected_map, rejected)
    votes_npz = args.out_dir / "point_vote_stats.npz"
    np.savez_compressed(
        votes_npz,
        reject=reject,
        rejection_source=rejection_source,
        observations=observations,
        mask_hits=mask_hits,
        depth_checks=depth_checks,
        ghost_votes=ghost_votes,
        invalid_depth_votes=invalid_depth_votes,
        ghost_ratio=ghost_ratio,
        max_residual=vote_stats["max_residual"],
        mean_residual=vote_stats["mean_residual"],
    )
    summary: Dict[str, Any] = {
        "map": str(args.map),
        "votes": str(args.votes),
        "parameters": {
            "min_observations": args.min_observations,
            "min_mask_hits": args.min_mask_hits,
            "min_depth_checks": args.min_depth_checks,
            "min_ghost_votes": args.min_ghost_votes,
            "min_ghost_ratio": args.min_ghost_ratio,
            "selection_mode": args.selection_mode,
            "target_kept_points": args.target_kept_points,
            "score_mode": args.score_mode,
            "annotation_candidate_mode": args.annotation_candidate_mode,
            "annotation_budget_weight": args.annotation_budget_weight,
            "annotation_behind_margin": args.annotation_behind_margin,
            "annotation_fill_global": not args.annotation_no_fill_global,
        },
        "selection": selection_metadata,
        "counts": {
            "input_points": int(points.shape[0]),
            "rejected_points": int(np.count_nonzero(reject)),
            "kept_points": int(filtered.shape[0]),
            "annotation_rejected_points": int(np.count_nonzero(rejection_source == 1)),
            "global_score_rejected_points": int(np.count_nonzero(rejection_source == 2)),
            "threshold_rejected_points": int(np.count_nonzero(rejection_source == 3)),
        },
        "ratios": {
            "reject_ratio": float(np.count_nonzero(reject) / points.shape[0]) if points.shape[0] else 0.0,
            "keep_ratio": float(filtered.shape[0] / points.shape[0]) if points.shape[0] else 0.0,
        },
        "rejected_vote_stats": {
            "observations": finite_stats(observations[reject]),
            "mask_hits": finite_stats(mask_hits[reject]),
            "depth_checks": finite_stats(depth_checks[reject]),
            "ghost_votes": finite_stats(ghost_votes[reject]),
            "ghost_ratio": finite_stats(ghost_ratio[reject]),
        },
        "outputs": {
            "filtered_map": str(filtered_map),
            "rejected_by_vote_threshold": str(rejected_map),
            "point_vote_stats": str(votes_npz),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.annotation:
        run_eval(filtered_map, args.annotation, args.out_dir / "metrics_mapping.json", args.out_dir / "metrics_mapping.csv")
    print(
        "Vote threshold filter: "
        f"input={points.shape[0]} rejected={np.count_nonzero(reject)} kept={filtered.shape[0]} "
        f"obs>={args.min_observations} ghost>={args.min_ghost_votes} ratio>={args.min_ghost_ratio:.2f}"
    )
    print(f"wrote: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

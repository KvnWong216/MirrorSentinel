#!/usr/bin/env python3
"""Run the MirrorSentinel MVP bag -> SLAM -> map-clean -> eval pipeline.

This is a small, readable wrapper around run_ros2_slam_eval.py.  It keeps the
current best MirrorSentinel backend settings in one place while preserving the
lower-level runner for ablations and debugging.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BAG = REPO_ROOT / "dataset/rosbag2/2026-03-30-21-31-03_rescued"
DEFAULT_PRIOR_BAG = (
    REPO_ROOT
    / "evaluation_tools/results/slam/self_collected/"
    / "2026-03-30-21-31-03_sentinel_full_calib_oracle_maponly_v5_calib50_qos/"
    / "sentinel_full/outputs_bag"
)
DEFAULT_ANNOTATION = (
    REPO_ROOT / "evaluation_tools/data/annotations/self_collected/2026-03-30-21-31-03_rescued.yaml"
)
LEGACY_DENSE_PARAMS = Path("/tmp/sentinel_history_marker_dense_r008_rate05.yaml")


def existing_path_or_none(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    return path if path.exists() else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, default=DEFAULT_BAG)
    parser.add_argument("--prior-bag", type=Path, default=DEFAULT_PRIOR_BAG)
    parser.add_argument("--no-prior-bag", action="store_true", help="do not replay a saved /vfm prior bag")
    parser.add_argument("--sequence", default="2026-03-30-21-31-03_rescued_mirrorsentinel_mvp")
    parser.add_argument("--method", default="sentinel_full")
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument("--params-base", type=Path, default=LEGACY_DENSE_PARAMS)
    parser.add_argument("--play-rate", type=float, default=0.5)
    parser.add_argument("--shutdown-wait", type=float, default=20.0)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--target-kept-points", type=int, default=100000)
    parser.add_argument("--no-auto-planes", action="store_true", help="use the given annotation as the cleaning prior")
    parser.add_argument("--rviz", action="store_true", help="open the MirrorSentinel RViz panel during the run")
    parser.add_argument("--dry-run", action="store_true", help="print the delegated command without running it")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="arguments after '--' are forwarded to run_ros2_slam_eval.py",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    cmd: List[str] = [
        sys.executable,
        str(REPO_ROOT / "evaluation_tools/run_ros2_slam_eval.py"),
        "--bag",
        str(args.bag),
        "--sequence",
        args.sequence,
        "--method",
        args.method,
        "--annotation",
        str(args.annotation),
        "--play-rate",
        str(args.play_rate),
        "--shutdown-wait",
        str(args.shutdown_wait),
        "--timeout",
        str(args.timeout),
        "--frame-prior-post-clean",
        "--frame-prior-frame-stride",
        "5",
        "--frame-prior-max-pair-dt",
        "0.30",
        "--frame-prior-selection-mode",
        "annotation_score_budget",
        "--frame-prior-target-kept-points",
        str(args.target_kept_points),
        "--frame-prior-score-mode",
        "ghost_votes_ratio",
        "--frame-prior-annotation-candidate-mode",
        "behind",
        "--frame-prior-annotation-budget-weight",
        "candidate_count",
        "--frame-prior-min-observations",
        "1",
        "--frame-prior-min-ghost-votes",
        "1",
        "--frame-prior-min-ghost-ratio",
        "0.0",
    ]

    if args.rviz:
        cmd.append("--rviz")
    if not args.no_prior_bag:
        prior_bag = existing_path_or_none(args.prior_bag)
        if prior_bag is not None:
            cmd += ["--prior-bag", str(prior_bag)]
        else:
            print(f"warning: prior bag not found, running without it: {args.prior_bag}", file=sys.stderr)
    params_base = existing_path_or_none(args.params_base)
    if params_base is not None:
        cmd += ["--params-base", str(params_base)]
    elif args.params_base == LEGACY_DENSE_PARAMS:
        print(f"warning: dense params not found, using repository default params: {args.params_base}", file=sys.stderr)
    else:
        raise SystemExit(f"params file not found: {args.params_base}")
    if not args.no_auto_planes:
        cmd += [
            "--auto-reflective-planes",
            "--auto-plane-z-min",
            "-0.3",
            "--auto-plane-z-max",
            "2.2",
            "--auto-plane-crop-radius",
            "2.3",
            "--auto-plane-compactness-weight",
            "180000",
        ]
    cmd += extra_args

    print("+", " ".join(cmd))
    if args.dry_run:
        return 0
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

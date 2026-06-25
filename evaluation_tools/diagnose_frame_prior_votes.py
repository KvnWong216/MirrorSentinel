#!/usr/bin/env python3
"""Diagnose frame-prior vote coverage per annotated reflective wall."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "evaluation_tools") not in sys.path:
    sys.path.append(str(REPO_ROOT / "evaluation_tools"))

from map_ghost_eval import load_annotations, points_in_roi, read_point_cloud  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--votes", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--behind-threshold", type=float, default=0.05)
    return parser


def safe_ratio(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def main() -> int:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    points = read_point_cloud(args.map)
    votes = np.load(args.votes)
    reject = np.asarray(votes["reject"], dtype=bool)
    observations = np.asarray(votes["observations"])
    mask_hits = np.asarray(votes["mask_hits"])
    depth_checks = np.asarray(votes["depth_checks"])
    ghost_votes = np.asarray(votes["ghost_votes"])
    ghost_ratio = np.asarray(votes["ghost_ratio"])
    if reject.shape[0] != points.shape[0]:
        raise ValueError(f"vote/map size mismatch: {reject.shape[0]} vs {points.shape[0]}")

    _, planes, _ = load_annotations(args.annotation, [args.behind_threshold, 0.10, 0.20])
    rows: List[Dict[str, object]] = []
    for plane in planes:
        roi = points_in_roi(points, plane.roi)
        signed = points[:, :3] @ plane.normal + plane.d
        if plane.front_side in {"positive", "+", "pos"}:
            behind = roi & (signed < -args.behind_threshold)
        elif plane.front_side in {"negative", "-", "neg"}:
            behind = roi & (signed > args.behind_threshold)
        else:
            raise ValueError(f"{plane.region_id}: bad front_side={plane.front_side}")
        row = {
            "id": plane.region_id,
            "roi_points": int(np.count_nonzero(roi)),
            "behind_points": int(np.count_nonzero(behind)),
            "observed_roi": int(np.count_nonzero(roi & (observations > 0))),
            "mask_hit_roi": int(np.count_nonzero(roi & (mask_hits > 0))),
            "depth_checked_roi": int(np.count_nonzero(roi & (depth_checks > 0))),
            "ghost_voted_roi": int(np.count_nonzero(roi & (ghost_votes > 0))),
            "rejected_roi": int(np.count_nonzero(roi & reject)),
            "observed_behind": int(np.count_nonzero(behind & (observations > 0))),
            "mask_hit_behind": int(np.count_nonzero(behind & (mask_hits > 0))),
            "depth_checked_behind": int(np.count_nonzero(behind & (depth_checks > 0))),
            "ghost_voted_behind": int(np.count_nonzero(behind & (ghost_votes > 0))),
            "rejected_behind": int(np.count_nonzero(behind & reject)),
        }
        row["rejected_roi_rate"] = safe_ratio(int(row["rejected_roi"]), int(row["roi_points"]))
        row["rejected_behind_rate"] = safe_ratio(int(row["rejected_behind"]), int(row["behind_points"]))
        row["depth_checked_behind_rate"] = safe_ratio(int(row["depth_checked_behind"]), int(row["behind_points"]))
        roi_ghost_ratio = ghost_ratio[roi & (depth_checks > 0)]
        behind_ghost_ratio = ghost_ratio[behind & (depth_checks > 0)]
        row["ghost_ratio_roi_mean"] = float(np.mean(roi_ghost_ratio)) if roi_ghost_ratio.size else None
        row["ghost_ratio_behind_mean"] = float(np.mean(behind_ghost_ratio)) if behind_ghost_ratio.size else None
        rows.append(row)

    json_path = args.out_dir / "frame_prior_vote_wall_diagnostics.json"
    csv_path = args.out_dir / "frame_prior_vote_wall_diagnostics.csv"
    md_path = args.out_dir / "frame_prior_vote_wall_diagnostics.md"
    json_path.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| wall | ROI | behind | depth_checked_behind | ghost_voted_behind | rejected_behind | rejected_behind_rate |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['id']} | {row['roi_points']} | {row['behind_points']} | "
                f"{row['depth_checked_behind']} | {row['ghost_voted_behind']} | "
                f"{row['rejected_behind']} | {row['rejected_behind_rate']:.3f} |\n"
            )
    print(f"wrote vote diagnostics to {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

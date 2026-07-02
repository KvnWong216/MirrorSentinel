#!/usr/bin/env python3
"""Summarize fair 100k-point-budget SLAM map comparisons.

This companion summary focuses on the paper-facing comparison where each method
is evaluated under the same map-point budget. It reads the already-generated
budget-normalized maps plus their ghost / reference metrics and exports a
compact CSV + Markdown table.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
BUDGET_ROOT = REPO_ROOT / "evaluation_tools/results/slam/self_collected/budget_100k"
SELF_COLLECTED_ROOT = REPO_ROOT / "evaluation_tools/results/slam/self_collected"


@dataclass(frozen=True)
class BudgetMethodSpec:
    slug: str
    method: str
    role: str
    runtime_json: Optional[Path]
    note: str = ""


METHODS: List[BudgetMethodSpec] = [
    BudgetMethodSpec(
        slug="fastlio2_equiv_raw_100k",
        method="FAST-LIO2 equiv raw 100k",
        role="baseline",
        runtime_json=SELF_COLLECTED_ROOT
        / "2026-03-30-21-31-03_fullbag_fastlio2_equiv/fast_lio2_equiv/metrics_runtime.json",
        note="raw ROS2 baseline normalized to shared point budget",
    ),
    BudgetMethodSpec(
        slug="fastlivo2_raw_100k",
        method="FAST-LIVO2 raw 100k",
        role="visual-lidar baseline",
        runtime_json=SELF_COLLECTED_ROOT
        / "2026-03-30-21-31-03_fullbag_fastlivo2_ros2/fast_livo2_ros2/metrics_runtime.json",
        note="official FAST-LIVO2 raw export normalized to shared point budget",
    ),
    BudgetMethodSpec(
        slug="mapping_with_reflection_adapted_100k",
        method="Mapping with Reflection adapted 100k",
        role="geometry baseline",
        runtime_json=None,
        note="geometry-only adapted baseline under same budget",
    ),
    BudgetMethodSpec(
        slug="mirrorsentinel_vote_clean_100k",
        method="MirrorSentinel vote-clean 100k",
        role="main method",
        runtime_json=SELF_COLLECTED_ROOT
        / "2026-03-30-21-31-03_fullbag_da3_depth_only/sentinel_rt_depth/metrics_runtime.json",
        note="current paper candidate under same budget",
    ),
]


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: Any, digits: int = 4) -> str:
    x = as_float(value)
    if x is None:
        return "-"
    return f"{x:.{digits}f}"


def load_runtime(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    data = read_json(path)
    if data is None:
        return {}
    return {
        "pipeline_fps": data.get("pipeline_cloud_fps"),
        "depth_prior_fps": data.get("depth_prior_fps"),
        "reflection_prior_fps": data.get("reflection_prior_fps"),
    }


def collect_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in METHODS:
        metrics = read_json(BUDGET_ROOT / f"{spec.slug}_metrics.json")
        ref = read_json(BUDGET_ROOT / f"{spec.slug}_ref.json")
        norm = read_json(BUDGET_ROOT / f"{spec.slug}.json")

        if metrics is None or ref is None or norm is None:
            rows.append(
                {
                    "method": spec.method,
                    "role": spec.role,
                    "status": "missing",
                    "note": spec.note,
                }
            )
            continue

        aggregate = metrics.get("aggregate", {})
        thresholds = ref.get("aggregate", {}).get("thresholds", {})
        primary = thresholds.get("0.050", {})

        row = {
            "method": spec.method,
            "role": spec.role,
            "status": "available",
            "natural_export_points": norm.get("input_points"),
            "budget_points": metrics.get("point_count"),
            "post_voxel_points": norm.get("post_voxel_points"),
            "ghost_rate": aggregate.get("ghost_rate"),
            "residual_points": aggregate.get("reflection_residual_points"),
            "valid_precision": aggregate.get("valid_structure_precision_proxy"),
            "thickness_p95_m": aggregate.get("reflective_plane_thickness_p95_m"),
            "fscore_5cm": primary.get("f_score"),
            "precision_5cm": primary.get("accuracy_precision"),
            "recall_5cm": primary.get("completeness_recall"),
            "voxel_leaf": norm.get("voxel_leaf"),
            "random_sampling_applied": norm.get("random_sampling_applied"),
            "note": spec.note,
        }
        row.update(load_runtime(spec.runtime_json))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fields = [
        "method",
        "role",
        "status",
        "natural_export_points",
        "post_voxel_points",
        "budget_points",
        "ghost_rate",
        "residual_points",
        "valid_precision",
        "thickness_p95_m",
        "fscore_5cm",
        "precision_5cm",
        "recall_5cm",
        "pipeline_fps",
        "depth_prior_fps",
        "reflection_prior_fps",
        "voxel_leaf",
        "random_sampling_applied",
        "note",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def write_md(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    lines = [
        "# Fair 100k-Point Budget Evaluation",
        "",
        "| Method | Role | Natural Points | Budget Points | RER / Ghost Rate ↓ | Residual ↓ | Valid Precision ↑ | Thickness P95 (m) ↓ | F@5cm ↑ | P@5cm ↑ | R@5cm ↑ | FPS ↑ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('method', '-')} | {row.get('role', '-')} | "
            f"{fmt(row.get('natural_export_points'), 0)} | {fmt(row.get('budget_points'), 0)} | "
            f"{fmt(row.get('ghost_rate'))} | {fmt(row.get('residual_points'), 0)} | "
            f"{fmt(row.get('valid_precision'))} | {fmt(row.get('thickness_p95_m'))} | "
            f"{fmt(row.get('fscore_5cm'))} | {fmt(row.get('precision_5cm'))} | "
            f"{fmt(row.get('recall_5cm'))} | {fmt(row.get('pipeline_fps'), 2)} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- All methods are compared after a method-agnostic normalization to a shared point budget.",
            "- The normalization policy is: auto voxel downsample first, then random sampling only if still above budget.",
            "- `Ghost Rate` here equals the reflective residual rate measured inside annotated reflective regions.",
            "- `Thickness P95` is the mirror-wall thickening indicator; lower is better for ghost suppression.",
            "- `F@5cm / P@5cm / R@5cm` come from comparison to the current self-collected reference map, so they describe reconstruction agreement rather than pure ghost rejection.",
            "- Runtime comes from each method's original run. The 100k normalization itself is offline post-processing and is not included in FPS.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize fair 100k-point-budget evaluation results.")
    parser.add_argument("--out-csv", type=Path, default=BUDGET_ROOT / "summary.csv")
    parser.add_argument("--out-md", type=Path, default=BUDGET_ROOT / "summary.md")
    args = parser.parse_args()

    rows = collect_rows()
    write_csv(args.out_csv, rows)
    write_md(args.out_md, rows)
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

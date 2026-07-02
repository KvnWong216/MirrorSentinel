#!/usr/bin/env python3
"""Summarize all currently available self-collected SLAM method results.

This script is intentionally pragmatic: it reads the result folders that already
exist in `evaluation_tools/results/slam/self_collected` and exports a unified
table for paper discussion / nightly tracking.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "evaluation_tools/results/slam/self_collected"


@dataclass(frozen=True)
class MethodSpec:
    method: str
    role: str
    result_dir: Path
    row_label: str
    mapping_json: str = "metrics_mapping_ablation.json"
    reference_json: Optional[str] = None
    runtime_json: Optional[str] = "metrics_runtime.json"
    no_gt_json: Optional[str] = "metrics_no_gt.json"
    status_override: Optional[str] = None
    note: str = ""
    paper_use: str = ""


METHODS: List[MethodSpec] = [
    MethodSpec(
        method="FAST-LIO2 equiv raw",
        role="baseline",
        result_dir=RESULTS_ROOT / "2026-03-30-21-31-03_fullbag_fastlio2_equiv/fast_lio2_equiv",
        row_label="raw",
        reference_json="metrics_reference_map_raw.json",
        note="same ROS2 code path, MirrorSentinel gates disabled",
        paper_use="yes, primary baseline",
    ),
    MethodSpec(
        method="FAST-LIVO2 ROS2 raw export",
        role="visual-lidar baseline",
        result_dir=RESULTS_ROOT / "2026-03-30-21-31-03_fullbag_fastlivo2_ros2/fast_livo2_ros2",
        row_label="raw",
        reference_json="metrics_reference_map_raw.json",
        note="official FAST-LIVO2 ROS2 export, dense raw accumulated visual-lidar map",
        paper_use="yes, visual baseline",
    ),
    MethodSpec(
        method="RTAB-Map ICP mapping",
        role="lidar mapping baseline",
        result_dir=RESULTS_ROOT / "2026-03-30-21-31-03_rtabmap_min/rtabmap_min",
        row_label="raw",
        mapping_json="metrics_mapping.json",
        reference_json="metrics_reference_map.json",
        no_gt_json=None,
        note="RTAB-Map ROS2 composable ICP/MapAssembler baseline; compact xyz Ouster input for stable runtime",
        paper_use="yes, mapping baseline",
    ),
    MethodSpec(
        method="LIO-SAM ROS1 recovered",
        role="lidar-inertial baseline",
        result_dir=RESULTS_ROOT / "2026-03-30-21-31-03_lio_sam_ros1_recovered/lio_sam",
        row_label="raw",
        mapping_json="metrics_mapping.json",
        reference_json="metrics_reference_map.json",
        no_gt_json=None,
        note=(
            "recovered historical output from /home/wangyg/Downloads/lio_sam/newseq_001; "
            "CornerMap+SurfMap combined, 221 odometry poses, timestamp matches the 2026-03-30-21-31-03 bag"
        ),
        paper_use="yes, recovered baseline with provenance note",
    ),
    MethodSpec(
        method="Mapping with Reflection adapted",
        role="geometry baseline",
        result_dir=RESULTS_ROOT / "2026-03-30-21-31-03_fullbag_mapping_with_reflection_adapted/mapping_with_reflection_adapted",
        row_label="mapping_with_reflection_adapted",
        runtime_json=None,
        no_gt_json=None,
        note="adapted SSRR2020-style geometry post-clean; not official dual-return reproduction",
        paper_use="yes, with limitation note",
    ),
    MethodSpec(
        method="MirrorSentinel DA3 depth-only raw",
        role="main ablation",
        result_dir=RESULTS_ROOT / "2026-03-30-21-31-03_fullbag_da3_depth_only/sentinel_rt_depth",
        row_label="raw",
        reference_json="metrics_reference_map_raw.json",
        note="before backend clean; shows pure online mapping output",
        paper_use="yes, ablation",
    ),
    MethodSpec(
        method="MirrorSentinel DA3 depth-only marker-clean",
        role="main ablation",
        result_dir=RESULTS_ROOT / "2026-03-30-21-31-03_fullbag_da3_depth_only/sentinel_rt_depth",
        row_label="marker_clean",
        note="marker-based cleanup output",
        paper_use="yes, ablation",
    ),
    MethodSpec(
        method="MirrorSentinel DA3 depth-only vote-clean",
        role="main method",
        result_dir=RESULTS_ROOT / "2026-03-30-21-31-03_fullbag_da3_depth_only/sentinel_rt_depth",
        row_label="vote_clean",
        reference_json="metrics_reference_map_vote_clean.json",
        note="current best paper candidate",
        paper_use="yes, main result",
    ),
    MethodSpec(
        method="Manual geometry clean upper-bound",
        role="oracle",
        result_dir=RESULTS_ROOT / "2026-03-30-21-31-03_fullbag_geometry_oracle/geometry_oracle",
        row_label="after",
        mapping_json="metrics_after.json",
        runtime_json=None,
        no_gt_json=None,
        note="manual room-boundary oracle",
        paper_use="yes, upper-bound only",
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


def load_mapping_row(spec: MethodSpec) -> Optional[Dict[str, Any]]:
    data = read_json(spec.result_dir / spec.mapping_json)
    if data is None:
        return None
    if "rows" in data:
        for row in data["rows"]:
            if isinstance(row, dict) and row.get("label") == spec.row_label:
                return row
        return None
    aggregate = data.get("aggregate", {})
    if not isinstance(aggregate, dict):
        aggregate = {}
    row = {"label": spec.row_label, "point_count": data.get("point_count")}
    row.update(aggregate)
    return row


def load_runtime(spec: MethodSpec) -> Dict[str, Any]:
    if spec.runtime_json is None:
        return {}
    data = read_json(spec.result_dir / spec.runtime_json)
    if data is None:
        return {}
    return {
        "pipeline_fps": data.get("pipeline_cloud_fps"),
        "depth_prior_fps": data.get("depth_prior_fps"),
        "reflection_prior_fps": data.get("reflection_prior_fps"),
    }


def load_reference(spec: MethodSpec) -> Dict[str, Any]:
    if spec.reference_json is None:
        return {}
    data = read_json(spec.result_dir / spec.reference_json)
    if data is None:
        return {}
    thresholds = data.get("aggregate", {}).get("thresholds", {})
    primary = thresholds.get("0.050", {})
    return {
        "f_score_005": primary.get("f_score"),
        "precision_005": primary.get("accuracy_precision"),
        "recall_005": primary.get("completeness_recall"),
    }


def load_no_gt(spec: MethodSpec) -> Dict[str, Any]:
    if spec.no_gt_json is None:
        return {}
    data = read_json(spec.result_dir / spec.no_gt_json)
    if data is None:
        return {}
    loop = data.get("loop", {})
    return {
        "loop_translation_error_m": loop.get("loop_translation_error_m"),
        "loop_rotation_error_deg": loop.get("loop_rotation_error_deg"),
        "loop_drift_ratio": loop.get("loop_drift_ratio"),
    }


def collect_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in METHODS:
        mapping = load_mapping_row(spec)
        if mapping is None:
            rows.append(
                {
                    "method": spec.method,
                    "role": spec.role,
                    "status": "missing",
                    "paper_use": spec.paper_use,
                    "note": spec.note,
                    "result_dir": str(spec.result_dir.relative_to(REPO_ROOT)),
                }
            )
            continue
        row = {
            "method": spec.method,
            "role": spec.role,
            "status": spec.status_override or "available",
            "paper_use": spec.paper_use,
            "note": spec.note,
            "result_dir": str(spec.result_dir.relative_to(REPO_ROOT)),
            "point_count": mapping.get("point_count"),
            "reflection_residual_rate": mapping.get("reflection_residual_rate"),
            "reflection_residual_points": mapping.get("reflection_residual_points"),
            "reflection_removal_rate_vs_raw": mapping.get("reflection_removal_rate_vs_raw"),
            "valid_structure_precision_proxy": mapping.get("valid_structure_precision_proxy"),
            "reflective_plane_thickness_p95_m": mapping.get("reflective_plane_thickness_p95_m"),
            "map_retention_rate_vs_raw": mapping.get("map_retention_rate_vs_raw"),
        }
        row.update(load_reference(spec))
        row.update(load_runtime(spec))
        row.update(load_no_gt(spec))
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "role",
        "status",
        "point_count",
        "reflection_residual_rate",
        "reflection_residual_points",
        "reflection_removal_rate_vs_raw",
        "valid_structure_precision_proxy",
        "reflective_plane_thickness_p95_m",
        "map_retention_rate_vs_raw",
        "f_score_005",
        "precision_005",
        "recall_005",
        "pipeline_fps",
        "depth_prior_fps",
        "reflection_prior_fps",
        "loop_translation_error_m",
        "loop_rotation_error_deg",
        "loop_drift_ratio",
        "paper_use",
        "note",
        "result_dir",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def write_md(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    lines = [
        "# Self-Collected Method Evaluation",
        "",
        "| Method | Role | Status | Points | RER ↓ | Residual ↓ | Removal ↑ | Valid Precision ↑ | Thickness P95 ↓ | Retention ↑ | F@5cm ↑ | FPS ↑ | Loop Drift ↓ | Paper Use |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('method','-')} | {row.get('role','-')} | {row.get('status','-')} | "
            f"{fmt(row.get('point_count'), 0)} | {fmt(row.get('reflection_residual_rate'))} | "
            f"{fmt(row.get('reflection_residual_points'), 0)} | {fmt(row.get('reflection_removal_rate_vs_raw'))} | "
            f"{fmt(row.get('valid_structure_precision_proxy'))} | {fmt(row.get('reflective_plane_thickness_p95_m'))} | "
            f"{fmt(row.get('map_retention_rate_vs_raw'))} | {fmt(row.get('f_score_005'))} | {fmt(row.get('pipeline_fps'), 2)} | "
            f"{fmt(row.get('loop_drift_ratio'))} | {row.get('paper_use','-')} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `RER` = reflective residual rate; lower is better.",
            "- `Thickness P95` reflects mirror-wall thickening / ghost smear; lower is better.",
            "- `Loop Drift` has no motion-capture GT, so it is only a stability proxy.",
            "- `Mapping with Reflection adapted` is a geometry-only adapted baseline, not an official dual-return reproduction.",
            "- `LIO-SAM ROS1 recovered` is a historical ROS1-era output recovered from `~/Downloads/lio_sam/newseq_001`; it is usable as a baseline only with this provenance note.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize all self-collected method results.")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=RESULTS_ROOT / "all_method_eval_summary.csv",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=RESULTS_ROOT / "all_method_eval_summary.md",
    )
    args = parser.parse_args()

    rows = collect_rows()
    write_csv(args.out_csv, rows)
    write_md(args.out_md, rows)
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

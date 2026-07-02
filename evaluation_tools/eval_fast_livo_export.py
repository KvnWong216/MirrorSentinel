#!/usr/bin/env python3
"""Evaluate an existing FAST-LIVO export with the MirrorSentinel protocol."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "evaluation_tools") not in sys.path:
    sys.path.append(str(REPO_ROOT / "evaluation_tools"))

from no_gt_trajectory_eval import sorted_poses, loop_metrics, trajectory_length  # noqa: E402


def run(cmd: List[str]) -> None:
    import subprocess

    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_mapping_metric_row(label: str, map_path: Path, metrics_path: Path) -> Dict[str, object]:
    metrics = read_json(metrics_path)
    aggregate = metrics.get("aggregate", {})
    return {
        "label": label,
        "map": str(map_path),
        "point_count": metrics.get("point_count"),
        "reflective_roi_point_count": aggregate.get("reflective_roi_point_count"),
        "behind_plane_point_count": aggregate.get("behind_plane_point_count"),
        "ghost_rate": aggregate.get("ghost_rate"),
        "reflective_plane_thickness_p95_mean_m": aggregate.get("reflective_plane_thickness_p95_mean_m"),
        "reflection_residual_points": aggregate.get("reflection_residual_points"),
        "reflection_residual_rate": aggregate.get("reflection_residual_rate"),
        "reflection_residual_density_m3": aggregate.get("reflection_residual_density_m3"),
        "valid_structure_precision_proxy": aggregate.get("valid_structure_precision_proxy"),
        "reflective_plane_thickness_p95_m": aggregate.get("reflective_plane_thickness_p95_m"),
    }


def add_ablation_relative_metrics(rows: List[Dict[str, object]]) -> None:
    raw = next((row for row in rows if row.get("label") == "raw"), None)
    if raw is None:
        return

    def as_float(row: Dict[str, object], key: str) -> Optional[float]:
        value = row.get(key)
        if isinstance(value, (float, int)):
            return float(value)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    raw_residual = as_float(raw, "reflection_residual_points")
    raw_points = as_float(raw, "point_count")
    raw_thickness = as_float(raw, "reflective_plane_thickness_p95_m")

    for row in rows:
        residual = as_float(row, "reflection_residual_points")
        points = as_float(row, "point_count")
        thickness = as_float(row, "reflective_plane_thickness_p95_m")
        row["reflection_removal_rate_vs_raw"] = (
            max(0.0, (raw_residual - residual) / raw_residual)
            if raw_residual and residual is not None
            else None
        )
        row["map_retention_rate_vs_raw"] = points / raw_points if raw_points and points is not None else None
        row["thickness_reduction_rate_vs_raw"] = (
            max(0.0, (raw_thickness - thickness) / raw_thickness)
            if raw_thickness and thickness is not None
            else None
        )


def write_mapping_ablation_summary(result_dir: Path, rows: List[Dict[str, object]]) -> None:
    add_ablation_relative_metrics(rows)
    json_path = result_dir / "metrics_mapping_ablation.json"
    csv_path = result_dir / "metrics_mapping_ablation.csv"
    md_path = result_dir / "metrics_mapping_ablation.md"
    json_path.write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fieldnames = [
        "label",
        "map",
        "point_count",
        "map_retention_rate_vs_raw",
        "reflective_roi_point_count",
        "reflection_residual_points",
        "reflection_residual_rate",
        "reflection_residual_density_m3",
        "reflection_removal_rate_vs_raw",
        "valid_structure_precision_proxy",
        "reflective_plane_thickness_p95_m",
        "thickness_reduction_rate_vs_raw",
        "behind_plane_point_count",
        "ghost_rate",
        "reflective_plane_thickness_p95_mean_m",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(
            "| label | points | retained | ROI | residual | residual_rate | removal_vs_raw | "
            "valid_precision | thickness_p95_m | map |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                "| {label} | {point_count} | {retained} | {roi} | {residual} | {residual_rate} | "
                "{removal} | {valid_precision} | {thickness} | {map} |\n".format(
                    label=row.get("label"),
                    point_count=row.get("point_count"),
                    retained=f"{row.get('map_retention_rate_vs_raw', 0):.3f}" if isinstance(row.get("map_retention_rate_vs_raw"), (float, int)) else row.get("map_retention_rate_vs_raw"),
                    roi=row.get("reflective_roi_point_count"),
                    residual=row.get("reflection_residual_points"),
                    residual_rate=f"{row.get('reflection_residual_rate', 0):.6f}" if isinstance(row.get("reflection_residual_rate"), (float, int)) else row.get("reflection_residual_rate"),
                    removal=f"{row.get('reflection_removal_rate_vs_raw', 0):.6f}" if isinstance(row.get("reflection_removal_rate_vs_raw"), (float, int)) else row.get("reflection_removal_rate_vs_raw"),
                    valid_precision=f"{row.get('valid_structure_precision_proxy', 0):.6f}" if isinstance(row.get("valid_structure_precision_proxy"), (float, int)) else row.get("valid_structure_precision_proxy"),
                    thickness=f"{row.get('reflective_plane_thickness_p95_m', 0):.6f}" if isinstance(row.get("reflective_plane_thickness_p95_m"), (float, int)) else row.get("reflective_plane_thickness_p95_m"),
                    map=row.get("map"),
                )
            )


def write_runtime_metrics(result_dir: Path, pipeline_fps: float) -> None:
    metrics = {
        "play_wall_time_s": None,
        "requested_play_rate": 1.0,
        "cloud_registered_frames": None,
        "odometry_frames": None,
        "frame_stats_frames": None,
        "pipeline_cloud_fps": pipeline_fps,
        "odometry_fps": pipeline_fps,
        "frame_stats_fps": None,
        "depth_prior_fps": 0.0,
        "reflection_prior_fps": 0.0,
    }
    (result_dir / "metrics_runtime.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (result_dir / "metrics_runtime.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def write_no_gt_metrics(result_dir: Path, trajectory: Path) -> None:
    poses = sorted_poses(trajectory)
    loop = loop_metrics(poses, None, None, None, None)
    result = {
        "trajectory": str(trajectory),
        "num_poses": len(poses),
        "duration_s": poses[-1].stamp - poses[0].stamp if len(poses) >= 2 else 0.0,
        "trajectory_length_m": trajectory_length(poses),
        "loop": loop,
    }
    (result_dir / "metrics_no_gt.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (result_dir / "metrics_no_gt.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in loop.items():
            writer.writerow({"metric": f"loop.{key}", "value": value})
        writer.writerow({"metric": "trajectory_length_m", "value": result["trajectory_length_m"]})
        writer.writerow({"metric": "num_poses", "value": result["num_poses"]})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-map", required=True, type=Path)
    parser.add_argument("--downsampled-map", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--pipeline-fps", type=float, default=19.80)
    args = parser.parse_args()

    args.result_dir.mkdir(parents=True, exist_ok=True)

    raw_json = args.result_dir / "metrics_mapping_raw.json"
    raw_csv = args.result_dir / "metrics_mapping_raw.csv"
    ds_json = args.result_dir / "metrics_mapping_downsampled.json"
    ds_csv = args.result_dir / "metrics_mapping_downsampled.csv"

    run(
        [
            sys.executable,
            str(REPO_ROOT / "evaluation_tools/map_ghost_eval.py"),
            "--map",
            str(args.raw_map),
            "--annotation",
            str(args.annotation),
            "--output-json",
            str(raw_json),
            "--output-csv",
            str(raw_csv),
        ]
    )
    run(
        [
            sys.executable,
            str(REPO_ROOT / "evaluation_tools/map_ghost_eval.py"),
            "--map",
            str(args.downsampled_map),
            "--annotation",
            str(args.annotation),
            "--output-json",
            str(ds_json),
            "--output-csv",
            str(ds_csv),
        ]
    )

    run(
        [
            sys.executable,
            str(REPO_ROOT / "evaluation_tools/reference_map_eval.py"),
            "--map",
            str(args.raw_map),
            "--reference",
            str(args.reference),
            "--output-json",
            str(args.result_dir / "metrics_reference_map_raw.json"),
            "--output-csv",
            str(args.result_dir / "metrics_reference_map_raw.csv"),
        ]
    )
    run(
        [
            sys.executable,
            str(REPO_ROOT / "evaluation_tools/reference_map_eval.py"),
            "--map",
            str(args.downsampled_map),
            "--reference",
            str(args.reference),
            "--output-json",
            str(args.result_dir / "metrics_reference_map_downsampled.json"),
            "--output-csv",
            str(args.result_dir / "metrics_reference_map_downsampled.csv"),
        ]
    )

    rows = [
        collect_mapping_metric_row("raw", args.raw_map, raw_json),
        collect_mapping_metric_row("downsampled", args.downsampled_map, ds_json),
    ]
    write_mapping_ablation_summary(args.result_dir, rows)
    write_runtime_metrics(args.result_dir, args.pipeline_fps)
    write_no_gt_metrics(args.result_dir, args.trajectory)

    print(f"FAST-LIVO export evaluation written to {args.result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

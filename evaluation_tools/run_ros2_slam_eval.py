#!/usr/bin/env python3
"""Run Sentinel-LIO on a ROS2 bag and evaluate the produced map/trajectory.

This is the one-command runner for the current mirror/glass SLAM protocol:
launch pipeline -> play bag -> record outputs -> collect map -> export TUM ->
run no-GT and map ghost/thickness evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
WS_ROOT = REPO_ROOT / "Sentinal-LIO_ws"
DEFAULT_PARAMS = WS_ROOT / "src/sentinel_lio/config/sentinel_ouster.yaml"
FIXED_PCD = WS_ROOT / "src/sentinel_lio/PCD/scans.pcd"


METHOD_OVERRIDES: Dict[str, Dict[str, object]] = {
    "sentinel_full": {
        "sentinel.explicit_mask_enable": True,
        "sentinel.use_depth_consistency": True,
        "sentinel.global_depth_prior_enable": False,
        "sentinel.frontend_confidence_enable": False,
        "sentinel.rgb_subscribe_enable": False,
        "sentinel.depth_calibration_enable": True,
        "sentinel.depth_calibration_use_non_mask": True,
    },
    "sentinel_rt_depth": {
        "sentinel.explicit_mask_enable": False,
        "sentinel.use_depth_consistency": True,
        "sentinel.global_depth_prior_enable": True,
        "sentinel.frontend_confidence_enable": True,
        "sentinel.rgb_subscribe_enable": False,
        "sentinel.depth_calibration_enable": True,
        "sentinel.depth_calibration_use_non_mask": False,
    },
    "sentinel_rt_depth_soft": {
        "sentinel.explicit_mask_enable": False,
        "sentinel.use_depth_consistency": True,
        "sentinel.global_depth_prior_enable": True,
        "sentinel.frontend_confidence_enable": True,
        "sentinel.rgb_subscribe_enable": False,
        "sentinel.depth_calibration_enable": True,
        "sentinel.depth_calibration_use_non_mask": False,
        "sentinel.map_depth_gate_enable": False,
        "sentinel.map_mask_gate_enable": False,
    },
    "fast_lio2_equiv": {
        "sentinel.explicit_mask_enable": False,
        "sentinel.use_depth_consistency": False,
        "sentinel.confidence_floor": 1.0,
        "sentinel.global_depth_prior_enable": False,
        "sentinel.frontend_confidence_enable": False,
        "sentinel.rgb_subscribe_enable": False,
        "sentinel.depth_calibration_enable": False,
        "sentinel.map_depth_gate_enable": False,
        "sentinel.map_mask_gate_enable": False,
    },
    "sentinel_no_mask": {
        "sentinel.explicit_mask_enable": False,
        "sentinel.use_depth_consistency": True,
        "sentinel.global_depth_prior_enable": True,
        "sentinel.frontend_confidence_enable": True,
        "sentinel.rgb_subscribe_enable": False,
        "sentinel.depth_calibration_enable": True,
        "sentinel.depth_calibration_use_non_mask": False,
    },
    "sentinel_no_depth": {
        "sentinel.explicit_mask_enable": True,
        "sentinel.use_depth_consistency": False,
        "sentinel.global_depth_prior_enable": False,
        "sentinel.frontend_confidence_enable": False,
        "sentinel.rgb_subscribe_enable": False,
        "sentinel.depth_calibration_enable": False,
    },
    "sentinel_hard_reject": {
        "sentinel.explicit_mask_enable": True,
        "sentinel.use_depth_consistency": True,
        "sentinel.global_depth_prior_enable": False,
        "sentinel.frontend_confidence_enable": True,
        "sentinel.rgb_subscribe_enable": False,
        "sentinel.depth_calibration_enable": True,
        "sentinel.depth_calibration_use_non_mask": True,
        "sentinel.confidence_floor": 0.0,
        "sentinel.reflective_invalid_depth_confidence": 0.0,
        "sentinel.mirror_surface_confidence": 0.0,
        "sentinel.mask_boundary_confidence": 0.0,
    },
}


def run(cmd: List[str], *, cwd: Path = REPO_ROOT, env: Optional[Dict[str, str]] = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected YAML mapping")
    return data


def dump_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def set_param(params: dict, key: str, value: object) -> None:
    ros_params = params.setdefault("laserMapping", {}).setdefault("ros__parameters", {})
    ros_params[key] = value


def make_params_file(base: Path, out: Path, method: str) -> Path:
    params = load_yaml(base)
    set_param(params, "pcd_save.pcd_save_en", True)
    set_param(params, "pcd_save.interval", -1)
    for key, value in METHOD_OVERRIDES[method].items():
        set_param(params, key, value)
    dump_yaml(out, params)
    return out


def default_annotation_for(args: argparse.Namespace) -> Path:
    annotation_dir = REPO_ROOT / "evaluation_tools/data/annotations/self_collected"
    sequence_annotation = annotation_dir / f"{args.sequence}.yaml"
    if sequence_annotation.exists():
        return sequence_annotation

    bag_annotation = annotation_dir / f"{args.bag.name}.yaml"
    if bag_annotation.exists():
        return bag_annotation

    return sequence_annotation


def shell_env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("ROS_LOG_DIR", "/tmp/yugong_ros2_log")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def ros_shell(command: str, env: Dict[str, str]) -> subprocess.Popen:
    setup = (
        "set -e; "
        "source /opt/ros/humble/setup.bash; "
        f"source {WS_ROOT}/install/setup.bash; "
    )
    return subprocess.Popen(
        ["bash", "-lc", setup + command],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )


def terminate_process(proc: subprocess.Popen, name: str, timeout_s: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    print(f"stopping {name} pid={proc.pid}")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()


def write_process_log(proc: subprocess.Popen, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output = proc.stdout.read() if proc.stdout else ""
    log_path.write_text(output, encoding="utf-8", errors="replace")
    return proc.poll() if proc.poll() is not None else -1


def wait_for_bag_play(proc: subprocess.Popen, timeout_s: int) -> int:
    start = time.time()
    while proc.poll() is None:
        if time.time() - start > timeout_s:
            print(f"bag play timeout after {timeout_s}s")
            terminate_process(proc, "bag_play")
            break
        time.sleep(1.0)
    return proc.poll() if proc.poll() is not None else -1


def safe_unlink_or_backup(path: Path, backup_dir: Path) -> None:
    if not path.exists():
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = backup_dir / f"{path.name}.{int(time.time())}.bak"
    shutil.move(str(path), str(backup))
    print(f"moved old {path} -> {backup}")


def run_map_eval(map_path: Path, annotation: Path, output_json: Path, output_csv: Path, env: Dict[str, str]) -> bool:
    try:
        run(
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
            env=env,
        )
        return True
    except subprocess.CalledProcessError as exc:
        print(f"warning: map ghost eval failed for {map_path}: {exc}")
        return False


def collect_mapping_metric_row(label: str, map_path: Path, metrics_path: Path) -> Optional[Dict[str, object]]:
    if not metrics_path.exists():
        return None
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: failed to read mapping metrics {metrics_path}: {exc}")
        return None
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
    if raw_residual is None:
        raw_residual = as_float(raw, "behind_plane_point_count")
    raw_points = as_float(raw, "point_count")
    raw_thickness = as_float(raw, "reflective_plane_thickness_p95_m")
    if raw_thickness is None:
        raw_thickness = as_float(raw, "reflective_plane_thickness_p95_mean_m")

    for row in rows:
        residual = as_float(row, "reflection_residual_points")
        if residual is None:
            residual = as_float(row, "behind_plane_point_count")
        points = as_float(row, "point_count")
        thickness = as_float(row, "reflective_plane_thickness_p95_m")
        if thickness is None:
            thickness = as_float(row, "reflective_plane_thickness_p95_mean_m")

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
    if not rows:
        return
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
        "reflection_removal_rate_vs_raw",
        "valid_structure_precision_proxy",
        "reflective_plane_thickness_p95_m",
        "thickness_reduction_rate_vs_raw",
        "behind_plane_point_count",
        "ghost_rate",
        "reflective_plane_thickness_p95_mean_m",
    ]
    for key in sorted({key for row in rows for key in row.keys()}):
        if key not in fieldnames:
            fieldnames.append(key)
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
            retained = row.get("map_retention_rate_vs_raw")
            residual_rate = row.get("reflection_residual_rate", row.get("ghost_rate"))
            removal = row.get("reflection_removal_rate_vs_raw")
            valid_precision = row.get("valid_structure_precision_proxy")
            thickness = row.get("reflective_plane_thickness_p95_m", row.get("reflective_plane_thickness_p95_mean_m"))
            f.write(
                "| {label} | {point_count} | {retained} | {roi} | {residual} | {residual_rate} | "
                "{removal} | {valid_precision} | {thickness} | {map} |\n".format(
                    label=row.get("label"),
                    point_count=row.get("point_count"),
                    retained=f"{retained:.3f}" if isinstance(retained, (float, int)) else retained,
                    roi=row.get("reflective_roi_point_count"),
                    residual=row.get("reflection_residual_points", row.get("behind_plane_point_count")),
                    residual_rate=f"{residual_rate:.6f}" if isinstance(residual_rate, (float, int)) else residual_rate,
                    removal=f"{removal:.6f}" if isinstance(removal, (float, int)) else removal,
                    valid_precision=f"{valid_precision:.6f}" if isinstance(valid_precision, (float, int)) else valid_precision,
                    thickness=f"{thickness:.6f}" if isinstance(thickness, (float, int)) else thickness,
                    map=row.get("map"),
                )
            )
    print(f"wrote mapping ablation summary: {md_path}")


def write_runtime_metrics(
    result_dir: Path,
    *,
    topic_counts: Dict[str, int],
    play_wall_time_s: float,
    play_rate: float,
) -> Dict[str, object]:
    def fps(topic: str) -> float:
        return float(topic_counts.get(topic, 0)) / play_wall_time_s if play_wall_time_s > 0.0 else 0.0

    metrics: Dict[str, object] = {
        "play_wall_time_s": float(play_wall_time_s),
        "requested_play_rate": float(play_rate),
        "cloud_registered_frames": int(topic_counts.get("/cloud_registered", 0)),
        "odometry_frames": int(topic_counts.get("/Odometry", 0)),
        "frame_stats_frames": int(topic_counts.get("/mirror_sentinel/frame_stats", 0)),
        "pipeline_cloud_fps": fps("/cloud_registered"),
        "odometry_fps": fps("/Odometry"),
        "frame_stats_fps": fps("/mirror_sentinel/frame_stats"),
        "depth_prior_fps": fps("/vfm/depth_image"),
        "reflection_prior_fps": fps("/vfm/mirror_mask"),
    }
    json_path = result_dir / "metrics_runtime.json"
    csv_path = result_dir / "metrics_runtime.csv"
    json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    print(
        "Runtime/FPS: "
        f"cloud={metrics['pipeline_cloud_fps']:.3f} Hz, "
        f"odom={metrics['odometry_fps']:.3f} Hz, "
        f"wall={play_wall_time_s:.3f}s"
    )
    return metrics


def count_rosbag2_topics(bag_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for db_path in sorted(bag_dir.glob("*.db3")):
        con = sqlite3.connect(str(db_path))
        try:
            rows = con.execute(
                """
                SELECT topics.name, COUNT(messages.id)
                FROM messages
                JOIN topics ON messages.topic_id = topics.id
                GROUP BY topics.name
                """
            ).fetchall()
        finally:
            con.close()
        for name, count in rows:
            counts[name] = counts.get(name, 0) + int(count)
    return counts


def validate_topic_counts(args: argparse.Namespace, counts: Dict[str, int]) -> List[str]:
    warnings: List[str] = []
    if counts.get("/Odometry", 0) == 0:
        warnings.append("missing /Odometry; SLAM did not publish trajectory")
    if args.run_da3_depth or args.prior_bag:
        if counts.get("/vfm/depth_image", 0) == 0:
            warnings.append("missing /vfm/depth_image; requested depth publisher did not produce output")
    if args.run_reflection_mask or args.run_mask_prior or args.prior_bag:
        if counts.get("/vfm/mirror_mask", 0) == 0:
            warnings.append("missing /vfm/mirror_mask; requested mask publisher did not produce output")
    if args.method in ("sentinel_full", "sentinel_no_depth", "sentinel_hard_reject"):
        if not (args.run_reflection_mask or args.run_mask_prior or args.prior_bag):
            warnings.append(
                f"{args.method} expects a real /vfm/mirror_mask; no mask publisher was enabled"
            )
        elif counts.get("/vfm/mirror_mask", 0) == 0:
            warnings.append(f"{args.method} did not receive /vfm/mirror_mask")
    return warnings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument(
        "--prior-bag",
        type=Path,
        default=None,
        help="optional ROS2 bag that replays precomputed /vfm/depth_image and /vfm/mirror_mask",
    )
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--method", choices=sorted(METHOD_OVERRIDES), default="sentinel_full")
    parser.add_argument("--annotation", type=Path, default=None)
    parser.add_argument("--params-base", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "evaluation_tools/results/slam/self_collected")
    parser.add_argument("--play-rate", type=float, default=1.0)
    parser.add_argument("--startup-wait", type=float, default=5.0)
    parser.add_argument("--shutdown-wait", type=float, default=8.0)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--rviz", action="store_true", help="open the Sentinel-LIO RViz mapping panel")
    parser.add_argument("--rviz-config", type=Path, default=None, help="optional custom RViz config")
    parser.add_argument("--skip-map-eval", action="store_true")
    parser.add_argument("--skip-no-gt-eval", action="store_true")
    parser.add_argument("--vfm-target-fps", type=float, default=10.0)
    parser.add_argument("--vfm-stats-log-interval", type=float, default=5.0)
    parser.add_argument("--run-da3-depth", action="store_true", help="enable monocular DA3 depth publisher")
    parser.add_argument("--run-reflection-mask", action="store_true", help="enable learned RGB/DA3 reflection mask publisher")
    parser.add_argument("--da3-backend", default="none", choices=["none", "heuristic", "module", "torchscript", "onnx"])
    parser.add_argument(
        "--da3-model",
        default="",
        help=(
            "module backend spec, e.g. "
            "depth_anything_3.api:DepthAnything3 for official Depth Anything 3"
        ),
    )
    parser.add_argument(
        "--da3-checkpoint",
        default="",
        help=(
            "checkpoint/model id. For official DA3 module backend use a HuggingFace id such as "
            "depth-anything/DA3Metric-Large"
        ),
    )
    parser.add_argument("--da3-input-width", type=int, default=518)
    parser.add_argument("--da3-input-height", type=int, default=518)
    parser.add_argument("--da3-metric-scale", type=float, default=1.0)
    parser.add_argument("--da3-metric-shift", type=float, default=0.0)
    parser.add_argument("--reflection-mask-backend", default="torch", choices=["torch", "model", "heuristic"])
    parser.add_argument("--reflection-mask-checkpoint", default="")
    parser.add_argument("--reflection-mask-width", type=int, default=512)
    parser.add_argument("--reflection-mask-height", type=int, default=288)
    parser.add_argument("--reflection-mask-threshold", type=float, default=0.5)
    parser.add_argument("--reflection-mask-target-fps", type=float, default=10.0)
    parser.add_argument("--reflection-mask-use-da3-prior", action="store_true")
    parser.add_argument("--run-mask-prior", action="store_true", help="enable directory/zero mask prior node")
    parser.add_argument("--mask-mode", default="zeros")
    parser.add_argument("--mask-dir", default="")
    parser.add_argument("--mask-prior-target-fps", type=float, default=10.0)
    parser.add_argument("--record-debug-topics", action="store_true", help="also record mask viz/overlay debug image topics")
    parser.add_argument(
        "--frame-prior-post-clean",
        action="store_true",
        help="after the run, filter scans_raw_accumulated.pcd by frame-wise depth/mask prior votes",
    )
    parser.add_argument("--frame-prior-frame-stride", type=int, default=5)
    parser.add_argument("--frame-prior-max-pair-dt", type=float, default=0.30)
    parser.add_argument("--frame-prior-min-observations", type=int, default=1)
    parser.add_argument("--frame-prior-min-ghost-votes", type=int, default=1)
    parser.add_argument("--frame-prior-min-ghost-ratio", type=float, default=0.0)
    parser.add_argument(
        "--frame-prior-selection-mode",
        choices=["threshold", "score_budget", "annotation_score_budget"],
        default="score_budget",
    )
    parser.add_argument("--frame-prior-target-kept-points", type=int, default=100000)
    parser.add_argument(
        "--frame-prior-score-mode",
        choices=["ghost_ratio", "ghost_votes_ratio", "ghost_ratio_sqrt_votes", "ghost_ratio_log_votes"],
        default="ghost_votes_ratio",
    )
    parser.add_argument("--frame-prior-margin-abs", type=float, default=None)
    parser.add_argument("--frame-prior-margin-rel", type=float, default=None)
    parser.add_argument(
        "--auto-reflective-planes",
        action="store_true",
        help="detect reflective room planes from map_raw.pcd and use them for frame-prior post-clean",
    )
    parser.add_argument("--auto-plane-z-min", type=float, default=-0.3)
    parser.add_argument("--auto-plane-z-max", type=float, default=2.2)
    parser.add_argument("--auto-plane-crop-radius", type=float, default=2.3)
    parser.add_argument("--auto-plane-compactness-weight", type=float, default=180000.0)
    parser.add_argument(
        "--auto-plane-max-side-length",
        type=float,
        default=0.0,
        help="optional absolute upper bound for cuboid side length; <=0 disables this scale prior",
    )
    parser.add_argument("--auto-plane-min-line-support", type=int, default=120)
    parser.add_argument("--auto-plane-min-line-length", type=float, default=0.45)
    parser.add_argument("--auto-plane-min-line-z-span", type=float, default=0.4)
    parser.add_argument("--frame-prior-annotation-candidate-mode", choices=["behind", "roi"], default="behind")
    parser.add_argument(
        "--frame-prior-annotation-budget-weight",
        choices=["candidate_count", "score_sum", "uniform"],
        default="candidate_count",
    )
    parser.add_argument("--frame-prior-annotation-behind-margin", type=float, default=None)
    parser.add_argument("--frame-prior-annotation-no-fill-global", action="store_true")
    parser.add_argument("--frame-prior-no-require-mask", action="store_true")
    parser.add_argument("--frame-prior-reject-invalid-depth-in-mask", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.bag.exists():
        raise SystemExit(f"bag not found: {args.bag}")
    if args.prior_bag and not args.prior_bag.exists():
        raise SystemExit(f"prior bag not found: {args.prior_bag}")
    if args.prior_bag and (args.run_da3_depth or args.run_reflection_mask or args.run_mask_prior):
        raise SystemExit(
            "--prior-bag already publishes /vfm priors; do not combine it with DA3/reflection/mask prior publishers"
        )
    if args.run_reflection_mask and args.reflection_mask_backend in ("torch", "model") and not args.reflection_mask_checkpoint:
        raise SystemExit("--run-reflection-mask with torch/model backend requires --reflection-mask-checkpoint")
    if args.run_da3_depth and args.da3_backend == "none":
        raise SystemExit("--run-da3-depth requires --da3-backend module|torchscript|onnx|heuristic")

    result_dir = args.results_root / args.sequence / args.method
    result_dir.mkdir(parents=True, exist_ok=True)
    params_file = make_params_file(args.params_base, result_dir / "params.yaml", args.method)
    env = shell_env()
    env["ROS_LOG_DIR"] = str(result_dir / "ros_logs")
    if args.run_da3_depth or args.run_reflection_mask:
        env.setdefault("SENTINEL_LIO_CONDA_SH", str(Path.home() / "anaconda3/etc/profile.d/conda.sh"))
        env.setdefault("SENTINEL_LIO_CONDA_ENV", "MirrSLAM_rtdepth")
    if args.run_da3_depth:
        env.setdefault("HF_ENDPOINT", "https://huggingface.co")

    safe_unlink_or_backup(FIXED_PCD, result_dir / "old_pcd_backups")

    output_bag = result_dir / "outputs_bag"
    if output_bag.exists():
        shutil.rmtree(output_bag)

    launch_args = [
        "ros2 launch sentinel_lio run_sentinel_ouster.launch.py",
        f"params_file:={params_file}",
        f"rviz:={'true' if args.rviz else 'false'}",
        f"enable_da3_depth:={'true' if args.run_da3_depth else 'false'}",
        f"enable_reflection_mask:={'true' if args.run_reflection_mask else 'false'}",
        f"mask_prior_mode:={args.mask_mode}",
        f"mask_prior_target_fps:={args.mask_prior_target_fps}",
        f"vfm_target_fps:={args.vfm_target_fps}",
        f"vfm_stats_log_interval:={args.vfm_stats_log_interval}",
        f"da3_backend:={args.da3_backend}",
        f"da3_input_width:={args.da3_input_width}",
        f"da3_input_height:={args.da3_input_height}",
        f"da3_metric_scale:={args.da3_metric_scale}",
        f"da3_metric_shift:={args.da3_metric_shift}",
        f"reflection_mask_backend:={args.reflection_mask_backend}",
        f"reflection_mask_width:={args.reflection_mask_width}",
        f"reflection_mask_height:={args.reflection_mask_height}",
        f"reflection_mask_threshold:={args.reflection_mask_threshold}",
        f"reflection_mask_target_fps:={args.reflection_mask_target_fps}",
        f"reflection_mask_use_da3_prior:={'true' if args.reflection_mask_use_da3_prior else 'false'}",
        f"enable_mask_prior:={'true' if args.run_mask_prior else 'false'}",
    ]
    if args.da3_model:
        launch_args.append(f"da3_model:={args.da3_model}")
    if args.da3_checkpoint:
        launch_args.append(f"da3_checkpoint:={args.da3_checkpoint}")
    if args.reflection_mask_checkpoint:
        launch_args.append(f"reflection_mask_checkpoint:={args.reflection_mask_checkpoint}")
    if args.mask_dir:
        launch_args.append(f"mask_dir:={args.mask_dir}")
    if args.rviz_config:
        launch_args.append(f"rviz_config:={args.rviz_config}")
    launch_cmd = " ".join(str(item) for item in launch_args)
    record_topics = [
        "/Odometry",
        "/path",
        "/cloud_registered",
        "/mirror_sentinel/frame_stats",
        "/vfm/depth_image",
        "/vfm/mirror_mask",
    ]
    if args.record_debug_topics:
        record_topics.extend(["/mirror_sentinel/mask_viz", "/mirror_sentinel/mask_overlay"])
    record_cmd = f"ros2 bag record -o {output_bag} " + " ".join(record_topics)
    play_cmd = (
        f"ros2 bag play {args.bag} --clock --rate {args.play_rate} "
        "--read-ahead-queue-size 100"
    )
    prior_play_cmd = None
    if args.prior_bag:
        prior_play_cmd = (
            f"ros2 bag play {args.prior_bag} --rate {args.play_rate} "
            "--topics /vfm/depth_image /vfm/mirror_mask "
            "--read-ahead-queue-size 100 --disable-keyboard-controls"
        )

    launch_proc = ros_shell(launch_cmd, env)
    time.sleep(args.startup_wait)
    record_proc = ros_shell(record_cmd, env)
    time.sleep(2.0)
    prior_play_proc = ros_shell(prior_play_cmd, env) if prior_play_cmd else None
    play_start_time = time.time()
    play_proc = ros_shell(play_cmd, env)
    play_code = wait_for_bag_play(play_proc, args.timeout)
    play_wall_time_s = max(0.0, time.time() - play_start_time)
    prior_play_code = None
    if prior_play_proc:
        prior_play_code = wait_for_bag_play(prior_play_proc, max(1, args.timeout))
    time.sleep(args.shutdown_wait)

    terminate_process(record_proc, "output_record")
    terminate_process(launch_proc, "sentinel_launch")
    if prior_play_proc:
        terminate_process(prior_play_proc, "prior_bag_play")

    write_process_log(play_proc, result_dir / "bag_play.log")
    if prior_play_proc:
        write_process_log(prior_play_proc, result_dir / "prior_bag_play.log")
    write_process_log(record_proc, result_dir / "output_record.log")
    write_process_log(launch_proc, result_dir / "sentinel_launch.log")

    map_path = result_dir / "map.pcd"
    map_marker_clean_path = result_dir / "map_marker_clean.pcd"
    map_raw_path = result_dir / "map_raw.pcd"
    map_vote_clean_path = result_dir / "map_vote_clean.pcd"
    if FIXED_PCD.exists():
        shutil.copy2(FIXED_PCD, map_path)
        print(f"copied map {FIXED_PCD} -> {map_path}")
        shutil.copy2(FIXED_PCD, map_marker_clean_path)
        print(f"copied marker-clean map {FIXED_PCD} -> {map_marker_clean_path}")
    else:
        print(f"warning: expected PCD not found: {FIXED_PCD}")
    for debug_name in (
        "scans_raw_accumulated.pcd",
        "history_deleted_markers.pcd",
        "history_anchor_markers.pcd",
        "history_surface_anchors.pcd",
    ):
        debug_src = FIXED_PCD.parent / debug_name
        if debug_src.exists():
            debug_dst = result_dir / debug_name
            shutil.copy2(debug_src, debug_dst)
            print(f"copied debug map {debug_src} -> {debug_dst}")
            if debug_name == "scans_raw_accumulated.pcd":
                shutil.copy2(debug_src, map_raw_path)
                print(f"copied raw accumulated map {debug_src} -> {map_raw_path}")

    if not map_raw_path.exists() and map_marker_clean_path.exists():
        shutil.copy2(map_marker_clean_path, map_raw_path)
        print(f"warning: raw accumulated map missing; copied marker-clean map as fallback -> {map_raw_path}")

    runtime_metrics: Dict[str, object] = {}
    if output_bag.exists():
        topic_counts = count_rosbag2_topics(output_bag)
        runtime_metrics = write_runtime_metrics(
            result_dir,
            topic_counts=topic_counts,
            play_wall_time_s=play_wall_time_s,
            play_rate=args.play_rate,
        )
        topic_warnings = validate_topic_counts(args, topic_counts)
        for warning in topic_warnings:
            print(f"warning: {warning}")
        try:
            run(
                [
                    sys.executable,
                    str(REPO_ROOT / "evaluation_tools/ros2_bag_to_tum.py"),
                    str(output_bag),
                    "--topic",
                    "/Odometry",
                    "--out",
                    str(result_dir / "trajectory_tum.txt"),
                ],
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            print(f"warning: failed to export /Odometry to TUM: {exc}")

    if not args.skip_no_gt_eval and (result_dir / "trajectory_tum.txt").exists():
        try:
            run(
                [
                    sys.executable,
                    str(REPO_ROOT / "evaluation_tools/no_gt_trajectory_eval.py"),
                    "--trajectory",
                    str(result_dir / "trajectory_tum.txt"),
                    "--output-json",
                    str(result_dir / "metrics_no_gt.json"),
                    "--output-csv",
                    str(result_dir / "metrics_no_gt.csv"),
                ],
                env=env,
            )
        except subprocess.CalledProcessError as exc:
            print(f"warning: no-GT trajectory eval failed: {exc}")

    frame_stats_warnings: List[str] = []
    if output_bag.exists() and topic_counts.get("/mirror_sentinel/frame_stats", 0) > 0:
        frame_stats_json = result_dir / "metrics_frame_stats.json"
        try:
            run(
                [
                    sys.executable,
                    str(REPO_ROOT / "evaluation_tools/summarize_frame_stats.py"),
                    str(output_bag),
                    "--output-json",
                    str(frame_stats_json),
                    "--output-csv",
                    str(result_dir / "metrics_frame_stats.csv"),
                ],
                env=env,
            )
            try:
                frame_stats = json.loads(frame_stats_json.read_text(encoding="utf-8"))
                frame_stats_warnings = list(frame_stats.get("warnings", []))
                for warning in frame_stats_warnings:
                    print(f"warning: frame_stats: {warning}")
            except (OSError, json.JSONDecodeError) as exc:
                print(f"warning: failed to read frame stats summary: {exc}")
        except subprocess.CalledProcessError as exc:
            print(f"warning: frame stats summary failed: {exc}")

    annotation = args.annotation
    if annotation is None:
        annotation = default_annotation_for(args)
    ablation_rows: List[Dict[str, object]] = []
    if not args.skip_map_eval and map_path.exists() and annotation.exists():
        run_map_eval(map_path, annotation, result_dir / "metrics_mapping.json", result_dir / "metrics_mapping.csv", env)
    elif not args.skip_map_eval:
        print(f"warning: skip map eval; map exists={map_path.exists()} annotation exists={annotation.exists()}")

    named_map_evals = [
        ("raw", map_raw_path, result_dir / "metrics_mapping_raw.json", result_dir / "metrics_mapping_raw.csv"),
        (
            "marker_clean",
            map_marker_clean_path,
            result_dir / "metrics_mapping_marker_clean.json",
            result_dir / "metrics_mapping_marker_clean.csv",
        ),
    ]
    if not args.skip_map_eval and annotation.exists():
        for label, named_map_path, output_json, output_csv in named_map_evals:
            if not named_map_path.exists():
                print(f"warning: skip {label} map eval; missing {named_map_path}")
                continue
            if run_map_eval(named_map_path, annotation, output_json, output_csv, env):
                row = collect_mapping_metric_row(label, named_map_path, output_json)
                if row:
                    ablation_rows.append(row)

    frame_prior_post_clean_dir = None
    raw_accumulated_map = map_raw_path
    frame_prior_annotation = annotation
    auto_plane_dir = None
    auto_plane_annotation = None
    if args.frame_prior_post_clean:
        if not output_bag.exists():
            print("warning: skip frame-prior post clean; output bag missing")
        elif not raw_accumulated_map.exists():
            print(f"warning: skip frame-prior post clean; raw accumulated map missing: {raw_accumulated_map}")
        else:
            if args.auto_reflective_planes:
                auto_plane_dir = result_dir / "auto_plane_detection"
                auto_plane_annotation = result_dir / "auto_reflective_planes.yaml"
                auto_cmd = [
                    sys.executable,
                    str(REPO_ROOT / "evaluation_tools/detect_reflective_planes.py"),
                    "--map",
                    str(raw_accumulated_map),
                    "--bag",
                    str(output_bag),
                    "--out-yaml",
                    str(auto_plane_annotation),
                    "--out-dir",
                    str(auto_plane_dir),
                    "--sequence",
                    args.bag.name,
                    "--bag-name",
                    str(args.bag),
                    "--map-frame",
                    "camera_init",
                    "--z-min",
                    str(args.auto_plane_z_min),
                    "--z-max",
                    str(args.auto_plane_z_max),
                    "--auto-crop-radius",
                    str(args.auto_plane_crop_radius),
                    "--compactness-weight",
                    str(args.auto_plane_compactness_weight),
                    "--min-line-support",
                    str(args.auto_plane_min_line_support),
                    "--min-line-length",
                    str(args.auto_plane_min_line_length),
                    "--min-line-z-span",
                    str(args.auto_plane_min_line_z_span),
                ]
                if args.auto_plane_max_side_length > 0.0:
                    auto_cmd += ["--max-side-length", str(args.auto_plane_max_side_length)]
                try:
                    run(auto_cmd, env=env)
                    frame_prior_annotation = auto_plane_annotation
                except subprocess.CalledProcessError as exc:
                    print(f"warning: auto reflective plane detection failed; using eval annotation for post-clean: {exc}")
            frame_prior_post_clean_dir = result_dir / "frame_prior_post_clean"
            cmd = [
                sys.executable,
                str(REPO_ROOT / "evaluation_tools/filter_map_by_frame_prior_votes.py"),
                "--map",
                str(raw_accumulated_map),
                "--bag",
                str(output_bag),
                "--params",
                str(params_file),
                "--out-dir",
                str(frame_prior_post_clean_dir),
                "--frame-stride",
                str(args.frame_prior_frame_stride),
                "--max-pair-dt",
                str(args.frame_prior_max_pair_dt),
                "--min-observations",
                str(args.frame_prior_min_observations),
                "--min-ghost-votes",
                str(args.frame_prior_min_ghost_votes),
                "--min-ghost-ratio",
                str(args.frame_prior_min_ghost_ratio),
                "--selection-mode",
                args.frame_prior_selection_mode,
                "--score-mode",
                args.frame_prior_score_mode,
            ]
            if args.frame_prior_target_kept_points is not None:
                cmd += ["--target-kept-points", str(args.frame_prior_target_kept_points)]
            if args.frame_prior_margin_abs is not None:
                cmd += ["--margin-abs", str(args.frame_prior_margin_abs)]
            if args.frame_prior_margin_rel is not None:
                cmd += ["--margin-rel", str(args.frame_prior_margin_rel)]
            cmd += [
                "--annotation-candidate-mode",
                args.frame_prior_annotation_candidate_mode,
                "--annotation-budget-weight",
                args.frame_prior_annotation_budget_weight,
            ]
            if args.frame_prior_annotation_behind_margin is not None:
                cmd += ["--annotation-behind-margin", str(args.frame_prior_annotation_behind_margin)]
            if args.frame_prior_annotation_no_fill_global:
                cmd += ["--annotation-no-fill-global"]
            if args.frame_prior_no_require_mask:
                cmd += ["--no-require-mask"]
            if args.frame_prior_reject_invalid_depth_in_mask:
                cmd += ["--reject-invalid-depth-in-mask"]
            if frame_prior_annotation.exists() and (
                not args.skip_map_eval or args.frame_prior_selection_mode == "annotation_score_budget"
            ):
                cmd += ["--annotation", str(frame_prior_annotation)]
            try:
                run(cmd, env=env)
                filtered_vote_map = frame_prior_post_clean_dir / "filtered_map.pcd"
                if filtered_vote_map.exists():
                    shutil.copy2(filtered_vote_map, map_vote_clean_path)
                    print(f"copied vote-clean map {filtered_vote_map} -> {map_vote_clean_path}")
                    vote_metrics = frame_prior_post_clean_dir / "metrics_mapping.json"
                    vote_metrics_csv = frame_prior_post_clean_dir / "metrics_mapping.csv"
                    if not args.skip_map_eval and annotation.exists():
                        # Always evaluate the final vote-clean map with the
                        # external protocol annotation, even when automatic
                        # planes were used as the cleaning prior.
                        external_vote_json = result_dir / "metrics_mapping_vote_clean.json"
                        external_vote_csv = result_dir / "metrics_mapping_vote_clean.csv"
                        if run_map_eval(map_vote_clean_path, annotation, external_vote_json, external_vote_csv, env):
                            row = collect_mapping_metric_row("vote_clean", map_vote_clean_path, external_vote_json)
                            if row:
                                ablation_rows.append(row)
                    elif not args.skip_map_eval and vote_metrics.exists():
                        shutil.copy2(vote_metrics, result_dir / "metrics_mapping_vote_clean.json")
                        if vote_metrics_csv.exists():
                            shutil.copy2(vote_metrics_csv, result_dir / "metrics_mapping_vote_clean.csv")
                        row = collect_mapping_metric_row("vote_clean", map_vote_clean_path, vote_metrics)
                        if row:
                            ablation_rows.append(row)
            except subprocess.CalledProcessError as exc:
                print(f"warning: frame-prior post clean failed: {exc}")

    write_mapping_ablation_summary(result_dir, ablation_rows)

    metadata = {
        "bag": str(args.bag),
        "prior_bag": str(args.prior_bag) if args.prior_bag else None,
        "sequence": args.sequence,
        "method": args.method,
        "params": str(params_file),
        "output_bag": str(output_bag),
        "map": str(map_path) if map_path.exists() else None,
        "map_raw": str(map_raw_path) if map_raw_path.exists() else None,
        "map_marker_clean": str(map_marker_clean_path) if map_marker_clean_path.exists() else None,
        "map_vote_clean": str(map_vote_clean_path) if map_vote_clean_path.exists() else None,
        "annotation": str(annotation) if annotation else None,
        "frame_prior_annotation": str(frame_prior_annotation) if frame_prior_annotation else None,
        "auto_reflective_planes": str(auto_plane_annotation) if auto_plane_annotation else None,
        "auto_plane_detection": str(auto_plane_dir) if auto_plane_dir else None,
        "play_return_code": play_code,
        "prior_play_return_code": prior_play_code,
        "topic_counts": topic_counts if output_bag.exists() else {},
        "topic_warnings": (topic_warnings + frame_stats_warnings) if output_bag.exists() else [],
        "runtime_metrics": runtime_metrics,
        "frame_prior_post_clean": str(frame_prior_post_clean_dir) if frame_prior_post_clean_dir else None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (result_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"done: {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

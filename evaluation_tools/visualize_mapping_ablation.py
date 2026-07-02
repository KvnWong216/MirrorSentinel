#!/usr/bin/env python3
"""Create paper quicklook figures for raw/marker/vote map ablations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "evaluation_tools") not in sys.path:
    sys.path.append(str(REPO_ROOT / "evaluation_tools"))

from map_ghost_eval import load_annotations, points_in_roi, read_point_cloud  # noqa: E402


COLORS = {
    "raw": "#4a5568",
    "marker_clean": "#2b6cb0",
    "vote_clean": "#2f855a",
    "rejected": "#c53030",
    "walls": "#d69e2e",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--sample", type=int, default=120000, help="max points per cloud in scatter plots")
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--rejected", type=Path, default=None, help="optional rejected point PCD")
    parser.add_argument("--xlim", type=float, nargs=2, default=None)
    parser.add_argument("--ylim", type=float, nargs=2, default=None)
    return parser


def sample_points(points: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if points.shape[0] <= limit:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=limit, replace=False)
    return points[idx]


def load_metrics(result_dir: Path, label: str) -> Optional[Dict[str, float]]:
    candidates = {
        "raw": [result_dir / "metrics_mapping_raw.json", result_dir / "raw_accumulated_metrics.json"],
        "marker_clean": [result_dir / "metrics_mapping_marker_clean.json", result_dir / "metrics_mapping.json"],
        "vote_clean": [
            result_dir / "metrics_mapping_vote_clean.json",
            result_dir / "frame_prior_post_clean/metrics_mapping.json",
        ],
    }[label]
    for path in candidates:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            aggregate = data.get("aggregate", {})
            return {
                "point_count": float(data.get("point_count", 0.0)),
                "roi": float(aggregate.get("reflective_roi_point_count", 0.0)),
                "residual": float(
                    aggregate.get("reflection_residual_points", aggregate.get("behind_plane_point_count", 0.0))
                ),
                "residual_rate": float(
                    aggregate.get("reflection_residual_rate", aggregate.get("ghost_rate", 0.0))
                ),
                "valid_precision": float(aggregate.get("valid_structure_precision_proxy", 0.0)),
                "thickness": float(
                    aggregate.get(
                        "reflective_plane_thickness_p95_m",
                        aggregate.get("reflective_plane_thickness_p95_mean_m", 0.0),
                    )
                ),
            }
    return None


def ensure_metrics(result_dir: Path, annotation: Path, label: str, map_path: Path) -> Optional[Dict[str, float]]:
    metrics = load_metrics(result_dir, label)
    if metrics is not None:
        return metrics
    out_json = result_dir / f"metrics_mapping_{label}.json"
    out_csv = result_dir / f"metrics_mapping_{label}.csv"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "evaluation_tools/map_ghost_eval.py"),
            "--map",
            str(map_path),
            "--annotation",
            str(annotation),
            "--output-json",
            str(out_json),
            "--output-csv",
            str(out_csv),
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )
    return load_metrics(result_dir, label)


def room_polygon(annotation: Path) -> Optional[np.ndarray]:
    data = json_like_yaml(annotation)
    room = data.get("room_bounds", {}) if isinstance(data, dict) else {}
    corners = room.get("floor_corners_xy")
    if corners is None:
        return None
    polygon = np.asarray(corners, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        return None
    return polygon


def json_like_yaml(path: Path) -> Dict:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def plot_room(ax, polygon: Optional[np.ndarray]) -> None:
    if polygon is None or polygon.shape[0] < 3:
        return
    closed = np.vstack([polygon, polygon[0]])
    ax.plot(closed[:, 0], closed[:, 1], color=COLORS["walls"], linewidth=1.8, zorder=5)


def set_equal_xy(ax, clouds: Iterable[np.ndarray], xlim: Optional[List[float]], ylim: Optional[List[float]]) -> None:
    if xlim is not None and ylim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal", adjustable="box")
        return
    xy_parts = [c[:, :2] for c in clouds if c.size]
    if not xy_parts:
        return
    xy = np.vstack(xy_parts)
    lo = np.percentile(xy, 1, axis=0)
    hi = np.percentile(xy, 99, axis=0)
    center = 0.5 * (lo + hi)
    radius = 0.55 * float(np.max(hi - lo))
    radius = max(radius, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_aspect("equal", adjustable="box")


def make_overview(
    clouds: Dict[str, np.ndarray],
    rejected: Optional[np.ndarray],
    polygon: Optional[np.ndarray],
    metrics: Dict[str, Dict[str, float]],
    out_path: Path,
    sample: int,
    dpi: int,
    xlim: Optional[List[float]],
    ylim: Optional[List[float]],
) -> None:
    labels = [("raw", "Raw accumulated"), ("marker_clean", "Marker clean"), ("vote_clean", "Vote clean")]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)
    sampled = {key: sample_points(value, sample, idx + 7) for idx, (key, _) in enumerate(labels) for value in [clouds[key]]}
    for ax, (key, title) in zip(axes, labels):
        pts = sampled[key]
        ax.scatter(pts[:, 0], pts[:, 1], s=0.15, c=COLORS[key], alpha=0.42, linewidths=0)
        if key == "vote_clean" and rejected is not None and rejected.size:
            rej = sample_points(rejected, min(sample // 2, 60000), 99)
            ax.scatter(rej[:, 0], rej[:, 1], s=0.2, c=COLORS["rejected"], alpha=0.35, linewidths=0)
        plot_room(ax, polygon)
        m = metrics.get(key, {})
        subtitle = (
            f"points={m.get('point_count', 0):.0f}\n"
            f"residual={m.get('residual', 0):.0f}, p95={m.get('thickness', 0):.3f} m"
        )
        ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, linewidth=0.25, alpha=0.25)
        set_equal_xy(ax, sampled.values(), xlim, ylim)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def signed_distances_in_reflective_rois(points: np.ndarray, annotation: Path) -> Dict[str, np.ndarray]:
    _, planes, _ = load_annotations(annotation, [0.05, 0.10, 0.20])
    out: Dict[str, List[np.ndarray]] = {"all": []}
    for plane in planes:
        roi = points_in_roi(points, plane.roi)
        if not np.any(roi):
            continue
        signed = points[:, :3] @ plane.normal + plane.d
        if plane.front_side in {"positive", "+", "pos"}:
            signed = -signed
        dist = signed[roi]
        out[plane.region_id] = dist
        out["all"].append(dist)
    if out["all"]:
        out["all"] = np.concatenate(out["all"])
    else:
        out["all"] = np.asarray([], dtype=np.float64)
    return out  # type: ignore[return-value]


def reflective_roi_masks(points: np.ndarray, annotation: Path, threshold: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    _, planes, _ = load_annotations(annotation, [threshold, 0.10, 0.20])
    in_any_roi = np.zeros(points.shape[0], dtype=bool)
    behind_any = np.zeros(points.shape[0], dtype=bool)
    for plane in planes:
        roi = points_in_roi(points, plane.roi)
        if not np.any(roi):
            continue
        signed = points[:, :3] @ plane.normal + plane.d
        if plane.front_side in {"positive", "+", "pos"}:
            signed = -signed
        in_any_roi |= roi
        behind_any |= roi & (signed > threshold)
    return in_any_roi, behind_any


def make_distance_hist(
    clouds: Dict[str, np.ndarray],
    annotation: Path,
    out_path: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    bins = np.linspace(-0.35, 1.25, 90)
    for label in ["raw", "marker_clean", "vote_clean"]:
        dist = signed_distances_in_reflective_rois(clouds[label], annotation)["all"]
        if dist.size == 0:
            continue
        ax.hist(
            dist,
            bins=bins,
            histtype="step",
            linewidth=1.8,
            density=True,
            label=label,
            color=COLORS[label],
        )
    ax.axvline(0.05, color="#111827", linewidth=1.0, linestyle="--", label="ghost threshold")
    ax.set_xlabel("signed distance behind reflective plane [m]")
    ax.set_ylabel("density in reflective ROIs")
    ax.set_title("Reflective-plane thickness / ghost distance distribution")
    ax.grid(True, linewidth=0.25, alpha=0.30)
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_metric_bars(metrics: Dict[str, Dict[str, float]], out_path: Path, dpi: int) -> None:
    labels = ["raw", "marker_clean", "vote_clean"]
    pretty = ["Raw", "Marker", "Vote"]
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.8), constrained_layout=True)
    raw_residual = metrics["raw"].get("residual", 0.0)
    for label in labels:
        residual = metrics[label].get("residual", 0.0)
        metrics[label]["removal_vs_raw"] = (
            max(0.0, (raw_residual - residual) / raw_residual) if raw_residual else 0.0
        )
    specs = [
        ("residual", "Reflection residual points"),
        ("removal_vs_raw", "Reflection removal vs raw"),
        ("thickness", "Plane thickness p95 [m]"),
    ]
    for ax, (key, title) in zip(axes, specs):
        values = [metrics[label].get(key, 0.0) for label in labels]
        ax.bar(pretty, values, color=[COLORS[label] for label in labels], width=0.65)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", linewidth=0.25, alpha=0.30)
        for i, value in enumerate(values):
            text = f"{value:.0f}" if key == "residual" else f"{value:.3f}"
            ax.text(i, value, text, ha="center", va="bottom", fontsize=8)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_reflective_roi_focus(
    clouds: Dict[str, np.ndarray],
    annotation: Path,
    polygon: Optional[np.ndarray],
    out_path: Path,
    sample: int,
    dpi: int,
    xlim: Optional[List[float]],
    ylim: Optional[List[float]],
) -> None:
    labels = [("raw", "Raw ROI ghosts"), ("marker_clean", "Marker-clean ROI ghosts"), ("vote_clean", "Vote-clean ROI ghosts")]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)
    for ax, (label, title) in zip(axes, labels):
        points = clouds[label]
        roi, behind = reflective_roi_masks(points, annotation)
        roi_points = sample_points(points[roi & ~behind], sample, 41)
        behind_points = sample_points(points[behind], sample, 43)
        if roi_points.size:
            ax.scatter(roi_points[:, 0], roi_points[:, 1], s=0.35, c="#718096", alpha=0.35, linewidths=0, label="ROI")
        if behind_points.size:
            ax.scatter(behind_points[:, 0], behind_points[:, 1], s=0.65, c=COLORS["rejected"], alpha=0.55, linewidths=0, label="behind")
        plot_room(ax, polygon)
        ax.set_title(f"{title}\nbehind={behind.sum()}, ROI={roi.sum()}", fontsize=10)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(True, linewidth=0.25, alpha=0.25)
        set_equal_xy(ax, [points[roi]] if np.any(roi) else [points], xlim, ylim)
        ax.legend(frameon=False, markerscale=5, loc="upper right")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def make_rejection_overlay(
    raw: np.ndarray,
    vote: np.ndarray,
    rejected: Optional[np.ndarray],
    polygon: Optional[np.ndarray],
    out_path: Path,
    sample: int,
    dpi: int,
    xlim: Optional[List[float]],
    ylim: Optional[List[float]],
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    raw_s = sample_points(raw, sample, 13)
    vote_s = sample_points(vote, sample, 17)
    ax.scatter(raw_s[:, 0], raw_s[:, 1], s=0.12, c="#a0aec0", alpha=0.20, linewidths=0, label="raw")
    ax.scatter(vote_s[:, 0], vote_s[:, 1], s=0.16, c=COLORS["vote_clean"], alpha=0.38, linewidths=0, label="vote clean")
    if rejected is not None and rejected.size:
        rej = sample_points(rejected, min(sample, 100000), 19)
        ax.scatter(rej[:, 0], rej[:, 1], s=0.22, c=COLORS["rejected"], alpha=0.42, linewidths=0, label="rejected")
    plot_room(ax, polygon)
    set_equal_xy(ax, [raw_s, vote_s] + ([rejected] if rejected is not None else []), xlim, ylim)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Frame-prior vote rejected map points")
    ax.grid(True, linewidth=0.25, alpha=0.25)
    ax.legend(frameon=False, markerscale=8, loc="upper right")
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def write_summary(out_dir: Path, metrics: Dict[str, Dict[str, float]]) -> None:
    md = out_dir / "quicklook_summary.md"
    raw_residual = metrics["raw"].get("residual", 0.0)
    with md.open("w", encoding="utf-8") as f:
        f.write("| label | points | ROI | residual | residual_rate | removal_vs_raw | valid_precision | thickness_p95_m |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for label in ["raw", "marker_clean", "vote_clean"]:
            m = metrics[label]
            removal = max(0.0, (raw_residual - m["residual"]) / raw_residual) if raw_residual else 0.0
            f.write(
                f"| {label} | {m['point_count']:.0f} | {m['roi']:.0f} | {m['residual']:.0f} | "
                f"{m['residual_rate']:.6f} | {removal:.6f} | {m['valid_precision']:.6f} | "
                f"{m['thickness']:.6f} |\n"
            )


def main() -> int:
    args = build_parser().parse_args()
    result_dir = args.result_dir
    out_dir = args.out_dir or result_dir / "paper_quicklook"
    out_dir.mkdir(parents=True, exist_ok=True)
    maps = {
        "raw": result_dir / "map_raw.pcd",
        "marker_clean": result_dir / "map_marker_clean.pcd",
        "vote_clean": result_dir / "map_vote_clean.pcd",
    }
    missing = [str(path) for path in maps.values() if not path.exists()]
    if missing:
        raise SystemExit("missing required maps:\n" + "\n".join(missing))
    clouds = {label: read_point_cloud(path) for label, path in maps.items()}
    metrics = {label: ensure_metrics(result_dir, args.annotation, label, path) for label, path in maps.items()}
    if any(value is None for value in metrics.values()):
        raise SystemExit("failed to load or compute metrics")
    metrics_clean: Dict[str, Dict[str, float]] = {key: value for key, value in metrics.items() if value is not None}

    rejected_path = args.rejected
    if rejected_path is None:
        candidates = [
            result_dir / "frame_prior_post_clean/rejected_by_frame_prior_votes.pcd",
            result_dir / "frame_prior_vote_sweep/stride5_m10_v6_r060/rejected_by_frame_prior_votes.pcd",
        ]
        rejected_path = next((path for path in candidates if path.exists()), None)
    rejected = read_point_cloud(rejected_path) if rejected_path and rejected_path.exists() else None
    polygon = room_polygon(args.annotation)
    xlim = list(args.xlim) if args.xlim else None
    ylim = list(args.ylim) if args.ylim else None

    make_overview(
        clouds,
        rejected,
        polygon,
        metrics_clean,
        out_dir / "mapping_ablation_overview.png",
        args.sample,
        args.dpi,
        xlim,
        ylim,
    )
    make_rejection_overlay(
        clouds["raw"],
        clouds["vote_clean"],
        rejected,
        polygon,
        out_dir / "vote_rejection_overlay.png",
        args.sample,
        args.dpi,
        xlim,
        ylim,
    )
    make_distance_hist(clouds, args.annotation, out_dir / "reflective_distance_hist.png", args.dpi)
    make_metric_bars(metrics_clean, out_dir / "mapping_metrics_bars.png", args.dpi)
    make_reflective_roi_focus(
        clouds,
        args.annotation,
        polygon,
        out_dir / "reflective_roi_ghost_focus.png",
        args.sample,
        args.dpi,
        xlim,
        ylim,
    )
    write_summary(out_dir, metrics_clean)
    print(f"wrote paper quicklook figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

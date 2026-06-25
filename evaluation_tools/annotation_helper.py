#!/usr/bin/env python3
"""Lightweight helpers for mirror/glass SLAM map annotation.

This script avoids external point-cloud viewers.  It can render orthographic
PNG previews, fit a plane from a manually chosen ROI, and export ROI/inlier
crops as plain ASCII PCD files for inspection in RViz or other tools.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from map_ghost_eval import (
    PlaneAnnotation,
    load_annotations,
    points_in_roi,
    read_point_cloud,
    signed_distances,
)


AXIS_NAMES = ("x", "y", "z")


@dataclass
class PlaneFit:
    normal: np.ndarray
    d: float
    centroid: np.ndarray
    inlier_mask: np.ndarray
    method: str
    threshold_m: float


@dataclass
class FitResult:
    region_id: str
    region_type: str
    normal: np.ndarray
    d: float
    front_side: str
    roi_min: np.ndarray
    roi_max: np.ndarray
    fit_roi_min: np.ndarray
    fit_roi_max: np.ndarray
    roi_polygon_xy: Optional[np.ndarray]
    fit_polygon_xy: Optional[np.ndarray]
    centroid: np.ndarray
    roi_point_count: int
    fit_point_count: int
    inlier_count: int
    method: str
    distance_abs_m: Mapping[str, Any]


def parse_vec3(values: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} must contain exactly three numbers")
    return arr


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def roi_mask(points: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    lo = np.minimum(minimum, maximum)
    hi = np.maximum(minimum, maximum)
    return np.logical_and(points >= lo, points <= hi).all(axis=1)


def polygon_area_xy(polygon_xy: np.ndarray) -> float:
    x = polygon_xy[:, 0]
    y = polygon_xy[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def points_in_polygon_xy(points_xy: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    x = points_xy[:, 0]
    y = points_xy[:, 1]
    inside = np.zeros(points_xy.shape[0], dtype=bool)
    px = polygon_xy[:, 0]
    py = polygon_xy[:, 1]
    j = polygon_xy.shape[0] - 1
    eps = 1e-12
    for i in range(polygon_xy.shape[0]):
        yi, yj = py[i], py[j]
        xi, xj = px[i], px[j]
        crosses = (yi > y) != (yj > y)
        x_intersection = (xj - xi) * (y - yi) / (yj - yi + eps) + xi
        inside ^= crosses & (x < x_intersection)

        edge = np.asarray([xj - xi, yj - yi], dtype=np.float64)
        edge_len_sq = float(np.dot(edge, edge))
        if edge_len_sq > 0.0:
            rel = points_xy - np.asarray([xi, yi], dtype=np.float64)
            t = np.clip((rel @ edge) / edge_len_sq, 0.0, 1.0)
            nearest = np.asarray([xi, yi], dtype=np.float64) + t[:, None] * edge
            inside |= np.linalg.norm(points_xy - nearest, axis=1) <= 1e-9
        j = i
    return inside


def prism_mask(points: np.ndarray, polygon_xy: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    z_lo, z_hi = sorted([float(z_min), float(z_max)])
    z_mask = (points[:, 2] >= z_lo) & (points[:, 2] <= z_hi)
    mask = np.zeros(points.shape[0], dtype=bool)
    idx = np.flatnonzero(z_mask)
    if idx.size:
        mask[idx] = points_in_polygon_xy(points[idx, :2], polygon_xy)
    return mask


def result_roi_mask(points: np.ndarray, result: FitResult) -> np.ndarray:
    if result.roi_polygon_xy is not None:
        return prism_mask(points, result.roi_polygon_xy, result.roi_min[2], result.roi_max[2])
    return roi_mask(points, result.roi_min, result.roi_max)


def result_fit_mask(points: np.ndarray, result: FitResult) -> np.ndarray:
    if result.fit_polygon_xy is not None:
        return prism_mask(points, result.fit_polygon_xy, result.fit_roi_min[2], result.fit_roi_max[2])
    return roi_mask(points, result.fit_roi_min, result.fit_roi_max)


def pcd_header(point_count: int) -> str:
    return "\n".join(
        [
            "# .PCD v0.7 - Point Cloud Data file format",
            "VERSION 0.7",
            "FIELDS x y z",
            "SIZE 4 4 4",
            "TYPE F F F",
            "COUNT 1 1 1",
            f"WIDTH {point_count}",
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            f"POINTS {point_count}",
            "DATA ascii",
            "",
        ]
    )


def write_pcd_ascii(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float64)
    with path.open("w", encoding="utf-8") as f:
        f.write(pcd_header(points.shape[0]))
        np.savetxt(f, points[:, :3], fmt="%.6f %.6f %.6f")


def fit_plane_pca(points: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    if points.shape[0] < 3:
        raise ValueError("need at least three points to fit a plane")
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)
    d = -float(np.dot(normal, centroid))
    return normal, d, centroid


def plane_from_three(points: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    a, b, c = points
    normal = np.cross(b - a, c - a)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        return None
    normal = normal / norm
    d = -float(np.dot(normal, a))
    return normal, d


def fit_plane_ransac(
    points: np.ndarray,
    *,
    threshold_m: float,
    iterations: int,
    seed: int,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    if points.shape[0] < 3:
        raise ValueError("need at least three points to fit a plane")
    rng = np.random.default_rng(seed)
    best_mask: Optional[np.ndarray] = None
    best_count = -1
    best_median = float("inf")

    for _ in range(max(1, iterations)):
        ids = rng.choice(points.shape[0], size=3, replace=False)
        candidate = plane_from_three(points[ids])
        if candidate is None:
            continue
        normal, d = candidate
        dist = np.abs(points @ normal + d)
        mask = dist <= threshold_m
        count = int(np.count_nonzero(mask))
        median = float(np.median(dist[mask])) if count else float("inf")
        if count > best_count or (count == best_count and median < best_median):
            best_mask = mask
            best_count = count
            best_median = median

    if best_mask is None or np.count_nonzero(best_mask) < 3:
        normal, d, centroid = fit_plane_pca(points)
        dist = np.abs(points @ normal + d)
        return normal, d, centroid, dist <= threshold_m

    normal, d, centroid = fit_plane_pca(points[best_mask])
    dist = np.abs(points @ normal + d)
    return normal, d, centroid, dist <= threshold_m


def orient_plane_to_front(
    normal: np.ndarray,
    d: float,
    front_point: Optional[np.ndarray],
) -> Tuple[np.ndarray, float, str, Optional[float]]:
    if front_point is None:
        return normal, d, "positive", None
    signed = float(np.dot(front_point, normal) + d)
    if signed < 0.0:
        normal = -normal
        d = -d
        signed = -signed
    return normal, d, "positive", signed


def distance_stats(points: np.ndarray, normal: np.ndarray, d: float) -> Mapping[str, Any]:
    if points.size == 0:
        return {"count": 0}
    dist = np.abs(points @ normal + d)
    return {
        "count": int(points.shape[0]),
        "mean": float(np.mean(dist)),
        "std": float(np.std(dist)),
        "median": float(np.median(dist)),
        "p90": float(np.percentile(dist, 90)),
        "p95": float(np.percentile(dist, 95)),
        "max": float(np.max(dist)),
    }


def yaml_vec(arr: np.ndarray, precision: int = 6) -> str:
    vals = ", ".join(f"{float(x):.{precision}f}" for x in arr)
    return f"[{vals}]"


def yaml_snippet(
    *,
    region_id: str,
    region_type: str,
    normal: np.ndarray,
    d: float,
    front_side: str,
    roi_min: np.ndarray,
    roi_max: np.ndarray,
    thresholds: Sequence[float],
    reflective: bool,
) -> str:
    key = "reflective_planes" if reflective else "planar_regions"
    lines = [
        f"{key}:",
        f"  - id: {region_id}",
        f"    type: {region_type}",
        f"    normal: {yaml_vec(normal)}",
        f"    d: {d:.6f}",
    ]
    if reflective:
        lines.append(f"    front_side: {front_side}")
    lines.extend(
        [
            "    roi:",
            f"      min: {yaml_vec(np.minimum(roi_min, roi_max))}",
            f"      max: {yaml_vec(np.maximum(roi_min, roi_max))}",
            "    thresholds_m: [" + ", ".join(f"{float(x):.2f}" for x in thresholds) + "]",
        ]
    )
    return "\n".join(lines)


def parse_thresholds(text: str) -> List[float]:
    thresholds = [float(x) for x in text.split(",") if x.strip()]
    if not thresholds:
        raise ValueError("thresholds must not be empty")
    return thresholds


def normalized_bounds(minimum: np.ndarray, maximum: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.minimum(minimum, maximum), np.maximum(minimum, maximum)


def determine_axis_range(
    args: argparse.Namespace,
    points: np.ndarray,
    axis: int,
    *,
    min_attr: str,
    max_attr: str,
) -> Tuple[float, float]:
    lo_value = getattr(args, min_attr)
    hi_value = getattr(args, max_attr)
    if lo_value is not None and hi_value is not None:
        lo = float(lo_value)
        hi = float(hi_value)
    elif args.crop_min is not None and args.crop_max is not None:
        crop_min = parse_vec3(args.crop_min, "crop-min")
        crop_max = parse_vec3(args.crop_max, "crop-max")
        lo = float(min(crop_min[axis], crop_max[axis]))
        hi = float(max(crop_min[axis], crop_max[axis]))
    else:
        lo = float(np.percentile(points[:, axis], 1.0))
        hi = float(np.percentile(points[:, axis], 99.0))
    if not hi > lo:
        raise ValueError(f"invalid {AXIS_NAMES[axis]} range: {lo}..{hi}")
    return lo, hi


def parse_projection(text: str) -> Tuple[int, int]:
    value = text.lower()
    table = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    if value not in table:
        raise ValueError(f"projection must be one of {sorted(table)}, got {text!r}")
    return table[value]


def select_display_points(points: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if args.crop_min is not None and args.crop_max is not None:
        crop_min = parse_vec3(args.crop_min, "crop-min")
        crop_max = parse_vec3(args.crop_max, "crop-max")
        points = points[roi_mask(points, crop_min, crop_max)]
        if points.shape[0] == 0:
            raise ValueError("crop contains no points")
    return sample_points(points, args.max_points, args.seed)


def collect_projected_rectangles(
    points: np.ndarray,
    *,
    axes: Tuple[int, int],
    color_axis: int,
    title: str,
    single: bool,
) -> List[Tuple[float, float, float, float]]:
    import os
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.widgets import RectangleSelector

    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise RuntimeError(
            "interactive bbox annotation needs a desktop display. "
            "Run this command in a local terminal, or pass --bbox-min/--bbox-max."
        )

    a, b = axes
    rects: List[Tuple[float, float, float, float]] = []
    artists: List[Any] = []

    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    sc = ax.scatter(
        points[:, a],
        points[:, b],
        c=points[:, color_axis],
        s=0.35,
        alpha=0.45,
        cmap="viridis",
        rasterized=True,
        linewidths=0,
    )
    fig.colorbar(sc, ax=ax, label=AXIS_NAMES[color_axis])
    ax.scatter([0.0], [0.0], marker="+", c="black", s=40, label="map origin")
    mode = "one bbox" if single else "one or more bboxes"
    ax.set_title(f"{title}\nDrag {mode}. Enter=finish, Backspace=undo, Esc=cancel")
    ax.set_xlabel(AXIS_NAMES[a])
    ax.set_ylabel(AXIS_NAMES[b])
    ax.axis("equal")
    ax.grid(True, linewidth=0.25, alpha=0.4)
    ax.legend(loc="best", fontsize=7)

    def redraw_labels() -> None:
        while artists:
            artists.pop().remove()
        for idx, (x0, x1, y0, y1) in enumerate(rects):
            lo_x, hi_x = min(x0, x1), max(x0, x1)
            lo_y, hi_y = min(y0, y1), max(y0, y1)
            rect = patches.Rectangle(
                (lo_x, lo_y),
                hi_x - lo_x,
                hi_y - lo_y,
                fill=False,
                linewidth=1.8,
                edgecolor="tab:red",
            )
            ax.add_patch(rect)
            artists.append(rect)
            label = ax.text(lo_x, hi_y, f"bbox_{idx}", color="tab:red", fontsize=8)
            artists.append(label)
        fig.canvas.draw_idle()

    def on_select(eclick: Any, erelease: Any) -> None:
        if eclick.xdata is None or eclick.ydata is None or erelease.xdata is None or erelease.ydata is None:
            return
        rect = (float(eclick.xdata), float(erelease.xdata), float(eclick.ydata), float(erelease.ydata))
        if single and rects:
            rects.clear()
        rects.append(rect)
        print(
            f"bbox_{len(rects) - 1}: "
            f"{AXIS_NAMES[a]}=[{min(rect[0], rect[1]):.4f}, {max(rect[0], rect[1]):.4f}], "
            f"{AXIS_NAMES[b]}=[{min(rect[2], rect[3]):.4f}, {max(rect[2], rect[3]):.4f}]"
        )
        redraw_labels()

    def on_key(event: Any) -> None:
        if event.key in {"enter", "return"}:
            plt.close(fig)
        elif event.key in {"backspace", "delete", "u"} and rects:
            removed = rects.pop()
            print(f"removed bbox: {removed}")
            redraw_labels()
        elif event.key in {"escape", "q"}:
            rects.clear()
            plt.close(fig)

    selector = RectangleSelector(
        ax,
        on_select,
        useblit=True,
        button=[1],
        minspanx=0.02,
        minspany=0.02,
        spancoords="data",
        interactive=False,
    )
    fig._annotation_selector = selector  # keep the widget alive while the window is open
    fig.canvas.mpl_connect("key_press_event", on_key)
    print("Interactive bbox window opened. Drag with left mouse, press Enter when done.")
    plt.show()
    return rects


def collect_xy_corners(
    points: np.ndarray,
    *,
    title: str,
    count: int = 4,
) -> np.ndarray:
    import os
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise RuntimeError(
            "interactive corner annotation needs a desktop display. "
            "Run this command in a local terminal, or pass --corners."
        )

    selected: List[Tuple[float, float]] = []
    artists: List[Any] = []

    fig, ax = plt.subplots(figsize=(10, 8), dpi=120)
    sc = ax.scatter(
        points[:, 0],
        points[:, 1],
        c=points[:, 2],
        s=0.35,
        alpha=0.45,
        cmap="viridis",
        rasterized=True,
        linewidths=0,
    )
    fig.colorbar(sc, ax=ax, label="z")
    ax.scatter([0.0], [0.0], marker="+", c="black", s=40, label="map origin")
    ax.set_title(f"{title}\nLeft-click {count} corners. Enter=finish, Backspace=undo, Esc=cancel")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(True, linewidth=0.25, alpha=0.4)
    ax.legend(loc="best", fontsize=7)

    def redraw() -> None:
        while artists:
            artists.pop().remove()
        if selected:
            xy = np.asarray(selected, dtype=np.float64)
            ax.scatter(xy[:, 0], xy[:, 1], c="tab:red", s=30, zorder=5)
            artists.append(ax.collections[-1])
            for idx, (x, y) in enumerate(selected):
                artists.append(ax.text(x, y, str(idx + 1), color="tab:red", fontsize=9))
            if len(selected) >= 2:
                line_xy = xy if len(selected) < count else np.vstack([xy, xy[0]])
                line, = ax.plot(line_xy[:, 0], line_xy[:, 1], color="tab:red", linewidth=1.5)
                artists.append(line)
            if len(selected) == count:
                poly = patches.Polygon(xy, closed=True, fill=False, edgecolor="tab:red", linewidth=2.0)
                ax.add_patch(poly)
                artists.append(poly)
        fig.canvas.draw_idle()

    def on_click(event: Any) -> None:
        if event.inaxes != ax or event.button != 1 or event.xdata is None or event.ydata is None:
            return
        if len(selected) >= count:
            return
        selected.append((float(event.xdata), float(event.ydata)))
        print(f"corner_{len(selected)}: [{event.xdata:.6f}, {event.ydata:.6f}]")
        redraw()

    def on_key(event: Any) -> None:
        if event.key in {"enter", "return"}:
            if len(selected) == count:
                plt.close(fig)
            else:
                print(f"need {count} corners, currently selected {len(selected)}")
        elif event.key in {"backspace", "delete", "u"} and selected:
            removed = selected.pop()
            print(f"removed corner: [{removed[0]:.6f}, {removed[1]:.6f}]")
            redraw()
        elif event.key in {"escape", "q"}:
            selected.clear()
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    print(f"Interactive corner window opened. Click {count} corners, press Enter when done.")
    plt.show()
    if len(selected) != count:
        raise ValueError(f"expected {count} corners, got {len(selected)}")
    return np.asarray(selected, dtype=np.float64)


def projected_rect_to_box(
    rect: Tuple[float, float, float, float],
    *,
    axes: Tuple[int, int],
    cross_range: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    a, b = axes
    c = ({0, 1, 2} - {a, b}).pop()
    lo = np.zeros(3, dtype=np.float64)
    hi = np.zeros(3, dtype=np.float64)
    lo[a], hi[a] = min(rect[0], rect[1]), max(rect[0], rect[1])
    lo[b], hi[b] = min(rect[2], rect[3]), max(rect[2], rect[3])
    lo[c], hi[c] = min(cross_range), max(cross_range)
    return lo, hi


def fit_plane_in_boxes(
    points: np.ndarray,
    *,
    region_id: str,
    region_type: str,
    roi_min: np.ndarray,
    roi_max: np.ndarray,
    fit_roi_min: np.ndarray,
    fit_roi_max: np.ndarray,
    front_point: Optional[np.ndarray],
    inlier_threshold: float,
    ransac_iters: int,
    fit_max_points: int,
    seed: int,
    no_ransac: bool,
) -> FitResult:
    roi_min, roi_max = normalized_bounds(roi_min, roi_max)
    fit_roi_min, fit_roi_max = normalized_bounds(fit_roi_min, fit_roi_max)
    fit_points_full = points[roi_mask(points, fit_roi_min, fit_roi_max)]
    if fit_points_full.shape[0] < 3:
        raise ValueError(
            f"{region_id}: fit bbox contains only {fit_points_full.shape[0]} points "
            f"({fit_roi_min.tolist()}..{fit_roi_max.tolist()})"
        )

    fit_points = sample_points(fit_points_full, fit_max_points, seed)
    if no_ransac:
        normal, d, centroid = fit_plane_pca(fit_points)
        method = "pca"
    else:
        normal, d, centroid, _ = fit_plane_ransac(
            fit_points,
            threshold_m=inlier_threshold,
            iterations=ransac_iters,
            seed=seed,
        )
        method = "ransac+pca"

    normal, d, front_side, _ = orient_plane_to_front(normal, d, front_point)
    roi_points = points[roi_mask(points, roi_min, roi_max)]
    full_dist = np.abs(roi_points @ normal + d) if roi_points.shape[0] else np.asarray([], dtype=np.float64)
    full_inliers = full_dist <= inlier_threshold
    return FitResult(
        region_id=region_id,
        region_type=region_type,
        normal=normal,
        d=d,
        front_side=front_side,
        roi_min=roi_min,
        roi_max=roi_max,
        fit_roi_min=fit_roi_min,
        fit_roi_max=fit_roi_max,
        roi_polygon_xy=None,
        fit_polygon_xy=None,
        centroid=centroid,
        roi_point_count=int(roi_points.shape[0]),
        fit_point_count=int(fit_points_full.shape[0]),
        inlier_count=int(np.count_nonzero(full_inliers)),
        method=method,
        distance_abs_m=distance_stats(roi_points, normal, d),
    )


def axis_wall_fit_result(
    points: np.ndarray,
    *,
    region_id: str,
    region_type: str,
    normal: np.ndarray,
    d: float,
    roi_min: np.ndarray,
    roi_max: np.ndarray,
    fit_roi_min: np.ndarray,
    fit_roi_max: np.ndarray,
    inlier_threshold: float,
) -> FitResult:
    normal = np.asarray(normal, dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        raise ValueError(f"{region_id}: normal must be non-zero")
    normal = normal / norm
    d = float(d) / norm
    roi_min, roi_max = normalized_bounds(roi_min, roi_max)
    fit_roi_min, fit_roi_max = normalized_bounds(fit_roi_min, fit_roi_max)
    roi_points = points[roi_mask(points, roi_min, roi_max)]
    fit_points = points[roi_mask(points, fit_roi_min, fit_roi_max)]
    if fit_points.shape[0]:
        centroid = np.mean(fit_points, axis=0)
    else:
        centroid = 0.5 * (fit_roi_min + fit_roi_max)
    full_dist = np.abs(roi_points @ normal + d) if roi_points.shape[0] else np.asarray([], dtype=np.float64)
    full_inliers = full_dist <= inlier_threshold
    return FitResult(
        region_id=region_id,
        region_type=region_type,
        normal=normal,
        d=d,
        front_side="positive",
        roi_min=roi_min,
        roi_max=roi_max,
        fit_roi_min=fit_roi_min,
        fit_roi_max=fit_roi_max,
        roi_polygon_xy=None,
        fit_polygon_xy=None,
        centroid=centroid,
        roi_point_count=int(roi_points.shape[0]),
        fit_point_count=int(fit_points.shape[0]),
        inlier_count=int(np.count_nonzero(full_inliers)),
        method="axis-aligned bbox",
        distance_abs_m=distance_stats(roi_points, normal, d),
    )


def oriented_wall_fit_result(
    points: np.ndarray,
    *,
    region_id: str,
    region_type: str,
    normal_xy: np.ndarray,
    d: float,
    roi_polygon_xy: np.ndarray,
    fit_polygon_xy: np.ndarray,
    z_min: float,
    z_max: float,
    inlier_threshold: float,
) -> FitResult:
    normal_xy = np.asarray(normal_xy, dtype=np.float64)
    norm = float(np.linalg.norm(normal_xy))
    if norm <= 0.0:
        raise ValueError(f"{region_id}: normal must be non-zero")
    normal_xy = normal_xy / norm
    normal = np.asarray([normal_xy[0], normal_xy[1], 0.0], dtype=np.float64)
    d = float(d) / norm
    roi_min = np.asarray([roi_polygon_xy[:, 0].min(), roi_polygon_xy[:, 1].min(), z_min], dtype=np.float64)
    roi_max = np.asarray([roi_polygon_xy[:, 0].max(), roi_polygon_xy[:, 1].max(), z_max], dtype=np.float64)
    fit_roi_min = np.asarray([fit_polygon_xy[:, 0].min(), fit_polygon_xy[:, 1].min(), z_min], dtype=np.float64)
    fit_roi_max = np.asarray([fit_polygon_xy[:, 0].max(), fit_polygon_xy[:, 1].max(), z_max], dtype=np.float64)
    roi_points = points[prism_mask(points, roi_polygon_xy, z_min, z_max)]
    fit_points = points[prism_mask(points, fit_polygon_xy, z_min, z_max)]
    if fit_points.shape[0]:
        centroid = np.mean(fit_points, axis=0)
    else:
        centroid = np.asarray([roi_polygon_xy[:, 0].mean(), roi_polygon_xy[:, 1].mean(), 0.5 * (z_min + z_max)])
    full_dist = np.abs(roi_points @ normal + d) if roi_points.shape[0] else np.asarray([], dtype=np.float64)
    full_inliers = full_dist <= inlier_threshold
    return FitResult(
        region_id=region_id,
        region_type=region_type,
        normal=normal,
        d=d,
        front_side="positive",
        roi_min=roi_min,
        roi_max=roi_max,
        fit_roi_min=fit_roi_min,
        fit_roi_max=fit_roi_max,
        roi_polygon_xy=roi_polygon_xy,
        fit_polygon_xy=fit_polygon_xy,
        centroid=centroid,
        roi_point_count=int(roi_points.shape[0]),
        fit_point_count=int(fit_points.shape[0]),
        inlier_count=int(np.count_nonzero(full_inliers)),
        method="oriented-corners",
        distance_abs_m=distance_stats(roi_points, normal, d),
    )


def order_corners_ccw(corners_xy: np.ndarray) -> np.ndarray:
    corners_xy = np.asarray(corners_xy, dtype=np.float64)
    if corners_xy.shape != (4, 2):
        raise ValueError(f"corners must be 4x2, got {corners_xy.shape}")
    center = corners_xy.mean(axis=0)
    angles = np.arctan2(corners_xy[:, 1] - center[1], corners_xy[:, 0] - center[0])
    ordered = corners_xy[np.argsort(angles)]
    if polygon_area_xy(ordered) <= 1e-6:
        raise ValueError("corner polygon area is too small")
    signed_area = 0.5 * float(
        np.dot(ordered[:, 0], np.roll(ordered[:, 1], -1))
        - np.dot(ordered[:, 1], np.roll(ordered[:, 0], -1))
    )
    if signed_area < 0.0:
        ordered = ordered[::-1]
    return ordered


def parse_corners(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size != 8:
        raise ValueError("--corners expects eight numbers: x1 y1 x2 y2 x3 y3 x4 y4")
    return arr.reshape(4, 2)


def oriented_wall_polygons_from_corners(
    corners_xy: np.ndarray,
    *,
    wall_thickness: float,
    ghost_margin: float,
    fit_thickness: float,
) -> List[Mapping[str, Any]]:
    ordered = order_corners_ccw(corners_xy)
    center = ordered.mean(axis=0)
    walls: List[Mapping[str, Any]] = []
    for idx in range(4):
        p0 = ordered[idx]
        p1 = ordered[(idx + 1) % 4]
        edge = p1 - p0
        length = float(np.linalg.norm(edge))
        if length <= 1e-6:
            raise ValueError(f"corner edge {idx} is degenerate")
        tangent = edge / length
        left_normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
        midpoint = 0.5 * (p0 + p1)
        inward = center - midpoint
        if float(np.dot(left_normal, inward)) < 0.0:
            inward_normal = -left_normal
        else:
            inward_normal = left_normal
        d = -float(np.dot(inward_normal, midpoint))
        roi_polygon = np.asarray(
            [
                p0 - inward_normal * ghost_margin,
                p1 - inward_normal * ghost_margin,
                p1 + inward_normal * wall_thickness,
                p0 + inward_normal * wall_thickness,
            ],
            dtype=np.float64,
        )
        fit_polygon = np.asarray(
            [
                p0 - inward_normal * fit_thickness,
                p1 - inward_normal * fit_thickness,
                p1 + inward_normal * fit_thickness,
                p0 + inward_normal * fit_thickness,
            ],
            dtype=np.float64,
        )
        walls.append(
            {
                "id": f"wall_{idx}",
                "normal_xy": inward_normal,
                "d": d,
                "roi_polygon_xy": roi_polygon,
                "fit_polygon_xy": fit_polygon,
                "edge": np.asarray([p0, p1], dtype=np.float64),
            }
        )
    return walls


def fit_result_to_yaml(result: FitResult, thresholds: Sequence[float]) -> Dict[str, Any]:
    roi: Dict[str, Any]
    if result.roi_polygon_xy is not None:
        roi = {
            "polygon_xy": [[float(x), float(y)] for x, y in result.roi_polygon_xy],
            "z": [float(result.roi_min[2]), float(result.roi_max[2])],
        }
    else:
        roi = {
            "min": [float(x) for x in result.roi_min],
            "max": [float(x) for x in result.roi_max],
        }
    return {
        "id": result.region_id,
        "type": result.region_type,
        "normal": [float(x) for x in result.normal],
        "d": float(result.d),
        "front_side": result.front_side,
        "roi": roi,
        "thresholds_m": [float(x) for x in thresholds],
    }


def write_annotation_yaml(
    path: Path,
    *,
    base_annotation: Optional[Path],
    results: Sequence[FitResult],
    thresholds: Sequence[float],
    sequence: Optional[str],
    bag: Optional[str],
    map_frame: Optional[str],
    append: bool,
    keep_planar: bool,
) -> None:
    if base_annotation is not None and base_annotation.exists():
        data = yaml.safe_load(base_annotation.read_text(encoding="utf-8")) or {}
        if not isinstance(data, Mapping):
            raise ValueError(f"{base_annotation}: expected YAML mapping")
        out: Dict[str, Any] = dict(data)
    else:
        out = {}

    if sequence is not None:
        out["sequence"] = sequence
    if bag is not None:
        out["bag"] = bag
    if map_frame is not None:
        out["map_frame"] = map_frame
    out["thresholds_m"] = [float(x) for x in thresholds]

    existing = list(out.get("reflective_planes", [])) if append else []
    out["reflective_planes"] = existing + [fit_result_to_yaml(x, thresholds) for x in results]
    if not keep_planar:
        out.pop("planar_regions", None)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(out, sort_keys=False, allow_unicode=False), encoding="utf-8")


def room_bounds_yaml_dict(
    *,
    corners_xy: np.ndarray,
    z_min: float,
    z_max: float,
    thresholds: Sequence[float],
    sequence: Optional[str],
    bag: Optional[str],
    map_frame: Optional[str],
    wall_types: Sequence[str],
    floor_type: str,
    ceiling_type: str,
    interior_margin: float,
    exterior_margin: float,
) -> Dict[str, Any]:
    ordered = order_corners_ccw(corners_xy)
    z_lo, z_hi = sorted([float(z_min), float(z_max)])
    if len(wall_types) != 4:
        raise ValueError("room_bounds needs exactly four wall types")
    bottom = [[float(x), float(y), z_lo] for x, y in ordered]
    top = [[float(x), float(y), z_hi] for x, y in ordered]
    faces: List[Dict[str, Any]] = []
    for idx, wall_type in enumerate(wall_types):
        j = (idx + 1) % 4
        faces.append(
            {
                "id": f"wall_{idx}",
                "type": str(wall_type),
                "corner_indices": [idx, j, j + 4, idx + 4],
            }
        )
    faces.extend(
        [
            {"id": "floor", "type": floor_type, "corner_indices": [0, 1, 2, 3]},
            {"id": "ceiling", "type": ceiling_type, "corner_indices": [4, 5, 6, 7]},
        ]
    )

    data: Dict[str, Any] = {
        "thresholds_m": [float(x) for x in thresholds],
        "room_bounds": {
            "type": "cuboid",
            "floor_corners_xy": [[float(x), float(y)] for x, y in ordered],
            "z": [z_lo, z_hi],
            "wall_types": [str(x) for x in wall_types],
            "floor_type": floor_type,
            "ceiling_type": ceiling_type,
            "interior_margin_m": float(interior_margin),
            "exterior_margin_m": float(exterior_margin),
            "thresholds_m": [float(x) for x in thresholds],
            "bottom_corners": bottom,
            "top_corners": top,
            "faces": faces,
        },
    }
    if sequence is not None:
        data["sequence"] = sequence
    if bag is not None:
        data["bag"] = bag
    if map_frame is not None:
        data["map_frame"] = map_frame
    return data


def write_room_bounds_yaml(
    path: Path,
    *,
    corners_xy: np.ndarray,
    z_min: float,
    z_max: float,
    thresholds: Sequence[float],
    sequence: Optional[str],
    bag: Optional[str],
    map_frame: Optional[str],
    wall_types: Sequence[str],
    floor_type: str,
    ceiling_type: str,
    interior_margin: float,
    exterior_margin: float,
) -> Dict[str, Any]:
    data = room_bounds_yaml_dict(
        corners_xy=corners_xy,
        z_min=z_min,
        z_max=z_max,
        thresholds=thresholds,
        sequence=sequence,
        bag=bag,
        map_frame=map_frame,
        wall_types=wall_types,
        floor_type=floor_type,
        ceiling_type=ceiling_type,
        interior_margin=interior_margin,
        exterior_margin=exterior_margin,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return data


def write_fit_artifacts(
    out_dir: Path,
    points: np.ndarray,
    results: Sequence[FitResult],
    *,
    inlier_threshold: float,
) -> None:
    ensure_output_dir(out_dir)
    summaries: List[Mapping[str, Any]] = []
    for result in results:
        roi_points = points[result_roi_mask(points, result)]
        fit_points = points[result_fit_mask(points, result)]
        dist = np.abs(roi_points @ result.normal + result.d) if roi_points.shape[0] else np.asarray([], dtype=np.float64)
        inliers = dist <= inlier_threshold
        write_pcd_ascii(out_dir / f"{result.region_id}_roi.pcd", roi_points)
        write_pcd_ascii(out_dir / f"{result.region_id}_fit_bbox.pcd", fit_points)
        write_pcd_ascii(out_dir / f"{result.region_id}_plane_inliers.pcd", roi_points[inliers])
        write_pcd_ascii(out_dir / f"{result.region_id}_plane_outliers.pcd", roi_points[~inliers])
        summaries.append(
            {
                "id": result.region_id,
                "type": result.region_type,
                "normal": result.normal.tolist(),
                "d": result.d,
                "front_side": result.front_side,
                "roi_min": result.roi_min.tolist(),
                "roi_max": result.roi_max.tolist(),
                "fit_roi_min": result.fit_roi_min.tolist(),
                "fit_roi_max": result.fit_roi_max.tolist(),
                "roi_polygon_xy": None if result.roi_polygon_xy is None else result.roi_polygon_xy.tolist(),
                "fit_polygon_xy": None if result.fit_polygon_xy is None else result.fit_polygon_xy.tolist(),
                "roi_point_count": result.roi_point_count,
                "fit_point_count": result.fit_point_count,
                "inlier_count": result.inlier_count,
                "method": result.method,
                "distance_abs_m": dict(result.distance_abs_m),
            }
        )
    (out_dir / "drag_annotation_summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n",
        encoding="utf-8",
    )


def print_fit_results(results: Sequence[FitResult]) -> None:
    print("\nFitted reflective planes")
    for result in results:
        p95 = result.distance_abs_m.get("p95")
        p95_text = "None" if p95 is None else f"{float(p95):.4f}"
        print(
            f"  {result.region_id}: roi_points={result.roi_point_count}, "
            f"fit_points={result.fit_point_count}, inliers={result.inlier_count}, "
            f"p95={p95_text} m, normal={yaml_vec(result.normal)}, d={result.d:.6f}"
        )


def load_annotation_regions(path: Optional[Path]) -> Tuple[List[PlaneAnnotation], List[PlaneAnnotation]]:
    if path is None or not path.exists():
        return [], []
    _, reflective, planar = load_annotations(path, [0.05, 0.10, 0.20])
    return reflective, planar


def add_roi_rectangles(ax: Any, axes: Tuple[int, int], regions: Iterable[PlaneAnnotation], color: str) -> None:
    import matplotlib.patches as patches

    a, b = axes
    for region in regions:
        lo = region.roi.minimum
        hi = region.roi.maximum
        if region.boundary_xy is not None and axes == (0, 1):
            ax.plot(
                region.boundary_xy[:, 0],
                region.boundary_xy[:, 1],
                color=color,
                linewidth=2.0,
            )
            label_x = float(np.mean(region.boundary_xy[:, 0]))
            label_y = float(np.mean(region.boundary_xy[:, 1]))
        elif region.roi.polygon_xy is not None and axes == (0, 1):
            patch = patches.Polygon(
                region.roi.polygon_xy,
                closed=True,
                fill=False,
                linewidth=1.2,
                edgecolor=color,
            )
            label_x = float(region.roi.polygon_xy[:, 0].min())
            label_y = float(region.roi.polygon_xy[:, 1].max())
        else:
            patch = patches.Rectangle(
                (lo[a], lo[b]),
                hi[a] - lo[a],
                hi[b] - lo[b],
                fill=False,
                linewidth=1.2,
                edgecolor=color,
            )
            label_x = float(lo[a])
            label_y = float(hi[b])
            ax.add_patch(patch)
        ax.text(label_x, label_y, region.region_id, color=color, fontsize=7)


def render_projection(
    points: np.ndarray,
    *,
    axes: Tuple[int, int],
    color_axis: int,
    output_path: Path,
    title: str,
    reflective: Sequence[PlaneAnnotation],
    planar: Sequence[PlaneAnnotation],
    candidate_roi: Optional[Tuple[np.ndarray, np.ndarray]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    a, b = axes
    fig, ax = plt.subplots(figsize=(10, 8), dpi=160)
    sc = ax.scatter(
        points[:, a],
        points[:, b],
        c=points[:, color_axis],
        s=0.25,
        alpha=0.45,
        cmap="viridis",
        rasterized=True,
        linewidths=0,
    )
    fig.colorbar(sc, ax=ax, label=AXIS_NAMES[color_axis])
    add_roi_rectangles(ax, axes, reflective, "tab:red")
    add_roi_rectangles(ax, axes, planar, "tab:blue")
    if candidate_roi is not None:
        lo, hi = candidate_roi
        rect = patches.Rectangle(
            (lo[a], lo[b]),
            hi[a] - lo[a],
            hi[b] - lo[b],
            fill=False,
            linestyle="--",
            linewidth=1.5,
            edgecolor="tab:orange",
        )
        ax.add_patch(rect)
        ax.text(lo[a], hi[b], "candidate_roi", color="tab:orange", fontsize=7)
    ax.scatter([0.0], [0.0], marker="+", c="black", s=40, label="map origin")
    ax.set_title(title)
    ax.set_xlabel(AXIS_NAMES[a])
    ax.set_ylabel(AXIS_NAMES[b])
    ax.axis("equal")
    ax.grid(True, linewidth=0.25, alpha=0.4)
    ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def command_render(args: argparse.Namespace) -> int:
    points = read_point_cloud(args.map)
    original_count = int(points.shape[0])
    if args.crop_min is not None and args.crop_max is not None:
        crop_min = parse_vec3(args.crop_min, "crop-min")
        crop_max = parse_vec3(args.crop_max, "crop-max")
        points = points[roi_mask(points, crop_min, crop_max)]
        if points.shape[0] == 0:
            raise ValueError("crop contains no points")
    sampled = sample_points(points, args.max_points, args.seed)
    reflective, planar = load_annotation_regions(args.annotation)
    candidate_roi = None
    if args.roi_min is not None and args.roi_max is not None:
        candidate_roi = (
            parse_vec3(args.roi_min, "roi-min"),
            parse_vec3(args.roi_max, "roi-max"),
        )

    out_dir = ensure_output_dir(args.output_dir)
    bounds = {
        "map": str(args.map),
        "point_count": int(points.shape[0]),
        "original_point_count": original_count,
        "rendered_point_count": int(sampled.shape[0]),
        "min": points.min(axis=0).tolist(),
        "max": points.max(axis=0).tolist(),
        "mean": points.mean(axis=0).tolist(),
    }
    (out_dir / "bounds.json").write_text(json.dumps(bounds, indent=2) + "\n", encoding="utf-8")

    render_projection(
        sampled,
        axes=(0, 1),
        color_axis=2,
        output_path=out_dir / f"{args.prefix}_xy_top.png",
        title="XY top view",
        reflective=reflective,
        planar=planar,
        candidate_roi=candidate_roi,
    )
    render_projection(
        sampled,
        axes=(0, 2),
        color_axis=1,
        output_path=out_dir / f"{args.prefix}_xz_side.png",
        title="XZ side view",
        reflective=reflective,
        planar=planar,
        candidate_roi=candidate_roi,
    )
    render_projection(
        sampled,
        axes=(1, 2),
        color_axis=0,
        output_path=out_dir / f"{args.prefix}_yz_side.png",
        title="YZ side view",
        reflective=reflective,
        planar=planar,
        candidate_roi=candidate_roi,
    )
    print(f"wrote previews to {out_dir}")
    print(f"point bounds: min={bounds['min']} max={bounds['max']}")
    return 0


def command_fit_plane(args: argparse.Namespace) -> int:
    points = read_point_cloud(args.map)
    roi_min = parse_vec3(args.roi_min, "roi-min")
    roi_max = parse_vec3(args.roi_max, "roi-max")
    mask = roi_mask(points, roi_min, roi_max)
    roi_points = points[mask]
    if roi_points.shape[0] < 3:
        raise ValueError(f"ROI contains only {roi_points.shape[0]} points")

    fit_points = sample_points(roi_points, args.fit_max_points, args.seed)
    if args.no_ransac:
        normal, d, centroid = fit_plane_pca(fit_points)
        sample_dist = np.abs(fit_points @ normal + d)
        inlier_mask_sample = sample_dist <= args.inlier_threshold
        method = "pca"
    else:
        normal, d, centroid, inlier_mask_sample = fit_plane_ransac(
            fit_points,
            threshold_m=args.inlier_threshold,
            iterations=args.ransac_iters,
            seed=args.seed,
        )
        method = "ransac+pca"

    front_point = parse_vec3(args.front_point, "front-point") if args.front_point else None
    normal, d, front_side, front_signed = orient_plane_to_front(normal, d, front_point)
    full_dist = np.abs(roi_points @ normal + d)
    full_inliers = full_dist <= args.inlier_threshold
    stats = distance_stats(roi_points, normal, d)
    thresholds = [float(x) for x in args.thresholds_m.split(",") if x.strip()]

    print("\nPlane fit summary")
    print(f"  map: {args.map}")
    print(f"  roi points: {roi_points.shape[0]}")
    print(f"  fit sample points: {fit_points.shape[0]}")
    print(f"  method: {method}")
    print(f"  inlier threshold: {args.inlier_threshold:.3f} m")
    print(f"  inliers in ROI: {int(np.count_nonzero(full_inliers))} / {roi_points.shape[0]}")
    print(f"  normal: {yaml_vec(normal)}")
    print(f"  d: {d:.6f}")
    print(f"  centroid: {yaml_vec(centroid)}")
    if front_signed is None:
        print("  front_side: positive  # VERIFY manually; pass --front-point to orient")
    else:
        print(f"  front_side: positive  # front point signed distance = {front_signed:.4f} m")
    print(f"  abs distance p95: {stats['p95']:.4f} m")

    snippet = yaml_snippet(
        region_id=args.id,
        region_type=args.type,
        normal=normal,
        d=d,
        front_side=front_side,
        roi_min=roi_min,
        roi_max=roi_max,
        thresholds=thresholds,
        reflective=not args.planar,
    )
    print("\nYAML snippet")
    print(snippet)

    if args.output_dir is not None:
        out_dir = ensure_output_dir(args.output_dir)
        write_pcd_ascii(out_dir / f"{args.id}_roi.pcd", roi_points)
        write_pcd_ascii(out_dir / f"{args.id}_plane_inliers.pcd", roi_points[full_inliers])
        write_pcd_ascii(out_dir / f"{args.id}_plane_outliers.pcd", roi_points[~full_inliers])
        summary = {
            "map": str(args.map),
            "id": args.id,
            "type": args.type,
            "roi_min": np.minimum(roi_min, roi_max).tolist(),
            "roi_max": np.maximum(roi_min, roi_max).tolist(),
            "roi_point_count": int(roi_points.shape[0]),
            "fit_sample_count": int(fit_points.shape[0]),
            "method": method,
            "normal": normal.tolist(),
            "d": d,
            "front_side": front_side,
            "front_point_signed_distance": front_signed,
            "inlier_threshold_m": args.inlier_threshold,
            "inlier_count": int(np.count_nonzero(full_inliers)),
            "distance_abs_m": dict(stats),
            "yaml_snippet": snippet,
        }
        (out_dir / f"{args.id}_fit_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote ROI crops and summary to {out_dir}")
    return 0


def command_drag_bbox(args: argparse.Namespace) -> int:
    points = read_point_cloud(args.map)
    display_points = select_display_points(points, args)
    axes = parse_projection(args.projection)
    color_axis = ({0, 1, 2} - set(axes)).pop()
    cross_range = determine_axis_range(
        args,
        points,
        color_axis,
        min_attr="cross_min",
        max_attr="cross_max",
    )

    if args.bbox_min is not None and args.bbox_max is not None:
        lo2 = parse_vec3([args.bbox_min[0], args.bbox_min[1], 0.0], "bbox-min")[:2]
        hi2 = parse_vec3([args.bbox_max[0], args.bbox_max[1], 0.0], "bbox-max")[:2]
        rects = [(float(lo2[0]), float(hi2[0]), float(lo2[1]), float(hi2[1]))]
    else:
        rects = collect_projected_rectangles(
            display_points,
            axes=axes,
            color_axis=color_axis,
            title=f"{args.projection.upper()} wall bbox annotation",
            single=False,
        )
    if not rects:
        raise ValueError("no bbox selected")

    front_point = parse_vec3(args.front_point, "front-point") if args.front_point else None
    thresholds = parse_thresholds(args.thresholds_m)
    results: List[FitResult] = []
    for idx, rect in enumerate(rects):
        roi_min, roi_max = projected_rect_to_box(rect, axes=axes, cross_range=cross_range)
        result = fit_plane_in_boxes(
            points,
            region_id=f"{args.id_prefix}_{idx}",
            region_type=args.type,
            roi_min=roi_min,
            roi_max=roi_max,
            fit_roi_min=roi_min,
            fit_roi_max=roi_max,
            front_point=front_point,
            inlier_threshold=args.inlier_threshold,
            ransac_iters=args.ransac_iters,
            fit_max_points=args.fit_max_points,
            seed=args.seed + idx,
            no_ransac=args.no_ransac,
        )
        results.append(result)

    print_fit_results(results)
    if args.output_yaml is not None:
        write_annotation_yaml(
            args.output_yaml,
            base_annotation=args.base_annotation,
            results=results,
            thresholds=thresholds,
            sequence=args.sequence,
            bag=args.bag,
            map_frame=args.map_frame,
            append=args.append,
            keep_planar=args.keep_planar,
        )
        print(f"\nwrote annotation YAML: {args.output_yaml}")
    if args.output_dir is not None:
        write_fit_artifacts(
            args.output_dir,
            points,
            results,
            inlier_threshold=args.inlier_threshold,
        )
        print(f"wrote bbox crops and summary to {args.output_dir}")
    return 0


def wall_boxes_from_elevator_bbox(
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    wall_thickness: float,
    ghost_margin: float,
    fit_thickness: float,
) -> List[Mapping[str, Any]]:
    mid_x = 0.5 * (x_min + x_max)
    mid_y = 0.5 * (y_min + y_max)
    return [
        {
            "id": "wall_y_min",
            "normal": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            "d": -y_min,
            "fit_min": np.asarray([x_min, y_min - fit_thickness, z_min], dtype=np.float64),
            "fit_max": np.asarray([x_max, y_min + fit_thickness, z_max], dtype=np.float64),
            "roi_min": np.asarray([x_min, y_min - ghost_margin, z_min], dtype=np.float64),
            "roi_max": np.asarray([x_max, y_min + wall_thickness, z_max], dtype=np.float64),
            "front_point": np.asarray([mid_x, mid_y, 0.5 * (z_min + z_max)], dtype=np.float64),
        },
        {
            "id": "wall_y_max",
            "normal": np.asarray([0.0, -1.0, 0.0], dtype=np.float64),
            "d": y_max,
            "fit_min": np.asarray([x_min, y_max - fit_thickness, z_min], dtype=np.float64),
            "fit_max": np.asarray([x_max, y_max + fit_thickness, z_max], dtype=np.float64),
            "roi_min": np.asarray([x_min, y_max - wall_thickness, z_min], dtype=np.float64),
            "roi_max": np.asarray([x_max, y_max + ghost_margin, z_max], dtype=np.float64),
            "front_point": np.asarray([mid_x, mid_y, 0.5 * (z_min + z_max)], dtype=np.float64),
        },
        {
            "id": "wall_x_min",
            "normal": np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            "d": -x_min,
            "fit_min": np.asarray([x_min - fit_thickness, y_min, z_min], dtype=np.float64),
            "fit_max": np.asarray([x_min + fit_thickness, y_max, z_max], dtype=np.float64),
            "roi_min": np.asarray([x_min - ghost_margin, y_min, z_min], dtype=np.float64),
            "roi_max": np.asarray([x_min + wall_thickness, y_max, z_max], dtype=np.float64),
            "front_point": np.asarray([mid_x, mid_y, 0.5 * (z_min + z_max)], dtype=np.float64),
        },
        {
            "id": "wall_x_max",
            "normal": np.asarray([-1.0, 0.0, 0.0], dtype=np.float64),
            "d": x_max,
            "fit_min": np.asarray([x_max - fit_thickness, y_min, z_min], dtype=np.float64),
            "fit_max": np.asarray([x_max + fit_thickness, y_max, z_max], dtype=np.float64),
            "roi_min": np.asarray([x_max - wall_thickness, y_min, z_min], dtype=np.float64),
            "roi_max": np.asarray([x_max + ghost_margin, y_max, z_max], dtype=np.float64),
            "front_point": np.asarray([mid_x, mid_y, 0.5 * (z_min + z_max)], dtype=np.float64),
        },
    ]


def command_drag_elevator(args: argparse.Namespace) -> int:
    points = read_point_cloud(args.map)
    display_points = select_display_points(points, args)
    z_min, z_max = determine_axis_range(
        args,
        points,
        2,
        min_attr="z_min",
        max_attr="z_max",
    )

    if args.bbox_min is not None and args.bbox_max is not None:
        lo = parse_vec3([args.bbox_min[0], args.bbox_min[1], 0.0], "bbox-min")[:2]
        hi = parse_vec3([args.bbox_max[0], args.bbox_max[1], 0.0], "bbox-max")[:2]
        rects = [(float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1]))]
    else:
        rects = collect_projected_rectangles(
            display_points,
            axes=(0, 1),
            color_axis=2,
            title="XY elevator footprint annotation",
            single=True,
        )
    if len(rects) != 1:
        raise ValueError("drag-elevator expects exactly one XY bbox")

    rect = rects[0]
    x_min, x_max = sorted([float(rect[0]), float(rect[1])])
    y_min, y_max = sorted([float(rect[2]), float(rect[3])])
    walls = wall_boxes_from_elevator_bbox(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
        wall_thickness=args.wall_thickness,
        ghost_margin=args.ghost_margin,
        fit_thickness=args.fit_thickness,
    )

    thresholds = parse_thresholds(args.thresholds_m)
    results: List[FitResult] = []
    skipped: List[str] = []
    for idx, wall in enumerate(walls):
        region_id = f"{args.id_prefix}_{wall['id']}"
        try:
            if args.fit_mode == "axis":
                result = axis_wall_fit_result(
                    points,
                    region_id=region_id,
                    region_type=args.type,
                    normal=wall["normal"],
                    d=wall["d"],
                    roi_min=wall["roi_min"],
                    roi_max=wall["roi_max"],
                    fit_roi_min=wall["fit_min"],
                    fit_roi_max=wall["fit_max"],
                    inlier_threshold=args.inlier_threshold,
                )
            else:
                result = fit_plane_in_boxes(
                    points,
                    region_id=region_id,
                    region_type=args.type,
                    roi_min=wall["roi_min"],
                    roi_max=wall["roi_max"],
                    fit_roi_min=wall["fit_min"],
                    fit_roi_max=wall["fit_max"],
                    front_point=wall["front_point"],
                    inlier_threshold=args.inlier_threshold,
                    ransac_iters=args.ransac_iters,
                    fit_max_points=args.fit_max_points,
                    seed=args.seed + idx,
                    no_ransac=args.no_ransac,
                )
        except ValueError as exc:
            if not args.allow_missing_walls:
                raise
            skipped.append(f"{region_id}: {exc}")
            continue
        results.append(result)

    if not results:
        raise ValueError("no wall plane was fitted")
    print_fit_results(results)
    if skipped:
        print("\nSkipped walls")
        for item in skipped:
            print(f"  {item}")

    if args.output_yaml is not None:
        write_annotation_yaml(
            args.output_yaml,
            base_annotation=args.base_annotation,
            results=results,
            thresholds=thresholds,
            sequence=args.sequence,
            bag=args.bag,
            map_frame=args.map_frame,
            append=args.append,
            keep_planar=args.keep_planar,
        )
        print(f"\nwrote annotation YAML: {args.output_yaml}")
    if args.output_dir is not None:
        write_fit_artifacts(
            args.output_dir,
            points,
            results,
            inlier_threshold=args.inlier_threshold,
        )
        print(f"wrote elevator crops and summary to {args.output_dir}")
    return 0


def command_corners_elevator(args: argparse.Namespace) -> int:
    points = read_point_cloud(args.map)
    display_points = select_display_points(points, args)
    z_min, z_max = determine_axis_range(
        args,
        points,
        2,
        min_attr="z_min",
        max_attr="z_max",
    )

    if args.corners is not None:
        corners_xy = parse_corners(args.corners)
    else:
        corners_xy = collect_xy_corners(
            display_points,
            title="XY rotated elevator footprint annotation",
            count=4,
        )
    ordered = order_corners_ccw(corners_xy)
    walls = oriented_wall_polygons_from_corners(
        ordered,
        wall_thickness=args.wall_thickness,
        ghost_margin=args.ghost_margin,
        fit_thickness=args.fit_thickness,
    )

    thresholds = parse_thresholds(args.thresholds_m)
    results: List[FitResult] = []
    for idx, wall in enumerate(walls):
        region_id = f"{args.id_prefix}_{wall['id']}"
        result = oriented_wall_fit_result(
            points,
            region_id=region_id,
            region_type=args.type,
            normal_xy=wall["normal_xy"],
            d=wall["d"],
            roi_polygon_xy=wall["roi_polygon_xy"],
            fit_polygon_xy=wall["fit_polygon_xy"],
            z_min=z_min,
            z_max=z_max,
            inlier_threshold=args.inlier_threshold,
        )
        results.append(result)

    print("\nOrdered footprint corners (counter-clockwise)")
    for idx, corner in enumerate(ordered):
        print(f"  corner_{idx + 1}: [{corner[0]:.6f}, {corner[1]:.6f}]")
    print_fit_results(results)

    if args.output_yaml is not None:
        write_annotation_yaml(
            args.output_yaml,
            base_annotation=args.base_annotation,
            results=results,
            thresholds=thresholds,
            sequence=args.sequence,
            bag=args.bag,
            map_frame=args.map_frame,
            append=args.append,
            keep_planar=args.keep_planar,
        )
        print(f"\nwrote annotation YAML: {args.output_yaml}")
    if args.output_dir is not None:
        write_fit_artifacts(
            args.output_dir,
            points,
            results,
            inlier_threshold=args.inlier_threshold,
        )
        (ensure_output_dir(args.output_dir) / "ordered_corners_xy.json").write_text(
            json.dumps({"corners_xy": ordered.tolist()}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote rotated elevator crops and summary to {args.output_dir}")
    return 0


def command_room_bounds(args: argparse.Namespace) -> int:
    points = read_point_cloud(args.map)
    display_points = select_display_points(points, args)
    z_min, z_max = determine_axis_range(
        args,
        points,
        2,
        min_attr="z_min",
        max_attr="z_max",
    )

    if args.corners is not None:
        corners_xy = parse_corners(args.corners)
    else:
        corners_xy = collect_xy_corners(
            display_points,
            title="XY room/elevator cuboid boundary annotation",
            count=4,
        )
    ordered = order_corners_ccw(corners_xy)
    thresholds = parse_thresholds(args.thresholds_m)
    wall_types = [str(x) for x in args.wall_types.split(",") if str(x).strip()]
    if len(wall_types) == 1:
        wall_types = wall_types * 4
    if len(wall_types) != 4:
        raise ValueError("--wall-types must contain one type or four comma-separated types")

    if args.output_yaml is not None:
        write_room_bounds_yaml(
            args.output_yaml,
            corners_xy=ordered,
            z_min=z_min,
            z_max=z_max,
            thresholds=thresholds,
            sequence=args.sequence,
            bag=args.bag,
            map_frame=args.map_frame,
            wall_types=wall_types,
            floor_type=args.floor_type,
            ceiling_type=args.ceiling_type,
            interior_margin=args.interior_margin,
            exterior_margin=args.exterior_margin,
        )
        print(f"\nwrote room_bounds YAML: {args.output_yaml}")

    print("\nOrdered room floor corners (counter-clockwise)")
    for idx, corner in enumerate(ordered):
        print(f"  corner_{idx + 1}: [{corner[0]:.6f}, {corner[1]:.6f}]")
    print(f"  z: [{float(z_min):.6f}, {float(z_max):.6f}]")
    print(f"  wall_types: {wall_types}")
    print(
        "  eval margins: "
        f"interior={float(args.interior_margin):.3f} m, exterior={float(args.exterior_margin):.3f} m"
    )

    if args.output_dir is not None:
        out_dir = ensure_output_dir(args.output_dir)
        (out_dir / "room_bounds_corners_xy.json").write_text(
            json.dumps(
                {
                    "corners_xy": ordered.tolist(),
                    "z": [float(z_min), float(z_max)],
                    "wall_types": wall_types,
                    "interior_margin_m": float(args.interior_margin),
                    "exterior_margin_m": float(args.exterior_margin),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if args.output_yaml is not None:
            reflective, planar = load_annotation_regions(args.output_yaml)
            render_projection(
                display_points,
                axes=(0, 1),
                color_axis=2,
                output_path=out_dir / "room_bounds_xy_top.png",
                title="XY room bounds",
                reflective=reflective,
                planar=planar,
                candidate_roi=None,
            )
            render_projection(
                display_points,
                axes=(0, 2),
                color_axis=1,
                output_path=out_dir / "room_bounds_xz_side.png",
                title="XZ room bounds",
                reflective=reflective,
                planar=planar,
                candidate_roi=None,
            )
            render_projection(
                display_points,
                axes=(1, 2),
                color_axis=0,
                output_path=out_dir / "room_bounds_yz_side.png",
                title="YZ room bounds",
                reflective=reflective,
                planar=planar,
                candidate_roi=None,
            )
        print(f"wrote room_bounds preview artifacts to {out_dir}")
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    points = read_point_cloud(args.map)
    reflective, planar = load_annotation_regions(args.annotation)
    out_dir = ensure_output_dir(args.output_dir)
    rows: List[Mapping[str, Any]] = []

    for region in list(reflective) + list(planar):
        mask = points_in_roi(points, region.roi)
        roi_points = points[mask]
        write_pcd_ascii(out_dir / f"{region.region_id}_roi.pcd", roi_points)
        dist = signed_distances(roi_points, region)
        row = {
            "id": region.region_id,
            "type": region.plane_type,
            "roi_point_count": int(roi_points.shape[0]),
            "normal": region.normal.tolist(),
            "d": region.d,
            "front_side": region.front_side,
            "distance_abs_m": distance_stats(roi_points, region.normal, region.d),
        }
        if region in reflective:
            margin = min(region.thresholds_m)
            if region.front_side in {"positive", "+", "pos"}:
                behind = dist < -margin
                front = dist >= -margin
            else:
                behind = dist > margin
                front = dist <= margin
            write_pcd_ascii(out_dir / f"{region.region_id}_front_or_plane.pcd", roi_points[front])
            write_pcd_ascii(out_dir / f"{region.region_id}_behind_ghost.pcd", roi_points[behind])
            row["behind_count"] = int(np.count_nonzero(behind))
        rows.append(row)

    (out_dir / "annotation_inspect_summary.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote annotation crops to {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    render = sub.add_parser("render", help="render XY/XZ/YZ PNG previews")
    render.add_argument("--map", required=True, type=Path)
    render.add_argument("--annotation", type=Path, default=None)
    render.add_argument("--output-dir", required=True, type=Path)
    render.add_argument("--prefix", default="map")
    render.add_argument("--max-points", type=int, default=300000)
    render.add_argument("--seed", type=int, default=7)
    render.add_argument("--crop-min", nargs=3, type=float, default=None)
    render.add_argument("--crop-max", nargs=3, type=float, default=None)
    render.add_argument("--roi-min", nargs=3, type=float, default=None)
    render.add_argument("--roi-max", nargs=3, type=float, default=None)
    render.set_defaults(func=command_render)

    fit = sub.add_parser("fit-plane", help="fit plane in a manually chosen 3D ROI")
    fit.add_argument("--map", required=True, type=Path)
    fit.add_argument("--roi-min", nargs=3, type=float, required=True)
    fit.add_argument("--roi-max", nargs=3, type=float, required=True)
    fit.add_argument("--front-point", nargs=3, type=float, default=None)
    fit.add_argument("--id", default="mirror_or_glass_0")
    fit.add_argument("--type", default="mirror")
    fit.add_argument("--planar", action="store_true", help="emit planar_regions snippet instead of reflective_planes")
    fit.add_argument("--thresholds-m", default="0.05,0.10,0.20")
    fit.add_argument("--inlier-threshold", type=float, default=0.05)
    fit.add_argument("--ransac-iters", type=int, default=1000)
    fit.add_argument("--fit-max-points", type=int, default=50000)
    fit.add_argument("--seed", type=int, default=7)
    fit.add_argument("--no-ransac", action="store_true")
    fit.add_argument("--output-dir", type=Path, default=None)
    fit.set_defaults(func=command_fit_plane)

    drag = sub.add_parser("drag-bbox", help="interactively drag one or more wall bboxes and fit reflective planes")
    drag.add_argument("--map", required=True, type=Path)
    drag.add_argument("--projection", default="xy", choices=["xy", "xz", "yz"])
    drag.add_argument("--crop-min", nargs=3, type=float, default=None)
    drag.add_argument("--crop-max", nargs=3, type=float, default=None)
    drag.add_argument("--bbox-min", nargs=2, type=float, default=None, help="non-interactive projected bbox minimum")
    drag.add_argument("--bbox-max", nargs=2, type=float, default=None, help="non-interactive projected bbox maximum")
    drag.add_argument("--cross-min", type=float, default=None, help="minimum along the axis not shown in the projection")
    drag.add_argument("--cross-max", type=float, default=None, help="maximum along the axis not shown in the projection")
    drag.add_argument("--front-point", nargs=3, type=float, default=None, help="point on the valid/front side of all planes")
    drag.add_argument("--id-prefix", default="mirror_wall")
    drag.add_argument("--type", default="mirror")
    drag.add_argument("--thresholds-m", default="0.05,0.10,0.20")
    drag.add_argument("--inlier-threshold", type=float, default=0.05)
    drag.add_argument("--ransac-iters", type=int, default=1000)
    drag.add_argument("--fit-max-points", type=int, default=50000)
    drag.add_argument("--seed", type=int, default=7)
    drag.add_argument("--no-ransac", action="store_true")
    drag.add_argument("--max-points", type=int, default=300000)
    drag.add_argument("--base-annotation", type=Path, default=None)
    drag.add_argument("--output-yaml", type=Path, default=None)
    drag.add_argument("--output-dir", type=Path, default=None)
    drag.add_argument("--sequence", default=None)
    drag.add_argument("--bag", default=None)
    drag.add_argument("--map-frame", default=None)
    drag.add_argument("--append", action="store_true", help="append to existing reflective_planes instead of replacing them")
    drag.add_argument("--keep-planar", action="store_true", help="keep planar_regions from base annotation")
    drag.set_defaults(func=command_drag_bbox)

    elevator = sub.add_parser(
        "drag-elevator",
        help="drag one XY elevator footprint and generate four reflective wall annotations",
    )
    elevator.add_argument("--map", required=True, type=Path)
    elevator.add_argument("--crop-min", nargs=3, type=float, default=None)
    elevator.add_argument("--crop-max", nargs=3, type=float, default=None)
    elevator.add_argument("--bbox-min", nargs=2, type=float, default=None, help="non-interactive XY footprint minimum")
    elevator.add_argument("--bbox-max", nargs=2, type=float, default=None, help="non-interactive XY footprint maximum")
    elevator.add_argument("--z-min", type=float, default=None)
    elevator.add_argument("--z-max", type=float, default=None)
    elevator.add_argument("--wall-thickness", type=float, default=0.25, help="front/interior side thickness included in each wall ROI")
    elevator.add_argument("--ghost-margin", type=float, default=1.20, help="behind-wall margin included for ghost statistics")
    elevator.add_argument("--fit-thickness", type=float, default=0.12, help="thin slab used to fit each wall plane")
    elevator.add_argument(
        "--fit-mode",
        default="axis",
        choices=["axis", "ransac"],
        help="axis uses the dragged footprint edges as mirror planes; ransac fits a plane from each wall slab",
    )
    elevator.add_argument("--id-prefix", default="elevator_mirror")
    elevator.add_argument("--type", default="mirror")
    elevator.add_argument("--thresholds-m", default="0.05,0.10,0.20")
    elevator.add_argument("--inlier-threshold", type=float, default=0.05)
    elevator.add_argument("--ransac-iters", type=int, default=1000)
    elevator.add_argument("--fit-max-points", type=int, default=50000)
    elevator.add_argument("--seed", type=int, default=7)
    elevator.add_argument("--no-ransac", action="store_true")
    elevator.add_argument("--max-points", type=int, default=300000)
    elevator.add_argument("--base-annotation", type=Path, default=None)
    elevator.add_argument("--output-yaml", type=Path, default=None)
    elevator.add_argument("--output-dir", type=Path, default=None)
    elevator.add_argument("--sequence", default=None)
    elevator.add_argument("--bag", default=None)
    elevator.add_argument("--map-frame", default=None)
    elevator.add_argument("--append", action="store_true", help="append to existing reflective_planes instead of replacing them")
    elevator.add_argument("--keep-planar", action="store_true", help="keep planar_regions from base annotation")
    elevator.add_argument("--allow-missing-walls", action="store_true", help="skip walls whose fit slabs contain too few points")
    elevator.set_defaults(func=command_drag_elevator)

    corners = sub.add_parser(
        "corners-elevator",
        help="click four XY corners of a rotated elevator footprint and generate four reflective wall annotations",
    )
    corners.add_argument("--map", required=True, type=Path)
    corners.add_argument("--crop-min", nargs=3, type=float, default=None)
    corners.add_argument("--crop-max", nargs=3, type=float, default=None)
    corners.add_argument(
        "--corners",
        nargs=8,
        type=float,
        default=None,
        metavar=("X1", "Y1", "X2", "Y2", "X3", "Y3", "X4", "Y4"),
        help="non-interactive XY corners; any order is accepted",
    )
    corners.add_argument("--z-min", type=float, default=None)
    corners.add_argument("--z-max", type=float, default=None)
    corners.add_argument("--wall-thickness", type=float, default=0.25, help="front/interior side thickness included in each wall ROI")
    corners.add_argument("--ghost-margin", type=float, default=1.20, help="behind-wall margin included for ghost statistics")
    corners.add_argument("--fit-thickness", type=float, default=0.12, help="thin slab used to count plane inliers")
    corners.add_argument("--id-prefix", default="elevator_mirror")
    corners.add_argument("--type", default="mirror")
    corners.add_argument("--thresholds-m", default="0.05,0.10,0.20")
    corners.add_argument("--inlier-threshold", type=float, default=0.05)
    corners.add_argument("--seed", type=int, default=7)
    corners.add_argument("--max-points", type=int, default=300000)
    corners.add_argument("--base-annotation", type=Path, default=None)
    corners.add_argument("--output-yaml", type=Path, default=None)
    corners.add_argument("--output-dir", type=Path, default=None)
    corners.add_argument("--sequence", default=None)
    corners.add_argument("--bag", default=None)
    corners.add_argument("--map-frame", default=None)
    corners.add_argument("--append", action="store_true", help="append to existing reflective_planes instead of replacing them")
    corners.add_argument("--keep-planar", action="store_true", help="keep planar_regions from base annotation")
    corners.set_defaults(func=command_corners_elevator)

    room = sub.add_parser(
        "room-bounds",
        help="click four XY floor corners and write true room_bounds cuboid annotation",
    )
    room.add_argument("--map", required=True, type=Path)
    room.add_argument("--crop-min", nargs=3, type=float, default=None)
    room.add_argument("--crop-max", nargs=3, type=float, default=None)
    room.add_argument(
        "--corners",
        nargs=8,
        type=float,
        default=None,
        metavar=("X1", "Y1", "X2", "Y2", "X3", "Y3", "X4", "Y4"),
        help="non-interactive XY corners; any order is accepted",
    )
    room.add_argument("--z-min", type=float, default=None)
    room.add_argument("--z-max", type=float, default=None)
    room.add_argument("--wall-types", default="mirror", help="one type for all walls or four comma-separated types")
    room.add_argument("--floor-type", default="floor")
    room.add_argument("--ceiling-type", default="ceiling")
    room.add_argument("--interior-margin", type=float, default=0.25, help="evaluation margin inside room")
    room.add_argument("--exterior-margin", type=float, default=1.20, help="evaluation margin outside mirror wall")
    room.add_argument("--thresholds-m", default="0.05,0.10,0.20")
    room.add_argument("--max-points", type=int, default=300000)
    room.add_argument("--seed", type=int, default=7)
    room.add_argument("--output-yaml", type=Path, default=None)
    room.add_argument("--output-dir", type=Path, default=None)
    room.add_argument("--sequence", default=None)
    room.add_argument("--bag", default=None)
    room.add_argument("--map-frame", default=None)
    room.set_defaults(func=command_room_bounds)

    inspect = sub.add_parser("inspect", help="export annotation ROI/front/behind PCD crops")
    inspect.add_argument("--map", required=True, type=Path)
    inspect.add_argument("--annotation", required=True, type=Path)
    inspect.add_argument("--output-dir", required=True, type=Path)
    inspect.set_defaults(func=command_inspect)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

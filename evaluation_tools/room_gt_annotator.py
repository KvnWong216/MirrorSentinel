#!/usr/bin/env python3
"""Simple cuboid reference-map annotator for mirror/glass SLAM.

This tool is intentionally small: select four floor corners, set z_min/z_max,
and export a dense six-face reference point cloud plus a YAML annotation.  The
result is a manual geometric reference map, not scanner-grade ground truth.
It is most useful for elevator or room-like mirror/glass scenes.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from map_ghost_eval import read_point_cloud


FACE_DEFS = [
    ("wall_0", [0, 1, 5, 4], "mirror"),
    ("wall_1", [1, 2, 6, 5], "mirror"),
    ("wall_2", [2, 3, 7, 6], "mirror"),
    ("wall_3", [3, 0, 4, 7], "mirror"),
    ("floor", [0, 1, 2, 3], "floor"),
    ("ceiling", [4, 5, 6, 7], "ceiling"),
]


def parse_values(values: Sequence[float], expected: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size != expected:
        raise ValueError(f"{name} expects {expected} numbers, got {arr.size}")
    return arr


def parse_thresholds(text: str) -> List[float]:
    values = [float(x) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("thresholds must not be empty")
    return values


def order_corners_ccw(corners_xy: np.ndarray) -> np.ndarray:
    corners_xy = np.asarray(corners_xy, dtype=np.float64).reshape(4, 2)
    center = corners_xy.mean(axis=0)
    angles = np.arctan2(corners_xy[:, 1] - center[1], corners_xy[:, 0] - center[0])
    ordered = corners_xy[np.argsort(angles)]
    signed_area = 0.5 * float(
        np.dot(ordered[:, 0], np.roll(ordered[:, 1], -1))
        - np.dot(ordered[:, 1], np.roll(ordered[:, 0], -1))
    )
    if abs(signed_area) <= 1e-9:
        raise ValueError("selected floor corners are nearly collinear")
    if signed_area < 0.0:
        ordered = ordered[::-1]
    return ordered


def sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def crop_points(
    points: np.ndarray,
    crop_min: Optional[Sequence[float]],
    crop_max: Optional[Sequence[float]],
) -> np.ndarray:
    if crop_min is None or crop_max is None:
        return points
    a = parse_values(crop_min, 3, "crop-min")
    b = parse_values(crop_max, 3, "crop-max")
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    mask = np.logical_and(points >= lo, points <= hi).all(axis=1)
    cropped = points[mask]
    if cropped.shape[0] == 0:
        raise ValueError("crop contains no points")
    return cropped


def collect_corners_interactive(points: np.ndarray, *, title: str) -> np.ndarray:
    import os
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise RuntimeError("interactive mode needs a desktop display; pass --corners")

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
    fig.colorbar(sc, ax=ax, label="z [m]")
    ax.scatter([0.0], [0.0], marker="+", c="black", s=45, label="map origin")
    ax.set_title(f"{title}\nLeft click four floor corners. Enter=save, u=undo, q=cancel")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True, linewidth=0.25, alpha=0.4)
    ax.legend(loc="best", fontsize=8)

    def redraw() -> None:
        while artists:
            artists.pop().remove()
        if selected:
            xy = np.asarray(selected, dtype=np.float64)
            artists.append(ax.scatter(xy[:, 0], xy[:, 1], c="tab:red", s=38, zorder=6))
            for idx, (x, y) in enumerate(selected):
                artists.append(ax.text(x, y, str(idx + 1), color="tab:red", fontsize=10, zorder=7))
            line_xy = xy if len(selected) < 4 else np.vstack([xy, xy[0]])
            line, = ax.plot(line_xy[:, 0], line_xy[:, 1], color="tab:red", linewidth=1.8, zorder=5)
            artists.append(line)
            if len(selected) == 4:
                poly = patches.Polygon(xy, closed=True, fill=False, edgecolor="tab:red", linewidth=2.0)
                ax.add_patch(poly)
                artists.append(poly)
        fig.canvas.draw_idle()

    def on_click(event: Any) -> None:
        if event.inaxes != ax or event.button != 1 or event.xdata is None or event.ydata is None:
            return
        if len(selected) >= 4:
            return
        selected.append((float(event.xdata), float(event.ydata)))
        print(f"corner_{len(selected)}: [{event.xdata:.6f}, {event.ydata:.6f}]")
        redraw()

    def on_key(event: Any) -> None:
        if event.key in {"enter", "return"}:
            if len(selected) == 4:
                plt.close(fig)
            else:
                print(f"need 4 corners, currently selected {len(selected)}")
        elif event.key in {"u", "backspace", "delete"} and selected:
            removed = selected.pop()
            print(f"removed corner: [{removed[0]:.6f}, {removed[1]:.6f}]")
            redraw()
        elif event.key in {"q", "escape"}:
            selected.clear()
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    print("Interactive GT corner window opened. Click 4 corners and press Enter.")
    plt.show()
    if len(selected) != 4:
        raise ValueError(f"expected 4 corners, got {len(selected)}")
    return np.asarray(selected, dtype=np.float64)


def cuboid_vertices(corners_xy: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    ordered = order_corners_ccw(corners_xy)
    z_lo, z_hi = sorted([float(z_min), float(z_max)])
    bottom = np.column_stack([ordered, np.full(4, z_lo)])
    top = np.column_stack([ordered, np.full(4, z_hi)])
    return np.vstack([bottom, top])


def face_normal(vertices: np.ndarray, indices: Sequence[int], center: np.ndarray) -> Tuple[np.ndarray, float]:
    pts = vertices[np.asarray(indices, dtype=np.int64)]
    normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        raise ValueError(f"degenerate face indices: {indices}")
    normal = normal / norm
    face_center = pts.mean(axis=0)
    # Existing reflective-wall annotations use inward normal as the positive side.
    if float(np.dot(normal, center - face_center)) < 0.0:
        normal = -normal
    d = -float(np.dot(normal, pts[0]))
    return normal, d


def sample_face_grid(corners: np.ndarray, spacing: float) -> np.ndarray:
    a, b, c, d = corners
    width = float(np.linalg.norm(b - a))
    height = float(np.linalg.norm(d - a))
    nu = max(2, int(math.ceil(width / spacing)) + 1)
    nv = max(2, int(math.ceil(height / spacing)) + 1)
    points: List[np.ndarray] = []
    for u in np.linspace(0.0, 1.0, nu):
        left = (1.0 - u) * a + u * b
        right = (1.0 - u) * d + u * c
        for v in np.linspace(0.0, 1.0, nv):
            points.append((1.0 - v) * left + v * right)
    return np.asarray(points, dtype=np.float64)


def generate_reference_cloud(
    vertices: np.ndarray,
    spacing: float,
    include_floor_ceiling: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    points: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for face_idx, (_face_id, indices, face_type) in enumerate(FACE_DEFS):
        if not include_floor_ceiling and face_type in {"floor", "ceiling"}:
            continue
        face_points = sample_face_grid(vertices[np.asarray(indices, dtype=np.int64)], spacing)
        points.append(face_points)
        labels.append(np.full(face_points.shape[0], face_idx, dtype=np.int32))
    if not points:
        return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.int32)
    return np.vstack(points), np.concatenate(labels)


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
    with path.open("w", encoding="utf-8") as f:
        f.write(pcd_header(points.shape[0]))
        if points.shape[0]:
            np.savetxt(f, points[:, :3], fmt="%.6f %.6f %.6f")


def write_ply_mesh(path: Path, vertices: np.ndarray) -> None:
    triangles: List[Tuple[int, int, int]] = []
    for _face_id, indices, _face_type in FACE_DEFS:
        i0, i1, i2, i3 = indices
        triangles.append((i0, i1, i2))
        triangles.append((i0, i2, i3))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {vertices.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(triangles)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for x, y, z in vertices:
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
        for tri in triangles:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def face_roi(vertices: np.ndarray, indices: Sequence[int], margin: float) -> Dict[str, List[float]]:
    pts = vertices[np.asarray(indices, dtype=np.int64)]
    lo = pts.min(axis=0) - margin
    hi = pts.max(axis=0) + margin
    return {"min": [float(x) for x in lo], "max": [float(x) for x in hi]}


def build_gt_yaml(
    *,
    vertices: np.ndarray,
    sequence: Optional[str],
    bag: Optional[str],
    map_frame: str,
    spacing: float,
    thresholds: Sequence[float],
    roi_margin: float,
    include_floor_ceiling: bool,
) -> Dict[str, Any]:
    center = vertices.mean(axis=0)
    reflective_planes: List[Dict[str, Any]] = []
    planar_regions: List[Dict[str, Any]] = []
    faces_yaml: List[Dict[str, Any]] = []
    for face_id, indices, face_type in FACE_DEFS:
        normal, d = face_normal(vertices, indices, center)
        face = {
            "id": face_id,
            "type": face_type,
            "corner_indices": [int(i) for i in indices],
            "normal": [float(x) for x in normal],
            "d": float(d),
        }
        faces_yaml.append(face)
        row = {
            "id": face_id,
            "type": face_type,
            "normal": [float(x) for x in normal],
            "d": float(d),
            "roi": face_roi(vertices, indices, roi_margin),
            "thresholds_m": [float(x) for x in thresholds],
        }
        if face_type == "mirror":
            row["front_side"] = "positive"
            reflective_planes.append(row)
        elif include_floor_ceiling:
            planar_regions.append(row)

    floor = vertices[:4, :2]
    data: Dict[str, Any] = {
        "sequence": sequence,
        "bag": bag,
        "map_frame": map_frame,
        "thresholds_m": [float(x) for x in thresholds],
        "gt_reference": {
            "type": "manual_cuboid_reference",
            "quality": "geometric_reference_not_laser_scan_gt",
            "spacing_m": float(spacing),
            "roi_margin_m": float(roi_margin),
            "vertices": [[float(x), float(y), float(z)] for x, y, z in vertices],
            "faces": faces_yaml,
        },
        "room_bounds": {
            "type": "cuboid",
            "floor_corners_xy": [[float(x), float(y)] for x, y in floor],
            "z": [float(vertices[:, 2].min()), float(vertices[:, 2].max())],
            "wall_types": ["mirror", "mirror", "mirror", "mirror"],
            "floor_type": "floor",
            "ceiling_type": "ceiling",
            "interior_margin_m": 0.25,
            "exterior_margin_m": 1.20,
            "thresholds_m": [float(x) for x in thresholds],
            "bottom_corners": [[float(x), float(y), float(z)] for x, y, z in vertices[:4]],
            "top_corners": [[float(x), float(y), float(z)] for x, y, z in vertices[4:]],
            "faces": [
                {"id": fid, "type": ftype, "corner_indices": [int(i) for i in indices]}
                for fid, indices, ftype in FACE_DEFS
            ],
        },
        "reflective_planes": reflective_planes,
    }
    if planar_regions:
        data["planar_regions"] = planar_regions
    return {key: value for key, value in data.items() if value is not None}


def render_preview(
    *,
    out_dir: Path,
    map_points: Optional[np.ndarray],
    ref_points: np.ndarray,
    vertices: np.ndarray,
    max_points: int,
    seed: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    ref_sample = sample_points(ref_points, max_points, seed)
    map_sample = sample_points(map_points, max_points, seed + 1) if map_points is not None else None
    floor = np.vstack([vertices[:4], vertices[0]])

    fig, ax = plt.subplots(figsize=(8, 7), dpi=180)
    if map_sample is not None and map_sample.size:
        ax.scatter(map_sample[:, 0], map_sample[:, 1], c="#a0aec0", s=0.18, alpha=0.28, linewidths=0, label="map")
    ax.scatter(ref_sample[:, 0], ref_sample[:, 1], c="#2b6cb0", s=0.22, alpha=0.50, linewidths=0, label="reference")
    ax.plot(floor[:, 0], floor[:, 1], color="#c53030", linewidth=1.8, label="floor boundary")
    ax.scatter(vertices[:4, 0], vertices[:4, 1], c="#c53030", s=18, zorder=5)
    for idx, (x, y, _z) in enumerate(vertices[:4]):
        ax.text(x, y, str(idx), color="#c53030", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Manual cuboid GT reference - top view")
    ax.grid(True, linewidth=0.25, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "gt_top_view.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
    if map_sample is not None and map_sample.size:
        ax.scatter(map_sample[:, 0], map_sample[:, 2], c="#a0aec0", s=0.18, alpha=0.28, linewidths=0, label="map")
    ax.scatter(ref_sample[:, 0], ref_sample[:, 2], c="#2b6cb0", s=0.22, alpha=0.50, linewidths=0, label="reference")
    for i in range(4):
        j = (i + 1) % 4
        ax.plot([vertices[i, 0], vertices[j, 0]], [vertices[i, 2], vertices[j, 2]], color="#c53030", linewidth=1.2)
        ax.plot(
            [vertices[i + 4, 0], vertices[j + 4, 0]],
            [vertices[i + 4, 2], vertices[j + 4, 2]],
            color="#c53030",
            linewidth=1.2,
        )
        ax.plot([vertices[i, 0], vertices[i + 4, 0]], [vertices[i, 2], vertices[i + 4, 2]], color="#c53030", linewidth=1.2)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("Manual cuboid GT reference - side view")
    ax.grid(True, linewidth=0.25, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "gt_xz_view.png")
    plt.close(fig)


def load_room_bounds(path: Path) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: expected YAML mapping")
    room = data.get("room_bounds")
    if not isinstance(room, Mapping):
        raise ValueError(f"{path}: missing room_bounds")
    corners = np.asarray(room["floor_corners_xy"], dtype=np.float64)
    if "z" in room:
        z = np.asarray(room["z"], dtype=np.float64)
        z_min, z_max = float(np.min(z)), float(np.max(z))
    else:
        z_min, z_max = sorted([float(room["z_min"]), float(room["z_max"])])
    return corners, z_min, z_max, dict(data)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, help="optional map point cloud for annotation and preview")
    parser.add_argument("--annotation", type=Path, help="existing room_bounds YAML; skips corner selection")
    parser.add_argument("--corners", type=float, nargs="*", help="eight numbers: x1 y1 x2 y2 x3 y3 x4 y4")
    parser.add_argument("--z-min", type=float, default=None)
    parser.add_argument("--z-max", type=float, default=None)
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--bag", default=None)
    parser.add_argument("--map-frame", default="camera_init")
    parser.add_argument("--spacing", type=float, default=0.03, help="reference cloud point spacing in meters")
    parser.add_argument("--roi-margin", type=float, default=0.05, help="face ROI margin in meters")
    parser.add_argument("--thresholds-m", default="0.05,0.10,0.20")
    parser.add_argument("--include-floor-ceiling", action="store_true", help="also write floor/ceiling planar regions")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-prefix", default="room_gt")
    parser.add_argument("--crop-min", type=float, nargs=3, default=None)
    parser.add_argument("--crop-max", type=float, nargs=3, default=None)
    parser.add_argument("--max-display-points", type=int, default=180000)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.spacing <= 0.0:
        raise SystemExit("--spacing must be positive")
    if args.roi_margin < 0.0:
        raise SystemExit("--roi-margin must be non-negative")

    map_points: Optional[np.ndarray] = None
    display_points: Optional[np.ndarray] = None
    if args.map is not None:
        map_points = read_point_cloud(args.map)
        display_points = sample_points(crop_points(map_points, args.crop_min, args.crop_max), args.max_display_points, args.seed)

    source_meta: Dict[str, Any] = {}
    if args.annotation is not None:
        corners_xy, z_min, z_max, source_meta = load_room_bounds(args.annotation)
    else:
        if args.corners is not None:
            corners_xy = parse_values(args.corners, 8, "corners").reshape(4, 2)
        else:
            if display_points is None:
                raise SystemExit("interactive corner selection needs --map, or pass --corners")
            corners_xy = collect_corners_interactive(display_points, title="Manual room GT annotation")
        if args.z_min is None or args.z_max is None:
            raise SystemExit("--z-min and --z-max are required unless --annotation has room_bounds.z")
        z_min, z_max = sorted([float(args.z_min), float(args.z_max)])

    thresholds = parse_thresholds(args.thresholds_m)
    sequence = args.sequence if args.sequence is not None else source_meta.get("sequence")
    bag = args.bag if args.bag is not None else source_meta.get("bag")
    map_frame = args.map_frame if args.map_frame is not None else source_meta.get("map_frame", "camera_init")

    vertices = cuboid_vertices(corners_xy, z_min, z_max)
    ref_points, face_labels = generate_reference_cloud(vertices, args.spacing, args.include_floor_ceiling)
    data = build_gt_yaml(
        vertices=vertices,
        sequence=sequence,
        bag=bag,
        map_frame=map_frame,
        spacing=args.spacing,
        thresholds=thresholds,
        roi_margin=args.roi_margin,
        include_floor_ceiling=args.include_floor_ceiling,
    )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    yaml_path = out_dir / f"{prefix}_annotation.yaml"
    pcd_path = out_dir / f"{prefix}_reference.pcd"
    ply_path = out_dir / f"{prefix}_mesh.ply"
    labels_path = out_dir / f"{prefix}_face_labels.npy"
    meta_path = out_dir / f"{prefix}_summary.json"

    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False), encoding="utf-8")
    write_pcd_ascii(pcd_path, ref_points)
    write_ply_mesh(ply_path, vertices)
    np.save(labels_path, face_labels)
    meta_path.write_text(
        json.dumps(
            {
                "yaml": str(yaml_path),
                "reference_pcd": str(pcd_path),
                "mesh_ply": str(ply_path),
                "face_labels": str(labels_path),
                "point_count": int(ref_points.shape[0]),
                "spacing_m": float(args.spacing),
                "z": [float(vertices[:, 2].min()), float(vertices[:, 2].max())],
                "floor_corners_xy_ccw": vertices[:4, :2].tolist(),
                "quality": "manual cuboid geometric reference; not laser-scanner GT",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    render_preview(
        out_dir=out_dir,
        map_points=map_points,
        ref_points=ref_points,
        vertices=vertices,
        max_points=args.max_display_points,
        seed=args.seed,
    )

    print("Manual cuboid GT reference written")
    print(f"  YAML annotation: {yaml_path}")
    print(f"  reference cloud: {pcd_path} ({ref_points.shape[0]} points)")
    print(f"  reference mesh:  {ply_path}")
    print(f"  previews:        {out_dir / 'gt_top_view.png'}, {out_dir / 'gt_xz_view.png'}")
    print("  note: this is a geometric reference map, not scanner-grade GT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

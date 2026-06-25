#!/usr/bin/env python3
"""Mirror/glass SLAM map-quality metrics.

This evaluator implements the main mapping protocol in SLAM_EVAL_PROTOCOL_CN.md.
It compares a saved map point cloud against manually annotated reflective
planes/ROIs and optional ordinary planar regions.

Supported point cloud inputs:
  - .pcd: PCL ascii or binary with x/y/z fields
  - .ply: ascii or binary_little_endian vertices with x/y/z properties
  - .npy/.npz: Nx3 or NxM arrays, first three columns are x/y/z
  - .txt/.csv/.xyz: first three numeric columns are x/y/z
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml


FLOAT_DTYPES = {
    ("F", 4): np.float32,
    ("F", 8): np.float64,
    ("I", 1): np.int8,
    ("I", 2): np.int16,
    ("I", 4): np.int32,
    ("I", 8): np.int64,
    ("U", 1): np.uint8,
    ("U", 2): np.uint16,
    ("U", 4): np.uint32,
    ("U", 8): np.uint64,
}

PLY_DTYPES = {
    "char": np.int8,
    "int8": np.int8,
    "uchar": np.uint8,
    "uint8": np.uint8,
    "short": np.int16,
    "int16": np.int16,
    "ushort": np.uint16,
    "uint16": np.uint16,
    "int": np.int32,
    "int32": np.int32,
    "uint": np.uint32,
    "uint32": np.uint32,
    "float": np.float32,
    "float32": np.float32,
    "double": np.float64,
    "float64": np.float64,
}


@dataclass
class Roi:
    minimum: np.ndarray
    maximum: np.ndarray
    polygon_xy: Optional[np.ndarray] = None


@dataclass
class PlaneAnnotation:
    region_id: str
    normal: np.ndarray
    d: float
    roi: Roi
    thresholds_m: List[float]
    plane_type: str = "plane"
    front_side: str = "positive"
    boundary_xy: Optional[np.ndarray] = None


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def finite_or_none(value: float) -> Optional[float]:
    return float(value) if math.isfinite(value) else None


def stats(values: np.ndarray) -> Dict[str, Optional[float]]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0.0,
            "mean": None,
            "std": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": float(values.size),
        "mean": finite_or_none(float(np.mean(values))),
        "std": finite_or_none(float(np.std(values))),
        "median": finite_or_none(float(np.median(values))),
        "p90": finite_or_none(float(np.percentile(values, 90))),
        "p95": finite_or_none(float(np.percentile(values, 95))),
        "max": finite_or_none(float(np.max(values))),
    }


def ensure_xyz(points: np.ndarray, source: Path) -> np.ndarray:
    points = np.asarray(points)
    if points.ndim == 1:
        if points.size % 3 != 0:
            raise ValueError(f"{source}: 1-D point array size is not divisible by 3")
        points = points.reshape(-1, 3)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"{source}: expected Nx3 or NxM point array, got {points.shape}")
    xyz = np.asarray(points[:, :3], dtype=np.float64)
    mask = np.isfinite(xyz).all(axis=1)
    return xyz[mask]


def parse_pcd_header(path: Path) -> tuple[Dict[str, List[str]], int, int]:
    header: Dict[str, List[str]] = {}
    offset = 0
    line_count = 0
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: PCD header ended before DATA line")
            line_count += 1
            offset = f.tell()
            text = line.decode("utf-8", errors="replace").strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split()
            key = parts[0].upper()
            header[key] = parts[1:]
            if key == "DATA":
                break
    return header, offset, line_count


def read_pcd(path: Path) -> np.ndarray:
    header, data_offset, header_lines = parse_pcd_header(path)
    fields = header.get("FIELDS")
    if not fields:
        raise ValueError(f"{path}: missing FIELDS in PCD")
    sizes = [int(x) for x in header.get("SIZE", ["4"] * len(fields))]
    types = header.get("TYPE", ["F"] * len(fields))
    counts = [int(x) for x in header.get("COUNT", ["1"] * len(fields))]
    width = int(header.get("WIDTH", ["0"])[0])
    height = int(header.get("HEIGHT", ["1"])[0])
    points = int(header.get("POINTS", [str(width * height)])[0])
    data_kind = header.get("DATA", [""])[0].lower()

    expanded_names: List[str] = []
    dtype_fields = []
    for field, typ, size, count in zip(fields, types, sizes, counts, strict=True):
        dtype = FLOAT_DTYPES.get((typ.upper(), size))
        if dtype is None:
            raise ValueError(f"{path}: unsupported PCD field type {field}: TYPE={typ} SIZE={size}")
        for i in range(count):
            name = field if count == 1 else f"{field}_{i}"
            expanded_names.append(name)
            dtype_fields.append((name, dtype))

    if data_kind == "ascii":
        arr = np.loadtxt(path, comments="#", skiprows=header_lines)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        index = [expanded_names.index(name) for name in ("x", "y", "z")]
        return ensure_xyz(arr[:, index], path)

    if data_kind == "binary":
        dtype = np.dtype(dtype_fields)
        raw = path.read_bytes()[data_offset:]
        structured = np.frombuffer(raw, dtype=dtype, count=points)
        xyz = np.column_stack([structured["x"], structured["y"], structured["z"]])
        return ensure_xyz(xyz, path)

    if data_kind == "binary_compressed":
        raise ValueError(f"{path}: PCD DATA binary_compressed is not supported yet")

    raise ValueError(f"{path}: unsupported PCD DATA kind {data_kind!r}")


def parse_ply_header(path: Path) -> tuple[List[str], int]:
    lines: List[str] = []
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: PLY header ended before end_header")
            lines.append(line.decode("utf-8", errors="replace").strip())
            if lines[-1] == "end_header":
                return lines, f.tell()


def read_ply(path: Path) -> np.ndarray:
    lines, offset = parse_ply_header(path)
    if not lines or lines[0] != "ply":
        raise ValueError(f"{path}: invalid PLY magic")
    fmt_line = next((line for line in lines if line.startswith("format ")), "")
    fmt = fmt_line.split()[1] if fmt_line else ""
    vertex_count = 0
    properties: List[tuple[str, np.dtype]] = []
    in_vertex = False
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
        elif in_vertex and parts[0] == "property":
            if parts[1] == "list":
                raise ValueError(f"{path}: list properties in vertex are not supported")
            dtype = PLY_DTYPES.get(parts[1])
            if dtype is None:
                raise ValueError(f"{path}: unsupported PLY property type {parts[1]!r}")
            properties.append((parts[2], np.dtype(dtype)))

    names = [name for name, _ in properties]
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"{path}: PLY vertex must contain x/y/z properties")

    if fmt == "ascii":
        data = np.loadtxt(path, skiprows=len(lines), max_rows=vertex_count)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        index = [names.index(name) for name in ("x", "y", "z")]
        return ensure_xyz(data[:, index], path)

    if fmt == "binary_little_endian":
        dtype = np.dtype(properties)
        raw = path.read_bytes()[offset:]
        structured = np.frombuffer(raw, dtype=dtype, count=vertex_count)
        xyz = np.column_stack([structured["x"], structured["y"], structured["z"]])
        return ensure_xyz(xyz, path)

    raise ValueError(f"{path}: unsupported PLY format {fmt!r}")


def read_text_points(path: Path) -> np.ndarray:
    rows: List[List[float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 3:
                raise ValueError(f"{path}:{line_no}: expected at least 3 columns")
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                if not rows:
                    continue
                raise
    return ensure_xyz(np.asarray(rows, dtype=np.float64), path)


def read_point_cloud(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".pcd":
        return read_pcd(path)
    if suffix == ".ply":
        return read_ply(path)
    if suffix == ".npy":
        return ensure_xyz(np.load(path), path)
    if suffix == ".npz":
        data = np.load(path)
        key = "points" if "points" in data else "arr_0" if "arr_0" in data else sorted(data.files)[0]
        return ensure_xyz(data[key], path)
    if suffix in {".txt", ".csv", ".xyz"}:
        return read_text_points(path)
    raise ValueError(f"unsupported point cloud suffix: {path}")


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
            on_edge = np.linalg.norm(points_xy - nearest, axis=1) <= 1e-9
            inside |= on_edge
        j = i
    return inside


def roi_volume(roi: Roi) -> float:
    if roi.polygon_xy is not None:
        height = float(roi.maximum[2] - roi.minimum[2])
        return polygon_area_xy(roi.polygon_xy) * height
    return float(np.prod(roi.maximum - roi.minimum))


def parse_roi(raw: Mapping[str, Any], region_id: str) -> Roi:
    if "polygon_xy" in raw:
        polygon_xy = np.asarray(raw["polygon_xy"], dtype=np.float64)
        if polygon_xy.ndim != 2 or polygon_xy.shape[1] != 2 or polygon_xy.shape[0] < 3:
            raise ValueError(f"{region_id}: roi polygon_xy must be an Nx2 array with at least 3 vertices")
        if "z" in raw:
            z = np.asarray(raw["z"], dtype=np.float64)
            if z.shape != (2,):
                raise ValueError(f"{region_id}: roi z must contain [z_min, z_max]")
            z_min, z_max = float(np.min(z)), float(np.max(z))
        elif "z_min" in raw and "z_max" in raw:
            z_min, z_max = sorted([float(raw["z_min"]), float(raw["z_max"])])
        else:
            raise ValueError(f"{region_id}: polygon_xy roi must contain z or z_min/z_max")
        lo = np.asarray([polygon_xy[:, 0].min(), polygon_xy[:, 1].min(), z_min], dtype=np.float64)
        hi = np.asarray([polygon_xy[:, 0].max(), polygon_xy[:, 1].max(), z_max], dtype=np.float64)
        return Roi(lo, hi, polygon_xy=polygon_xy)

    if "min" not in raw or "max" not in raw:
        raise ValueError(f"{region_id}: roi must contain min/max or polygon_xy/z")
    minimum = np.asarray(raw["min"], dtype=np.float64)
    maximum = np.asarray(raw["max"], dtype=np.float64)
    if minimum.shape != (3,) or maximum.shape != (3,):
        raise ValueError(f"{region_id}: roi min/max must be 3-vectors")
    lo = np.minimum(minimum, maximum)
    hi = np.maximum(minimum, maximum)
    return Roi(lo, hi)


def parse_plane(raw: Mapping[str, Any], *, default_type: str, default_thresholds: Sequence[float]) -> PlaneAnnotation:
    region_id = str(raw.get("id", "unnamed"))
    normal = np.asarray(raw["normal"], dtype=np.float64)
    if normal.shape != (3,):
        raise ValueError(f"{region_id}: normal must be a 3-vector")
    norm = float(np.linalg.norm(normal))
    if norm <= 0.0:
        raise ValueError(f"{region_id}: normal must be non-zero")
    d = float(raw["d"]) / norm
    normal = normal / norm
    thresholds = [float(x) for x in raw.get("thresholds_m", default_thresholds)]
    if not thresholds:
        raise ValueError(f"{region_id}: thresholds_m must not be empty")
    return PlaneAnnotation(
        region_id=region_id,
        normal=normal,
        d=d,
        roi=parse_roi(raw["roi"], region_id),
        thresholds_m=thresholds,
        plane_type=str(raw.get("type", default_type)),
        front_side=str(raw.get("front_side", "positive")).lower(),
        boundary_xy=np.asarray(raw["boundary_xy"], dtype=np.float64) if "boundary_xy" in raw else None,
    )


def order_corners_ccw(corners_xy: np.ndarray) -> np.ndarray:
    corners_xy = np.asarray(corners_xy, dtype=np.float64)
    if corners_xy.shape != (4, 2):
        raise ValueError(f"room_bounds floor_corners_xy must be 4x2, got {corners_xy.shape}")
    center = corners_xy.mean(axis=0)
    angles = np.arctan2(corners_xy[:, 1] - center[1], corners_xy[:, 0] - center[0])
    ordered = corners_xy[np.argsort(angles)]
    signed_area = 0.5 * float(
        np.dot(ordered[:, 0], np.roll(ordered[:, 1], -1))
        - np.dot(ordered[:, 1], np.roll(ordered[:, 0], -1))
    )
    if abs(signed_area) <= 1e-9:
        raise ValueError("room_bounds floor polygon area is too small")
    if signed_area < 0.0:
        ordered = ordered[::-1]
    return ordered


def wall_from_room_edge(
    *,
    region_id: str,
    p0: np.ndarray,
    p1: np.ndarray,
    center: np.ndarray,
    z_min: float,
    z_max: float,
    wall_type: str,
    thresholds: Sequence[float],
    interior_margin: float,
    exterior_margin: float,
) -> PlaneAnnotation:
    edge = p1 - p0
    length = float(np.linalg.norm(edge))
    if length <= 1e-9:
        raise ValueError(f"{region_id}: degenerate room edge")
    tangent = edge / length
    left_normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float64)
    midpoint = 0.5 * (p0 + p1)
    inward = center - midpoint
    inward_normal = left_normal if float(np.dot(left_normal, inward)) >= 0.0 else -left_normal
    d = -float(np.dot(inward_normal, midpoint))
    roi_polygon = np.asarray(
        [
            p0 - inward_normal * exterior_margin,
            p1 - inward_normal * exterior_margin,
            p1 + inward_normal * interior_margin,
            p0 + inward_normal * interior_margin,
        ],
        dtype=np.float64,
    )
    roi = Roi(
        np.asarray([roi_polygon[:, 0].min(), roi_polygon[:, 1].min(), z_min], dtype=np.float64),
        np.asarray([roi_polygon[:, 0].max(), roi_polygon[:, 1].max(), z_max], dtype=np.float64),
        polygon_xy=roi_polygon,
    )
    return PlaneAnnotation(
        region_id=region_id,
        normal=np.asarray([inward_normal[0], inward_normal[1], 0.0], dtype=np.float64),
        d=d,
        roi=roi,
        thresholds_m=[float(x) for x in thresholds],
        plane_type=wall_type,
        front_side="positive",
        boundary_xy=np.asarray([p0, p1], dtype=np.float64),
    )


def parse_room_bounds(
    data: Mapping[str, Any],
    *,
    default_thresholds: Sequence[float],
) -> List[PlaneAnnotation]:
    raw = data.get("room_bounds")
    if not raw:
        return []
    if not isinstance(raw, Mapping):
        raise ValueError("room_bounds must be a mapping")
    corners_xy = order_corners_ccw(np.asarray(raw["floor_corners_xy"], dtype=np.float64))
    if "z" in raw:
        z = np.asarray(raw["z"], dtype=np.float64)
        if z.shape != (2,):
            raise ValueError("room_bounds.z must contain [z_min, z_max]")
        z_min, z_max = sorted([float(z[0]), float(z[1])])
    else:
        z_min, z_max = sorted([float(raw["z_min"]), float(raw["z_max"])])

    thresholds = [float(x) for x in raw.get("thresholds_m", data.get("thresholds_m", default_thresholds))]
    interior_margin = float(raw.get("interior_margin_m", 0.25))
    exterior_margin = float(raw.get("exterior_margin_m", raw.get("ghost_margin_m", 1.20)))
    wall_types = list(raw.get("wall_types", ["mirror", "mirror", "mirror", "mirror"]))
    if len(wall_types) != 4:
        raise ValueError("room_bounds.wall_types must contain four entries")
    center = corners_xy.mean(axis=0)
    planes: List[PlaneAnnotation] = []
    for idx in range(4):
        wall_type = str(wall_types[idx]).lower()
        if wall_type in {"none", "skip", "open"}:
            continue
        planes.append(
            wall_from_room_edge(
                region_id=f"room_wall_{idx}",
                p0=corners_xy[idx],
                p1=corners_xy[(idx + 1) % 4],
                center=center,
                z_min=z_min,
                z_max=z_max,
                wall_type=wall_type,
                thresholds=thresholds,
                interior_margin=interior_margin,
                exterior_margin=exterior_margin,
            )
        )
    return planes


def load_annotations(path: Path, fallback_thresholds: Sequence[float]) -> tuple[Dict[str, Any], List[PlaneAnnotation], List[PlaneAnnotation]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: expected YAML mapping")
    default_thresholds = [float(x) for x in data.get("thresholds_m", fallback_thresholds)]
    reflective = [
        parse_plane(x, default_type="reflective", default_thresholds=default_thresholds)
        for x in data.get("reflective_planes", [])
    ]
    room_planes = parse_room_bounds(data, default_thresholds=default_thresholds)
    reflective.extend([x for x in room_planes if x.plane_type in {"mirror", "glass", "reflective"}])
    planar = [
        parse_plane(x, default_type="planar", default_thresholds=default_thresholds)
        for x in data.get("planar_regions", [])
    ]
    if not reflective and not planar:
        raise ValueError(f"{path}: expected reflective_planes or planar_regions")
    return dict(data), reflective, planar


def points_in_roi(points: np.ndarray, roi: Roi) -> np.ndarray:
    in_bounds = np.logical_and(points >= roi.minimum, points <= roi.maximum).all(axis=1)
    if roi.polygon_xy is None:
        return in_bounds
    mask = np.zeros(points.shape[0], dtype=bool)
    idx = np.flatnonzero(in_bounds)
    if idx.size:
        mask[idx] = points_in_polygon_xy(points[idx, :2], roi.polygon_xy)
    return mask


def signed_distances(points: np.ndarray, plane: PlaneAnnotation) -> np.ndarray:
    return points @ plane.normal + plane.d


def behind_mask(dist: np.ndarray, front_side: str, margin: float) -> np.ndarray:
    if front_side in {"positive", "+", "pos"}:
        return dist < -margin
    if front_side in {"negative", "-", "neg"}:
        return dist > margin
    raise ValueError(f"front_side must be positive or negative, got {front_side!r}")


def evaluate_reflective_plane(points: np.ndarray, plane: PlaneAnnotation) -> Dict[str, Any]:
    roi_mask = points_in_roi(points, plane.roi)
    roi_points = points[roi_mask]
    dist = signed_distances(roi_points, plane)
    abs_dist = np.abs(dist)
    volume = roi_volume(plane.roi)

    threshold_rows: Dict[str, Dict[str, float]] = {}
    primary_margin = min(plane.thresholds_m)
    behind_any = behind_mask(dist, plane.front_side, primary_margin)
    behind_dist = np.abs(dist[behind_any])

    for tau in plane.thresholds_m:
        inlier = abs_dist <= tau
        behind = behind_mask(dist, plane.front_side, tau)
        key = f"{tau:.3f}"
        threshold_rows[key] = {
            "tau_m": tau,
            "inlier_count": float(np.count_nonzero(inlier)),
            "outlier_count": float(np.count_nonzero(~inlier)),
            "inlier_ratio": safe_div(float(np.count_nonzero(inlier)), float(roi_points.shape[0])),
            "outlier_ratio": safe_div(float(np.count_nonzero(~inlier)), float(roi_points.shape[0])),
            "behind_plane_point_count": float(np.count_nonzero(behind)),
            "ghost_rate": safe_div(float(np.count_nonzero(behind)), float(roi_points.shape[0])),
            "behind_plane_density": safe_div(float(np.count_nonzero(behind)), volume),
        }

    result: Dict[str, Any] = {
        "id": plane.region_id,
        "type": plane.plane_type,
        "front_side": plane.front_side,
        "roi_point_count": float(roi_points.shape[0]),
        "roi_volume_m3": volume,
        "primary_margin_m": primary_margin,
        "behind_plane_point_count": float(np.count_nonzero(behind_any)),
        "ghost_rate": safe_div(float(np.count_nonzero(behind_any)), float(roi_points.shape[0])),
        "behind_plane_density": safe_div(float(np.count_nonzero(behind_any)), volume),
        "ghost_distance_m": stats(behind_dist),
        "plane_abs_distance_m": stats(abs_dist),
        "thresholds": threshold_rows,
    }
    return result


def evaluate_planar_region(points: np.ndarray, plane: PlaneAnnotation) -> Dict[str, Any]:
    roi_mask = points_in_roi(points, plane.roi)
    roi_points = points[roi_mask]
    abs_dist = np.abs(signed_distances(roi_points, plane))

    threshold_rows: Dict[str, Dict[str, float]] = {}
    for tau in plane.thresholds_m:
        inlier = abs_dist <= tau
        key = f"{tau:.3f}"
        threshold_rows[key] = {
            "tau_m": tau,
            "inlier_count": float(np.count_nonzero(inlier)),
            "outlier_count": float(np.count_nonzero(~inlier)),
            "inlier_ratio": safe_div(float(np.count_nonzero(inlier)), float(roi_points.shape[0])),
            "outlier_ratio": safe_div(float(np.count_nonzero(~inlier)), float(roi_points.shape[0])),
        }

    return {
        "id": plane.region_id,
        "type": plane.plane_type,
        "roi_point_count": float(roi_points.shape[0]),
        "plane_abs_distance_m": stats(abs_dist),
        "thresholds": threshold_rows,
    }


def aggregate_results(reflective: Sequence[Dict[str, Any]], planar: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total_reflective_points = sum(float(x["roi_point_count"]) for x in reflective)
    total_behind = sum(float(x["behind_plane_point_count"]) for x in reflective)
    ghost_rates = [float(x["ghost_rate"]) for x in reflective if float(x["roi_point_count"]) > 0.0]
    reflective_p95 = [
        x["plane_abs_distance_m"]["p95"]
        for x in reflective
        if x["plane_abs_distance_m"]["p95"] is not None
    ]
    planar_p95 = [
        x["plane_abs_distance_m"]["p95"]
        for x in planar
        if x["plane_abs_distance_m"]["p95"] is not None
    ]
    return {
        "reflective_region_count": float(len(reflective)),
        "planar_region_count": float(len(planar)),
        "reflective_roi_point_count": total_reflective_points,
        "behind_plane_point_count": total_behind,
        "ghost_rate": safe_div(total_behind, total_reflective_points),
        "ghost_rate_mean_per_region": finite_or_none(float(np.mean(ghost_rates))) if ghost_rates else None,
        "reflective_plane_thickness_p95_mean_m": finite_or_none(float(np.mean(reflective_p95))) if reflective_p95 else None,
        "planar_thickness_p95_mean_m": finite_or_none(float(np.mean(planar_p95))) if planar_p95 else None,
    }


def flatten_for_csv(prefix: str, obj: Mapping[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"region": prefix}
    for key, value in obj.items():
        if isinstance(value, Mapping):
            for subkey, subvalue in value.items():
                if isinstance(subvalue, Mapping):
                    continue
                row[f"{key}.{subkey}"] = subvalue
        elif not isinstance(value, (list, tuple)):
            row[key] = value
    return row


def write_outputs(output_json: Path, output_csv: Optional[Path], result: Dict[str, Any]) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not output_csv:
        return
    rows: List[Dict[str, Any]] = []
    rows.append(flatten_for_csv("aggregate", result["aggregate"]))
    for region in result["reflective_planes"]:
        rows.append(flatten_for_csv(f"reflective:{region['id']}", region))
    for region in result["planar_regions"]:
        rows.append(flatten_for_csv(f"planar:{region['id']}", region))
    fieldnames = sorted({key for row in rows for key in row.keys()})
    if "region" in fieldnames:
        fieldnames.remove("region")
        fieldnames.insert(0, "region")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, type=Path, help="map point cloud: pcd/ply/npy/npz/txt/csv/xyz")
    parser.add_argument("--annotation", required=True, type=Path, help="annotation YAML")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument(
        "--thresholds-m",
        default="0.05,0.10,0.20",
        help="fallback comma-separated thresholds if annotation does not specify thresholds_m",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    fallback_thresholds = [float(x) for x in args.thresholds_m.split(",") if x.strip()]
    points = read_point_cloud(args.map)
    annotation_meta, reflective_annotations, planar_annotations = load_annotations(args.annotation, fallback_thresholds)

    reflective_results = [evaluate_reflective_plane(points, ann) for ann in reflective_annotations]
    planar_results = [evaluate_planar_region(points, ann) for ann in planar_annotations]
    result: Dict[str, Any] = {
        "map": str(args.map),
        "annotation": str(args.annotation),
        "sequence": annotation_meta.get("sequence"),
        "map_frame": annotation_meta.get("map_frame"),
        "point_count": float(points.shape[0]),
        "aggregate": aggregate_results(reflective_results, planar_results),
        "reflective_planes": reflective_results,
        "planar_regions": planar_results,
    }
    write_outputs(args.output_json, args.output_csv, result)
    agg = result["aggregate"]
    print(
        "Ghost/behind/thickness: "
        f"{agg['ghost_rate']:.6f}, "
        f"{agg['behind_plane_point_count']:.0f}, "
        f"{agg['reflective_plane_thickness_p95_mean_m']}"
    )
    print(f"wrote: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

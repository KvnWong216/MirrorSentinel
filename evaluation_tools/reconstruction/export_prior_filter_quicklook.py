#!/usr/bin/env python3
"""Export a colored PLY quicklook for NeuralRecon-prior map filtering."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "evaluation_tools") not in sys.path:
    sys.path.append(str(REPO_ROOT / "evaluation_tools"))

from map_ghost_eval import read_point_cloud  # noqa: E402


def sample_points(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[np.sort(idx)]


def write_colored_ply(path: Path, clouds: List[Tuple[np.ndarray, Tuple[int, int, int]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for points, color in clouds:
        points = np.asarray(points, dtype=np.float64)
        if points.size == 0:
            continue
        rgb = np.broadcast_to(np.asarray(color, dtype=np.uint8), (points.shape[0], 3))
        rows.append((points[:, :3], rgb))

    point_count = sum(points.shape[0] for points, _ in rows)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {point_count}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for points, rgb in rows:
            out = np.column_stack([points, rgb.astype(np.float64)])
            np.savetxt(f, out, fmt="%.6f %.6f %.6f %d %d %d")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered-map", required=True, type=Path)
    parser.add_argument("--rejected", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-filtered", type=int, default=80000)
    parser.add_argument("--max-rejected", type=int, default=80000)
    parser.add_argument("--max-mesh", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    filtered = sample_points(read_point_cloud(args.filtered_map), args.max_filtered, args.seed)
    rejected = sample_points(read_point_cloud(args.rejected), args.max_rejected, args.seed + 1)
    mesh = sample_points(read_point_cloud(args.mesh), args.max_mesh, args.seed + 2)

    write_colored_ply(
        args.output,
        [
            (filtered, (165, 165, 165)),
            (rejected, (245, 42, 30)),
            (mesh, (0, 185, 255)),
        ],
    )
    print(
        f"wrote {args.output} "
        f"(filtered={filtered.shape[0]}, rejected={rejected.shape[0]}, mesh={mesh.shape[0]})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

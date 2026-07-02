#!/usr/bin/env python3
"""Generate minimal stereo calibration YAMLs for RTAB-Map from known ZED intrinsics.

This helper is intentionally simple: it creates ROS camera calibration files
compatible with `rtabmap_util/yaml_to_camera_info.py` when a rosbag contains
rectified stereo images but no camera_info topics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def make_camera_yaml(
    *,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    tx: float,
) -> dict:
    return {
        "image_width": width,
        "image_height": height,
        "camera_name": "zed2_rectified",
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        },
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {"rows": 1, "cols": 5, "data": [0.0, 0.0, 0.0, 0.0, 0.0]},
        "rectification_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        },
        "projection_matrix": {
            "rows": 3,
            "cols": 4,
            "data": [fx, 0.0, cx, tx, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fx", type=float, default=541.56)
    parser.add_argument("--fy", type=float, default=541.56)
    parser.add_argument("--cx", type=float, default=496.91)
    parser.add_argument("--cy", type=float, default=264.66)
    parser.add_argument(
        "--baseline-m",
        type=float,
        default=0.12,
        help="Rectified stereo baseline in meters. Right projection uses -fx*baseline.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    left = make_camera_yaml(
        width=args.width,
        height=args.height,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        tx=0.0,
    )
    right = make_camera_yaml(
        width=args.width,
        height=args.height,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        tx=-args.fx * args.baseline_m,
    )

    left_path = args.out_dir / "zed_left.yaml"
    right_path = args.out_dir / "zed_right.yaml"
    left_path.write_text(yaml.safe_dump(left, sort_keys=False), encoding="utf-8")
    right_path.write_text(yaml.safe_dump(right, sort_keys=False), encoding="utf-8")
    print(left_path)
    print(right_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

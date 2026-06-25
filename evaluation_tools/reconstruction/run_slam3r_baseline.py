#!/usr/bin/env python3
"""Run SLAM3R baseline on an extracted image folder."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SLAM3R_ROOT = REPO_ROOT / "third_party/reconstruction/SLAM3R"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="image folder or video")
    parser.add_argument("--slam3r-root", type=Path, default=SLAM3R_ROOT)
    parser.add_argument("--test-name", default="sentinel_slam3r")
    parser.add_argument("--gpu-id", type=int, default=-1)
    parser.add_argument("--keyframe-stride", type=int, default=3)
    parser.add_argument("--win-r", type=int, default=5)
    parser.add_argument("--max-num-register", type=int, default=10)
    parser.add_argument("--num-scene-frame", type=int, default=10)
    parser.add_argument("--initial-winsize", type=int, default=5)
    parser.add_argument("--conf-thres-l2w", type=float, default=12.0)
    parser.add_argument("--conf-thres-i2p", type=float, default=1.5)
    parser.add_argument("--num-points-save", type=int, default=1000000)
    parser.add_argument("--retrieve-freq", type=int, default=1)
    parser.add_argument("--update-buffer-intv", type=int, default=1)
    parser.add_argument("--buffer-size", type=int, default=100)
    parser.add_argument("--buffer-strategy", default="reservoir")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cmd = [
        "python",
        "recon.py",
        "--test_name",
        args.test_name,
        "--dataset",
        str(args.dataset),
        "--gpu_id",
        str(args.gpu_id),
        "--keyframe_stride",
        str(args.keyframe_stride),
        "--win_r",
        str(args.win_r),
        "--num_scene_frame",
        str(args.num_scene_frame),
        "--initial_winsize",
        str(args.initial_winsize),
        "--conf_thres_l2w",
        str(args.conf_thres_l2w),
        "--conf_thres_i2p",
        str(args.conf_thres_i2p),
        "--num_points_save",
        str(args.num_points_save),
        "--update_buffer_intv",
        str(args.update_buffer_intv),
        "--buffer_size",
        str(args.buffer_size),
        "--buffer_strategy",
        args.buffer_strategy,
        "--max_num_register",
        str(args.max_num_register),
        "--retrieve_freq",
        str(args.retrieve_freq),
        "--save_preds",
    ]
    if args.online:
        cmd.append("--online")

    print("+ cd", args.slam3r_root)
    print("+", " ".join(cmd))
    if args.dry_run:
        return 0
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(cmd, cwd=str(args.slam3r_root), env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

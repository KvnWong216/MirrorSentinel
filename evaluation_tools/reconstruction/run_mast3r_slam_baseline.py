#!/usr/bin/env python3
"""Run MASt3R-SLAM baseline on an extracted image folder."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
MAST3R_ROOT = REPO_ROOT / "third_party/reconstruction/MASt3R-SLAM"


def load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_calib(path: Path, metadata_path: Path) -> None:
    meta = load_json(metadata_path)
    K = meta["intrinsics"]
    width, height = meta["image_size"]
    calib = {
        "width": int(width),
        "height": int(height),
        "calibration": [
            float(K[0][0]),
            float(K[1][1]),
            float(K[0][2]),
            float(K[1][2]),
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(calib, sort_keys=False), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="image folder or video")
    parser.add_argument("--mast3r-root", type=Path, default=MAST3R_ROOT)
    parser.add_argument("--config", type=Path, default=MAST3R_ROOT / "config/base.yaml")
    parser.add_argument("--metadata", type=Path, default=None, help="conversion_metadata.json for intrinsics")
    parser.add_argument("--calib", type=Path, default=None)
    parser.add_argument("--no-calib", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    calib = args.calib
    if calib is None and args.metadata is not None and not args.no_calib:
        calib = args.metadata.parent / "mast3r_intrinsics.yaml"
        write_calib(calib, args.metadata)

    cmd = [
        "python",
        "main.py",
        "--dataset",
        str(args.dataset),
        "--config",
        str(args.config),
    ]
    if calib is not None and not args.no_calib:
        cmd.extend(["--calib", str(calib)])
    if args.extra:
        cmd.extend(args.extra)

    print("+ cd", args.mast3r_root)
    print("+", " ".join(cmd))
    if args.dry_run:
        return 0
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(cmd, cwd=str(args.mast3r_root), env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run NeuralRecon demo on prepared Sentinel-LIO data."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
NEURALRECON_ROOT = REPO_ROOT / "third_party/reconstruction/NeuralRecon"


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected YAML mapping")
    return data


def dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def build_config(args: argparse.Namespace) -> Path:
    config = load_yaml(args.base_cfg)
    config["DATASET"] = "demo"
    config["BATCH_SIZE"] = 1
    config["SAVE_SCENE_MESH"] = bool(args.save_scene_mesh)
    config["SAVE_INCREMENTAL"] = bool(args.save_incremental)
    config["VIS_INCREMENTAL"] = bool(args.vis_incremental)
    config["REDUCE_GPU_MEM"] = bool(args.reduce_gpu_mem)
    config["LOGDIR"] = str(args.checkpoint_dir)
    config.setdefault("TEST", {})
    config["TEST"]["PATH"] = str(args.data_dir)
    config["TEST"]["N_VIEWS"] = int(args.n_views)
    config["TEST"]["N_WORKERS"] = int(args.workers)
    config.setdefault("MODEL", {})
    if args.voxel_size is not None:
        config["MODEL"]["VOXEL_SIZE"] = float(args.voxel_size)
    if args.n_vox:
        config["MODEL"]["N_VOX"] = [int(x) for x in args.n_vox.split(",")]

    out_cfg = args.output_cfg
    if out_cfg is None:
        out_cfg = args.data_dir / "neuralrecon_demo.yaml"
    dump_yaml(out_cfg, config)
    return out_cfg


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    args.data_dir = args.data_dir.resolve()
    args.neuralrecon_root = args.neuralrecon_root.resolve()
    args.base_cfg = args.base_cfg.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    if args.output_cfg is not None:
        args.output_cfg = args.output_cfg.resolve()
    return args


def validate_inputs(args: argparse.Namespace, *, dry_run: bool) -> None:
    missing = []
    for path, label in [
        (args.data_dir / "images", "prepared image directory"),
        (args.data_dir / "fragments.pkl", "NeuralRecon fragments.pkl"),
        (args.data_dir / "SyncedPoses.txt", "NeuralRecon SyncedPoses.txt"),
        (args.base_cfg, "NeuralRecon base config"),
        (args.neuralrecon_root / "demo.py", "NeuralRecon demo.py"),
    ]:
        if not path.exists():
            missing.append(f"{label}: {path}")

    if missing:
        raise FileNotFoundError("Missing required NeuralRecon inputs:\n  " + "\n  ".join(missing))

    if not dry_run and not any(args.checkpoint_dir.glob("*.ckpt")):
        raise FileNotFoundError(
            "No NeuralRecon checkpoint found under "
            f"{args.checkpoint_dir}. Run:\n"
            "  bash evaluation_tools/reconstruction/download_neuralrecon_checkpoint.sh\n"
            "or manually place the official .ckpt file there."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--neuralrecon-root", type=Path, default=NEURALRECON_ROOT)
    parser.add_argument("--base-cfg", type=Path, default=NEURALRECON_ROOT / "config/demo.yaml")
    parser.add_argument("--checkpoint-dir", type=Path, default=NEURALRECON_ROOT / "checkpoints")
    parser.add_argument("--output-cfg", type=Path, default=None)
    parser.add_argument("--n-views", type=int, default=9)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--voxel-size", type=float, default=None)
    parser.add_argument("--n-vox", default="", help="comma-separated MODEL.N_VOX override, e.g. 96,96,96")
    parser.add_argument("--save-scene-mesh", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-incremental", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vis-incremental", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--reduce-gpu-mem", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = resolve_args(build_parser().parse_args())
    try:
        validate_inputs(args, dry_run=args.dry_run)
    except FileNotFoundError as exc:
        print(f"[neuralrecon] {exc}", file=sys.stderr)
        return 2
    cfg = build_config(args)
    cmd = ["python", "demo.py", "--cfg", str(cfg)]
    print("+ cd", args.neuralrecon_root)
    print("+", " ".join(cmd))
    if args.dry_run:
        return 0
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    subprocess.run(cmd, cwd=str(args.neuralrecon_root), env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

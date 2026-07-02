#!/usr/bin/env python3
"""Run reflection mask inference on an image directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from da3_adapter import DA3Config, DA3DepthRunner, normalize_depth_prior
from reflection_seg_model import build_model_from_checkpoint


SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def make_da3(args: argparse.Namespace) -> DA3DepthRunner | None:
    if not args.use_da3_prior:
        return None
    return DA3DepthRunner(
        DA3Config(
            backend=args.da3_backend,
            model=args.da3_model,
            checkpoint=args.da3_checkpoint,
            device=args.device,
            input_width=args.da3_input_width,
            input_height=args.da3_input_height,
        )
    )


def make_input(rgb: np.ndarray, width: int, height: int, device: str, da3: DA3DepthRunner | None) -> torch.Tensor:
    resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = (x - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    x = x.transpose(2, 0, 1)
    if da3 is not None:
        depth = da3.infer(rgb)
        depth = cv2.resize(normalize_depth_prior(depth), (width, height), interpolation=cv2.INTER_LINEAR)
        x = np.concatenate([x, depth[None]], axis=0)
    return torch.from_numpy(x[None]).to(device)


@torch.inference_mode()
def run(args: argparse.Namespace) -> int:
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    model = build_model_from_checkpoint(str(args.checkpoint), device=device)
    da3 = make_da3(args)

    images = [p for p in sorted(args.image_dir.rglob("*")) if p.is_file() and p.suffix.lower() in SUFFIXES]
    if args.max_samples > 0:
        images = images[: args.max_samples]
    if not images:
        raise RuntimeError(f"no images found under {args.image_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for idx, path in enumerate(images, start=1):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"failed to read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        x = make_input(rgb, args.image_width, args.image_height, device, da3)
        logits = model(x)
        prob = torch.sigmoid(logits)[0, 0].detach().float().cpu().numpy()
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = (prob > args.threshold).astype(np.uint8) * 255

        rel = path.relative_to(args.image_dir).with_suffix(".png")
        out_path = args.output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), mask)
        if args.prob_dir:
            prob_path = args.prob_dir / rel
            prob_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(prob_path), np.clip(prob * 255.0, 0, 255).astype(np.uint8))
        if idx % 100 == 0 or idx == len(images):
            print(f"[{idx}/{len(images)}] wrote {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--image-dir", type=Path, default=Path("dataset/public/3DRef/reflection/rgb/alllabel/test/image"))
    p.add_argument("--output-dir", type=Path, default=Path("evaluation_tools/results/3dref/predictions/da3_light_mask/alllabel"))
    p.add_argument("--prob-dir", type=Path, default=None)
    p.add_argument("--image-width", type=int, default=512)
    p.add_argument("--image-height", type=int, default=288)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--use-da3-prior", action="store_true")
    p.add_argument("--da3-backend", default="none", choices=["none", "heuristic", "module", "torchscript", "onnx"])
    p.add_argument("--da3-model", default="")
    p.add_argument("--da3-checkpoint", default="")
    p.add_argument("--da3-input-width", type=int, default=518)
    p.add_argument("--da3-input-height", type=int, default=518)
    return p


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

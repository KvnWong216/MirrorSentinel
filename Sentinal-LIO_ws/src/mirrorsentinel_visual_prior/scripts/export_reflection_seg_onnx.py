#!/usr/bin/env python3
"""Export a trained reflection mask head to ONNX."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from reflection_seg_model import build_model_from_checkpoint, load_reflection_checkpoint


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", type=Path, default=Path("Sentinal-LIO_ws/models/reflection_mask/reflection_mask.onnx"))
    p.add_argument("--image-width", type=int, default=None)
    p.add_argument("--image-height", type=int, default=None)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--device", default="cpu")
    return p


def main() -> int:
    args = build_parser().parse_args()
    ckpt = load_reflection_checkpoint(str(args.checkpoint), device="cpu")
    in_channels = int(ckpt.get("in_channels", 3))
    width = args.image_width or int(ckpt.get("image_width", 512))
    height = args.image_height or int(ckpt.get("image_height", 288))
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    model = build_model_from_checkpoint(str(args.checkpoint), device=device)
    dummy = torch.randn(1, in_channels, height, width, device=device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(args.output),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"}, "logits": {0: "batch", 2: "height", 3: "width"}},
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

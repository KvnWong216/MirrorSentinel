#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


class DatasetSpec:
    num_classes = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run EBLNet on 3DRef RGB images and export binary masks.')
    parser.add_argument('--eblnet-root', type=Path, default=Path('baselines/reflection/EBLNet'))
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--image-dir', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--input-size', type=int, default=416)
    parser.add_argument('--num-cascade', type=int, default=3)
    parser.add_argument('--num-points', type=int, default=96)
    parser.add_argument('--thres-gcn', type=float, default=0.9)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--suffix', default='.png')
    return parser


def install_apex_stub() -> None:
    if 'apex' in sys.modules:
        return
    apex = types.ModuleType('apex')
    amp = types.ModuleType('apex.amp')

    def float_function(func):
        return func

    amp.float_function = float_function
    apex.amp = amp
    sys.modules['apex'] = apex
    sys.modules['apex.amp'] = amp


def patch_resnext_pretrained_load() -> None:
    import network.resnext as resnext

    def resnext101_32x8_no_pretrained(**kwargs):
        kwargs['deep_stem'] = False
        kwargs['avg_down'] = False
        kwargs['groups'] = 32
        kwargs['width_per_group'] = 8
        return resnext.ResNet(resnext.Bottleneck, [3, 4, 23, 3], **kwargs)

    resnext.resnext101_32x8 = resnext101_32x8_no_pretrained


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key.replace('module.', '', 1) if key.startswith('module.') else key: value for key, value in state_dict.items()}


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    install_apex_stub()
    sys.path.insert(0, str(args.eblnet_root.resolve()))

    from config import cfg

    cfg.MODEL.BNFUNC = torch.nn.BatchNorm2d
    cfg.immutable(True)

    patch_resnext_pretrained_load()

    from network.EBLNet import EBLNet_resnext101_os8

    model = EBLNet_resnext101_os8(
        num_classes=DatasetSpec.num_classes,
        criterion=None,
        num_cascade=args.num_cascade,
        num_points=args.num_points,
        threshold=args.thres_gcn,
    )
    checkpoint = torch.load(str(args.checkpoint), map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    missing, unexpected = model.load_state_dict(strip_module_prefix(state_dict), strict=False)
    if missing or unexpected:
        print(f'warning: missing={len(missing)} unexpected={len(unexpected)}')
    model.to(args.device)
    model.eval()
    return model


def preprocess(image_path: Path, input_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    image = Image.open(image_path).convert('RGB')
    orig_w, orig_h = image.size
    resized = image.resize((input_size, input_size), Image.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    tensor = (tensor - MEAN) / STD
    return tensor.unsqueeze(0), (orig_h, orig_w)


def main() -> int:
    args = build_parser().parse_args()
    model = build_model(args)
    images = sorted([p for p in args.image_dir.rglob('*') if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}])
    if args.limit > 0:
        images = images[:args.limit]
    if not images:
        raise SystemExit(f'no images found under {args.image_dir}')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for image_path in tqdm(images, desc=f'EBLNet {args.checkpoint.stem}', unit='img'):
            tensor, (orig_h, orig_w) = preprocess(image_path, args.input_size)
            logits = model(tensor.to(args.device, non_blocking=True))
            logits = F.interpolate(logits, size=(orig_h, orig_w), mode='bilinear', align_corners=True)
            pred = torch.argmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
            mask = (pred > 0).astype(np.uint8) * 255
            out_path = args.out_dir / f'{image_path.stem}{args.suffix}'
            if not cv2.imwrite(str(out_path), mask):
                raise RuntimeError(f'failed to write {out_path}')

    print(f'wrote {len(images)} masks to {args.out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run SATNet on 3DRef RGB images and export binary masks.')
    parser.add_argument('--satnet-root', type=Path, default=Path('baselines/reflection/SATNet'))
    parser.add_argument('--config', type=Path, default=Path('baselines/reflection/SATNet/configs/satnet/msd_satnet.py'))
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--image-dir', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--threshold', type=int, default=0, help='Class id threshold. Default writes class>0 as foreground.')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--suffix', default='.png')
    return parser


def main() -> int:
    args = build_parser().parse_args()
    satnet_root = args.satnet_root.resolve()
    sys.path.insert(0, str(satnet_root))

    import mmcv  # noqa: WPS433
    from mmcv.runner import load_checkpoint  # noqa: WPS433
    from mmseg.apis import inference_segmentor  # noqa: WPS433
    from mmseg.models import build_segmentor  # noqa: WPS433

    cfg = mmcv.Config.fromfile(str(args.config))
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, str(args.checkpoint), map_location='cpu')
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', ('background', 'foreground'))
    model.PALETTE = checkpoint.get('meta', {}).get('PALETTE', [[0, 0, 0], [255, 255, 255]])
    model.cfg = cfg
    model.to(args.device)
    model.eval()

    images = sorted([p for p in args.image_dir.rglob('*') if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}])
    if args.limit > 0:
        images = images[:args.limit]
    if not images:
        raise SystemExit(f'no images found under {args.image_dir}')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    with torch.no_grad():
        for image_path in tqdm(images, desc=f'SATNet {args.checkpoint.parent.name}', unit='img'):
            result = inference_segmentor(model, str(image_path))[0]
            mask = (np.asarray(result) > args.threshold).astype(np.uint8) * 255
            out_path = args.out_dir / f'{image_path.stem}{args.suffix}'
            if not cv2.imwrite(str(out_path), mask):
                raise RuntimeError(f'failed to write {out_path}')

    print(f'wrote {len(images)} masks to {args.out_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

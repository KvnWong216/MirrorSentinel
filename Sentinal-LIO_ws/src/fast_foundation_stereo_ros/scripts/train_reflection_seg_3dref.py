#!/usr/bin/env python3
"""Train a lightweight reflection-mask head on 3DRef RGB masks."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from da3_adapter import DA3Config, DA3DepthRunner, normalize_depth_prior
from reflection_seg_model import create_reflection_model, dice_loss_with_logits


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def find_pairs(split_root: Path) -> List[Tuple[Path, Path]]:
    image_dir = split_root / "image"
    mask_dir = split_root / "mask"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(f"expected image/ and mask/ under {split_root}")

    masks: Dict[str, Path] = {}
    for path in sorted(mask_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            masks[path.stem] = path

    pairs = []
    for image_path in sorted(image_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        mask_path = masks.get(image_path.stem)
        if mask_path is not None:
            pairs.append((image_path, mask_path))
    if not pairs:
        raise RuntimeError(f"no image/mask pairs found under {split_root}")
    return pairs


class ReflectionMaskDataset(Dataset):
    def __init__(
        self,
        split_root: Path,
        image_size: Tuple[int, int],
        train: bool,
        mask_threshold: float = 0.0,
        use_da3_prior: bool = False,
        da3: Optional[DA3DepthRunner] = None,
        max_samples: int = 0,
    ):
        self.pairs = find_pairs(split_root)
        if max_samples > 0:
            self.pairs = self.pairs[:max_samples]
        self.width, self.height = image_size
        self.train = train
        self.mask_threshold = mask_threshold
        self.use_da3_prior = use_da3_prior
        self.da3 = da3

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[idx]
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if bgr is None:
            raise ValueError(f"failed to read image: {image_path}")
        if mask is None:
            raise ValueError(f"failed to read mask: {mask_path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.train and random.random() < 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])

        rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        image = rgb.astype(np.float32) / 255.0
        image = (image - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        image_chw = image.transpose(2, 0, 1)

        if self.use_da3_prior:
            if self.da3 is not None and self.da3.enabled():
                depth = self.da3.infer(rgb)
            else:
                from da3_adapter import heuristic_depth

                depth = heuristic_depth(rgb)
            depth = normalize_depth_prior(depth)[None]
            image_chw = np.concatenate([image_chw, depth], axis=0)

        target = (mask.astype(np.float32) > self.mask_threshold).astype(np.float32)[None]
        return torch.from_numpy(image_chw), torch.from_numpy(target)


def binary_counts(gt: torch.Tensor, pred: torch.Tensor) -> Tuple[int, int, int, int]:
    gt_b = gt.bool()
    pred_b = pred.bool()
    tp = torch.logical_and(gt_b, pred_b).sum().item()
    fp = torch.logical_and(~gt_b, pred_b).sum().item()
    fn = torch.logical_and(gt_b, ~pred_b).sum().item()
    tn = torch.logical_and(~gt_b, ~pred_b).sum().item()
    return int(tp), int(fp), int(fn), int(tn)


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    def div(a: float, b: float) -> float:
        return float(a / b) if b else 0.0

    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    return {
        "iou": div(tp, tp + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": div(2.0 * precision * recall, precision + recall),
        "accuracy": div(tp + tn, tp + fp + fn + tn),
    }


def make_da3(args: argparse.Namespace) -> Optional[DA3DepthRunner]:
    if not args.use_da3_prior:
        return None
    cfg = DA3Config(
        backend=args.da3_backend,
        model=args.da3_model,
        checkpoint=args.da3_checkpoint,
        device=args.device,
        input_width=args.da3_input_width,
        input_height=args.da3_input_height,
    )
    return DA3DepthRunner(cfg)


def train_one_epoch(model, loader, optimizer, device: str, pos_weight: float, amp: bool) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(images)
            bce = F.binary_cross_entropy_with_logits(
                logits, masks, pos_weight=torch.tensor([pos_weight], device=device).view(1, 1, 1, 1)
            )
            loss = bce + dice_loss_with_logits(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().cpu()) * images.shape[0]
    return {"loss": total_loss / max(len(loader.dataset), 1)}


@torch.inference_mode()
def evaluate(model, loader, device: str, threshold: float) -> Dict[str, float]:
    model.eval()
    tp = fp = fn = tn = 0
    total_loss = 0.0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        logits = model(images)
        total_loss += float((F.binary_cross_entropy_with_logits(logits, masks) + dice_loss_with_logits(logits, masks)).cpu()) * images.shape[0]
        pred = torch.sigmoid(logits) > threshold
        c = binary_counts(masks > 0.5, pred)
        tp += c[0]
        fp += c[1]
        fn += c[2]
        tn += c[3]
    out = metrics_from_counts(tp, fp, fn, tn)
    out["loss"] = total_loss / max(len(loader.dataset), 1)
    out["tp"] = float(tp)
    out["fp"] = float(fp)
    out["fn"] = float(fn)
    out["tn"] = float(tn)
    return out


def save_checkpoint(path: Path, model, args: argparse.Namespace, metrics: Dict[str, float], epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "arch": args.arch,
            "in_channels": 4 if args.use_da3_prior else 3,
            "base_channels": args.base_channels,
            "use_da3_prior": args.use_da3_prior,
            "image_width": args.image_width,
            "image_height": args.image_height,
            "threshold": args.pred_threshold,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, default=Path("dataset/public/3DRef/reflection/rgb/alllabel"))
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="test")
    p.add_argument("--output-dir", type=Path, default=Path("Sentinal-LIO_ws/models/reflection_mask"))
    p.add_argument("--arch", default="tiny", choices=["tiny", "mobilenetv3"])
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--pretrained", action="store_true", help="use ImageNet weights when arch supports it")
    p.add_argument("--image-width", type=int, default=512)
    p.add_argument("--image-height", type=int, default=288)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--mask-threshold", type=float, default=0.0)
    p.add_argument("--pred-threshold", type=float, default=0.5)
    p.add_argument("--pos-weight", type=float, default=3.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--use-da3-prior", action="store_true", help="train RGB+DA3-depth guided mask head")
    p.add_argument("--da3-backend", default="none", choices=["none", "heuristic", "module", "torchscript", "onnx"])
    p.add_argument("--da3-model", default="")
    p.add_argument("--da3-checkpoint", default="")
    p.add_argument("--da3-input-width", type=int, default=518)
    p.add_argument("--da3-input-height", type=int, default=518)
    return p


def main() -> int:
    args = build_parser().parse_args()
    seed_all(args.seed)
    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    da3 = make_da3(args)

    train_ds = ReflectionMaskDataset(
        args.dataset_root / args.train_split,
        (args.image_width, args.image_height),
        train=True,
        mask_threshold=args.mask_threshold,
        use_da3_prior=args.use_da3_prior,
        da3=da3,
        max_samples=args.max_train_samples,
    )
    val_ds = ReflectionMaskDataset(
        args.dataset_root / args.val_split,
        (args.image_width, args.image_height),
        train=False,
        mask_threshold=args.mask_threshold,
        use_da3_prior=args.use_da3_prior,
        da3=da3,
        max_samples=args.max_val_samples,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.startswith("cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.startswith("cuda"))

    model = create_reflection_model(
        args.arch,
        in_channels=4 if args.use_da3_prior else 3,
        base_channels=args.base_channels,
        pretrained=args.pretrained,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "history.csv"
    best_f1 = -1.0
    rows: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args.pos_weight, args.amp and device.startswith("cuda"))
        val_metrics = evaluate(model, val_loader, device, args.pred_threshold)
        row = {"epoch": float(epoch), **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        rows.append(row)
        print(
            f"epoch {epoch:03d}: train_loss={train_metrics['loss']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_p={val_metrics['precision']:.4f} val_r={val_metrics['recall']:.4f}"
        )

        save_checkpoint(args.output_dir / "last.pt", model, args, val_metrics, epoch)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            save_checkpoint(args.output_dir / "best.pt", model, args, val_metrics, epoch)

        with history_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {"best_f1": best_f1, "num_train": len(train_ds), "num_val": len(val_ds), "args": vars(args)}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"best checkpoint: {args.output_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

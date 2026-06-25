#!/usr/bin/env python3
"""Reflection mask/point-label benchmark for 3DRef-style evaluation.

Modes:
  image           Compare binary predicted masks against GT image masks.
  semantickitti   Compare SemanticKITTI .label files or .npy label arrays.

The aggregate metrics are the usual reflection-detection benchmark numbers:
IoU, precision, recall, F1, accuracy, and balanced accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy", ".npz"}
LABEL_SUFFIXES = {".label", ".npy", ".npz", ".txt"}


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update(self, other: "Counts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.tn += other.tn


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def metrics_from_counts(c: Counts) -> Dict[str, float]:
    precision = safe_div(c.tp, c.tp + c.fp)
    recall = safe_div(c.tp, c.tp + c.fn)
    specificity = safe_div(c.tn, c.tn + c.fp)
    return {
        "tp": float(c.tp),
        "fp": float(c.fp),
        "fn": float(c.fn),
        "tn": float(c.tn),
        "iou": safe_div(c.tp, c.tp + c.fp + c.fn),
        "precision": precision,
        "recall": recall,
        "f1": safe_div(2.0 * precision * recall, precision + recall),
        "accuracy": safe_div(c.tp + c.tn, c.tp + c.fp + c.fn + c.tn),
        "balanced_accuracy": 0.5 * (recall + specificity),
    }


def binary_counts(gt: np.ndarray, pred: np.ndarray) -> Counts:
    if gt.shape != pred.shape:
        raise ValueError(f"shape mismatch: gt {gt.shape}, pred {pred.shape}")
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    return Counts(
        tp=int(np.logical_and(gt, pred).sum()),
        fp=int(np.logical_and(~gt, pred).sum()),
        fn=int(np.logical_and(gt, ~pred).sum()),
        tn=int(np.logical_and(~gt, ~pred).sum()),
    )


def read_image_like(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        key = "arr_0" if "arr_0" in data else sorted(data.files)[0]
        return data[key]

    try:
        import cv2  # type: ignore

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"cv2 could not read {path}")
        if img.ndim == 3:
            img = img[..., 0]
        return img
    except ImportError:
        from PIL import Image  # type: ignore

        img = Image.open(path).convert("L")
        return np.asarray(img)


def read_label_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".label":
        raw = np.fromfile(path, dtype=np.uint32)
        return raw & 0xFFFF
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        data = np.load(path)
        key = "arr_0" if "arr_0" in data else sorted(data.files)[0]
        return data[key]
    if suffix == ".txt":
        return np.loadtxt(path, dtype=np.int64)
    raise ValueError(f"unsupported label suffix: {path}")


def threshold_binary(arr: np.ndarray, threshold: float) -> np.ndarray:
    if arr.dtype == np.bool_:
        return arr
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float64) > threshold


def positive_label_mask(arr: np.ndarray, positive_labels: Sequence[int]) -> np.ndarray:
    labels = np.asarray(positive_labels, dtype=arr.dtype)
    return np.isin(arr, labels)


def collect_files(root: Path, suffixes: Sequence[str]) -> Dict[str, Path]:
    suffix_set = {s.lower() for s in suffixes}
    files: Dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffix_set:
            rel = str(path.relative_to(root).with_suffix(""))
            files[rel] = path
    return files


def match_files(gt_dir: Path, pred_dir: Path, suffixes: Sequence[str]) -> List[Tuple[str, Path, Path]]:
    gt_files = collect_files(gt_dir, suffixes)
    pred_files = collect_files(pred_dir, suffixes)
    common = sorted(set(gt_files) & set(pred_files))
    if not common:
        raise ValueError(f"no matched files under {gt_dir} and {pred_dir}")
    missing_pred = sorted(set(gt_files) - set(pred_files))
    if missing_pred:
        print(f"warning: {len(missing_pred)} GT files have no prediction")
    return [(key, gt_files[key], pred_files[key]) for key in common]


def write_outputs(output_json: Path, output_csv: Optional[Path], aggregate: Dict[str, float], rows: List[Dict[str, float]]) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump({"aggregate": aggregate, "per_sample": rows}, f, indent=2, sort_keys=True)
        f.write("\n")
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["sample", "tp", "fp", "fn", "tn", "iou", "precision", "recall", "f1", "accuracy", "balanced_accuracy"]
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def run_image(args: argparse.Namespace) -> Dict[str, float]:
    pairs = match_files(args.gt_dir, args.pred_dir, sorted(IMAGE_SUFFIXES))
    total = Counts()
    rows: List[Dict[str, float]] = []
    for key, gt_path, pred_path in pairs:
        gt = threshold_binary(read_image_like(gt_path), args.gt_threshold)
        pred = threshold_binary(read_image_like(pred_path), args.pred_threshold)
        counts = binary_counts(gt, pred)
        total.update(counts)
        row = {"sample": key}
        row.update(metrics_from_counts(counts))
        rows.append(row)
    aggregate = metrics_from_counts(total)
    aggregate["num_samples"] = float(len(rows))
    write_outputs(args.output_json, args.output_csv, aggregate, rows)
    return aggregate


def run_semantickitti(args: argparse.Namespace) -> Dict[str, float]:
    positive_labels = [int(x) for x in args.positive_labels.split(",") if x.strip()]
    pairs = match_files(args.gt_dir, args.pred_dir, sorted(LABEL_SUFFIXES))
    total = Counts()
    rows: List[Dict[str, float]] = []
    for key, gt_path, pred_path in pairs:
        gt = positive_label_mask(read_label_array(gt_path), positive_labels)
        pred = positive_label_mask(read_label_array(pred_path), positive_labels)
        counts = binary_counts(gt, pred)
        total.update(counts)
        row = {"sample": key}
        row.update(metrics_from_counts(counts))
        rows.append(row)
    aggregate = metrics_from_counts(total)
    aggregate["num_samples"] = float(len(rows))
    aggregate["positive_labels"] = float(len(positive_labels))
    write_outputs(args.output_json, args.output_csv, aggregate, rows)
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    image = sub.add_parser("image", help="evaluate RGB mask folders")
    image.add_argument("--gt-dir", required=True, type=Path)
    image.add_argument("--pred-dir", required=True, type=Path)
    image.add_argument("--gt-threshold", type=float, default=0.0)
    image.add_argument("--pred-threshold", type=float, default=127.0)
    image.add_argument("--output-json", type=Path, default=Path("evaluation_tools/results/3dref_image.json"))
    image.add_argument("--output-csv", type=Path, default=Path("evaluation_tools/results/3dref_image_per_sample.csv"))

    sem = sub.add_parser("semantickitti", help="evaluate point labels in SemanticKITTI-style folders")
    sem.add_argument("--gt-dir", required=True, type=Path)
    sem.add_argument("--pred-dir", required=True, type=Path)
    sem.add_argument(
        "--positive-labels",
        default="1",
        help="comma-separated semantic IDs treated as reflective; set this to the 3DRef label IDs you use",
    )
    sem.add_argument("--output-json", type=Path, default=Path("evaluation_tools/results/3dref_semantickitti.json"))
    sem.add_argument("--output-csv", type=Path, default=Path("evaluation_tools/results/3dref_semantickitti_per_sample.csv"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "image":
        aggregate = run_image(args)
    elif args.command == "semantickitti":
        aggregate = run_semantickitti(args)
    else:
        raise AssertionError(args.command)

    print(
        "IoU/F1/Precision/Recall/Accuracy: "
        f"{aggregate['iou']:.6f} {aggregate['f1']:.6f} "
        f"{aggregate['precision']:.6f} {aggregate['recall']:.6f} {aggregate['accuracy']:.6f}"
    )
    print(f"wrote: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

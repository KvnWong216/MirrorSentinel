#!/usr/bin/env python3
"""Reflection-removal metrics for map filtering outputs.

This is not a standalone segmentation benchmark.  It evaluates whether a map
cleaning method removes 3DRef-style reflection / virtual points while retaining
ordinary indoor structure.  It can be used with 3DRef SemanticKITTI labels or
with equivalent per-point labels from self-collected annotated maps.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


DEFAULT_REFLECTION_LABELS = "5"
DEFAULT_INDOOR_LABELS = "1,2,3,4,6"


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def parse_labels(text: str) -> List[int]:
    labels = [int(x) for x in text.split(",") if x.strip()]
    if not labels:
        raise ValueError("label list must not be empty")
    return labels


def read_label_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".label":
        return np.fromfile(path, dtype=np.uint32) & 0xFFFF
    if suffix == ".npy":
        return np.load(path)
    if suffix == ".npz":
        data = np.load(path)
        key = "labels" if "labels" in data else "arr_0" if "arr_0" in data else sorted(data.files)[0]
        return data[key]
    if suffix in {".txt", ".csv"}:
        return np.loadtxt(path, dtype=np.int64)
    raise ValueError(f"unsupported label suffix: {path}")


def read_index_mask(path: Path, n: int) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        key = "kept_mask" if "kept_mask" in data else "kept_indices" if "kept_indices" in data else sorted(data.files)[0]
        arr = data[key]
    elif suffix in {".txt", ".csv"}:
        arr = np.loadtxt(path)
    else:
        raise ValueError(f"unsupported kept-index suffix: {path}")

    arr = np.asarray(arr)
    if arr.dtype == np.bool_:
        mask = arr.reshape(-1)
        if mask.size != n:
            raise ValueError(f"kept boolean mask has {mask.size} entries, expected {n}")
        return mask
    if arr.size == n and np.all(np.isin(arr, [0, 1])):
        return arr.astype(bool).reshape(-1)
    indices = arr.astype(np.int64).reshape(-1)
    if indices.size and (indices.min() < 0 or indices.max() >= n):
        raise ValueError(f"kept indices out of range for {n} labels")
    mask = np.zeros(n, dtype=bool)
    mask[indices] = True
    return mask


def compute_metrics(
    labels: np.ndarray,
    kept_mask: np.ndarray,
    *,
    reflection_labels: Sequence[int],
    indoor_labels: Sequence[int],
) -> Dict[str, float]:
    labels = np.asarray(labels).reshape(-1)
    kept_mask = np.asarray(kept_mask, dtype=bool).reshape(-1)
    if labels.shape[0] != kept_mask.shape[0]:
        raise ValueError(f"label/kept length mismatch: {labels.shape[0]} vs {kept_mask.shape[0]}")

    reflection = np.isin(labels, np.asarray(reflection_labels, dtype=labels.dtype))
    indoor = np.isin(labels, np.asarray(indoor_labels, dtype=labels.dtype))
    non_reflection = ~reflection
    removed = ~kept_mask

    reflection_total = float(np.count_nonzero(reflection))
    indoor_total = float(np.count_nonzero(indoor))
    non_reflection_total = float(np.count_nonzero(non_reflection))
    kept_total = float(np.count_nonzero(kept_mask))
    removed_total = float(np.count_nonzero(removed))
    reflection_removed = float(np.count_nonzero(reflection & removed))
    reflection_retained = float(np.count_nonzero(reflection & kept_mask))
    indoor_retained = float(np.count_nonzero(indoor & kept_mask))
    non_reflection_retained = float(np.count_nonzero(non_reflection & kept_mask))
    false_removed_non_reflection = float(np.count_nonzero(non_reflection & removed))

    return {
        "point_count": float(labels.shape[0]),
        "kept_point_count": kept_total,
        "removed_point_count": removed_total,
        "reflection_point_count": reflection_total,
        "indoor_point_count": indoor_total,
        "non_reflection_point_count": non_reflection_total,
        "reflection_removed_points": reflection_removed,
        "reflection_residual_points": reflection_retained,
        "reflection_removal_rate": safe_div(reflection_removed, reflection_total),
        "reflection_residual_rate": safe_div(reflection_retained, reflection_total),
        "indoor_retention_rate": safe_div(indoor_retained, indoor_total),
        "non_reflection_retention_rate": safe_div(non_reflection_retained, non_reflection_total),
        "non_reflection_precision": safe_div(non_reflection_retained, kept_total),
        "indoor_precision": safe_div(indoor_retained, kept_total),
        "false_removal_rate_non_reflection": safe_div(false_removed_non_reflection, non_reflection_total),
        "map_retention_rate": safe_div(kept_total, float(labels.shape[0])),
    }


def collect_label_files(root: Path) -> Dict[str, Path]:
    suffixes = {".label", ".npy", ".npz", ".txt", ".csv"}
    return {str(p.relative_to(root).with_suffix("")): p for p in sorted(root.rglob("*")) if p.suffix.lower() in suffixes}


def write_outputs(output_json: Path, output_csv: Path | None, aggregate: Dict[str, float], rows: List[Dict[str, float]]) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps({"aggregate": aggregate, "per_sample": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if output_csv is None:
        return
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample"] + sorted(aggregate.keys())
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def average_rows(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    keys = sorted({key for row in rows for key in row if key != "sample"})
    return {key: float(np.mean([float(row[key]) for row in rows if key in row])) for key in keys}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, help="single .label/.npy/.npz/.txt label array")
    parser.add_argument("--kept", type=Path, help="single kept boolean mask or kept-index file")
    parser.add_argument("--label-dir", type=Path, help="folder of GT label arrays")
    parser.add_argument("--kept-dir", type=Path, help="folder of kept mask/index arrays with matching relative names")
    parser.add_argument("--reflection-labels", default=DEFAULT_REFLECTION_LABELS)
    parser.add_argument("--indoor-labels", default=DEFAULT_INDOOR_LABELS)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    reflection_labels = parse_labels(args.reflection_labels)
    indoor_labels = parse_labels(args.indoor_labels)

    rows: List[Dict[str, float]] = []
    if args.labels and args.kept:
        labels = read_label_array(args.labels)
        kept = read_index_mask(args.kept, labels.size)
        row: Dict[str, float] = {"sample": str(args.labels)}
        row.update(
            compute_metrics(labels, kept, reflection_labels=reflection_labels, indoor_labels=indoor_labels)
        )
        rows.append(row)
    elif args.label_dir and args.kept_dir:
        label_files = collect_label_files(args.label_dir)
        kept_files = collect_label_files(args.kept_dir)
        common = sorted(set(label_files) & set(kept_files))
        if not common:
            raise SystemExit(f"no matched label/kept files under {args.label_dir} and {args.kept_dir}")
        for key in common:
            labels = read_label_array(label_files[key])
            kept = read_index_mask(kept_files[key], labels.size)
            row = {"sample": key}
            row.update(
                compute_metrics(labels, kept, reflection_labels=reflection_labels, indoor_labels=indoor_labels)
            )
            rows.append(row)
    else:
        raise SystemExit("pass either --labels/--kept or --label-dir/--kept-dir")

    aggregate = average_rows(rows)
    aggregate["num_samples"] = float(len(rows))
    write_outputs(args.output_json, args.output_csv, aggregate, rows)
    print(
        "Reflection removal: "
        f"RRR={aggregate.get('reflection_removal_rate', 0.0):.6f}, "
        f"indoor_precision={aggregate.get('indoor_precision', 0.0):.6f}, "
        f"retention={aggregate.get('map_retention_rate', 0.0):.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

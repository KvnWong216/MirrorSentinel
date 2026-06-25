#!/usr/bin/env python3
"""Batch 3DRef-style reflection mask evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


def load_config(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    if yaml is None:
        raise SystemExit("PyYAML is required for YAML configs. Install pyyaml or pass a JSON config.")
    return yaml.safe_load(text)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["split", "gt_dir", "pred_dir", "iou", "f1", "precision", "recall", "accuracy", "num_samples", "status", "message"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("evaluation_tools/configs/benchmark_plan.yaml"))
    parser.add_argument("--pred-threshold", type=float, default=127.0)
    parser.add_argument("--gt-threshold", type=float, default=0.0)
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    section = cfg["reflection_benchmark"]
    script = Path(__file__).with_name("mirror_segmentation_eval.py")
    eval_root = Path(section["eval_root"])
    rows: List[Dict[str, Any]] = []

    for split in section["splits"]:
        name = split["name"]
        gt_dir = Path(split["gt_dir"])
        pred_dir = Path(split["pred_dir"])
        out_json = eval_root / f"{name}.json"
        out_csv = eval_root / f"{name}_per_sample.csv"
        row: Dict[str, Any] = {
            "split": name,
            "gt_dir": str(gt_dir),
            "pred_dir": str(pred_dir),
            "status": "pending",
            "message": "",
        }
        if not gt_dir.exists() or not pred_dir.exists():
            row["status"] = "missing"
            row["message"] = f"missing {'gt_dir' if not gt_dir.exists() else 'pred_dir'}"
            rows.append(row)
            if args.strict:
                raise SystemExit(row["message"])
            continue
        cmd = [
            sys.executable,
            str(script),
            "image",
            "--gt-dir",
            str(gt_dir),
            "--pred-dir",
            str(pred_dir),
            "--gt-threshold",
            str(args.gt_threshold),
            "--pred-threshold",
            str(args.pred_threshold),
            "--output-json",
            str(out_json),
            "--output-csv",
            str(out_csv),
        ]
        try:
            subprocess.run(cmd, check=True)
            result = json.loads(out_json.read_text(encoding="utf-8"))["aggregate"]
            row.update(
                {
                    "iou": result["iou"],
                    "f1": result["f1"],
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "accuracy": result["accuracy"],
                    "num_samples": result["num_samples"],
                    "status": "ok",
                }
            )
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            row["message"] = str(exc)
            if args.strict:
                raise
        rows.append(row)

    summary_csv = eval_root / "summary.csv"
    summary_json = eval_root / "summary.json"
    write_csv(summary_csv, rows)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote: {summary_csv}")
    print(f"wrote: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

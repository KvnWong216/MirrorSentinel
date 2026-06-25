#!/usr/bin/env python3
"""Batch ATE/RPE evaluation for public SLAM benchmarks.

This reads a small YAML/JSON manifest and calls tum_trajectory_eval.py for each
method/sequence pair with existing TUM-format trajectory files.
"""

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
    if path.suffix.lower() in {".json"}:
        return json.loads(text)
    if yaml is None:
        raise SystemExit("PyYAML is required for YAML configs. Install pyyaml or pass a JSON config.")
    return yaml.safe_load(text)


def metric(summary: Dict[str, Any], group: str, name: str) -> float:
    return float(summary[group]["translation_error_m"][name])


def run_one(
    script: Path,
    gt: Path,
    est: Path,
    out_dir: Path,
    max_difference: float,
    fixed_delta: float,
    delta_unit: str,
) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(script),
        "all",
        "--groundtruth",
        str(gt),
        "--estimate",
        str(est),
        "--max-difference",
        str(max_difference),
        "--fixed-delta",
        "--delta",
        str(fixed_delta),
        "--delta-unit",
        delta_unit,
        "--output-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)
    summary_path = out_dir / "summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "sequence",
        "method",
        "gt",
        "estimate",
        "ate_rmse_m",
        "ate_mean_m",
        "ate_median_m",
        "rpe_trans_rmse_m",
        "rpe_trans_mean_m",
        "rpe_rot_mean_deg",
        "num_matches",
        "num_rpe_pairs",
        "status",
        "message",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("evaluation_tools/configs/benchmark_plan.yaml"))
    parser.add_argument("--section", choices=["public_slam"], default="public_slam")
    parser.add_argument("--fixed-delta", type=float, default=1.0)
    parser.add_argument("--delta-unit", choices=["s", "m", "rad", "f"], default="s")
    parser.add_argument("--strict", action="store_true", help="fail immediately on missing files or eval errors")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    section = cfg[args.section]
    root = Path(".")
    script = Path(__file__).with_name("tum_trajectory_eval.py")
    dataset = section["dataset"]
    gt_root = root / section["tum_groundtruth_root"]
    est_root = root / section["exported_estimate_root"]
    eval_root = root / section["eval_root"]
    max_difference = float(section.get("max_difference", 0.02))
    rows: List[Dict[str, Any]] = []

    for seq in section["sequences"]:
        gt = gt_root / f"{seq}.txt"
        for method in section["methods"]:
            est = est_root / method / f"{seq}.txt"
            out_dir = eval_root / method / seq
            row: Dict[str, Any] = {
                "dataset": dataset,
                "sequence": seq,
                "method": method,
                "gt": str(gt),
                "estimate": str(est),
                "status": "pending",
                "message": "",
            }
            if not gt.exists() or not est.exists():
                row["status"] = "missing"
                row["message"] = f"missing {'gt' if not gt.exists() else 'estimate'}"
                rows.append(row)
                if args.strict:
                    raise SystemExit(row["message"])
                continue
            try:
                summary = run_one(script, gt, est, out_dir, max_difference, args.fixed_delta, args.delta_unit)
                ate = summary["ate"]
                rpe = summary["rpe"]
                row.update(
                    {
                        "ate_rmse_m": metric(ate, "translation_error_m", "rmse"),
                        "ate_mean_m": metric(ate, "translation_error_m", "mean"),
                        "ate_median_m": metric(ate, "translation_error_m", "median"),
                        "rpe_trans_rmse_m": metric(rpe, "translation_error_m", "rmse"),
                        "rpe_trans_mean_m": metric(rpe, "translation_error_m", "mean"),
                        "rpe_rot_mean_deg": float(rpe["rotation_error_deg"]["mean"]),
                        "num_matches": int(ate["num_matches"]),
                        "num_rpe_pairs": int(rpe["num_pairs"]),
                        "status": "ok",
                    }
                )
            except Exception as exc:  # noqa: BLE001
                row["status"] = "error"
                row["message"] = str(exc)
                if args.strict:
                    raise
            rows.append(row)

    csv_path = eval_root / "summary.csv"
    json_path = eval_root / "summary.json"
    write_csv(csv_path, rows)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote: {csv_path}")
    print(f"wrote: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

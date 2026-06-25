#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


DEFAULT_ROWS = [
    ("SATNet", "mirror", "evaluation_tools/results/3dref/baselines/satnet_mirror_eval.json", "official RGB reflection segmentation baseline"),
    ("SATNet", "alllabel", "evaluation_tools/results/3dref/baselines/satnet_alllabel_eval.json", "official RGB reflection segmentation baseline"),
    ("EBLNet", "glass", "evaluation_tools/results/3dref/baselines/eblnet_glass_eval.json", "official RGB reflection segmentation baseline"),
    ("EBLNet", "alllabel", "evaluation_tools/results/3dref/baselines/eblnet_alllabel_eval.json", "official RGB reflection segmentation baseline"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Summarize 3DRef baseline mask metrics.')
    parser.add_argument('--out-csv', type=Path, default=Path('evaluation_tools/results/3dref/baselines/summary.csv'))
    parser.add_argument('--out-md', type=Path, default=Path('evaluation_tools/results/3dref/baselines/summary.md'))
    return parser


def load_aggregate(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text(encoding='utf-8'))
    return data['aggregate']


def format_float(value: float) -> str:
    return f'{value:.6f}'


def main() -> int:
    args = build_parser().parse_args()
    rows: List[Dict[str, str]] = []
    for method, split, json_path, role in DEFAULT_ROWS:
        path = Path(json_path)
        row = {
            'method': method,
            'split': split,
            'role': role,
            'eval_json': str(path),
        }
        if not path.exists():
            row.update({'status': 'missing'})
        else:
            agg = load_aggregate(path)
            row.update(
                {
                    'status': 'ok',
                    'num_samples': str(int(agg.get('num_samples', 0))),
                    'iou': format_float(agg['iou']),
                    'f1': format_float(agg['f1']),
                    'precision': format_float(agg['precision']),
                    'recall': format_float(agg['recall']),
                    'accuracy': format_float(agg['accuracy']),
                }
            )
        rows.append(row)

    fields = ['method', 'split', 'role', 'status', 'num_samples', 'iou', 'f1', 'precision', 'recall', 'accuracy', 'eval_json']
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    md_lines = [
        '# 3DRef RGB Reflection Segmentation Baselines',
        '',
        'These numbers evaluate RGB reflection-mask prediction quality only. They are not SLAM mapping-quality results.',
        '',
        '| Method | Split | N | IoU | F1 | Precision | Recall | Accuracy |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        if row['status'] != 'ok':
            md_lines.append(f"| {row['method']} | {row['split']} | missing | | | | | |")
            continue
        md_lines.append(
            f"| {row['method']} | {row['split']} | {row['num_samples']} | {row['iou']} | "
            f"{row['f1']} | {row['precision']} | {row['recall']} | {row['accuracy']} |"
        )
    md_lines.extend(
        [
            '',
            'Interpretation: use this table to justify the quality of the non-Lambertian visual prior. ',
            'For the paper main claim, the SLAM evidence still needs bag replay, map saving, ghost-point/thickness metrics, and ablations against FAST-LIO/LIO-SAM on the same sensor stream.',
        ]
    )
    args.out_md.write_text('\n'.join(md_lines) + '\n', encoding='utf-8')
    print(f'wrote: {args.out_csv}')
    print(f'wrote: {args.out_md}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

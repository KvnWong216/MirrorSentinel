#!/usr/bin/env python3
"""Generate timestamp-aligned directory masks for a ROS2 bag.

This is intended for oracle or hand-annotated self-collected experiments.  The
output directory can be played by `mask_prior_node.py` as `/vfm/mirror_mask`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import yaml
from PIL import Image, ImageDraw
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

from extract_bag_images import ros_image_to_pil


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def stamp_to_name(stamp) -> str:
    return f"{int(stamp.sec)}.{int(stamp.nanosec):09d}"


def normalize_box(box: Iterable[float], width: int, height: int) -> Tuple[int, int, int, int]:
    vals = [float(v) for v in box]
    if len(vals) != 4:
        raise ValueError(f"box expects [x0, y0, x1, y1], got: {box}")
    x0, y0, x1, y1 = vals
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) <= 1.0:
        x0, x1 = x0 * width, x1 * width
        y0, y1 = y0 * height, y1 * height
    xa = int(round(max(0.0, min(width - 1, min(x0, x1)))))
    xb = int(round(max(0.0, min(width, max(x0, x1)))))
    ya = int(round(max(0.0, min(height - 1, min(y0, y1)))))
    yb = int(round(max(0.0, min(height, max(y0, y1)))))
    return xa, ya, xb, yb


def normalize_polygon(poly: Iterable[Iterable[float]], width: int, height: int) -> List[Tuple[int, int]]:
    points = []
    for point in poly:
        vals = [float(v) for v in point]
        if len(vals) != 2:
            raise ValueError(f"polygon point expects [x, y], got: {point}")
        x, y = vals
        if abs(x) <= 1.0 and abs(y) <= 1.0:
            x *= width
            y *= height
        points.append((int(round(max(0.0, min(width - 1, x)))), int(round(max(0.0, min(height - 1, y))))))
    return points


def load_config(path: Path) -> Dict[str, object]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected YAML mapping")
    data.setdefault("segments", [])
    return data


def segment_matches(segment: Dict[str, object], frame_idx: int, stamp_s: float) -> bool:
    frame_range = segment.get("frame_range")
    if frame_range is not None:
        vals = [int(v) for v in frame_range]
        if len(vals) != 2:
            raise ValueError(f"frame_range expects [start, end], got: {frame_range}")
        if not (vals[0] <= frame_idx <= vals[1]):
            return False

    time_range = segment.get("time_range_s")
    if time_range is not None:
        vals = [float(v) for v in time_range]
        if len(vals) != 2:
            raise ValueError(f"time_range_s expects [start, end], got: {time_range}")
        if not (vals[0] <= stamp_s <= vals[1]):
            return False

    return True


def draw_segments(mask: Image.Image, segments: List[Dict[str, object]], frame_idx: int, stamp_s: float) -> None:
    draw = ImageDraw.Draw(mask)
    width, height = mask.size
    for segment in segments:
        if not segment_matches(segment, frame_idx, stamp_s):
            continue
        fill = int(segment.get("value", 255))
        for box in segment.get("boxes", []) or []:
            draw.rectangle(normalize_box(box, width, height), fill=fill)
        for poly in segment.get("polygons", []) or []:
            points = normalize_polygon(poly, width, height)
            if len(points) >= 3:
                draw.polygon(points, fill=fill)


def dilate_mask(mask: Image.Image, radius: int) -> Image.Image:
    if radius <= 0:
        return mask
    try:
        from PIL import ImageFilter

        return mask.filter(ImageFilter.MaxFilter(radius * 2 + 1))
    except Exception:
        return mask


def make_overlay(image: Image.Image, mask: Image.Image, alpha: float = 0.40) -> Image.Image:
    rgb = image.convert("RGB")
    mask_l = mask.convert("L")
    red = Image.new("RGB", rgb.size, (255, 0, 0))
    tinted = Image.blend(rgb, red, alpha)
    out = rgb.copy()
    out.paste(tinted, mask=mask_l)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--topic", default="/zed2/zed_node/left/image_rect_color")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dilate", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--save-preview", action="store_true")
    parser.add_argument("--save-overlay-preview", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    segments = list(config.get("segments", []))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = args.out_dir / "_preview"
    if args.save_preview or args.save_overlay_preview:
        preview_dir.mkdir(parents=True, exist_ok=True)

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    seen = 0
    saved = 0
    active = 0
    with AnyReader([args.bag], default_typestore=typestore) as reader:
        connections = [conn for conn in reader.connections if conn.topic == args.topic]
        if not connections:
            topics = ", ".join(sorted({conn.topic for conn in reader.connections}))
            raise SystemExit(f"topic {args.topic!r} not found. Available topics: {topics}")
        for conn, _timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, conn.msgtype)
            stamp_s = stamp_to_sec(msg.header.stamp)
            mask = Image.fromarray(np.zeros((int(msg.height), int(msg.width)), dtype=np.uint8), mode="L")
            draw_segments(mask, segments, seen, stamp_s)
            mask = dilate_mask(mask, args.dilate)
            if np.asarray(mask).max() > 0:
                active += 1
            name = stamp_to_name(msg.header.stamp)
            mask.save(args.out_dir / f"{name}.png")
            if args.save_preview and (seen % int(config.get("preview_every", 80)) == 0):
                mask.convert("RGB").save(preview_dir / f"{seen:06d}_{name}.jpg", quality=92)
            if args.save_overlay_preview and (seen % int(config.get("preview_every", 80)) == 0):
                image = ros_image_to_pil(msg)
                overlay = make_overlay(image, mask)
                overlay.save(preview_dir / f"{seen:06d}_{name}_overlay.jpg", quality=92)
            saved += 1
            seen += 1
            if args.max_images > 0 and saved >= args.max_images:
                break

    metadata = {
        "bag": str(args.bag),
        "topic": args.topic,
        "config": str(args.config),
        "num_masks": saved,
        "active_masks": active,
    }
    (args.out_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
    print(f"seen={seen} saved={saved} active={active} out={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

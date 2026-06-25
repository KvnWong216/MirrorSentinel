#!/usr/bin/env python3
"""Extract sampled images from a ROS2 bag using pure Python dependencies."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Set

import numpy as np
from PIL import Image
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


def stamp_to_name(stamp) -> str:
    return f"{int(stamp.sec)}.{int(stamp.nanosec):09d}"


def ros_image_to_pil(msg) -> Image.Image:
    width = int(msg.width)
    height = int(msg.height)
    step = int(msg.step)
    encoding = str(msg.encoding).lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)
    rows = data.reshape(height, step)

    if encoding in ("bgra8", "rgba8"):
        arr = rows[:, : width * 4].reshape(height, width, 4)
        if encoding == "bgra8":
            arr = arr[..., [2, 1, 0, 3]]
        return Image.fromarray(arr, mode="RGBA").convert("RGB")
    if encoding in ("bgr8", "rgb8"):
        arr = rows[:, : width * 3].reshape(height, width, 3)
        if encoding == "bgr8":
            arr = arr[..., ::-1]
        return Image.fromarray(arr, mode="RGB")
    if encoding in ("mono8", "8uc1"):
        arr = rows[:, :width]
        return Image.fromarray(arr, mode="L")
    raise ValueError(f"unsupported image encoding: {msg.encoding}")


def parse_indices(text: str) -> Optional[Set[int]]:
    if not text:
        return None
    indices: Set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            bits = [int(x) if x else None for x in part.split(":")]
            if len(bits) == 2:
                start, stop = bits
                step = 1
            elif len(bits) == 3:
                start, stop, step = bits
            else:
                raise ValueError(f"invalid index range: {part}")
            if start is None or stop is None:
                raise ValueError(f"range needs explicit start and stop: {part}")
            indices.update(range(start, stop, step or 1))
        else:
            indices.add(int(part))
    return indices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag", type=Path)
    parser.add_argument("--topic", default="/zed2/zed_node/left/image_rect_color")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--every", type=int, default=0, help="save every Nth image; 0 disables periodic sampling")
    parser.add_argument("--indices", default="", help="comma-separated indices or start:stop:step ranges")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--quality", type=int, default=92)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    wanted = parse_indices(args.indices)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    seen = 0
    saved = 0
    with AnyReader([args.bag], default_typestore=typestore) as reader:
        connections = [conn for conn in reader.connections if conn.topic == args.topic]
        if not connections:
            topics = ", ".join(sorted({conn.topic for conn in reader.connections}))
            raise SystemExit(f"topic {args.topic!r} not found. Available topics: {topics}")

        for conn, _timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, conn.msgtype)
            save = False
            if wanted is not None and seen in wanted:
                save = True
            if args.every > 0 and seen % args.every == 0:
                save = True
            if save:
                image = ros_image_to_pil(msg)
                stamp = stamp_to_name(msg.header.stamp)
                out = args.out_dir / f"{seen:06d}_{stamp}.jpg"
                image.save(out, quality=args.quality)
                saved += 1
                if args.max_images > 0 and saved >= args.max_images:
                    break
            seen += 1

    print(f"seen={seen} saved={saved} out={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

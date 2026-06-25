#!/usr/bin/env python3
"""Prepare NeuralRecon demo input from a ROS2 bag and a TUM trajectory.

NeuralRecon's custom demo expects:

  DATAROOT/
    images/0.jpg, 1.jpg, ...
    fragments.pkl
    SyncedPoses.txt

The generated fragments use camera-to-world poses in the Sentinel-LIO map frame.
Images are extracted from a ROS2 image topic and matched to the nearest TUM pose.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml
from PIL import Image
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_TOOLS = REPO_ROOT / "evaluation_tools"
if str(EVAL_TOOLS) not in sys.path:
    sys.path.append(str(EVAL_TOOLS))

from extract_bag_images import ros_image_to_pil  # noqa: E402


@dataclass
class Pose:
    stamp: float
    matrix: np.ndarray


@dataclass
class ImageRecord:
    image_id: int
    stamp: float
    source_index: int
    path: str
    pose_index: int
    pose_stamp: float
    pose_dt: float


def stamp_to_sec(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def quat_xyzw_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0.0:
        raise ValueError("zero-length quaternion")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.asarray(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def transform_from_rt(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return T


def read_tum(path: Path) -> List[Pose]:
    poses: List[Pose] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                raise ValueError(f"{path}:{line_no}: expected 8 TUM columns")
            stamp, tx, ty, tz, qx, qy, qz, qw = [float(x) for x in parts]
            T = transform_from_rt(quat_xyzw_to_rot(qx, qy, qz, qw), [tx, ty, tz])
            poses.append(Pose(stamp=stamp, matrix=T))
    if not poses:
        raise ValueError(f"{path}: no poses")
    poses.sort(key=lambda p: p.stamp)
    return poses


def nearest_pose_index(times: np.ndarray, stamp: float) -> int:
    idx = int(np.searchsorted(times, stamp))
    if idx <= 0:
        return 0
    if idx >= len(times):
        return len(times) - 1
    before = idx - 1
    return before if abs(times[before] - stamp) <= abs(times[idx] - stamp) else idx


def load_sentinel_camera(params_path: Path) -> Tuple[np.ndarray, np.ndarray, int, int]:
    data = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    ros_params = data.get("laserMapping", {}).get("ros__parameters", {})
    fx = float(ros_params.get("sentinel.camera_matrix.fx", 541.56))
    fy = float(ros_params.get("sentinel.camera_matrix.fy", 541.56))
    cx = float(ros_params.get("sentinel.camera_matrix.cx", 496.91))
    cy = float(ros_params.get("sentinel.camera_matrix.cy", 264.66))
    width = int(ros_params.get("sentinel.image_width", 960))
    height = int(ros_params.get("sentinel.image_height", 540))
    K = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    R = np.asarray(ros_params.get("sentinel.extrinsic_R"), dtype=np.float64).reshape(3, 3)
    t = np.asarray(ros_params.get("sentinel.extrinsic_t"), dtype=np.float64).reshape(3)
    T_body_to_cam = transform_from_rt(R, t)
    return K, T_body_to_cam, width, height


def scale_intrinsics(K: np.ndarray, src_size: Tuple[int, int], dst_size: Tuple[int, int]) -> np.ndarray:
    src_w, src_h = src_size
    dst_w, dst_h = dst_size
    out = K.copy()
    out[0, :] *= float(dst_w) / float(src_w)
    out[1, :] *= float(dst_h) / float(src_h)
    out[2, :] = [0.0, 0.0, 1.0]
    return out


def should_save(index: int, every: int, max_images: int, saved: int) -> bool:
    if every > 0 and index % every != 0:
        return False
    if max_images > 0 and saved >= max_images:
        return False
    return True


def extract_images_and_match_poses(
    *,
    bag: Path,
    topic: str,
    out_dir: Path,
    poses_body: Sequence[Pose],
    T_body_to_cam_pose: np.ndarray,
    every: int,
    max_images: int,
    image_size: Tuple[int, int],
    quality: int,
    max_pose_dt: float,
) -> Tuple[List[ImageRecord], Tuple[int, int]]:
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    times = np.asarray([p.stamp for p in poses_body], dtype=np.float64)
    records: List[ImageRecord] = []
    source_count = 0
    skipped_pose = 0
    raw_size: Optional[Tuple[int, int]] = None

    with AnyReader([bag], default_typestore=typestore) as reader:
        connections = [conn for conn in reader.connections if conn.topic == topic]
        if not connections:
            topics = ", ".join(sorted({conn.topic for conn in reader.connections}))
            raise SystemExit(f"topic {topic!r} not found. Available topics: {topics}")

        for conn, _timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, conn.msgtype)
            if not should_save(source_count, every, max_images, len(records)):
                source_count += 1
                continue

            stamp = stamp_to_sec(msg.header.stamp)
            pose_idx = nearest_pose_index(times, stamp)
            pose = poses_body[pose_idx]
            pose_dt = abs(pose.stamp - stamp)
            if max_pose_dt > 0.0 and pose_dt > max_pose_dt:
                skipped_pose += 1
                source_count += 1
                continue

            pil = ros_image_to_pil(msg)
            if raw_size is None:
                raw_size = pil.size
            if pil.size != image_size:
                pil = pil.resize(image_size, resample=Image.BILINEAR)

            image_id = len(records)
            out_name = f"{image_id}.jpg"
            pil.save(image_dir / out_name, quality=quality)
            records.append(
                ImageRecord(
                    image_id=image_id,
                    stamp=stamp,
                    source_index=source_count,
                    path=f"images/{out_name}",
                    pose_index=pose_idx,
                    pose_stamp=pose.stamp,
                    pose_dt=pose_dt,
                )
            )
            source_count += 1

    if raw_size is None:
        raise ValueError(f"no images extracted from {bag}:{topic}")
    if skipped_pose:
        print(f"warning: skipped {skipped_pose} images with pose dt > {max_pose_dt:.3f}s")
    return records, raw_size


def make_fragments(
    *,
    scene: str,
    records: Sequence[ImageRecord],
    poses_body: Sequence[Pose],
    T_body_to_cam_pose: np.ndarray,
    K: np.ndarray,
    n_views: int,
    stride: int,
) -> List[Dict[str, Any]]:
    if n_views <= 0:
        raise ValueError("n_views must be positive")
    if len(records) < n_views:
        raise ValueError(f"need at least {n_views} images, got {len(records)}")
    if stride <= 0:
        stride = n_views

    fragments: List[Dict[str, Any]] = []
    for start in range(0, len(records) - n_views + 1, stride):
        window = records[start : start + n_views]
        extrinsics = []
        intrinsics = []
        ids = []
        for rec in window:
            T_world_body = poses_body[rec.pose_index].matrix
            T_world_cam = T_world_body @ T_body_to_cam_pose
            extrinsics.append(T_world_cam.astype(np.float32))
            intrinsics.append(K.astype(np.float32))
            ids.append(rec.image_id)
        fragments.append(
            {
                "scene": scene,
                "fragment_id": len(fragments),
                "image_ids": ids,
                "extrinsics": extrinsics,
                "intrinsics": intrinsics,
            }
        )
    return fragments


def write_sidecars(out_dir: Path, records: Sequence[ImageRecord], metadata: Mapping[str, Any]) -> None:
    rows = ["image_id stamp source_index pose_index pose_stamp pose_dt path"]
    for r in records:
        rows.append(
            f"{r.image_id} {r.stamp:.9f} {r.source_index} {r.pose_index} "
            f"{r.pose_stamp:.9f} {r.pose_dt:.9f} {r.path}"
        )
    (out_dir / "timestamps.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (out_dir / "image_metadata.json").write_text(
        json.dumps([r.__dict__ for r in records], indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "conversion_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    # NeuralRecon demo.py only checks for existence to skip ARKit pre-processing.
    (out_dir / "SyncedPoses.txt").write_text(
        "# Placeholder generated by prepare_neuralrecon_demo.py; fragments.pkl is authoritative.\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--image-topic", default="/zed2/zed_node/left/image_rect_color")
    parser.add_argument("--trajectory-tum", required=True, type=Path)
    parser.add_argument("--sentinel-params", type=Path, default=REPO_ROOT / "Sentinal-LIO_ws/src/sentinel_lio/config/sentinel_ouster.yaml")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--scene", default="")
    parser.add_argument("--every", type=int, default=3, help="save every Nth image from the bag")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--n-views", type=int, default=9)
    parser.add_argument("--fragment-stride", type=int, default=9)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--max-pose-dt", type=float, default=0.08)
    parser.add_argument(
        "--extrinsic-direction",
        choices=("body_to_camera", "camera_to_body"),
        default="body_to_camera",
        help=(
            "direction of sentinel.extrinsic_R/t in the params file. "
            "body_to_camera means p_camera = T * p_body and is inverted "
            "before writing NeuralRecon camera-to-world poses."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    scene = args.scene or args.bag.name
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    poses_body = read_tum(args.trajectory_tum)
    K_raw, T_body_to_cam_raw, raw_w, raw_h = load_sentinel_camera(args.sentinel_params)
    if args.extrinsic_direction == "body_to_camera":
        T_body_to_cam_pose = np.linalg.inv(T_body_to_cam_raw)
    else:
        T_body_to_cam_pose = T_body_to_cam_raw

    image_size = (args.image_width, args.image_height)
    K = scale_intrinsics(K_raw, (raw_w, raw_h), image_size)
    records, actual_raw_size = extract_images_and_match_poses(
        bag=args.bag,
        topic=args.image_topic,
        out_dir=out_dir,
        poses_body=poses_body,
        T_body_to_cam_pose=T_body_to_cam_pose,
        every=args.every,
        max_images=args.max_images,
        image_size=image_size,
        quality=args.jpeg_quality,
        max_pose_dt=args.max_pose_dt,
    )
    fragments = make_fragments(
        scene=scene,
        records=records,
        poses_body=poses_body,
        T_body_to_cam_pose=T_body_to_cam_pose,
        K=K,
        n_views=args.n_views,
        stride=args.fragment_stride,
    )

    with (out_dir / "fragments.pkl").open("wb") as f:
        pickle.dump(fragments, f)

    metadata: Dict[str, Any] = {
        "bag": str(args.bag),
        "image_topic": args.image_topic,
        "trajectory_tum": str(args.trajectory_tum),
        "sentinel_params": str(args.sentinel_params),
        "scene": scene,
        "image_count": len(records),
        "fragment_count": len(fragments),
        "n_views": args.n_views,
        "fragment_stride": args.fragment_stride,
        "every": args.every,
        "image_size": list(image_size),
        "bag_image_size": list(actual_raw_size),
        "configured_image_size": [raw_w, raw_h],
        "intrinsics": K.tolist(),
        "raw_intrinsics": K_raw.tolist(),
        "T_body_to_camera_raw": T_body_to_cam_raw.tolist(),
        "T_body_to_camera_pose": T_body_to_cam_pose.tolist(),
        "extrinsic_direction": args.extrinsic_direction,
        "max_pose_dt": args.max_pose_dt,
        "max_observed_pose_dt": max((r.pose_dt for r in records), default=0.0),
    }
    write_sidecars(out_dir, records, metadata)

    print(f"wrote NeuralRecon demo data: {out_dir}")
    print(f"images={len(records)} fragments={len(fragments)} n_views={args.n_views}")
    print(f"intrinsics={K.tolist()}")
    print(f"max_pose_dt={metadata['max_observed_pose_dt']:.6f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

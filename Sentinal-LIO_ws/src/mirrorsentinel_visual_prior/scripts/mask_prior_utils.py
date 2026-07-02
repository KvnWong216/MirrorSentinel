from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


MASK_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class MaskLookup:
    mask: np.ndarray
    path: Optional[Path]
    stamp_delta: Optional[float]


def stamp_to_float(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def stamp_candidate_stems(stamp) -> List[str]:
    sec = int(stamp.sec)
    nsec = int(stamp.nanosec)
    candidates = [
        f"{sec}.{nsec:09d}",
        f"{sec}.{nsec // 1000:06d}",
        f"{sec}.{nsec // 1000000:03d}",
        str(sec),
    ]
    trimmed = f"{sec}.{nsec:09d}".rstrip("0").rstrip(".")
    if trimmed not in candidates:
        candidates.insert(1, trimmed)
    return candidates


def normalize_mask(mask: np.ndarray, width: int, height: int, threshold: float) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    if threshold >= 0.0:
        mask = np.where(mask.astype(np.float32) > threshold, 255, 0)
    return mask.astype(np.uint8)


def empty_mask(width: int, height: int) -> np.ndarray:
    return np.zeros((height, width), dtype=np.uint8)


class DirectoryMaskPrior:
    def __init__(self, mask_dir: Path, threshold: float = 0.0, tolerance_s: float = 0.05) -> None:
        self.mask_dir = Path(mask_dir).expanduser()
        self.threshold = float(threshold)
        self.tolerance_s = float(tolerance_s)
        self.by_stem: Dict[str, Path] = {}
        timed: List[Tuple[float, Path]] = []

        if not self.mask_dir.exists():
            raise FileNotFoundError(f"mask_dir does not exist: {self.mask_dir}")

        for path in sorted(self.mask_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in MASK_SUFFIXES:
                continue
            self.by_stem.setdefault(path.stem, path)
            try:
                timed.append((float(path.stem), path))
            except ValueError:
                pass

        timed.sort(key=lambda item: item[0])
        self.times = [item[0] for item in timed]
        self.paths = [item[1] for item in timed]

    def __len__(self) -> int:
        return len(self.by_stem)

    def find(self, stamp) -> Tuple[Optional[Path], Optional[float]]:
        for stem in stamp_candidate_stems(stamp):
            path = self.by_stem.get(stem)
            if path is not None:
                return path, 0.0

        if not self.times or self.tolerance_s < 0.0:
            return None, None

        target = stamp_to_float(stamp)
        pos = bisect_left(self.times, target)
        best: Optional[Tuple[float, Path]] = None
        for idx in (pos - 1, pos):
            if 0 <= idx < len(self.times):
                delta = abs(self.times[idx] - target)
                if best is None or delta < best[0]:
                    best = (delta, self.paths[idx])

        if best is not None and best[0] <= self.tolerance_s:
            return best[1], best[0]
        return None, None

    def load(self, stamp, width: int, height: int) -> Optional[MaskLookup]:
        path, delta = self.find(stamp)
        if path is None:
            return None
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"failed to read mask image: {path}")
        return MaskLookup(normalize_mask(mask, width, height, self.threshold), path, delta)

#!/usr/bin/env python3
"""MirrorSentinel map cleaner entry point.

This runs the historical frame-prior vote backend: accumulated map points are
reprojected into recorded depth/mask prior frames, and repeatedly contradicted
points behind reflective geometry are removed.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> int:
    target = Path(__file__).resolve().with_name("filter_map_by_frame_prior_votes.py")
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

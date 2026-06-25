#!/usr/bin/env python3
"""MirrorSentinel saved-vote replay entry point.

Use this when point_vote_stats.npz already exists and only the rejection policy
or 100k map budget needs to be swept.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> int:
    target = Path(__file__).resolve().with_name("apply_frame_prior_vote_threshold.py")
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

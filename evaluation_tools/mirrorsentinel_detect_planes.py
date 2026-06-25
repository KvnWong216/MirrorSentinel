#!/usr/bin/env python3
"""MirrorSentinel reflective-plane detector entry point.

This is the user-facing name for the automatic room-boundary detector used by
the MirrorSentinel map backend.  The implementation is kept in
detect_reflective_planes.py for compatibility with existing experiment logs.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> int:
    target = Path(__file__).resolve().with_name("detect_reflective_planes.py")
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

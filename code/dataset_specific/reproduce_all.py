#!/usr/bin/env python3
"""Run one public pipeline stage for all four configured datasets."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("calibrate", "match", "export", "masks"), required=True)
    args = parser.parse_args()
    for config in sorted((ROOT / "configs").glob("000[1-4].json")):
        command = [
            sys.executable,
            str(ROOT / "code/run_pipeline.py"),
            "--config", str(config),
            "--stage", args.stage,
        ]
        subprocess.run(command, check=True, cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

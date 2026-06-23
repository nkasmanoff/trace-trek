#!/usr/bin/env python3
"""Sync Cursor agent transcripts into raw/cursor/ using the existing collector.

Thin wrapper around cursor-traces/collect_traces.py so this repo doesn't
duplicate its discovery logic.

Usage:
    python collect/pull_cursor.py \
        --collector ~/Desktop/cursor-traces/collect_traces.py \
        --dest raw/cursor
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--collector",
        type=Path,
        default=Path.home() / "Desktop" / "cursor-traces" / "collect_traces.py",
        help="Path to cursor-traces/collect_traces.py",
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "raw" / "cursor",
        help="Destination folder for transcripts",
    )
    args = p.parse_args()

    collector = args.collector.expanduser().resolve()
    if not collector.is_file():
        print(f"error: collector not found: {collector}", file=sys.stderr)
        return 1

    dest = args.dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(collector), "--dest", str(dest)]
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

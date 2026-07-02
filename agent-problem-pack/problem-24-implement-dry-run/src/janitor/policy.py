"""Candidate selection policy."""

import time
from pathlib import Path

SUFFIXES = (".tmp", ".log")


def candidates(root, max_age_seconds, now=None):
    """Stale files under root: matching suffix, older than the cutoff.

    Returned in sorted path order so runs are deterministic.
    """
    now = time.time() if now is None else now
    found = []
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or path.suffix not in SUFFIXES:
            continue
        if now - path.stat().st_mtime > max_age_seconds:
            found.append(path)
    return found

"""Timing helpers."""

import time
from contextlib import contextmanager


@contextmanager
def timed(label, sink=None):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if sink is not None:
        sink(f"{label}: {elapsed:.3f}s")

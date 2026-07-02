"""Artifact path conventions for pipeline runs."""

from pathlib import Path


def run_dir(base, run_name):
    return Path(base) / "runs" / run_name


def summary_path(base, run_name):
    return run_dir(base, run_name) / "summary.json"

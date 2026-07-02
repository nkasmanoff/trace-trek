"""Minimal CSV helpers for numeric columns."""

import csv
from pathlib import Path


def read_column(path, column):
    with open(Path(path), newline="", encoding="utf-8") as handle:
        return [float(row[column]) for row in csv.DictReader(handle)]

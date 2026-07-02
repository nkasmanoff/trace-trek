"""JSONL read/write helpers."""

import json
from pathlib import Path


def read_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    text = "".join(json.dumps(row) + "\n" for row in rows)
    Path(path).write_text(text, encoding="utf-8")

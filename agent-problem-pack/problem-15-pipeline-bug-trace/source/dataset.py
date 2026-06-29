"""Stage 1: load eval tasks from a JSONL manifest.

Each line is one task:
    {"task_id": "t1", "type": "code", "prompt": "...", "weight": 2.0}

`weight` is optional and defaults to 1.0. Weights let some tasks count more
than others when the suite computes a weighted score downstream.
"""

import json


def load_tasks(lines):
    tasks = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        record.setdefault("weight", 1.0)
        tasks.append(record)
    return tasks

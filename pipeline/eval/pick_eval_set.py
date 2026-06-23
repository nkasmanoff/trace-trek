#!/usr/bin/env python3
"""Pick a small, stable evaluation set from the held-out test tasks.

The eval set is a handful of tasks that several opencode agents are run
against so their pass rates can be compared side by side.

Selection is deterministic: tasks are ordered by a seeded hash of their id and
drawn round-robin across repos so the handful spans different codebases. Adding
tasks to the pool later never reshuffles an already-chosen set for the same
--seed/--n (the ordering is stable per id).

Usage:
    python eval/pick_eval_set.py [--tasks tasks-test.jsonl] [--n 6] [--seed 0]
        [--out eval/eval-tasks.jsonl]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent


def rank(task_id: str, seed: int) -> int:
    h = hashlib.sha256(f"{seed}:{task_id}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def pick(tasks: list[dict], n: int, seed: int) -> list[dict]:
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for t in sorted(tasks, key=lambda t: rank(t["id"], seed)):
        by_repo[t.get("repo", "")].append(t)
    # round-robin across repos for breadth, then truncate to n
    order = sorted(by_repo, key=lambda r: rank(r, seed))
    chosen: list[dict] = []
    while len(chosen) < n and any(by_repo[r] for r in order):
        for r in order:
            if by_repo[r]:
                chosen.append(by_repo[r].pop(0))
                if len(chosen) >= n:
                    break
    return sorted(chosen, key=lambda t: rank(t["id"], seed))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tasks", type=Path, default=BASE / "tasks-test.jsonl")
    p.add_argument("--n", type=int, default=10, help="size of the eval set")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=BASE / "eval" / "eval-tasks.jsonl")
    args = p.parse_args()

    tasks = [json.loads(l) for l in args.tasks.read_text().splitlines()
             if l.strip()]
    chosen = pick(tasks, args.n, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for t in chosen:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")

    by_type: dict[str, int] = defaultdict(int)
    for t in chosen:
        by_type[t["type"]] += 1
    print(f"eval set: {len(chosen)} tasks {dict(by_type)} -> {args.out}")
    for t in chosen:
        print(f"  [{t['type']:9}] {Path(t['repo']).name:24} "
              f"{t['id']}  {t['prompt'][:60]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

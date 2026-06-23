#!/usr/bin/env python3
"""Harvest held-out evaluation tasks from the SFT eval split.

Each opencode-sft session is a real coding task: a user request followed by the
assistant's ground-truth trajectory. We turn the held-out eval split (the rows
pull_dataset.py reserved and never trains on) into knowledge-style eval tasks:

    {"id": ..., "type": "knowledge",
     "prompt": "<first user turn>",
     "reference": "<ground-truth assistant reasoning + answer + tool calls>"}

These feed eval/run_baseline.py, which runs each prompt through the opencode
harness against a served model and grades the answer by how well it covers the
entities (file paths, symbols, identifiers) in the reference trajectory — the
same grader collect/replay.py uses for knowledge tasks.

Usage:
    python eval/harvest_eval_tasks.py [--eval dataset/sft-eval.jsonl]
        [--out eval/eval-tasks.jsonl] [--max-prompt-chars 4000] [--limit 0]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from collect.replay import extract_entities  # noqa: E402


def first_user_turn(messages: list[dict]) -> str | None:
    for m in messages:
        if m.get("role") == "user" and isinstance(m.get("content"), str) \
                and m["content"].strip():
            return m["content"].strip()
    return None


def reference_text(messages: list[dict]) -> str:
    """Concatenate the ground-truth assistant trajectory (reasoning, content,
    and tool-call names/args) into one reference string to grade against."""
    parts: list[str] = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        if m.get("reasoning_content"):
            parts.append(str(m["reasoning_content"]))
        if isinstance(m.get("content"), str) and m["content"].strip():
            parts.append(m["content"])
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            parts.append(f"{fn.get('name', '')} {args}")
    return "\n".join(parts)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--eval", type=Path, default=BASE / "dataset" / "sft-eval.jsonl")
    p.add_argument("--out", type=Path, default=BASE / "eval" / "eval-tasks.jsonl")
    p.add_argument("--max-prompt-chars", type=int, default=4000,
                   help="cap the harvested prompt length")
    p.add_argument("--min-entities", type=int, default=5,
                   help="skip tasks whose reference has too few gradable entities")
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    args = p.parse_args()

    if not args.eval.is_file():
        raise SystemExit(
            f"{args.eval} not found — run `make split` (pull_dataset.py) first")

    tasks: list[dict] = []
    skipped = 0
    for line in args.eval.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        messages = rec.get("messages") or []
        prompt = first_user_turn(messages)
        reference = reference_text(messages)
        if not prompt or len(extract_entities(reference)) < args.min_entities:
            skipped += 1
            continue
        tid = "sft-eval-" + hashlib.sha256(
            prompt.encode("utf-8")).hexdigest()[:12]
        tasks.append({
            "id": tid,
            "type": "knowledge",
            "prompt": prompt[:args.max_prompt_chars],
            "reference": reference,
            "source": rec.get("source", "opencode-sft"),
        })

    if args.limit:
        tasks = tasks[:args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"harvested {len(tasks)} eval tasks ({skipped} skipped) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

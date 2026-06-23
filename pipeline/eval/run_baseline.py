#!/usr/bin/env python3
"""Baseline opencode-harness eval — run BEFORE training.

Runs each harvested eval task (eval/harvest_eval_tasks.py) through the opencode
harness against a served model and grades the answer, establishing the
base-model score to beat. Run the same command after deploying a fine-tuned
checkpoint (point --model at it) to measure the lift from training.

Each task runs `opencode run -m <model> "<prompt>"` in a fresh scratch dir (no
git repo needed — these are open-ended coding requests) and is graded by how
well the answer covers the entities cited in the ground-truth trajectory, using
collect/replay.py's knowledge grader.

Requires `opencode` on PATH and a reachable model. Serve base Laguna first, e.g.
    python inference/laguna_mlx.py            # local (Apple Silicon)
    modal deploy inference/laguna_modal.py    # cloud (H100 + vLLM)
then point opencode at it; --model is the opencode model id (see eval/agents.json).

Usage:
    python eval/run_baseline.py [--model frontier/poolside/laguna-xs.2]
        [--tasks eval/eval-tasks.jsonl] [--limit 0] [--threshold 0.3]
        [--run-label laguna-base]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from collect.replay import grade_knowledge, run_opencode  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tasks", type=Path, default=BASE / "eval" / "eval-tasks.jsonl")
    p.add_argument("--model", default="frontier/poolside/laguna-xs.2",
                   help="opencode model id (see eval/agents.json)")
    p.add_argument("--run-label", default="laguna-base",
                   help="label recorded on each result row")
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    p.add_argument("--threshold", type=float, default=0.3,
                   help="knowledge pass threshold (entity coverage)")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--results", type=Path, default=None,
                   help="output JSON (default eval/baseline-<label>-<ts>.json)")
    args = p.parse_args()

    if shutil.which("opencode") is None:
        raise SystemExit(
            "opencode is not on PATH — install it and serve the model first "
            "(see inference/laguna_mlx.py or inference/laguna_modal.py).")
    if not args.tasks.is_file():
        raise SystemExit(
            f"{args.tasks} not found — run eval/harvest_eval_tasks.py first")

    tasks = [json.loads(l) for l in args.tasks.read_text().splitlines()
             if l.strip()]
    if args.limit:
        tasks = tasks[:args.limit]

    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = args.results or (
        BASE / "eval" / f"baseline-{args.run_label}-{ts}.json")

    rows, passed, gradable = [], 0, 0
    print(f"running {len(tasks)} tasks through opencode (model={args.model})")
    for i, task in enumerate(tasks, 1):
        workdir = Path(tempfile.mkdtemp(prefix="laguna-baseline-"))
        row = {"task_id": task["id"], "type": task.get("type", "knowledge"),
               "run": args.run_label, "model": args.model}
        try:
            answer = run_opencode(task["prompt"], workdir, args.model,
                                  args.timeout, sandbox=False)
            score, ok_gradable, detail = grade_knowledge(
                task.get("reference", ""), answer)
            row.update(answer_chars=len(answer), score=score,
                       gradable=ok_gradable, detail=detail,
                       passed=bool(ok_gradable and score is not None
                                   and score >= args.threshold))
        except subprocess.TimeoutExpired:
            row.update(passed=False, error="timeout")
        except Exception as exc:  # noqa: BLE001
            row.update(passed=False, error=repr(exc))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        rows.append(row)
        gradable += bool(row.get("gradable"))
        passed += bool(row.get("passed"))
        status = "PASS" if row.get("passed") else "fail"
        print(f"  [{i}/{len(tasks)}] {status} score={row.get('score')} "
              f"{task['prompt'][:60]!r} {row.get('error', '')}")

    summary = {
        "timestamp": ts, "model": args.model, "run": args.run_label,
        "threshold": args.threshold, "total": len(rows),
        "gradable": gradable, "passed": passed,
        "pass_rate": round(passed / len(rows), 3) if rows else 0.0,
        "results": rows,
    }
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nbaseline: {passed}/{len(rows)} passed "
          f"({gradable} gradable) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run the eval set through several opencode agents and collect the results.

Comparison harness for running agents against the eval task set.

  1. a small handful of tasks is picked once (eval/pick_eval_set.py ->
     eval/eval-tasks.jsonl)
  2. each agent in eval/agents.json is run over that same set via
     collect/replay.py, which runs `opencode run` in an isolated git worktree,
     grades the outcome (code: changed-file overlap; knowledge: entity/view
     coverage), and writes one result row per task
  3. every agent's rows are merged into eval/eval-results.json (a JSON array of
     eval rows, one per task per agent, each tagged with its run label)

Each agent is a JSON object in eval/agents.json:
    {"label": "...",        # run label shown on the Eval tab (must be unique)
     "model": "...",        # opencode model id passed to `opencode run -m`
     "agent": null,         # optional opencode agent (`opencode run --agent`)
     "no_synth": false,     # skip knowledge-task artifact reconstruction
     "extra": []}           # any extra args forwarded to replay.py

Per-agent rows land in eval/runs/<label>.jsonl; replay.py skips task ids it has
already recorded there, so a re-run resumes where it left off. Delete a label's
file (or pass --fresh) to re-run that agent from scratch.

Usage:
    python eval/run_eval_set.py                       # all agents, then merge
    python eval/run_eval_set.py --embed               # ...and bake into the HTML
    python eval/run_eval_set.py --only agent-2        # just one agent
    python eval/run_eval_set.py --merge-only --embed  # skip running; re-bake
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
EVAL_DIR = BASE / "eval"
REPLAY = BASE / "collect" / "replay.py"


def load_agents(path: Path) -> list[dict]:
    agents = json.loads(path.read_text())
    if not isinstance(agents, list) or not agents:
        raise SystemExit(f"{path}: expected a non-empty JSON array of agents")
    labels = [a.get("label") for a in agents]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"{path}: agent labels must be unique, got {labels}")
    return agents


def run_agent(agent: dict, tasks: Path, runs_dir: Path, timeout: int,
              fresh: bool) -> Path:
    label = agent["label"]
    out = runs_dir / f"{label}.jsonl"
    if fresh:
        out.unlink(missing_ok=True)
    cmd = [sys.executable, str(REPLAY),
           "--tasks", str(tasks),
           "--model", agent.get("model", "llamacpp/local-code-model"),
           "--run-label", label,
           "--results", str(out),
           "--timeout", str(timeout)]
    if agent.get("agent"):
        cmd += ["--agent", agent["agent"]]
    if agent.get("no_synth"):
        cmd.append("--no-synth")
    cmd += list(agent.get("extra", []))
    print(f"\n=== agent '{label}' (model={agent.get('model')}"
          + (f", opencode-agent={agent['agent']}" if agent.get("agent") else "")
          + ") ===")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=False)
    return out


def merge(runs_dir: Path, out: Path) -> int:
    rows: list[dict] = []
    for f in sorted(runs_dir.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    runs = {}
    for r in rows:
        lbl = r.get("run") or "(unlabeled)"
        b = runs.setdefault(lbl, [0, 0])
        b[0] += 1
        b[1] += bool(r.get("passed"))
    print(f"\nmerged {len(rows)} rows -> {out}")
    for lbl, (tot, ok) in sorted(runs.items()):
        print(f"  {lbl:20} {ok}/{tot} passed")
    return len(rows)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tasks", type=Path, default=EVAL_DIR / "eval-tasks.jsonl")
    p.add_argument("--agents", type=Path, default=EVAL_DIR / "agents.json")
    p.add_argument("--out", type=Path, default=EVAL_DIR / "eval-results.json")
    p.add_argument("--runs-dir", type=Path, default=EVAL_DIR / "runs")
    p.add_argument("--only", default=None, help="run only this agent label")
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--fresh", action="store_true",
                   help="re-run agents from scratch (ignore prior rows)")
    p.add_argument("--merge-only", action="store_true",
                   help="skip running; just merge existing per-agent rows")

    args = p.parse_args()

    if not args.tasks.is_file():
        raise SystemExit(f"{args.tasks} not found — run eval/pick_eval_set.py first")

    args.runs_dir.mkdir(parents=True, exist_ok=True)
    agents = load_agents(args.agents)
    if args.only:
        agents = [a for a in agents if a["label"] == args.only]
        if not agents:
            raise SystemExit(f"no agent labelled {args.only!r} in {args.agents}")

    if not args.merge_only:
        for agent in agents:
            run_agent(agent, args.tasks, args.runs_dir, args.timeout, args.fresh)

    merge(args.runs_dir, args.out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

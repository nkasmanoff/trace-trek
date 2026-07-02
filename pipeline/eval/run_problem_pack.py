#!/usr/bin/env python3
"""Headless agent-problem-pack runner for the training eval-gate.

Runs a subset of the agent-problem-pack against a model served over an
OpenAI-compatible endpoint (e.g. the in-training `eval_server.InProcessModelServer`)
and reports the pass rate as JSON, so `train/train.py --eval-gate-problem-pack`
can log a benchmark trend line to W&B while the model trains.

For each selected problem we:
  1. `pack_tools.prepare_run` -> an isolated git-baselined workspace (hidden
     tests withheld, exactly as a real eval),
  2. `opencode run --model <model> --dir <workspace> <task_prompt>`,
  3. `pack_tools.capture_run` -> inject hidden tests, run the verify command,
     capture diff/verification; pass == verify exit code 0.

When `--base-url` is given we point opencode at it via an isolated opencode
config + data dir (`--opencode-home`), so the gate never touches your real
opencode.db (avoiding the giant-DB lock contention) and needs no provider auth.

Usage:
    python run_problem_pack.py --base-url http://127.0.0.1:8848/v1 \
        --model local/local-code-model --subset smoke --out ../eval \
        --opencode-home /tmp/pack-gate-home
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACK_ROOT = REPO_ROOT / "agent-problem-pack"
sys.path.insert(0, str(PACK_ROOT / "scripts"))

OPENCODE = os.environ.get(
    "OPENCODE_BIN", str(Path.home() / ".opencode" / "bin" / "opencode"))


def select_problems(subset: str, explicit: list[str] | None) -> list[str]:
    import pack_tools
    problems = pack_tools.PROBLEMS
    if explicit:
        missing = [i for i in explicit if i not in problems]
        if missing:
            raise SystemExit(f"unknown problem ids: {missing}")
        return list(explicit)
    ids = list(problems.keys())
    if subset == "all":
        return ids
    if subset in ("easy", "medium", "hard"):
        return [i for i, p in problems.items() if p.difficulty == subset]
    if subset in ("repair", "comprehension", "implement", "grounding"):
        return [i for i, p in problems.items() if p.kind == subset]
    if subset == "new":
        # the hard additions (15+): harness-driving tasks aligned with the
        # opencode SFT trace distribution
        return [i for i in ids if int(i.split("-")[1]) >= 15]
    if subset == "smoke":
        # a couple of quick repairs + the first comprehension: cheap signal
        repairs = [i for i, p in problems.items()
                   if p.kind == "repair" and p.difficulty == "easy"][:2]
        comp = next((i for i, p in problems.items()
                     if p.kind == "comprehension"), None)
        return repairs + ([comp] if comp else [])
    raise SystemExit(f"unknown subset: {subset}")


def write_opencode_config(home: Path, base_url: str, served: str) -> Path:
    """Write an isolated opencode config pointing the `local` provider at the
    in-training server, and return the config path (also exported via
    OPENCODE_CONFIG so it's picked up regardless of cwd)."""
    cfg = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "local": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Local (in-training)",
                "options": {"baseURL": base_url},
                "models": {served: {"name": served}},
            }
        },
    }
    cfg_dir = home / ".config" / "opencode"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "opencode.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg_path


def session_stats(home: Path, workspace: Path) -> dict | None:
    """Best-effort harness-mechanics stats from the isolated opencode.db for
    the session opencode created in this workspace: tokens and steps, plus
    per-tool call counts, tool-call error counts, and tool-call loops (the
    same signals build_dataset.py uses as training-data quality filters).
    Returns None on any problem."""
    import sqlite3
    db = home / ".local" / "share" / "opencode" / "opencode.db"
    if not db.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        wsf = str(workspace.resolve())
        rows = con.execute(
            "SELECT id FROM session WHERE directory = ? "
            "ORDER BY time_updated DESC LIMIT 1", (wsf,)).fetchall()
        if not rows:
            return None
        sid = rows[0]["id"]
        ids = [r["id"] for r in con.execute(
            "SELECT id FROM session WHERE id = ? OR parent_id = ?",
            (sid, sid)).fetchall()]
        ph = ",".join("?" * len(ids))
        tok = con.execute(
            f"SELECT COALESCE(sum(tokens_input),0) i, "
            f"COALESCE(sum(tokens_output),0) o FROM session WHERE id IN ({ph})",
            ids).fetchone()
        steps = con.execute(
            f"SELECT count(*) c FROM message WHERE session_id IN ({ph}) "
            f"AND json_extract(data,'$.role')='assistant'", ids).fetchone()["c"]

        tool_calls: dict[str, int] = {}
        for r in con.execute(
                f"SELECT json_extract(data,'$.tool') t, count(*) c FROM part "
                f"WHERE session_id IN ({ph}) AND "
                f"json_extract(data,'$.type')='tool' GROUP BY 1", ids):
            if r["t"]:
                tool_calls[r["t"]] = int(r["c"])
        tool_errors = con.execute(
            f"SELECT count(*) c FROM part WHERE session_id IN ({ph}) AND "
            f"json_extract(data,'$.type')='tool' AND "
            f"json_extract(data,'$.state.status')='error'", ids).fetchone()["c"]
        edit_errors = con.execute(
            f"SELECT count(*) c FROM part WHERE session_id IN ({ph}) AND "
            f"json_extract(data,'$.type')='tool' AND "
            f"json_extract(data,'$.tool') IN ('edit','write') AND "
            f"json_extract(data,'$.state.status')='error'", ids).fetchone()["c"]

        # Loop detection: 3+ consecutive identical (tool, input) calls.
        seq = [(r["t"], r["i"]) for r in con.execute(
            f"SELECT json_extract(data,'$.tool') t, "
            f"json_extract(data,'$.state.input') i FROM part "
            f"WHERE session_id IN ({ph}) AND "
            f"json_extract(data,'$.type')='tool' ORDER BY rowid", ids)]
        loops, streak = 0, 1
        for prev, cur in zip(seq, seq[1:]):
            if cur == prev and cur[0] is not None:
                streak += 1
                if streak == 3:
                    loops += 1
            else:
                streak = 1
        con.close()
        return {"tokens": int(tok["i"]) + int(tok["o"]),
                "steps": int(steps), "sessionId": sid,
                "toolCalls": sum(tool_calls.values()),
                "toolCallsByName": tool_calls,
                "toolErrors": int(tool_errors),
                "editErrors": int(edit_errors),
                "toolLoops": int(loops)}
    except Exception:  # noqa: BLE001 — stats are optional
        return None


def run_one(problem_id: str, model: str, env: dict, home: Path | None,
            timeout: int, keep_runs: bool) -> dict:
    import pack_tools
    problem = pack_tools.PROBLEMS[problem_id]
    run_name = f"gate-{int(time.time() * 1000)}"
    run_dir = pack_tools.prepare_run(PACK_ROOT, problem_id, run_name)
    workspace = (run_dir / "workspace").resolve()
    prompts = [pack_tools.task_prompt_text(problem)] + list(problem.turns)

    t0 = time.time()
    timed_out = False
    stderr = ""
    try:
        for turn_index, prompt in enumerate(prompts):
            cmd = [OPENCODE, "run", "--model", model,
                   "--dangerously-skip-permissions", "--dir", str(workspace)]
            if turn_index > 0:
                cmd.append("--continue")
            cmd.append(prompt)
            remaining = timeout - (time.time() - t0)
            if remaining <= 0:
                raise subprocess.TimeoutExpired(cmd, timeout)
            proc = subprocess.run(cmd, cwd=str(workspace), env=env,
                                  capture_output=True, text=True,
                                  timeout=remaining)
            stderr = (proc.stderr or "")[-2000:]
    except subprocess.TimeoutExpired:
        timed_out = True
        stderr = f"timeout after {timeout}s"
    seconds = round(time.time() - t0, 1)

    try:
        verification = pack_tools.capture_run(run_dir, root=PACK_ROOT)
        passed = verification.returncode == 0
    except Exception as exc:  # noqa: BLE001
        passed = False
        stderr = (stderr + f"\ncapture failed: {exc!r}")[-2000:]

    stats = session_stats(home, workspace) if home else None
    result = {
        "id": problem_id,
        "kind": problem.kind,
        "difficulty": problem.difficulty,
        "passed": bool(passed),
        "timed_out": timed_out,
        "seconds": seconds,
        "turns": len(prompts),
        "tokens": (stats or {}).get("tokens"),
        "steps": (stats or {}).get("steps"),
        "tool_calls": (stats or {}).get("toolCalls"),
        "tool_calls_by_name": (stats or {}).get("toolCallsByName"),
        "tool_errors": (stats or {}).get("toolErrors"),
        "edit_errors": (stats or {}).get("editErrors"),
        "tool_loops": (stats or {}).get("toolLoops"),
        "stderr": stderr if not passed else "",
    }
    if not keep_runs:
        shutil.rmtree(run_dir, ignore_errors=True)
    return result


def summarize(results: list[dict]) -> dict:
    total = len(results)
    passed = sum(r["passed"] for r in results)
    by_diff: dict[str, dict] = {}
    by_kind: dict[str, dict] = {}
    for r in results:
        for key, bucket in (("difficulty", by_diff), ("kind", by_kind)):
            d = bucket.setdefault(r[key], {"passed": 0, "total": 0})
            d["total"] += 1
            d["passed"] += int(r["passed"])
    toks = [r["tokens"] for r in results if r.get("tokens")]
    steps = [r["steps"] for r in results if r.get("steps")]
    tool_errors = [r["tool_errors"] for r in results
                   if r.get("tool_errors") is not None]
    edit_errors = [r["edit_errors"] for r in results
                   if r.get("edit_errors") is not None]
    loops = [r["tool_loops"] for r in results
             if r.get("tool_loops") is not None]
    tool_calls = [r["tool_calls"] for r in results if r.get("tool_calls")]
    # Cost-adjusted headline numbers: the thesis is "smaller CHEAPER models
    # become viable drivers of the harness", so tokens spent per solved task
    # matters as much as the raw pass rate.
    total_tokens = sum(toks) if toks else None
    return {
        "passed": passed,
        "total": total,
        "rate": (passed / total) if total else 0.0,
        "by_difficulty": by_diff,
        "by_kind": by_kind,
        "mean_tokens": round(sum(toks) / len(toks)) if toks else None,
        "mean_steps": round(sum(steps) / len(steps), 1) if steps else None,
        "total_tokens": total_tokens,
        "tokens_per_solve": (round(total_tokens / passed)
                             if total_tokens and passed else None),
        "tool_error_rate": (round(sum(tool_errors) / sum(tool_calls), 4)
                            if tool_calls and tool_errors else None),
        "edit_errors": sum(edit_errors) if edit_errors else 0,
        "tool_loops": sum(loops) if loops else 0,
        "problems": results,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default="local/local-code-model",
                   help="opencode model id to evaluate")
    p.add_argument("--base-url", default=None,
                   help="OpenAI-compatible base URL to route the `local` "
                        "provider at (in-training server). When set, an "
                        "isolated opencode config + data dir is used.")
    p.add_argument("--served-model", default="local-code-model",
                   help="model name the server advertises (for the config)")
    p.add_argument("--subset", default="smoke",
                   choices=["smoke", "easy", "medium", "hard", "repair",
                            "comprehension", "implement", "grounding",
                            "new", "all"])
    p.add_argument("--problems", default="",
                   help="comma-separated problem ids (overrides --subset)")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent,
                   help="directory to write pack-results-*.json into")
    p.add_argument("--opencode-home", type=Path, default=None,
                   help="isolated HOME/XDG dir for opencode state (a fresh "
                        "opencode.db, avoiding contention with your real one)")
    p.add_argument("--timeout", type=int, default=900,
                   help="per-problem opencode wall-clock timeout (seconds)")
    p.add_argument("--max-problems", type=int, default=0,
                   help="cap number of problems (0 = no cap)")
    p.add_argument("--keep-runs", action="store_true",
                   help="keep agent-problem-pack/runs/* workspaces after grading")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    explicit = [s.strip() for s in args.problems.split(",") if s.strip()]
    problem_ids = select_problems(args.subset, explicit or None)
    if args.max_problems:
        problem_ids = problem_ids[:args.max_problems]
    if not problem_ids:
        print("no problems selected", file=sys.stderr)
        return 1

    env = dict(os.environ)
    env["PATH"] = (str(Path(OPENCODE).parent) + os.pathsep + env.get("PATH", ""))
    home = args.opencode_home.resolve() if args.opencode_home else None
    if args.base_url:
        if home is None:
            home = (args.out / "pack-opencode-home").resolve()
        home.mkdir(parents=True, exist_ok=True)
        cfg_path = write_opencode_config(home, args.base_url, args.served_model)
        env["HOME"] = str(home)
        env["XDG_DATA_HOME"] = str(home / ".local" / "share")
        env["XDG_CONFIG_HOME"] = str(home / ".config")
        env["OPENCODE_CONFIG"] = str(cfg_path)
        print(f"[pack] isolated opencode home: {home}  config: {cfg_path}")

    print(f"[pack] model={args.model} subset={args.subset} "
          f"problems={len(problem_ids)} timeout={args.timeout}s")
    results = []
    for pid in problem_ids:
        r = run_one(pid, args.model, env, home, args.timeout, args.keep_runs)
        flag = "PASS" if r["passed"] else ("TIMEOUT" if r["timed_out"] else "FAIL")
        print(f"[pack] {flag:7} {pid}  ({r['seconds']}s"
              f"{', ' + str(r['tokens']) + ' tok' if r.get('tokens') else ''})")
        results.append(r)

    summary = summarize(results)
    out = {"problem_pack": summary, "model": args.model,
           "created": int(time.time())}
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"pack-results-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[pack] pass {summary['passed']}/{summary['total']} "
          f"({summary['rate']*100:.0f}%) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

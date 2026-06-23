#!/usr/bin/env python3
"""Replay tasks through opencode and verify the outcomes.

For each task from make_tasks.py:
  1. create an isolated git worktree
       code      -> at the commit's PARENT (the pre-change state)
       knowledge -> at the chat-era commit (or HEAD if none found)
  2. run `opencode run` inside it (requests flow through the logging proxy,
     so full trajectories land in raw/opencode/)
  3. verify:
       code      -> overlap between files the model changed and the real
                    commit's changed files (>= --code-threshold)
       knowledge -> coverage of file paths/symbols cited by the reference
                    answer in the model's answer (>= --knowledge-threshold)
  4. annotate the trace files written during the run with
     {"replay": {"task_id", "type", "pass", "score"}} so build_dataset.py
     can keep passes and drop fails
  5. failed/ungradable knowledge tasks fall back to a distillation pair
     (query -> reference answer) in raw/replay/distill.jsonl

Multi-turn knowledge tasks replay every human turn in the same opencode
session (`opencode run --continue`), grading each turn against its reference
and passing on the mean score over gradable turns.

Model choice: defaults to the local model (on-policy rejection sampling). For
distillation-quality trajectories, run with a frontier model traced through
the proxy, e.g. --model frontier/anthropic/claude-sonnet-4.6 (requires
OPENROUTER_API_KEY in the proxy's environment).

opencode 1.15.x hangs when spawned without a TTY, so runs go through
`script -q` (pseudo-TTY).

Usage:
    python collect/replay.py --tasks tasks.jsonl [--limit 5] [--only code]
        [--model frontier/anthropic/claude-sonnet-4.6] [--timeout 600]
        [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
TRACE_DIR = BASE / "raw" / "opencode"
REPLAY_DIR = BASE / "raw" / "replay"
SANDBOX_PROFILE = Path(__file__).resolve().parent / "opencode-sandbox.sb"

STOPWORDS = {
    "the", "this", "that", "with", "from", "your", "file", "files", "code",
    "function", "class", "method", "true", "false", "none", "null", "self",
    "return", "import", "main", "test", "tests", "data", "model", "value",
}


# ------------------------------------------------------------- git helpers


def git(repo: Path, *args: str, check: bool = False) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args],
                         capture_output=True, text=True, timeout=120)
    if check and out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout.strip()


def make_worktree(repo: Path, ref: str) -> Path:
    wt = Path(tempfile.mkdtemp(prefix="improver-replay-"))
    wt.rmdir()  # git worktree wants to create it
    git(repo, "worktree", "add", "--detach", str(wt), ref, check=True)
    return wt


def remove_worktree(repo: Path, wt: Path) -> None:
    subprocess.run(["git", "-C", str(repo), "worktree", "remove",
                    "--force", str(wt)],
                   capture_output=True, timeout=120)


# ------------------------------------------------------------- opencode


def sandbox_wrapper() -> list[str]:
    """On macOS, confine opencode's file writes to tmp dirs (worktrees) and
    its own state dirs via sandbox-exec, so a replayed agent can never modify
    real repositories. Empty wrapper elsewhere."""
    import platform
    import shutil
    if platform.system() != "Darwin" or not shutil.which("sandbox-exec") \
            or not SANDBOX_PROFILE.is_file():
        return []
    return ["sandbox-exec", "-f", str(SANDBOX_PROFILE),
            "-D", f"HOME={Path.home()}"]


def run_opencode(prompt: str, cwd: Path, model: str, timeout: int,
                 continue_session: bool = False,
                 sandbox: bool = True,
                 agent: str | None = None) -> str:
    """Run opencode under a pseudo-TTY (it hangs without one) and return the
    rendered output text. continue_session resumes the last session in cwd,
    which is how multi-turn conversations are replayed. agent selects an
    opencode agent (`opencode run --agent <name>`) when comparing agents."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
        transcript = Path(tf.name)
    cmd = ["script", "-q", str(transcript)]
    if sandbox:
        cmd += sandbox_wrapper()
    cmd += ["opencode", "run", "-m", model]
    if agent:
        cmd += ["--agent", agent]
    if continue_session:
        cmd.append("--continue")
    cmd.append(prompt)
    # opencode resolves its working directory from $PWD, not getcwd(), so it
    # must be overridden or it inherits the replay runner's directory
    env = {**os.environ, "PWD": str(cwd)}
    try:
        subprocess.run(
            cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
        raw = transcript.read_text(encoding="utf-8", errors="replace")
    finally:
        transcript.unlink(missing_ok=True)
    # strip ANSI escapes and control chars from the pty capture
    text = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", raw)
    return text.replace("\x00", "").strip()


# ------------------------------------------------------ artifact synthesis


def load_openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key
    env_file = BASE / ".env"
    if env_file.is_file():
        for line in env_file.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"")
    return ""


SYNTH_SYSTEM = (
    "You reconstruct missing files in a repository snapshot so an AI coding "
    "assistant can faithfully replay a historical conversation. Git restores "
    "tracked files only, so run artifacts (results files, reports, generated "
    "data) the conversation depends on may be absent.\n\n"
    "Given the conversation (user requests + the original assistant's "
    "ground-truth answers) and the snapshot's directory listing, decide which "
    "artifact files the requests depend on that are missing, and reconstruct "
    "their contents CONSISTENT with the ground-truth answers — reuse their "
    "numbers, names, and facts verbatim wherever available; invent as little "
    "as possible.\n\n"
    "Respond with ONLY a JSON object mapping relative file paths to string "
    "file contents. Respond with {} if nothing relevant is missing. Keep each "
    "file under 200 lines. Never include paths outside the repository."
)


def synthesize_artifacts(task: dict, wt: Path, model: str,
                         timeout: int = 180) -> list[str]:
    """Reconstruct missing run artifacts from the reference answers (direct
    OpenRouter call — deliberately NOT through the proxy, so these synthesis
    requests never pollute the training traces). Returns written paths."""
    key = load_openrouter_key()
    if not key:
        return []
    listing = subprocess.run(
        ["find", ".", "-maxdepth", "3", "-not", "-path", "./.git*"],
        cwd=wt, capture_output=True, text=True, timeout=60,
    ).stdout.splitlines()[:400]
    turns = task.get("turns") or [
        {"prompt": task["prompt"], "reference": task.get("reference", "")}]
    convo = "\n\n".join(
        f"USER: {t['prompt'][:2000]}\n"
        f"ASSISTANT (ground truth): {(t.get('reference') or '')[:6000]}"
        for t in turns
    )
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYNTH_SYSTEM},
            {"role": "user", "content":
                f"Conversation to replay:\n{convo}\n\n"
                f"Repository snapshot listing (depth 3):\n"
                + "\n".join(listing)},
        ],
        "temperature": 0.2,
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    content = data["choices"][0]["message"]["content"] or "{}"
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if not m:
        return []
    try:
        mapping = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    written: list[str] = []
    wt_resolved = wt.resolve()
    for rel, text in mapping.items():
        if not isinstance(text, str) or len(text) > 100_000:
            continue
        target = (wt / rel).resolve()
        if not str(target).startswith(str(wt_resolved)) or target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        written.append(rel)
    return written


# ------------------------------------------------------------- verification


ENTITY_PATTERNS = [
    re.compile(r"[\w./-]+\.(?:py|ts|tsx|js|jsx|md|json|yaml|yml|toml|sh|go|rs)\b"),
    re.compile(r"`([^`\n]{2,60})`"),
    re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b"),          # snake_case
    re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b"),      # CamelCase
]


def extract_entities(text: str) -> set[str]:
    out: set[str] = set()
    for pat in ENTITY_PATTERNS:
        for m in pat.finditer(text):
            ent = (m.group(1) if m.groups() else m.group(0)).strip()
            if 2 < len(ent) <= 80 and ent.lower() not in STOPWORDS:
                out.add(ent.lower())
    return out


FILE_PATTERN = ENTITY_PATTERNS[0]
VIEW_TOOLS = {"read", "grep", "glob", "list", "edit", "write"}


def reference_files(reference: str) -> set[str]:
    """File paths the reference answer cites — a grounded agent should have
    opened these (Cursor transcripts carry no tool calls, so cited paths are
    the best available proxy for what the original agent viewed)."""
    return {m.group(0).lower().lstrip("./@") for m in
            FILE_PATTERN.finditer(reference)}


def viewed_files_from_traces(start_ts: str, end_ts: str) -> set[str]:
    """File paths the replayed agent actually touched with view/edit tools,
    parsed from the proxy traces written during the task window."""
    viewed: set[str] = set()
    for path in TRACE_DIR.glob("*.json"):
        if not (start_ts <= path.name[:15] <= end_ts):
            continue
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for msg in (rec.get("request") or {}).get("messages", []):
            for tc in (msg or {}).get("tool_calls") or []:
                fn = tc.get("function") or {}
                if fn.get("name") not in VIEW_TOOLS:
                    continue
                try:
                    fnargs = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    continue
                for key in ("filePath", "path"):
                    v = fnargs.get(key)
                    if isinstance(v, str) and v:
                        viewed.add(v.lower())
    return viewed


def file_match(ref: str, viewed: set[str]) -> bool:
    """Suffix-tolerant: traces hold absolute worktree paths, references hold
    repo-relative ones."""
    ref = ref.strip("/")
    base = ref.rsplit("/", 1)[-1]
    return any(v.rstrip("/").endswith(ref) or v.rsplit("/", 1)[-1] == base
               for v in viewed)


def grade_knowledge(reference: str, answer: str,
                    viewed: set[str] | None = None,
                    ) -> tuple[float | None, bool, dict]:
    """Return (score, gradable, detail).

    Score blends two signals:
      - entity coverage: fraction of reference entities present in the answer
      - view overlap: fraction of files cited by the reference that the agent
        actually opened (tool-call evidence from the traces)
    Equal weight when both are measurable; entity-only otherwise."""
    ref_entities = extract_entities(reference)
    if len(ref_entities) < 3:
        return None, False, {}
    answer_lower = answer.lower()
    hit = sum(1 for e in ref_entities if e in answer_lower)
    entity_cov = hit / len(ref_entities)

    ref_files = reference_files(reference)
    detail: dict = {"entity_coverage": round(entity_cov, 3),
                    "ref_entities": len(ref_entities)}
    if viewed is not None and ref_files:
        seen = sum(1 for f in ref_files if file_match(f, viewed))
        view_cov = seen / len(ref_files)
        detail.update(view_overlap=round(view_cov, 3),
                      ref_files=sorted(ref_files), files_viewed=seen)
        return 0.5 * entity_cov + 0.5 * view_cov, True, detail
    return entity_cov, True, detail


def grade_code(worktree: Path, expected_files: list[str]) -> tuple[float, dict]:
    # NUL-separated porcelain: immune to the whitespace-stripping in git()
    # that silently ate the first character of the first status line.
    out = subprocess.run(
        ["git", "-C", str(worktree), "status", "--porcelain=v1", "-z"],
        capture_output=True, text=True, timeout=120).stdout
    entries = [e for e in out.split("\0") if e]
    changed_files = set()
    i = 0
    while i < len(entries):
        status, path = entries[i][:2], entries[i][3:]
        changed_files.add(path)
        if status and status[0] in "RC":
            i += 1  # rename/copy: skip the following original-path entry
        i += 1
    expected = set(expected_files)
    if not changed_files:
        return 0.0, {"changed": [], "expected": sorted(expected)}
    overlap = len(changed_files & expected) / max(len(expected), 1)
    return overlap, {"changed": sorted(changed_files),
                     "expected": sorted(expected)}


# ------------------------------------------------------------- trace tagging


def tag_traces(start_ts: str, end_ts: str, annotation: dict) -> int:
    """Annotate proxy trace files written within [start_ts, end_ts]."""
    n = 0
    for path in sorted(TRACE_DIR.glob("*.json")):
        stamp = path.name[:15]  # YYYYmmdd-HHMMSS
        if not (start_ts <= stamp <= end_ts):
            continue
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if "replay" in rec:
            continue
        rec["replay"] = annotation
        path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        n += 1
    return n


# ------------------------------------------------------------- main


def run_task(task: dict, args) -> dict:
    repo = Path(task["repo"])
    if task["type"] == "code":
        ref = task["parent"]
    else:
        ref = ""
        if task.get("chat_date"):
            ref = git(repo, "rev-list", "-1",
                      f"--before={task['chat_date']}", "HEAD")
        ref = ref or "HEAD"

    wt = make_worktree(repo, ref)
    start_ts = time.strftime("%Y%m%d-%H%M%S")
    result: dict = {"task_id": task["id"], "type": task["type"],
                    "repo": str(repo), "started": start_ts}
    result["run"] = args.run_label if args.run_label else args.model
    try:
        if task["type"] == "code":
            answer = run_opencode(task["prompt"], wt, args.model, args.timeout,
                                  sandbox=not args.no_sandbox, agent=args.agent)
            result["answer_chars"] = len(answer)
            score, detail = grade_code(wt, task["changed_files"])
            result.update(score=score, detail=detail,
                          passed=score >= args.code_threshold)
        else:
            # reconstruct untracked run artifacts the conversation depends on
            # (git time-travel only restores tracked files)
            if not args.no_synth:
                try:
                    synth = synthesize_artifacts(task, wt, args.synth_model)
                    if synth:
                        result["synth_artifacts"] = synth
                except Exception as exc:  # noqa: BLE001
                    result["synth_error"] = repr(exc)
            # replay every human turn in the same opencode session
            turns = task.get("turns") or [
                {"prompt": task["prompt"], "reference": task["reference"]}]
            turn_results = []
            for i, turn in enumerate(turns):
                answer = run_opencode(turn["prompt"], wt, args.model,
                                      args.timeout, continue_session=i > 0,
                                      sandbox=not args.no_sandbox,
                                      agent=args.agent)
                # files viewed so far in the session ground every turn
                viewed = viewed_files_from_traces(
                    start_ts, time.strftime("%Y%m%d-%H%M%S"))
                t_score, t_gradable, t_detail = grade_knowledge(
                    turn["reference"], answer, viewed)
                turn_results.append({"score": t_score, "gradable": t_gradable,
                                     "answer_chars": len(answer),
                                     **t_detail})
            result["turns"] = turn_results
            graded = [t["score"] for t in turn_results if t["gradable"]]
            score = sum(graded) / len(graded) if graded else None
            result.update(score=score, gradable=bool(graded),
                          passed=bool(graded
                                      and score >= args.knowledge_threshold))
            if not result["passed"]:
                # fallback distillation: the full reference conversation
                messages = []
                for turn in turns:
                    messages.append({"role": "user", "content": turn["prompt"]})
                    messages.append({"role": "assistant",
                                     "content": turn["reference"]})
                with (REPLAY_DIR / "distill.jsonl").open("a") as fh:
                    fh.write(json.dumps({
                        "task_id": task["id"],
                        "messages": messages,
                        "source": "cursor-distill",
                    }, ensure_ascii=False) + "\n")
                result["distilled"] = True
    except subprocess.TimeoutExpired:
        result.update(passed=False, error="timeout")
    except Exception as exc:  # noqa: BLE001
        result.update(passed=False, error=repr(exc))
    finally:
        end_ts = time.strftime("%Y%m%d-%H%M%S")
        remove_worktree(repo, wt)

    result["ended"] = end_ts
    result["tagged_traces"] = tag_traces(start_ts, end_ts, {
        "task_id": task["id"], "type": task["type"],
        "pass": bool(result.get("passed")), "score": result.get("score"),
    })
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tasks", type=Path, default=BASE / "tasks.jsonl")
    p.add_argument("--model", default="llamacpp/local-code-model")
    p.add_argument("--run-label", default=None,
                   help="label for this run (e.g. 'base', 'v1-sft', 'v2-rl'). "
                        "Defaults to the --model value.")
    p.add_argument("--agent", default=None,
                   help="opencode agent to run as (opencode run --agent <name>)")
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    p.add_argument("--only", choices=["code", "knowledge"], default=None)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--code-threshold", type=float, default=0.5)
    p.add_argument("--knowledge-threshold", type=float, default=0.3)
    p.add_argument("--no-synth", action="store_true",
                   help="skip reconstructing missing run artifacts")
    p.add_argument("--synth-model", default="openai/gpt-5.4-mini",
                   help="OpenRouter model for artifact reconstruction")
    p.add_argument("--results", type=Path, default=None,
                   help="results file (default raw/replay/results.jsonl); "
                        "use a fresh file to re-run tasks already recorded")
    p.add_argument("--no-sandbox", action="store_true",
                   help="disable the macOS write-sandbox around opencode")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    import urllib.request
    try:
        urllib.request.urlopen("http://127.0.0.1:8765/health", timeout=5)
    except Exception:
        print("warning: proxy on :8765 not reachable — trajectories will NOT "
              "be captured. Start it with `make collect`.")

    tasks = [json.loads(l) for l in args.tasks.read_text().splitlines()
             if l.strip()]
    if args.only:
        tasks = [t for t in tasks if t["type"] == args.only]
    done_ids = set()
    results_path = args.results or (REPLAY_DIR / "results.jsonl")
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    if results_path.is_file():
        for line in results_path.read_text().splitlines():
            with_suppress = json.loads(line) if line.strip() else {}
            if with_suppress.get("task_id"):
                done_ids.add(with_suppress["task_id"])
    tasks = [t for t in tasks if t["id"] not in done_ids]
    if args.limit:
        tasks = tasks[:args.limit]

    print(f"{len(tasks)} tasks to run ({len(done_ids)} already done)")
    if args.dry_run:
        for t in tasks:
            print(f"  [{t['type']}] {Path(t['repo']).name}: "
                  f"{t['prompt'][:80]!r}")
        return 0

    passed = 0
    dead_streak = 0  # consecutive no-output tasks => API likely down/out of credits
    for i, task in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {task['type']} @ {Path(task['repo']).name}: "
              f"{task['prompt'][:70]!r}")
        result = run_task(task, args)
        with results_path.open("a") as fh:
            fh.write(json.dumps(result, ensure_ascii=False) + "\n")
        status = "PASS" if result.get("passed") else "fail"
        passed += bool(result.get("passed"))
        print(f"    {status} score={result.get('score')} "
              f"traces={result.get('tagged_traces')} "
              f"{result.get('error', '')}")
        got_output = result.get("answer_chars") or any(
            t.get("answer_chars") for t in result.get("turns", []))
        # zero captured traces means opencode never reached the API —
        # broken harness, not a model failure (timeouts excepted)
        dead = not result.get("tagged_traces") and \
            result.get("error") != "timeout"
        if result.get("error") or not got_output or dead:
            dead_streak += 1
        else:
            dead_streak = 0
        if dead_streak >= 3:
            print("ABORT: 3 consecutive tasks produced no output -- "
                  "upstream API likely failing (credits?). Remove their "
                  "fail lines from the results file before resuming.")
            break

    print(f"done: {passed}/{len(tasks)} passed -> {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

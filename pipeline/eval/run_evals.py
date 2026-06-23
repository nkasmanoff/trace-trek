#!/usr/bin/env python3
"""Eval gate for a deployed local code model.

Three sections:
  1. tool-validity  — server must emit parseable, schema-valid tool calls
  2. opencode-tasks — small end-to-end tasks run through `opencode run` in a
                      temp dir, scored by verifiable side effects
  3. chat-sanity    — plain answers: non-empty, no leaked template tokens

Writes eval/results-<ts>.json. Compare against a previous run with --baseline.
Exit code is non-zero if any section regresses vs the baseline (or, without a
baseline, if hard floors are missed), making it usable as a deploy gate.

Usage:
    python eval/run_evals.py [--base-url http://127.0.0.1:8080/v1]
                             [--model llamacpp/local-code-model]
                             [--baseline eval/results-<old>.json]
                             [--skip-opencode]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

LEAK_TOKENS = [
    "<|END_OF_TURN_TOKEN|>",
    "<|START_ACTION|>",
    "<|END_ACTION|>",
    "<|START_THINKING|>",
    "<|END_THINKING|>",
    "<|CHATBOT_TOKEN|>",
]

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command and return its output",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}

TOOL_PROMPTS = [
    "List the files in the current directory.",
    "Show the first 5 lines of README.md using a shell command.",
    "Count how many python files exist under the current directory.",
    "Check whether git is installed and print its version.",
    "Create an empty file named marker.txt in /tmp.",
]

CHAT_PROMPTS = [
    "Who are you? One sentence.",
    "Write a one-line Python lambda that squares a number. Code only.",
    "What does HTTP status 404 mean? One sentence.",
]

# (prompt, verifier) — verifier runs inside the temp workdir
OPENCODE_TASKS = [
    (
        "Create a file named hello.py that prints exactly: hello world",
        lambda d: (d / "hello.py").is_file()
        and "hello world"
        in subprocess.run(
            [sys.executable, str(d / "hello.py")],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout,
    ),
    (
        "Create a file fib.py with a function fib(n) returning the nth Fibonacci "
        "number (fib(0)=0, fib(1)=1). No prints.",
        lambda d: (d / "fib.py").is_file()
        and subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.path.insert(0,'.'); from fib import fib; "
                "assert fib(10)==55; print('ok')",
            ],
            cwd=d,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        == "ok",
    ),
    (
        "Create notes.md containing a markdown list of exactly three fruits.",
        lambda d: (d / "notes.md").is_file()
        and sum(
            1
            for line in (d / "notes.md").read_text().splitlines()
            if line.strip().startswith(("-", "*"))
        )
        == 3,
    ),
]


def post_chat(base_url: str, payload: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def eval_tool_validity(base_url: str) -> dict:
    passed, details = 0, []
    for prompt in TOOL_PROMPTS:
        ok, why = False, ""
        try:
            data = post_chat(
                base_url,
                {
                    "model": "local-code-model",
                    "messages": [{"role": "user", "content": prompt}],
                    "tools": [BASH_TOOL],
                    "max_tokens": 2048,
                },
            )
            msg = data["choices"][0]["message"]
            tcs = msg.get("tool_calls") or []
            if not tcs:
                why = "no tool_calls"
            else:
                args = json.loads(tcs[0]["function"]["arguments"])
                if tcs[0]["function"]["name"] == "bash" and "command" in args:
                    ok = True
                else:
                    why = f"bad call shape: {tcs[0]['function']}"
        except Exception as exc:  # noqa: BLE001
            why = repr(exc)
        passed += ok
        details.append({"prompt": prompt, "ok": ok, "why": why})
    return {"passed": passed, "total": len(TOOL_PROMPTS), "details": details}


def eval_chat_sanity(base_url: str) -> dict:
    passed, details = 0, []
    for prompt in CHAT_PROMPTS:
        ok, why = False, ""
        try:
            data = post_chat(
                base_url,
                {
                    "model": "local-code-model",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2048,
                },
            )
            content = data["choices"][0]["message"].get("content") or ""
            leaked = [t for t in LEAK_TOKENS if t in content]
            if not content.strip():
                why = "empty content"
            elif leaked:
                why = f"leaked tokens: {leaked}"
            else:
                ok = True
        except Exception as exc:  # noqa: BLE001
            why = repr(exc)
        passed += ok
        details.append({"prompt": prompt, "ok": ok, "why": why})
    return {"passed": passed, "total": len(CHAT_PROMPTS), "details": details}


def run_opencode(prompt: str, model: str, cwd: Path, timeout: int = 600) -> None:
    """opencode 1.15.x hangs when spawned without a TTY; run it under
    `script -q` to give it a pseudo-TTY."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tf:
        transcript = Path(tf.name)
    # opencode resolves its working directory from $PWD, not getcwd()
    env = {**os.environ, "PWD": str(cwd)}
    try:
        subprocess.run(
            ["script", "-q", str(transcript), "opencode", "run", "-m", model, prompt],
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    finally:
        transcript.unlink(missing_ok=True)


def eval_opencode_tasks(model: str) -> dict:
    if shutil.which("opencode") is None:
        return {
            "passed": 0,
            "total": 0,
            "details": [],
            "skipped": "opencode not on PATH",
        }
    passed, details = 0, []
    for prompt, verify in OPENCODE_TASKS:
        ok, why = False, ""
        workdir = Path(tempfile.mkdtemp(prefix="improver-eval-"))
        try:
            run_opencode(prompt, model, workdir)
            ok = bool(verify(workdir))
            if not ok:
                why = (
                    f"verifier failed; dir={sorted(p.name for p in workdir.iterdir())}"
                )
        except Exception as exc:  # noqa: BLE001
            why = repr(exc)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        passed += ok
        details.append({"prompt": prompt, "ok": ok, "why": why})
    return {"passed": passed, "total": len(OPENCODE_TASKS), "details": details}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-url", default="http://127.0.0.1:63450/v1")
    p.add_argument(
        "--model",
        default="llamacpp/local-code-model",
        help="opencode model id for the task section",
    )
    p.add_argument("--baseline", type=Path, default=None)
    p.add_argument("--skip-opencode", action="store_true")
    args = p.parse_args()

    results = {"timestamp": time.strftime("%Y%m%d-%H%M%S"), "base_url": args.base_url}

    print("== tool-validity ==")
    results["tool_validity"] = eval_tool_validity(args.base_url)
    print(f"  {results['tool_validity']['passed']}/{results['tool_validity']['total']}")

    print("== chat-sanity ==")
    results["chat_sanity"] = eval_chat_sanity(args.base_url)
    print(f"  {results['chat_sanity']['passed']}/{results['chat_sanity']['total']}")

    if args.skip_opencode:
        results["opencode_tasks"] = {"passed": 0, "total": 0, "skipped": True}
    else:
        print("== opencode-tasks (slow) ==")
        results["opencode_tasks"] = eval_opencode_tasks(args.model)
        print(
            f"  {results['opencode_tasks']['passed']}/{results['opencode_tasks']['total']}"
        )

    out = Path(__file__).resolve().parent / f"results-{results['timestamp']}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"results -> {out}")

    sections = ["tool_validity", "chat_sanity", "opencode_tasks"]
    if args.baseline and args.baseline.is_file():
        base = json.loads(args.baseline.read_text())
        regressed = []
        for s in sections:
            new, old = results.get(s, {}), base.get(s, {})
            if old.get("total") and new.get("passed", 0) < old.get("passed", 0):
                regressed.append(f"{s}: {new.get('passed')} < {old.get('passed')}")
        if regressed:
            print("REGRESSION vs baseline:\n  " + "\n  ".join(regressed))
            return 1
        print("no regression vs baseline — OK to keep")
        return 0

    # no baseline: hard floors
    floors = {"tool_validity": 4, "chat_sanity": 3}
    failed = [s for s, floor in floors.items() if results[s]["passed"] < floor]
    if failed:
        print(f"FAILED hard floors: {failed}")
        return 1
    print("passed hard floors — OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

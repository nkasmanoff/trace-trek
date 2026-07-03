#!/usr/bin/env python
"""Check an SFT trace file for contamination against the problem pack.

Scans every row of a traces JSONL (rows with a `messages` list, opencode SFT
format) for distinctive markers of the pack's problems: file names, symbol
names, and prompt phrases that would only appear in a trace if the session
actually touched that problem's material. A model fine-tuned on a trace that
discusses a problem's source would partially memorize the eval.

Usage:
    uv run python scripts/check_contamination.py ../pipeline/dataset/sft.jsonl
    uv run python scripts/check_contamination.py sft.jsonl --write-clean clean.jsonl

Exit code 1 when contaminated rows are found (so it can gate dataset builds).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# problem id -> distinctive, low-false-positive markers (matched
# case-insensitively as substrings of the full serialized trace)
MARKERS: dict[str, tuple[str, ...]] = {
    "problem-01-tokenizer-regression": ("test_tokenizer.py",),
    "problem-02-shell-command-injection": ("run_user_command",),
    "problem-03-cross-platform-task-path": (
        "ollama_tool_reasoning_bench",
        "personal_tool_reasoning_tasks",
    ),
    "problem-04-import-error-after-refactor": ("project.settings import DEFAULT_TIMEOUT",),
    "problem-05-mutable-default-cache": ("collect_metrics(",),
    "problem-06-config-merge-priority": ("load_env()",),
    "problem-07-thread-safe-cache": ("get_or_compute", "ComputeCache"),
    "problem-08-pack-lifecycle": ("pack_tools", "agent-problem-pack"),
    "problem-09-eval-pipeline-viewer": ("aggregateEval", "EvalView"),
    "problem-10-flatten-depth": ("def flatten(nested",),
    "problem-11-eval-scoring-pipeline": ("normalize_result", "evalkit"),
    "problem-12-merge-latest-property": ("merge_latest",),
    # Source-specific symbols only: a passing "persona vector" mention in an
    # unrelated repo is not trace-on-eval contamination.
    "problem-13-persona-vector-extraction": (
        "activation_steer",
        "generate_vec",
        "ActivationSteerer",
        "response_avg_diff",
    ),
    "problem-14-agent-eval-suite": ("eval-suite-design",),
    "problem-15-pipeline-bug-trace": ("weighted_score",),
    "problem-16-grounded-audit": ("AUDIT_TOKEN", "audit_token("),
    "problem-17-stage-localization": ("flowkit",),
    "problem-18-edit-gauntlet": ("handle_heartbeat", "handle_audit"),
    "problem-19-follow-the-pattern": ("line-stats", "registered_services"),
    "problem-20-limiter-follow-ups": ("TokenBucket",),
    "problem-21-js-eval-aggregate": ("findFlips", "aggregateEval"),
    "problem-22-implement-jsonl-support": ("EXT_LOADERS", "RecordParseError"),
    "problem-23-implement-retry-backoff": ("BACKOFF_BASE",),
    "problem-24-implement-dry-run": ("would delete",),
    "problem-25-implement-parse-lineno": ("logkit", "bad key in"),
    "problem-26-lru-pin-revision": ("lrupin", "all entries pinned"),
    "problem-27-implement-rates-section": ("section_rates", "REPORT_FORMAT.md"),
    "problem-28-spreadsheet-grounding": ("REPORT_TOKEN", "report_token("),
    "problem-29-web-research-hackathon": ("nba hackathon", "frank hu"),
}


def scan_row(text: str) -> list[tuple[str, str]]:
    lowered = text.lower()
    hits = []
    for problem_id, markers in MARKERS.items():
        for marker in markers:
            if marker.lower() in lowered:
                hits.append((problem_id, marker))
                break
    return hits


def summarize_row(row: dict) -> str:
    for message in row.get("messages", []):
        if message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str):
                return content.replace("\n", " ")[:90]
    return "<no user message>"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("traces", type=Path, help="SFT traces JSONL")
    parser.add_argument("--write-clean", type=Path, default=None,
                        help="write a copy with contaminated rows removed")
    args = parser.parse_args()

    clean_lines: list[str] = []
    contaminated = 0
    total = 0
    for index, line in enumerate(
            args.traces.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        total += 1
        row = json.loads(line)
        hits = scan_row(line)
        if hits:
            contaminated += 1
            names = ", ".join(f"{pid} (marker {marker!r})" for pid, marker in hits)
            print(f"row {index}: {names}")
            print(f"        task: {summarize_row(row)}")
        else:
            clean_lines.append(line)

    print(f"\n{contaminated}/{total} rows reference pack problem material")
    if args.write_clean is not None:
        args.write_clean.write_text(
            "".join(l + "\n" for l in clean_lines), encoding="utf-8")
        print(f"clean copy ({len(clean_lines)} rows) -> {args.write_clean}")
    return 1 if contaminated else 0


if __name__ == "__main__":
    raise SystemExit(main())

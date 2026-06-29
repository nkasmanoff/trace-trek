"""Reference fixes used by pack validation — not shipped to agent workspaces."""

from __future__ import annotations

from pathlib import Path

PACK_ROOT = Path(__file__).resolve().parents[1]

# Maps problem id -> relative path -> file contents after a correct minimal fix.
GOLDEN_FILES: dict[str, dict[str, str]] = {
    "problem-01-tokenizer-regression": {
        "tokenizer.py": """\
def tokenize(text):
    return [part.lower() for part in text.strip().split(",") if part]
""",
    },
    "problem-02-shell-command-injection": {
        "runner.py": """\
import subprocess


def run_user_command(command):
    if isinstance(command, str):
        raise TypeError("string commands are not allowed; pass an argv list")
    return subprocess.check_output(command, text=True)
""",
    },
    "problem-03-cross-platform-task-path": {
        "code/tool-reasoning-benchmark/ollama_tool_reasoning_bench.py": """\
from pathlib import Path


TASKS = Path(__file__).with_name("personal_tool_reasoning_tasks.jsonl")


def read_default_tasks():
    return TASKS.read_text(encoding="utf-8")
""",
    },
    "problem-04-import-error-after-refactor": {
        "src/project/config.py": """\
from project.settings import DEFAULT_TIMEOUT

__all__ = ["DEFAULT_TIMEOUT"]
""",
    },
    "problem-05-mutable-default-cache": {
        "metrics.py": """\
def collect_metrics(name, value, cache=None):
    if cache is None:
        cache = {}
    cache[name] = value
    return cache
""",
    },
    "problem-06-config-merge-priority": {
        "src/config.py": """\
from src import loader


class Config:
    DEFAULTS = {"host": "localhost", "port": 8080, "debug": False}

    def __init__(self, config_path=None):
        self._values = dict(self.DEFAULTS)
        if config_path:
            self._values.update(loader.load_file(config_path))
        self._values.update(loader.load_env())

    def get(self, key, default=None):
        return self._values.get(key, default)
""",
    },
    "problem-07-thread-safe-cache": {
        "src/cache.py": """\
import threading


class ComputeCache:
    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()

    def get_or_compute(self, key, factory):
        with self._lock:
            if key not in self._cache:
                self._cache[key] = factory()
            return self._cache[key]
""",
    },
    "problem-11-eval-scoring-pipeline": {
        "src/evalkit/scoring.py": '''\
"""Normalize raw task results emitted by the eval runner.

Each raw record looks like:

    {"task_id": "t1", "type": "code", "status": "ok", "score": 0.9}

`status` is one of: "ok", "error", "skip".
- "ok"    -> the task ran; it passes when score meets the threshold.
- "error" -> the task crashed; it counts as a failure.
- "skip"  -> the task was not run for this config; it must be EXCLUDED from
             pass-rate and mean-score aggregation (passed is None).

`score` is a model score that callers may report outside the [0, 1] range
(some runners emit raw logits or summed sub-scores). Normalized scores must
always be clamped to [0, 1].
"""

DEFAULT_THRESHOLD = 0.5


def normalize_result(raw, run, threshold=DEFAULT_THRESHOLD):
    status = raw.get("status", "ok")
    score = max(0.0, min(1.0, raw.get("score", 0.0)))

    if status == "skip":
        passed = None
    elif status == "error":
        passed = False
    else:
        passed = score >= threshold

    return {
        "task_id": raw["task_id"],
        "type": raw["type"],
        "run": run,
        "passed": passed,
        "score": score,
    }
''',
        "src/evalkit/aggregate.py": '''\
"""Aggregate normalized results into per-(run, type) statistics.

A bucket summarizes every normalized result that shares a run and task type:

    {
        "run": "baseline",
        "type": "code",
        "total": 3,        # tasks that actually ran (skips excluded)
        "passed": 2,
        "pass_rate": 0.666...,
        "mean_score": 0.74,
    }

Skipped tasks (passed is None) must not contribute to total, passed,
pass_rate, or mean_score. A bucket whose tasks were all skipped reports
total 0 with pass_rate 0.0 and mean_score 0.0.
"""

from collections import defaultdict


def aggregate_results(results):
    buckets = defaultdict(list)
    for result in results:
        buckets[(result["run"], result["type"])].append(result)

    summaries = []
    for (run, task_type), items in sorted(buckets.items()):
        scored = [item for item in items if item["passed"] is not None]
        total = len(scored)
        passed = sum(1 for item in scored if item["passed"])
        score_sum = sum(item["score"] for item in scored)
        pass_rate = passed / total if total else 0.0
        mean_score = score_sum / total if total else 0.0
        summaries.append(
            {
                "run": run,
                "type": task_type,
                "total": total,
                "passed": passed,
                "pass_rate": pass_rate,
                "mean_score": mean_score,
            }
        )
    return summaries
''',
    },
    "problem-12-merge-latest-property": {
        "src/merge.py": '''\
"""Deduplicate eval records, keeping the latest attempt per task.

Records look like: {"task_id": "t1", "attempt": 3, "score": 0.7}

`merge_latest` collapses many records into one record per task_id. For each
task_id the winner is the record with the HIGHEST `attempt`. Ties (equal
attempt for the same task_id) keep the record seen later in the input.

Guarantees (regardless of input order):
- The result contains exactly one record per distinct task_id.
- Each kept record is the highest-attempt record for its task_id.
- The result is sorted by task_id ascending.
- The input list is not mutated.
"""


def merge_latest(records):
    best = {}
    for record in records:
        task_id = record["task_id"]
        current = best.get(task_id)
        if current is None or record["attempt"] >= current["attempt"]:
            best[task_id] = record
    return sorted(best.values(), key=lambda record: record["task_id"])
''',
    },
    "problem-10-flatten-depth": {
        "src/flatten.py": """\
def flatten(nested, depth=None):
    result = []
    for item in nested:
        if isinstance(item, list) and (depth is None or depth > 0):
            sub = depth - 1 if depth is not None else None
            result.extend(flatten(item, sub))
        else:
            result.append(item)
    return result
""",
    },
}


def apply_golden_fix(problem_id: str, workspace: Path) -> None:
    patches = GOLDEN_FILES.get(problem_id)
    if not patches:
        raise KeyError(f"no golden fix registered for {problem_id}")
    for relative_path, contents in patches.items():
        target = workspace / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents, encoding="utf-8")

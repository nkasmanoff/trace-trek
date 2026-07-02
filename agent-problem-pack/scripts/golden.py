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
    "problem-15-pipeline-bug-trace": {
        "src/pipeline/runner.py": '''\
"""Stage 2: run each task and emit a raw result.

The runner calls a model adapter for every task and records a per-task score
in [0, 1]. The task's `weight` must be copied onto the result UNCHANGED so
later stages can compute a weighted aggregate without re-reading the
manifest. A weight of 0.0 is a valid value and must survive this stage.
"""


def run_tasks(tasks, model):
    results = []
    for task in tasks:
        score = model.score(task["prompt"])
        results.append(
            {
                "task_id": task["task_id"],
                "type": task["type"],
                "score": score,
                "weight": task.get("weight", 1.0),
            }
        )
    return results
''',
        "src/pipeline/aggregate.py": '''\
"""Stage 3: aggregate raw results into a weighted suite score.

The suite score is the WEIGHTED mean of task scores:

    weighted_score = sum(score_i * weight_i) / sum(weight_i)

computed over tasks whose weight is strictly positive. Tasks with weight 0.0
are excluded from BOTH the numerator and the denominator, but still counted
in the `recorded` field. If no task has positive weight, weighted_score is
0.0.

Returned summary:

    {"recorded": <all results>, "scored": <positive-weight results>,
     "weighted_score": <float>}
"""


def aggregate(results):
    scored = [r for r in results if r["weight"] > 0]
    denominator = sum(r["weight"] for r in scored)
    if denominator:
        weighted_score = sum(r["score"] * r["weight"] for r in scored) / denominator
    else:
        weighted_score = 0.0
    return {
        "recorded": len(results),
        "scored": len(scored),
        "weighted_score": weighted_score,
    }
''',
    },
    "problem-17-stage-localization": {
        "src/flowkit/transforms/windows.py": '''\
"""Trailing-window aggregations used by pipeline stages.

rolling_mean(values, window) returns a list of the same length as `values`
where output element i is the mean of the TRAILING window ending at i
(inclusive): mean(values[max(0, i - window + 1) : i + 1]). The first
window - 1 elements therefore average over a shorter prefix. window must be
at least 1.
"""


def rolling_mean(values, window=3):
    if window < 1:
        raise ValueError("window must be >= 1")
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out
''',
    },
    "problem-18-edit-gauntlet": {
        "src/handlers.py": '''\
"""Event handlers for the ingestion service.

Every handler follows the same contract:

- The event body lives under the "data" key of the incoming event dict.
- A handler accepts an event when its value is at or below the kind's limit
  (events exactly AT the limit are accepted), and rejects it otherwise.
- Handlers that manage liveness (heartbeat) decrement the event's ttl by 1
  when forwarding. All other handlers forward ttl UNCHANGED.
- Accepted values are appended to state[kind]; rejected events are counted
  in state["rejected"].
"""

LIMITS = {
    "metric": 100,
    "log": 100,
    "trace": 250,
    "alert": 10,
    "audit": 50,
    "heartbeat": 1,
}

DEFAULT_TTL = 8


def handle_metric(event, state):
    payload = event.get("data", {})
    value = payload.get("value", 0)
    limit = LIMITS["metric"]
    accepted = value <= limit
    if accepted:
        state.setdefault("metric", []).append(value)
    else:
        state["rejected"] = state.get("rejected", 0) + 1
    ttl = event.get("ttl", DEFAULT_TTL)
    return {"kind": "metric", "accepted": accepted, "value": value, "ttl": ttl}


def handle_log(event, state):
    payload = event.get("data", {})
    value = payload.get("value", 0)
    limit = LIMITS["log"]
    accepted = value <= limit
    if accepted:
        state.setdefault("log", []).append(value)
    else:
        state["rejected"] = state.get("rejected", 0) + 1
    ttl = event.get("ttl", DEFAULT_TTL)
    return {"kind": "log", "accepted": accepted, "value": value, "ttl": ttl}


def handle_trace(event, state):
    payload = event.get("data", {})
    value = payload.get("value", 0)
    limit = LIMITS["trace"]
    accepted = value <= limit
    if accepted:
        state.setdefault("trace", []).append(value)
    else:
        state["rejected"] = state.get("rejected", 0) + 1
    ttl = event.get("ttl", DEFAULT_TTL)
    return {"kind": "trace", "accepted": accepted, "value": value, "ttl": ttl}


def handle_alert(event, state):
    payload = event.get("data", {})
    value = payload.get("value", 0)
    limit = LIMITS["alert"]
    accepted = value <= limit
    if accepted:
        state.setdefault("alert", []).append(value)
    else:
        state["rejected"] = state.get("rejected", 0) + 1
    ttl = event.get("ttl", DEFAULT_TTL)
    return {"kind": "alert", "accepted": accepted, "value": value, "ttl": ttl}


def handle_audit(event, state):
    payload = event.get("data", {})
    value = payload.get("value", 0)
    limit = LIMITS["audit"]
    accepted = value <= limit
    if accepted:
        state.setdefault("audit", []).append(value)
    else:
        state["rejected"] = state.get("rejected", 0) + 1
    ttl = event.get("ttl", DEFAULT_TTL)
    return {"kind": "audit", "accepted": accepted, "value": value, "ttl": ttl}


def handle_heartbeat(event, state):
    payload = event.get("data", {})
    value = payload.get("value", 0)
    limit = LIMITS["heartbeat"]
    accepted = value <= limit
    if accepted:
        state.setdefault("heartbeat", []).append(value)
    else:
        state["rejected"] = state.get("rejected", 0) + 1
    ttl = event.get("ttl", DEFAULT_TTL) - 1
    return {"kind": "heartbeat", "accepted": accepted, "value": value, "ttl": ttl}


HANDLERS = {
    "metric": handle_metric,
    "log": handle_log,
    "trace": handle_trace,
    "alert": handle_alert,
    "audit": handle_audit,
    "heartbeat": handle_heartbeat,
}


def dispatch(event, state):
    handler = HANDLERS.get(event.get("kind"))
    if handler is None:
        raise ValueError(f"unknown event kind: {event.get('kind')!r}")
    return handler(event, state)
''',
    },
    "problem-19-follow-the-pattern": {
        "src/services/line_stats_service.py": '''\
from src.services.base import BaseService


class LineStatsService(BaseService):
    name = "line-stats"
    schema = {"text": str}

    def _process(self, payload):
        text = payload["text"]
        lines = [line for line in text.splitlines() if line.strip()]
        words = text.split()
        return {
            "lines": len(lines),
            "words": len(words),
            "unique_words": len({word.lower() for word in words}),
        }
''',
        "src/services/__init__.py": '''\
"""Service package.

Importing this package imports every service module, which registers each
service with the registry in base.py. New services must be imported here or
they will never be registered.
"""

from src.services import dedupe_service, line_stats_service, word_count_service  # noqa: F401
from src.services.base import get_service, registered_services  # noqa: F401
''',
    },
    "problem-20-limiter-follow-ups": {
        "src/limiter.py": '''\
"""Token-bucket rate limiter.

A bucket holds at most `capacity` tokens and refills at `rate` tokens per
second. Tokens NEVER exceed capacity, no matter how long the bucket sits
idle. Each allowed request spends tokens; a request that cannot be paid for
in full is denied and spends nothing.
"""

import time


class TokenBucket:
    def __init__(self, capacity, rate, clock=time.monotonic):
        self.capacity = float(capacity)
        self.rate = float(rate)
        self._clock = clock
        self._tokens = float(capacity)
        self._last = clock()

    def _refill(self):
        now = self._clock()
        elapsed = now - self._last
        self._last = now
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)

    def allow(self, cost=1):
        self._refill()
        if self._tokens >= cost:
            self._tokens -= cost
            return True
        return False

    def remaining(self):
        self._refill()
        return self._tokens
''',
    },
    "problem-21-js-eval-aggregate": {
        "src/aggregate.js": '''\
/**
 * Aggregate normalized eval results into per-(run, type) buckets and detect
 * run-to-run regressions ("flips").
 *
 * A result record: { taskId, run, type, passed, score }
 * `passed` is true, false, or null. Null means the task was SKIPPED for that
 * run: skipped records must not count toward total, passed, or passRate,
 * and must never be treated as failures.
 *
 * findFlips(results, baseRun, candRun) returns the taskIds that passed in
 * baseRun but FAILED (passed === false) in candRun. Tasks that are missing
 * or skipped in either run are not flips: a flip requires an actual pass in
 * base and an actual fail in candidate.
 */

export function aggregateEval(results) {
  const buckets = new Map()
  for (const r of results) {
    const key = `${r.run}::${r.type}`
    if (!buckets.has(key)) {
      buckets.set(key, { run: r.run, type: r.type, total: 0, passed: 0, records: [] })
    }
    const bucket = buckets.get(key)
    if (r.passed === null) continue
    bucket.records.push(r)
    bucket.total += 1
    if (r.passed) bucket.passed += 1
  }
  return [...buckets.values()].map(b => ({
    run: b.run,
    type: b.type,
    total: b.total,
    passed: b.passed,
    passRate: b.total ? b.passed / b.total : 0,
    meanScore: b.records.length
      ? b.records.reduce((sum, r) => sum + (r.score ?? 0), 0) / b.records.length
      : 0,
  }))
}

export function findFlips(results, baseRun, candRun) {
  const base = new Map()
  const cand = new Map()
  for (const r of results) {
    if (r.run === baseRun) base.set(r.taskId, r)
    if (r.run === candRun) cand.set(r.taskId, r)
  }
  const flips = []
  for (const [taskId, record] of base) {
    if (record.passed !== true) continue
    const candidate = cand.get(taskId)
    if (candidate && candidate.passed === false) flips.push(taskId)
  }
  return flips.sort()
}
''',
    },
    "problem-22-implement-jsonl-support": {
        "src/datakit/loaders.py": '''\
"""File loaders.

Loaders are registered by file extension in EXT_LOADERS. A loader takes a
Path and returns a list of records (dicts). load_any() picks the loader for
the path's suffix and raises UnsupportedFormatError for unknown extensions.

Malformed content raises RecordParseError; for line-oriented formats the
error message includes the file path and the 1-based line number of the bad
line. Blank lines in line-oriented formats are skipped, not errors.
"""

import json
from pathlib import Path

from src.datakit.errors import RecordParseError, UnsupportedFormatError


def load_json_file(path):
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RecordParseError(f"{path}: {exc}") from None
    if isinstance(data, list):
        return data
    return [data]


def load_jsonl_file(path):
    path = Path(path)
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RecordParseError(
                f"{path}: line {line_number}: {exc}"
            ) from None
    return records


EXT_LOADERS = {
    ".json": load_json_file,
    ".jsonl": load_jsonl_file,
}


def get_loader(path):
    suffix = Path(path).suffix.lower()
    try:
        return EXT_LOADERS[suffix]
    except KeyError:
        raise UnsupportedFormatError(
            f"no loader registered for {suffix!r} ({path})"
        ) from None


def load_any(path):
    return get_loader(path)(Path(path))
''',
    },
    "problem-23-implement-retry-backoff": {
        "src/apiclient/client.py": '''\
"""ApiClient: the public entry point for issuing requests."""

import time

from src.apiclient import config
from src.apiclient.errors import TransientError


class ApiClient:
    def __init__(self, transport):
        self._transport = transport

    def send(self, request):
        delay = config.BACKOFF_BASE
        for attempt in range(1, config.MAX_ATTEMPTS + 1):
            try:
                return self._transport.send(request)
            except TransientError:
                if attempt == config.MAX_ATTEMPTS:
                    raise
                time.sleep(delay)
                delay *= 2
''',
    },
    "problem-24-implement-dry-run": {
        "src/janitor/cleanup.py": '''\
"""Deletion pass over the policy's candidates."""

from src.janitor.policy import candidates


def run_cleanup(root, max_age_seconds, emit=print, dry_run=False):
    removed = []
    for path in candidates(root, max_age_seconds):
        if dry_run:
            emit(f"would delete {path}")
            continue
        path.unlink()
        emit(f"deleted {path}")
        removed.append(path)
    return removed
''',
        "src/janitor/cli.py": '''\
"""Command-line entry point."""

import argparse

from src.janitor.cleanup import run_cleanup


def main(argv=None):
    parser = argparse.ArgumentParser(description="Delete stale scratch files.")
    parser.add_argument("root")
    parser.add_argument("--max-age-seconds", type=float, default=3600.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be deleted without deleting anything",
    )
    args = parser.parse_args(argv)
    run_cleanup(args.root, args.max_age_seconds, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
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
    "problem-25-implement-parse-lineno": {
        "src/logkit/errors.py": '''\
"""Errors raised by logkit.

Contract for parse failures (the parser must follow this exactly):

- ``ParseError(reason, lineno)`` — ``lineno`` is the 1-BASED line number of
  the offending line in the original input, counting every line including
  blank lines and comments.
- ``str(error)`` must be exactly ``"line {lineno}: {reason}"``.
- The error object must expose the line number as an ``int`` attribute
  named ``lineno`` and the bare reason as an attribute named ``reason``.
- Parsing stops at the FIRST invalid line; later invalid lines are not
  reported.
"""


class ParseError(Exception):
    def __init__(self, reason, lineno):
        super().__init__(f"line {lineno}: {reason}")
        self.reason = reason
        self.lineno = lineno
''',
        "src/logkit/parser.py": '''\
"""Parser for the simple ``KEY=VALUE`` config format.

Rules:

- One ``KEY=VALUE`` assignment per line. Keys are non-empty and contain no
  whitespace; values may be empty and keep inner whitespace.
- Blank lines and lines starting with ``#`` (after stripping leading
  whitespace) are skipped.
- When a key repeats, the LAST assignment wins.
- Invalid lines (no ``=``, or an empty/whitespace-containing key) raise
  ``ParseError`` — see ``errors.py`` for the exact failure contract.
"""

from src.logkit.errors import ParseError


def parse(text):
    result = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ParseError(f"missing '=' in {line!r}", lineno)
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or any(ch.isspace() for ch in key):
            raise ParseError(f"bad key in {line!r}", lineno)
        result[key] = value.strip()
    return result
''',
    },
    "problem-26-lru-pin-revision": {
        "src/lrupin.py": '''\
"""LRU cache with pinnable entries.

Contract:

- ``get(key)`` returns the value and marks the key as most recently used.
  Missing keys raise ``KeyError``.
- ``put(key, value)`` inserts or overwrites. Overwriting an existing key
  updates its value, marks it most recently used, and NEVER evicts.
- Inserting a new key when the cache is full evicts the least recently
  used UNPINNED key first. Pinned keys are never evicted.
- If every entry is pinned and the cache is full, inserting a new key
  raises ``RuntimeError("all entries pinned")`` and changes nothing.
- ``pin(key)`` / ``unpin(key)`` toggle protection; both raise ``KeyError``
  for keys not in the cache. Pinning does not change recency. An unpinned
  key competes for eviction based on its existing recency, not the time
  it was unpinned.
"""


class LRUCache:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._data = {}
        self._pinned = set()

    def get(self, key):
        value = self._data.pop(key)
        self._data[key] = value
        return value

    def put(self, key, value):
        if key in self._data:
            self._data.pop(key)
            self._data[key] = value
            return
        if len(self._data) >= self.capacity:
            self._evict()
        self._data[key] = value

    def pin(self, key):
        if key not in self._data:
            raise KeyError(key)
        self._pinned.add(key)

    def unpin(self, key):
        if key not in self._data:
            raise KeyError(key)
        self._pinned.discard(key)

    def _evict(self):
        for key in self._data:
            if key not in self._pinned:
                del self._data[key]
                return
        raise RuntimeError("all entries pinned")

    def keys(self):
        return list(self._data)

    def __len__(self):
        return len(self._data)

    def __contains__(self, key):
        return key in self._data
''',
    },
    "problem-27-implement-rates-section": {
        "src/reportkit/sections.py": '''\
"""Report sections.

Every section is a function ``(suite, records) -> list[str]`` returning the
section's body lines (without the banner). Sections are registered in
``SECTIONS`` as ``(name, function)`` pairs; the renderer emits them in list
order. See docs/REPORT_FORMAT.md for the exact output contract.
"""

from collections import Counter

KNOWN_STATUSES = ("pass", "fail", "error", "skip")


def section_header(suite, records):
    return [f"suite: {suite}", f"records: {len(records)}"]


def section_counts(suite, records):
    counts = Counter()
    for record in records:
        status = record.get("status")
        counts[status if status in KNOWN_STATUSES else "error"] += 1
    return [f"{status}: {counts[status]}" for status in KNOWN_STATUSES]


def section_rates(suite, records):
    counts = Counter()
    for record in records:
        status = record.get("status")
        counts[status if status in KNOWN_STATUSES else "error"] += 1
    attempted = counts["pass"] + counts["fail"] + counts["error"]
    if attempted == 0:
        return ["pass_rate: n/a", "error_rate: n/a"]
    pass_rate = 100.0 * counts["pass"] / attempted
    error_rate = 100.0 * (counts["fail"] + counts["error"]) / attempted
    return [f"pass_rate: {pass_rate:.1f}%", f"error_rate: {error_rate:.1f}%"]


SECTIONS = [
    ("header", section_header),
    ("counts", section_counts),
    ("rates", section_rates),
]
''',
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

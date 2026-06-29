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
    score = raw.get("score", 0.0)

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

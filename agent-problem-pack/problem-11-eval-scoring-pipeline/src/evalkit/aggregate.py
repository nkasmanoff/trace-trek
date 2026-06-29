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
        scored = items
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

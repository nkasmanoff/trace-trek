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
                "weight": task.get("weight") or 1.0,
            }
        )
    return results

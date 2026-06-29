"""Stage 2: run each task and emit a raw result.

The runner calls a model adapter for every task and records a per-task score
in [0, 1]. The task's `weight` is copied onto the result so later stages can
compute a weighted aggregate without re-reading the manifest.
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

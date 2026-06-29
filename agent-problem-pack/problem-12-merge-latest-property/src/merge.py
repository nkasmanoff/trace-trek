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
        if task_id not in best or record["attempt"] > best[task_id]["attempt"]:
            best[task_id] = record
    return list(best.values())

from src.merge import merge_latest


def test_single_record():
    assert merge_latest([{"task_id": "t1", "attempt": 1, "score": 0.5}]) == [
        {"task_id": "t1", "attempt": 1, "score": 0.5}
    ]


def test_keeps_highest_attempt_when_ascending():
    records = [
        {"task_id": "t1", "attempt": 1, "score": 0.2},
        {"task_id": "t1", "attempt": 2, "score": 0.9},
    ]
    assert merge_latest(records) == [{"task_id": "t1", "attempt": 2, "score": 0.9}]


def test_distinct_tasks_sorted():
    records = [
        {"task_id": "t2", "attempt": 1, "score": 0.1},
        {"task_id": "t1", "attempt": 1, "score": 0.4},
    ]
    assert merge_latest(records) == [
        {"task_id": "t1", "attempt": 1, "score": 0.4},
        {"task_id": "t2", "attempt": 1, "score": 0.1},
    ]

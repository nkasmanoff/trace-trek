from hypothesis import given
from hypothesis import strategies as st

from src.merge import merge_latest


records_strategy = st.lists(
    st.fixed_dictionaries(
        {
            "task_id": st.sampled_from(["t1", "t2", "t3", "t4"]),
            "attempt": st.integers(min_value=0, max_value=8),
            "score": st.floats(min_value=0.0, max_value=1.0),
        }
    ),
    max_size=20,
)


def _reference(records):
    best = {}
    for index, record in enumerate(records):
        task_id = record["task_id"]
        current = best.get(task_id)
        if current is None or (record["attempt"], index) >= (current[0]["attempt"], current[1]):
            best[task_id] = (record, index)
    return sorted((value[0] for value in best.values()), key=lambda r: r["task_id"])


def test_result_sorted_by_task_id():
    records = [
        {"task_id": "t3", "attempt": 5, "score": 0.1},
        {"task_id": "t1", "attempt": 2, "score": 0.2},
    ]
    result = merge_latest(records)
    assert [r["task_id"] for r in result] == sorted(r["task_id"] for r in result)


def test_does_not_mutate_input():
    records = [
        {"task_id": "t1", "attempt": 1, "score": 0.2},
        {"task_id": "t1", "attempt": 3, "score": 0.8},
    ]
    snapshot = [dict(r) for r in records]
    merge_latest(records)
    assert records == snapshot


@given(records=records_strategy)
def test_matches_reference_regardless_of_order(records):
    result = merge_latest(records)
    expected = _reference(records)
    assert result == expected


@given(records=records_strategy)
def test_one_record_per_task_and_highest_attempt(records):
    result = merge_latest(records)
    ids = [r["task_id"] for r in result]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)
    for kept in result:
        max_attempt = max(r["attempt"] for r in records if r["task_id"] == kept["task_id"])
        assert kept["attempt"] == max_attempt

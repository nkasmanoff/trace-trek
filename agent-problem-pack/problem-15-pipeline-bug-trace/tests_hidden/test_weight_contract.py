from hypothesis import given
from hypothesis import strategies as st

from src.pipeline.aggregate import aggregate
from src.pipeline.dataset import load_tasks
from src.pipeline.runner import run_tasks


class ConstModel:
    def __init__(self, value):
        self._value = value

    def score(self, prompt):
        return self._value


def test_zero_weight_survives_the_runner():
    tasks = load_tasks(
        ['{"task_id": "t1", "type": "code", "prompt": "p1", "weight": 0.0}']
    )
    results = run_tasks(tasks, ConstModel(1.0))
    assert results[0]["weight"] == 0.0


def test_zero_weight_excluded_from_aggregate_but_recorded():
    results = [
        {"task_id": "t1", "type": "code", "score": 1.0, "weight": 0.0},
        {"task_id": "t2", "type": "code", "score": 0.5, "weight": 1.0},
    ]
    summary = aggregate(results)
    assert summary["recorded"] == 2
    assert summary["scored"] == 1
    assert abs(summary["weighted_score"] - 0.5) < 1e-9


def test_all_zero_weights_scores_zero():
    results = [
        {"task_id": "t1", "type": "code", "score": 1.0, "weight": 0.0},
        {"task_id": "t2", "type": "code", "score": 1.0, "weight": 0.0},
    ]
    summary = aggregate(results)
    assert summary["recorded"] == 2
    assert summary["scored"] == 0
    assert summary["weighted_score"] == 0.0


def test_end_to_end_zero_weight_task():
    lines = [
        '{"task_id": "t1", "type": "code", "prompt": "p1", "weight": 0.0}',
        '{"task_id": "t2", "type": "code", "prompt": "p2", "weight": 3.0}',
        '{"task_id": "t3", "type": "code", "prompt": "p3"}',
    ]
    tasks = load_tasks(lines)

    class Model:
        def score(self, prompt):
            return {"p1": 1.0, "p2": 0.6, "p3": 0.2}[prompt]

    summary = aggregate(run_tasks(tasks, Model()))
    expected = (0.6 * 3.0 + 0.2 * 1.0) / (3.0 + 1.0)
    assert summary["recorded"] == 3
    assert summary["scored"] == 2
    assert abs(summary["weighted_score"] - expected) < 1e-9


weights = st.one_of(st.just(0.0), st.floats(min_value=0.1, max_value=5.0))
results_strategy = st.lists(
    st.builds(
        lambda i, s, w: {"task_id": f"t{i}", "type": "code", "score": s, "weight": w},
        st.integers(min_value=0, max_value=99),
        st.floats(min_value=0.0, max_value=1.0),
        weights,
    ),
    max_size=25,
)


@given(results=results_strategy)
def test_matches_reference_weighted_mean(results):
    summary = aggregate(results)
    positive = [r for r in results if r["weight"] > 0]
    denominator = sum(r["weight"] for r in positive)
    if denominator:
        expected = sum(r["score"] * r["weight"] for r in positive) / denominator
    else:
        expected = 0.0
    assert abs(summary["weighted_score"] - expected) < 1e-9
    assert summary["recorded"] == len(results)
    assert summary["scored"] == len(positive)

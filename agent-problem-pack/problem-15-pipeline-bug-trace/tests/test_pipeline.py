from src.pipeline.aggregate import aggregate
from src.pipeline.dataset import load_tasks
from src.pipeline.report import render_summary
from src.pipeline.runner import run_tasks


class FixedModel:
    def __init__(self, scores):
        self._scores = dict(scores)

    def score(self, prompt):
        return self._scores[prompt]


def test_load_tasks_defaults_missing_weight():
    tasks = load_tasks(['{"task_id": "t1", "type": "code", "prompt": "p1"}'])
    assert tasks == [
        {"task_id": "t1", "type": "code", "prompt": "p1", "weight": 1.0}
    ]


def test_uniform_weights_match_plain_mean():
    tasks = load_tasks(
        [
            '{"task_id": "t1", "type": "code", "prompt": "p1"}',
            '{"task_id": "t2", "type": "code", "prompt": "p2"}',
        ]
    )
    results = run_tasks(tasks, FixedModel({"p1": 0.4, "p2": 0.8}))
    summary = aggregate(results)
    assert summary["weighted_score"] == (0.4 + 0.8) / 2


def test_weighted_mean_uses_weights():
    tasks = load_tasks(
        [
            '{"task_id": "t1", "type": "code", "prompt": "p1", "weight": 2.0}',
            '{"task_id": "t2", "type": "code", "prompt": "p2", "weight": 1.0}',
        ]
    )
    results = run_tasks(tasks, FixedModel({"p1": 0.9, "p2": 0.3}))
    summary = aggregate(results)
    expected = (0.9 * 2.0 + 0.3 * 1.0) / (2.0 + 1.0)
    assert abs(summary["weighted_score"] - expected) < 1e-9


def test_render_summary_mentions_counts():
    summary = {"recorded": 3, "scored": 2, "weighted_score": 0.5}
    line = render_summary(summary)
    assert "2/3" in line
    assert "0.500" in line

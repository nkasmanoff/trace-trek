from pathlib import Path

ANSWER = Path(__file__).parents[1] / "AGENT_FINAL_ANSWER.md"


def test_answer_exists():
    assert ANSWER.exists(), "AGENT_FINAL_ANSWER.md not found"


def test_covers_eval_results_schema():
    text = ANSWER.read_text(encoding="utf-8")
    assert "task_id" in text, "Answer should mention task_id"
    assert "type" in text, "Answer should mention type (code/knowledge)"
    assert "score" in text.lower(), "Answer should mention score"
    assert "passed" in text.lower(), "Answer should mention passed"


def test_covers_aggregation():
    text = ANSWER.read_text(encoding="utf-8")
    assert "aggregateEval" in text or "aggregat" in text.lower(), "Answer should mention aggregation"
    assert "bytype" in text.lower() or "by_type" in text.lower() or "by type" in text.lower() or "per-type" in text.lower() or "per type" in text.lower(), "Answer should mention grouping by type"


def test_covers_viewer_display():
    text = ANSWER.read_text(encoding="utf-8")
    assert "EvalView" in text, "Answer should mention EvalView component"
    assert "flip" in text.lower() or "comparison" in text.lower(), "Answer should mention run comparison"


def test_covers_data_flow():
    text = ANSWER.read_text(encoding="utf-8")
    assert "pipeline" in text.lower(), "Answer should mention the pipeline that generates the data"
    assert "run" in text.lower(), "Answer should mention runs"
    assert "pass rate" in text.lower() or "pass_rate" in text.lower(), "Answer should mention pass rate"

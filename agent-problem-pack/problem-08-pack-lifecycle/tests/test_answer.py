from pathlib import Path

ANSWER = Path(__file__).parents[1] / "AGENT_FINAL_ANSWER.md"


def test_answer_exists():
    assert ANSWER.exists(), "AGENT_FINAL_ANSWER.md not found"


def test_covers_prepare_stage():
    text = ANSWER.read_text(encoding="utf-8")
    assert "prepare" in text.lower(), "Answer should describe the prepare step"
    assert "workspace" in text.lower(), "Answer should mention the workspace"
    assert "task-prompt" in text.lower(), "Answer should mention the task prompt"


def test_covers_capture_stage():
    text = ANSWER.read_text(encoding="utf-8")
    assert "capture" in text.lower(), "Answer should describe the capture step"
    assert "diff" in text.lower(), "Answer should mention diff capture"
    assert "verification" in text.lower(), "Answer should mention verification"


def test_covers_agent_work_stage():
    text = ANSWER.read_text(encoding="utf-8")
    assert "agent" in text.lower(), "Answer should describe the agent's role"
    assert "AGENT_FINAL_ANSWER" in text or "final answer" in text.lower(), "Answer should mention the final answer"


def test_covers_structure():
    text = ANSWER.read_text(encoding="utf-8")
    assert "metadata.json" in text.lower(), "Answer should mention metadata.json"
    assert "usage.json" in text.lower() or "usage" in text.lower(), "Answer should mention usage tracking"

from pathlib import Path

ANSWER = Path(__file__).parents[1] / "AGENT_FINAL_ANSWER.md"


def _text():
    return ANSWER.read_text(encoding="utf-8").lower()


def test_answer_exists():
    assert ANSWER.exists(), "AGENT_FINAL_ANSWER.md not found"


def test_covers_task_taxonomy():
    text = _text()
    assert "repair" in text, "Answer should include the repair task kind"
    assert "comprehension" in text, "Answer should include the comprehension task kind"
    assert "pytest" in text or "test" in text, "Answer should explain test-based verification"
    assert "rubric" in text or "judge" in text, (
        "Answer should mention rubric/judge scoring beyond raw tests"
    )


def test_covers_isolation_lifecycle():
    text = _text()
    assert "workspace" in text and "isolat" in text, (
        "Answer should explain isolated per-run workspaces"
    )
    assert "prepare" in text and "capture" in text, (
        "Answer should describe the prepare -> agent -> capture lifecycle"
    )
    assert "git" in text or "diff" in text or "baseline" in text, (
        "Answer should mention git baseline / diff capture"
    )


def test_covers_overfit_resistance():
    text = _text()
    assert "hidden" in text, "Answer should mention hidden tests"
    assert "overfit" in text or "gaming" in text or "leak" in text or "memoriz" in text, (
        "Answer should explain resistance to overfitting/gaming"
    )


def test_covers_validation():
    text = _text()
    assert "baseline" in text, "Answer should mention the failing baseline check"
    assert "golden" in text or "validat" in text, (
        "Answer should mention golden-fix / validation integrity check"
    )


def test_covers_scoring_repro_and_usage():
    text = _text()
    assert "token" in text, "Answer should mention token-usage capture"
    assert (
        "pass rate" in text
        or "pass_rate" in text
        or "score" in text
        or "aggregat" in text
    ), "Answer should mention scoring/aggregation (pass rate / score)"
    assert "reproduc" in text, "Answer should explain what makes the suite reproducible"

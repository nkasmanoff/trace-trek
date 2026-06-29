from pathlib import Path

ANSWER = Path(__file__).parents[1] / "AGENT_FINAL_ANSWER.md"


def _text():
    return ANSWER.read_text(encoding="utf-8").lower()


def test_answer_exists():
    assert ANSWER.exists(), "AGENT_FINAL_ANSWER.md not found"


def test_covers_extraction():
    text = _text()
    assert "persona vector" in text, "Answer should name the persona vector"
    assert "activation" in text or "hidden state" in text or "hidden_state" in text, (
        "Answer should explain that the vector comes from model activations / hidden states"
    )
    assert "difference" in text or "diff" in text, (
        "Answer should describe the mean-difference construction"
    )
    assert "positive" in text and "negative" in text, (
        "Answer should mention the positive vs negative (contrastive) prompts"
    )
    assert "response" in text and "prompt" in text, (
        "Answer should distinguish response vs prompt activation summaries"
    )


def test_covers_filtering_and_layers():
    text = _text()
    assert "coherence" in text or "threshold" in text or "effective" in text, (
        "Answer should mention filtering effective examples (threshold / coherence)"
    )
    assert "layer" in text, "Answer should mention that vectors are computed per layer"


def test_covers_steering():
    text = _text()
    assert "steer" in text, "Answer should describe steering"
    assert "coef" in text, "Answer should mention the steering coefficient (coef/coeff)"
    assert "hook" in text or "activationsteerer" in text, (
        "Answer should mention the forward hook / ActivationSteerer mechanism"
    )
    assert "all" in text and ("prompt" in text and "response" in text), (
        "Answer should mention the positions option (all / prompt / response)"
    )


def test_covers_monitoring_and_training():
    text = _text()
    assert "monitor" in text or "projection" in text or "project" in text, (
        "Answer should cover monitoring via projection"
    )
    assert "control" in text or "steer" in text, "Answer should cover control via steering"
    assert (
        "preventative" in text
        or "caft" in text
        or "ablat" in text
        or ("training" in text and "steer" in text)
    ), "Answer should cover training-time / preventative steering (steer vs ablate/CAFT)"

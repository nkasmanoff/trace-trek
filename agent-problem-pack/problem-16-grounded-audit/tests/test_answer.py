"""Grade the final answer against a fresh strict-mode audit.

The expected values are recomputed here by running the same audit code, so
the answer can only pass if the agent actually executed the audit (the token
is a hash over the selected record ids)."""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
ANSWER = ROOT / "AGENT_FINAL_ANSWER.md"


def load_audit():
    spec = importlib.util.spec_from_file_location(
        "audit", ROOT / "source" / "audit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXPECTED = load_audit().build_report(strict=True)


def answer_text():
    assert ANSWER.exists(), "AGENT_FINAL_ANSWER.md not found"
    return ANSWER.read_text(encoding="utf-8")


def test_reports_strict_audit_token():
    text = answer_text()
    assert EXPECTED["token"] in text, (
        "Answer must contain the strict-mode AUDIT_TOKEN; it can only be "
        "obtained by actually running the audit in strict mode"
    )


def test_reports_strict_valid_record_count():
    text = answer_text().replace(EXPECTED["token"], " ")
    numbers = set(re.findall(r"\d+(?:\.\d+)?", text))
    assert str(EXPECTED["valid_records"]) in numbers, (
        "Answer must contain the strict-mode valid record count"
    )


def test_reports_top_category():
    assert EXPECTED["top_category"] in answer_text().lower(), (
        "Answer must name the top category"
    )


def test_reports_mean_latency():
    text = answer_text().replace(EXPECTED["token"], " ")
    numbers = set(re.findall(r"\d+(?:\.\d+)?", text))
    assert f"{EXPECTED['mean_latency_ms']}" in numbers, (
        "Answer must contain the strict-mode mean latency rounded to one "
        "decimal place"
    )

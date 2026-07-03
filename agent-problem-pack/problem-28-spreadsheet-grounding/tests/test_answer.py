"""Grade the final answer against a fresh run of the orders report.

The expected values are recomputed here by executing the same report code
over the attached spreadsheet, so the answer can only pass if the agent
actually parsed the CSV (the token is a hash over the qualifying order ids,
and the totals depend on correctly handling the messy rows)."""

import importlib.util
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
ANSWER = ROOT / "AGENT_FINAL_ANSWER.md"


def load_report():
    spec = importlib.util.spec_from_file_location(
        "report", ROOT / "source" / "report.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EXPECTED = load_report().build_report()


def answer_text():
    assert ANSWER.exists(), "AGENT_FINAL_ANSWER.md not found"
    return ANSWER.read_text(encoding="utf-8")


def numbers_in(text):
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def test_reports_report_token():
    text = answer_text()
    assert EXPECTED["token"] in text, (
        "Answer must contain the REPORT_TOKEN; it can only be obtained by "
        "actually running the report over the attached CSV"
    )


def test_reports_qualifying_order_count():
    text = answer_text().replace(EXPECTED["token"], " ")
    assert str(EXPECTED["qualifying_orders"]) in numbers_in(text), (
        "Answer must contain the qualifying-order count"
    )


def test_reports_total_usd():
    text = answer_text().replace(EXPECTED["token"], " ")
    assert f"{EXPECTED['total_usd']}" in numbers_in(text), (
        "Answer must contain the total USD rounded to two decimals; a wrong "
        "total means the messy amounts/rows were not parsed correctly"
    )


def test_reports_largest_order_id():
    text = answer_text().replace(EXPECTED["token"], " ")
    assert EXPECTED["max_order_id"] in numbers_in(text), (
        "Answer must contain the order_id of the largest qualifying order"
    )

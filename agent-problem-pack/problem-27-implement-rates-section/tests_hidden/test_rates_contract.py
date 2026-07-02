"""Hidden grading tests for the planned `rates` section.

Grades the exact contract in docs/REPORT_FORMAT.md: placement immediately
after `counts`, skip-exclusion, `.1f` percent formatting, and the `n/a`
degenerate case.
"""

from src.reportkit.render import render_report
from src.reportkit.sections import SECTIONS


def test_rates_registered_immediately_after_counts():
    names = [name for name, _ in SECTIONS]
    assert "rates" in names
    assert names.index("rates") == names.index("counts") + 1


def test_rates_exact_output_excludes_skips():
    records = [
        {"task": "t1", "status": "pass"},
        {"task": "t2", "status": "pass"},
        {"task": "t3", "status": "pass"},
        {"task": "t4", "status": "fail"},
        {"task": "t5", "status": "error"},
        {"task": "t6", "status": "skip"},
        {"task": "t7", "status": "skip"},
        {"task": "t8", "status": "pass"},
    ]
    # attempted = 6, pass = 4 (66.7%), fail+error = 2 (33.3%)
    out = render_report("s", records)
    assert "== rates ==\npass_rate: 66.7%\nerror_rate: 33.3%" in out


def test_unknown_status_counts_as_error_in_rates():
    records = [
        {"task": "t1", "status": "pass"},
        {"task": "t2", "status": "mystery"},
    ]
    out = render_report("s", records)
    assert "pass_rate: 50.0%" in out
    assert "error_rate: 50.0%" in out


def test_all_skipped_renders_na():
    records = [
        {"task": "t1", "status": "skip"},
        {"task": "t2", "status": "skip"},
    ]
    out = render_report("s", records)
    assert "== rates ==\npass_rate: n/a\nerror_rate: n/a" in out


def test_empty_records_renders_na():
    out = render_report("s", [])
    assert "pass_rate: n/a" in out
    assert "error_rate: n/a" in out


def test_hundred_percent_formatting():
    records = [{"task": "t1", "status": "pass"}]
    out = render_report("s", records)
    assert "pass_rate: 100.0%\nerror_rate: 0.0%" in out

from src.reportkit.render import render_report

RECORDS = [
    {"task": "t1", "status": "pass"},
    {"task": "t2", "status": "pass"},
    {"task": "t3", "status": "fail"},
    {"task": "t4", "status": "skip"},
]


def test_header_section():
    out = render_report("nightly", RECORDS)
    assert "== header ==\nsuite: nightly\nrecords: 4\n" in out


def test_counts_section():
    out = render_report("nightly", RECORDS)
    assert "== counts ==\npass: 2\nfail: 1\nerror: 0\nskip: 1\n" in out


def test_unknown_status_counts_as_error():
    out = render_report("s", [{"task": "t", "status": "wat"}])
    assert "error: 1" in out


def test_rates_section_present():
    out = render_report("nightly", RECORDS)
    assert "== rates ==" in out

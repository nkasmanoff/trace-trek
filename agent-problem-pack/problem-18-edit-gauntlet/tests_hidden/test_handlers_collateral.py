"""Hidden: catch collateral damage to the three handlers that were correct.

A blanket find/replace fix (e.g. replaceAll on the ttl or boundary lines)
passes the visible tests but breaks one of these."""

from src.handlers import DEFAULT_TTL, LIMITS, dispatch


def make_event(kind, value, ttl=None):
    event = {"kind": kind, "data": {"value": value}}
    if ttl is not None:
        event["ttl"] = ttl
    return event


def test_metric_ttl_not_decremented():
    result = dispatch(make_event("metric", 1, ttl=9), {})
    assert result["ttl"] == 9


def test_metric_over_limit_rejected():
    state = {}
    result = dispatch(make_event("metric", LIMITS["metric"] + 1), state)
    assert result["accepted"] is False
    assert state["rejected"] == 1


def test_log_at_limit_is_accepted():
    result = dispatch(make_event("log", LIMITS["log"]), {})
    assert result["accepted"] is True


def test_log_ttl_not_decremented():
    result = dispatch(make_event("log", 1, ttl=6), {})
    assert result["ttl"] == 6


def test_trace_over_limit_rejected():
    state = {}
    result = dispatch(make_event("trace", LIMITS["trace"] + 1), state)
    assert result["accepted"] is False
    assert state["rejected"] == 1


def test_trace_ttl_not_decremented():
    result = dispatch(make_event("trace", 1, ttl=2), {})
    assert result["ttl"] == 2


def test_alert_at_limit_is_accepted():
    state = {}
    result = dispatch(make_event("alert", LIMITS["alert"]), state)
    assert result["accepted"] is True
    assert state["alert"] == [LIMITS["alert"]]


def test_alert_default_ttl_unchanged():
    result = dispatch(make_event("alert", 1), {})
    assert result["ttl"] == DEFAULT_TTL


def test_audit_reads_value_from_data_key():
    state = {}
    result = dispatch(make_event("audit", 5), state)
    assert result["value"] == 5
    assert state["audit"] == [5]


def test_audit_at_limit_is_accepted():
    result = dispatch(make_event("audit", LIMITS["audit"]), {})
    assert result["accepted"] is True


def test_heartbeat_at_limit_is_accepted():
    result = dispatch(make_event("heartbeat", LIMITS["heartbeat"]), {})
    assert result["accepted"] is True


def test_dispatch_unknown_kind_raises():
    try:
        dispatch({"kind": "nope", "data": {}}, {})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown kind")

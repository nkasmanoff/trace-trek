from src.handlers import DEFAULT_TTL, LIMITS, dispatch


def make_event(kind, value, ttl=None):
    event = {"kind": kind, "data": {"value": value}}
    if ttl is not None:
        event["ttl"] = ttl
    return event


def test_metric_at_limit_is_accepted():
    state = {}
    result = dispatch(make_event("metric", LIMITS["metric"]), state)
    assert result["accepted"] is True
    assert state["metric"] == [LIMITS["metric"]]


def test_log_reads_value_from_data_key():
    state = {}
    result = dispatch(make_event("log", 7), state)
    assert result["accepted"] is True
    assert result["value"] == 7
    assert state["log"] == [7]


def test_trace_at_limit_is_accepted():
    state = {}
    result = dispatch(make_event("trace", LIMITS["trace"]), state)
    assert result["accepted"] is True
    assert state["trace"] == [LIMITS["trace"]]


def test_alert_forwards_ttl_unchanged():
    state = {}
    result = dispatch(make_event("alert", 3, ttl=5), state)
    assert result["accepted"] is True
    assert result["ttl"] == 5


def test_audit_over_limit_rejected():
    state = {}
    result = dispatch(make_event("audit", LIMITS["audit"] + 1), state)
    assert result["accepted"] is False
    assert state["rejected"] == 1


def test_heartbeat_decrements_ttl():
    state = {}
    result = dispatch(make_event("heartbeat", 1, ttl=4), state)
    assert result["accepted"] is True
    assert result["ttl"] == 3


def test_heartbeat_default_ttl_decremented():
    state = {}
    result = dispatch(make_event("heartbeat", 0), state)
    assert result["ttl"] == DEFAULT_TTL - 1

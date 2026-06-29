from evalkit import aggregate_results, format_summary, normalize_result


def normalize_all(raw_records, run):
    return [normalize_result(raw, run) for raw in raw_records]


def test_ok_task_passes_at_threshold():
    result = normalize_result({"task_id": "t1", "type": "code", "status": "ok", "score": 0.5}, "baseline")
    assert result["passed"] is True


def test_error_task_is_failure():
    result = normalize_result({"task_id": "t2", "type": "code", "status": "error", "score": 0.0}, "baseline")
    assert result["passed"] is False


def test_skip_task_is_excluded_from_aggregation():
    raw = [
        {"task_id": "t1", "type": "code", "status": "ok", "score": 0.8},
        {"task_id": "t2", "type": "code", "status": "skip", "score": 0.0},
    ]
    summary = aggregate_results(normalize_all(raw, "baseline"))
    code = next(b for b in summary if b["type"] == "code")
    assert code["total"] == 1
    assert code["passed"] == 1
    assert code["pass_rate"] == 1.0


def test_summary_is_formatted():
    raw = [{"task_id": "t1", "type": "code", "status": "ok", "score": 0.9}]
    text = format_summary(aggregate_results(normalize_all(raw, "baseline")))
    assert "baseline/code" in text
    assert "1/1 passed" in text

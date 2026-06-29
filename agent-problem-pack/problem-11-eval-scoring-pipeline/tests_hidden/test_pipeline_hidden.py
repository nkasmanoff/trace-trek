from evalkit import aggregate_results, normalize_result


def normalize_all(raw_records, run):
    return [normalize_result(raw, run) for raw in raw_records]


def test_score_clamped_to_unit_interval():
    high = normalize_result({"task_id": "t1", "type": "code", "status": "ok", "score": 4.2}, "baseline")
    low = normalize_result({"task_id": "t2", "type": "code", "status": "ok", "score": -1.0}, "baseline")
    assert high["score"] == 1.0
    assert low["score"] == 0.0


def test_skip_excluded_from_mean_score():
    raw = [
        {"task_id": "t1", "type": "code", "status": "ok", "score": 1.0},
        {"task_id": "t2", "type": "code", "status": "skip", "score": 0.0},
    ]
    code = aggregate_results(normalize_all(raw, "baseline"))[0]
    assert code["total"] == 1
    assert code["mean_score"] == 1.0


def test_all_skipped_bucket_is_empty():
    raw = [
        {"task_id": "t1", "type": "knowledge", "status": "skip", "score": 0.0},
        {"task_id": "t2", "type": "knowledge", "status": "skip", "score": 0.9},
    ]
    bucket = aggregate_results(normalize_all(raw, "baseline"))[0]
    assert bucket["total"] == 0
    assert bucket["pass_rate"] == 0.0
    assert bucket["mean_score"] == 0.0


def test_mean_score_uses_clamped_values():
    raw = [
        {"task_id": "t1", "type": "code", "status": "ok", "score": 2.0},
        {"task_id": "t2", "type": "code", "status": "ok", "score": 0.0},
    ]
    code = aggregate_results(normalize_all(raw, "baseline"))[0]
    assert code["mean_score"] == 0.5


def test_error_score_still_clamped_and_excludes_nothing():
    raw = [
        {"task_id": "t1", "type": "code", "status": "error", "score": 9.0},
        {"task_id": "t2", "type": "code", "status": "ok", "score": 0.5},
    ]
    code = aggregate_results(normalize_all(raw, "baseline"))[0]
    assert code["total"] == 2
    assert code["passed"] == 1
    assert code["mean_score"] == 0.75

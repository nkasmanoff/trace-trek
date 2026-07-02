from src.flowkit.core.pipeline import Pipeline


def test_smooth_stage_uses_trailing_window():
    pipeline = Pipeline({"stages": [{"stage": "smooth", "params": {"window": 3}}]})
    assert pipeline.run([2, 4, 6]) == [2.0, 3.0, 4.0]


def test_smooth_then_clip_pipeline():
    pipeline = Pipeline(
        {
            "stages": [
                {"stage": "smooth", "params": {"window": 3}},
                {"stage": "clip", "params": {"hi": 3.5}},
            ]
        }
    )
    assert pipeline.run([1, 2, 3, 4, 5]) == [1.0, 1.5, 2.0, 3.0, 3.5]

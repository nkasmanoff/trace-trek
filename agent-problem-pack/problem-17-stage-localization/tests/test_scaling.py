from src.flowkit.transforms.scaling import clip, minmax_scale


def test_clip_bounds():
    assert clip([-1, 0.5, 2], lo=0, hi=1) == [0, 0.5, 1]


def test_minmax_scale_range():
    assert minmax_scale([0, 5, 10]) == [0.0, 0.5, 1.0]

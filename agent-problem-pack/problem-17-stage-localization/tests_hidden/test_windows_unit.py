"""Hidden: pin the fix to the right function in the right module."""

from src.flowkit.core.registry import get_stage
from src.flowkit.stats.window_stats import rolling_mean as report_rolling_mean
from src.flowkit.transforms.windows import rolling_mean


def test_trailing_window_basic():
    assert rolling_mean([1, 2, 3, 4, 5], 3) == [1.0, 1.5, 2.0, 3.0, 4.0]


def test_window_one_is_identity():
    assert rolling_mean([5.0, 1.0, 9.0], 1) == [5.0, 1.0, 9.0]


def test_window_larger_than_prefix_uses_available_values():
    assert rolling_mean([10, 20], 5) == [10.0, 15.0]


def test_report_helper_not_rewired():
    assert report_rolling_mean([1, 2, 3, 4, 5], 3) == [1.0, 1.5, 2.0, 3.0, 4.0]


def test_smooth_stage_still_bound_to_transforms_windows():
    assert get_stage("smooth") is rolling_mean

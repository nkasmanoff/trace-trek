"""Summary statistics for report rendering."""

from src.flowkit.stats.window_stats import rolling_mean


def describe(values, window=3):
    if not values:
        return {"count": 0, "mean": 0.0, "smoothed_tail": None}
    mean = sum(values) / len(values)
    return {
        "count": len(values),
        "mean": mean,
        "smoothed_tail": rolling_mean(values, window)[-1],
    }

"""Exponential smoothing (EWMA) stage."""


def exponential_smoothing(values, alpha=0.5):
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    out = []
    level = None
    for value in values:
        level = value if level is None else alpha * value + (1 - alpha) * level
        out.append(level)
    return out

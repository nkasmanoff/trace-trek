"""Value scaling stages."""


def clip(values, lo=None, hi=None):
    out = []
    for value in values:
        if lo is not None and value < lo:
            value = lo
        if hi is not None and value > hi:
            value = hi
        out.append(value)
    return out


def minmax_scale(values):
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]

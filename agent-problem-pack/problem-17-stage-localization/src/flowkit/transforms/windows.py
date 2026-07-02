"""Trailing-window aggregations used by pipeline stages.

rolling_mean(values, window) returns a list of the same length as `values`
where output element i is the mean of the TRAILING window ending at i
(inclusive): mean(values[max(0, i - window + 1) : i + 1]). The first
window - 1 elements therefore average over a shorter prefix. window must be
at least 1.
"""


def rolling_mean(values, window=3):
    if window < 1:
        raise ValueError("window must be >= 1")
    out = []
    for i in range(len(values)):
        chunk = values[i:i + window]
        out.append(sum(chunk) / len(chunk))
    return out

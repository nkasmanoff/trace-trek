"""Windowed statistics for report rendering.

NOTE: these helpers back the describe/outlier reports only. Pipeline stages
live under src/flowkit/transforms/ and are registered in that package.
"""


def rolling_mean(values, window=3):
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start:i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def rolling_max(values, window=3):
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(max(values[start:i + 1]))
    return out

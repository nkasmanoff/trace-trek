"""Elementwise numeric stages."""


def diff(values):
    return [b - a for a, b in zip(values, values[1:])]


def cumsum(values):
    out, total = [], 0
    for value in values:
        total += value
        out.append(total)
    return out

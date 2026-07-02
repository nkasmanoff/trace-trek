"""Simple z-score outlier detection."""


def zscore_outliers(values, threshold=3.0):
    if len(values) < 2:
        return []
    mean = sum(values) / len(values)
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    std = var ** 0.5
    if std == 0:
        return []
    return [i for i, value in enumerate(values) if abs(value - mean) / std > threshold]

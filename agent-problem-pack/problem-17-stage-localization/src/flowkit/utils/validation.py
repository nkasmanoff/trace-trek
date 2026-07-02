"""Value validation helpers."""


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def require_numbers(values):
    bad = [value for value in values if not is_number(value)]
    if bad:
        raise TypeError(f"non-numeric values: {bad!r}")
    return values

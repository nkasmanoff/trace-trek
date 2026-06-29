def flatten(nested, depth=None):
    result = []
    for item in nested:
        if isinstance(item, list) and (depth is None or depth > 1):
            sub = depth - 1 if depth is not None else None
            result.extend(flatten(item, sub))
        else:
            result.append(item)
    return result

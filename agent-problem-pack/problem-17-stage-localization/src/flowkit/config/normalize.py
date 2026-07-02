"""Key normalization for configs loaded from JSON/YAML-ish sources."""


def normalize_keys(obj):
    if isinstance(obj, dict):
        return {key.replace("-", "_"): normalize_keys(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [normalize_keys(item) for item in obj]
    return obj

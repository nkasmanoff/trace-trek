"""Config validation for pipelines."""

from src.flowkit.config.normalize import normalize_keys
from src.flowkit.core.errors import ConfigError


def validate_config(config):
    config = normalize_keys(config)
    if not isinstance(config, dict) or "stages" not in config:
        raise ConfigError("config must be a dict with a 'stages' list")
    stages = config["stages"]
    if not isinstance(stages, list) or not stages:
        raise ConfigError("'stages' must be a non-empty list")
    for entry in stages:
        if not isinstance(entry, dict) or not isinstance(entry.get("stage"), str):
            raise ConfigError(f"bad stage entry: {entry!r}")
        params = entry.get("params", {})
        if not isinstance(params, dict):
            raise ConfigError(f"'params' must be a dict: {entry!r}")
    return config

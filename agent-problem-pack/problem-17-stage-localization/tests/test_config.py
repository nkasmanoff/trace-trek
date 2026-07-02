import pytest

from src.flowkit.config.normalize import normalize_keys
from src.flowkit.config.schema import validate_config
from src.flowkit.core.errors import ConfigError


def test_normalize_keys_recurses():
    assert normalize_keys({"a-b": [{"c-d": 1}]}) == {"a_b": [{"c_d": 1}]}


def test_validate_rejects_missing_stages():
    with pytest.raises(ConfigError):
        validate_config({})


def test_validate_accepts_dashed_keys():
    config = validate_config({"stages": [{"stage": "clip", "params": {"hi": 1}}]})
    assert config["stages"][0]["stage"] == "clip"

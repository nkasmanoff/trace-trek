import json
import pytest

from src.config import Config


def test_defaults_used_when_no_file_or_env():
    cfg = Config()
    assert cfg.get("host") == "localhost"
    assert cfg.get("port") == 8080
    assert cfg.get("debug") is False


def test_config_file_overrides_defaults(tmp_path):
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"host": "example.com", "port": 3000}))
    cfg = Config(str(cfg_file))
    assert cfg.get("host") == "example.com"
    assert cfg.get("port") == 3000
    assert cfg.get("debug") is False


def test_env_var_overrides_config_file(monkeypatch, tmp_path):
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"port": 8080, "debug": True}))
    monkeypatch.setenv("APP_PORT", "9090")
    cfg = Config(str(cfg_file))
    assert cfg.get("port") == "9090", "env var should override file value"
    assert cfg.get("debug") is True, "file value should apply when no env var"


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("APP_HOST", "0.0.0.0")
    cfg = Config()
    assert cfg.get("host") == "0.0.0.0"

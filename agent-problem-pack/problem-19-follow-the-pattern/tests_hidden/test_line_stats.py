"""Hidden: grade the new line-stats service against the framework conventions."""

import pytest

import src.services  # noqa: F401  (imports register services)
from src.services.base import get_service, registered_services
from src.services.errors import ServiceInputError


def test_line_stats_is_registered():
    assert "line-stats" in registered_services()


def test_line_stats_counts():
    result = get_service("line-stats").run(
        {"text": "Hello world\nhello there\n\n"}
    )
    assert result == {
        "ok": True,
        "service": "line-stats",
        "result": {"lines": 2, "words": 4, "unique_words": 3},
    }


def test_line_stats_empty_text():
    result = get_service("line-stats").run({"text": ""})
    assert result["result"] == {"lines": 0, "words": 0, "unique_words": 0}


def test_line_stats_unique_is_case_insensitive():
    result = get_service("line-stats").run({"text": "A a A"})
    assert result["result"]["unique_words"] == 1


def test_line_stats_missing_text_raises_service_input_error():
    with pytest.raises(ServiceInputError):
        get_service("line-stats").run({})


def test_line_stats_wrong_type_raises_service_input_error():
    with pytest.raises(ServiceInputError):
        get_service("line-stats").run({"text": 42})


def test_existing_services_still_registered():
    names = registered_services()
    assert "word-count" in names
    assert "dedupe" in names

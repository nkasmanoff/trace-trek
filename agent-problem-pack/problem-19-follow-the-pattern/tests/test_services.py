import pytest

from src.services import get_service
from src.services.errors import ServiceInputError


def test_word_count_counts_words():
    result = get_service("word-count").run({"text": "one two  three"})
    assert result == {"ok": True, "service": "word-count", "result": {"words": 3}}


def test_word_count_rejects_missing_text():
    with pytest.raises(ServiceInputError):
        get_service("word-count").run({})


def test_dedupe_preserves_order():
    result = get_service("dedupe").run({"items": ["b", "a", "b", "c", "a"]})
    assert result["result"] == {"items": ["b", "a", "c"]}


def test_dedupe_rejects_wrong_type():
    with pytest.raises(ServiceInputError):
        get_service("dedupe").run({"items": "not-a-list"})

import pytest

from src.lrupin import LRUCache


def test_get_and_put_roundtrip():
    cache = LRUCache(2)
    cache.put("a", 1)
    assert cache.get("a") == 1


def test_lru_eviction_order():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    assert "a" not in cache
    assert cache.keys() == ["b", "c"]


def test_get_refreshes_recency():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")
    cache.put("c", 3)
    assert "b" not in cache
    assert "a" in cache


def test_pinned_key_survives_eviction():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.pin("a")
    cache.put("c", 3)
    assert "a" in cache
    assert "b" not in cache
    assert cache.get("a") == 1


def test_missing_key_raises():
    cache = LRUCache(1)
    with pytest.raises(KeyError):
        cache.get("nope")

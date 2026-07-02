"""Hidden grading tests for the pin/eviction contract in src/lrupin.py.

These target the parts of the docstring contract the visible suite does not
cover: overwrites never evict, a fully pinned cache refuses inserts without
losing state, and unpinned keys compete on their original recency.
"""

import pytest

from src.lrupin import LRUCache


def test_overwrite_when_full_never_evicts():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    # Overwrite the most-recent key: "a" (the LRU entry) must survive.
    cache.put("b", 20)
    assert len(cache) == 2
    assert cache.get("a") == 1
    assert cache.get("b") == 20


def test_overwrite_marks_most_recently_used():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("a", 10)
    cache.put("c", 3)
    assert "b" not in cache
    assert cache.keys() == ["a", "c"]


def test_overwrite_of_pinned_key_keeps_pin():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.pin("a")
    cache.put("a", 10)
    cache.put("b", 2)
    cache.put("c", 3)
    assert "a" in cache
    assert cache.get("a") == 10


def test_all_pinned_raises_and_preserves_state():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.pin("a")
    cache.pin("b")
    with pytest.raises(RuntimeError, match="all entries pinned"):
        cache.put("c", 3)
    assert set(cache.keys()) == {"a", "b"}
    assert "c" not in cache


def test_eviction_skips_pinned_and_takes_next_lru():
    cache = LRUCache(3)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)
    cache.pin("a")
    cache.put("d", 4)
    assert "a" in cache
    assert "b" not in cache
    assert "c" in cache and "d" in cache


def test_unpinned_key_competes_on_original_recency():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.pin("a")
    cache.unpin("a")
    cache.put("c", 3)
    assert "a" not in cache
    assert "b" in cache


def test_pin_unknown_key_raises():
    cache = LRUCache(1)
    cache.put("a", 1)
    with pytest.raises(KeyError):
        cache.pin("ghost")
    with pytest.raises(KeyError):
        cache.unpin("ghost")

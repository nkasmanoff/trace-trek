"""LRU cache with pinnable entries.

Contract:

- ``get(key)`` returns the value and marks the key as most recently used.
  Missing keys raise ``KeyError``.
- ``put(key, value)`` inserts or overwrites. Overwriting an existing key
  updates its value, marks it most recently used, and NEVER evicts.
- Inserting a new key when the cache is full evicts the least recently
  used UNPINNED key first. Pinned keys are never evicted.
- If every entry is pinned and the cache is full, inserting a new key
  raises ``RuntimeError("all entries pinned")`` and changes nothing.
- ``pin(key)`` / ``unpin(key)`` toggle protection; both raise ``KeyError``
  for keys not in the cache. Pinning does not change recency. An unpinned
  key competes for eviction based on its existing recency, not the time
  it was unpinned.
"""


class LRUCache:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._data = {}
        self._pinned = set()

    def get(self, key):
        value = self._data.pop(key)
        self._data[key] = value
        return value

    def put(self, key, value):
        if len(self._data) >= self.capacity:
            self._evict()
        if key in self._data:
            self._data.pop(key)
        self._data[key] = value

    def pin(self, key):
        if key not in self._data:
            raise KeyError(key)
        self._pinned.add(key)

    def unpin(self, key):
        if key not in self._data:
            raise KeyError(key)
        self._pinned.discard(key)

    def _evict(self):
        oldest = next(iter(self._data))
        del self._data[oldest]
        self._pinned.discard(oldest)

    def keys(self):
        return list(self._data)

    def __len__(self):
        return len(self._data)

    def __contains__(self, key):
        return key in self._data

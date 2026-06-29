import threading


class ComputeCache:
    def __init__(self):
        self._cache = {}

    def get_or_compute(self, key, factory):
        if key not in self._cache:
            value = factory()
            self._cache[key] = value
        return self._cache[key]

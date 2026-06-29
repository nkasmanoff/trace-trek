import threading
import time

from src.cache import ComputeCache


def test_single_threaded():
    cache = ComputeCache()
    assert cache.get_or_compute("x", lambda: 42) == 42


def test_returns_cached_value():
    cache = ComputeCache()
    cache.get_or_compute("k", lambda: "first")
    assert cache.get_or_compute("k", lambda: "second") == "first"


def test_concurrent_computes_do_not_duplicate():
    cache = ComputeCache()
    results = []
    lock = threading.Lock()

    def factory():
        with lock:
            results.append(True)
        time.sleep(0.05)
        return 42

    threads = [
        threading.Thread(target=cache.get_or_compute, args=("k", factory))
        for _ in range(10)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 1, f"factory called {len(results)} times, expected 1"

"""Token-bucket rate limiter.

A bucket holds at most `capacity` tokens and refills at `rate` tokens per
second. Tokens NEVER exceed capacity, no matter how long the bucket sits
idle. Each allowed request spends tokens; a request that cannot be paid for
in full is denied and spends nothing.
"""

import time


class TokenBucket:
    def __init__(self, capacity, rate, clock=time.monotonic):
        self.capacity = float(capacity)
        self.rate = float(rate)
        self._clock = clock
        self._tokens = float(capacity)
        self._last = clock()

    def _refill(self):
        now = self._clock()
        elapsed = now - self._last
        self._last = now
        self._tokens = self._tokens + elapsed * self.rate

    def allow(self):
        self._refill()
        if self._tokens >= 1:
            self._tokens -= 1
            return True
        return False

"""Hidden: grade the follow-up API requested in the later turns."""

import inspect

from src.limiter import TokenBucket


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make_bucket(capacity=5, rate=1.0):
    clock = FakeClock()
    return TokenBucket(capacity=capacity, rate=rate, clock=clock), clock


def test_allow_accepts_cost_parameter():
    bucket, _ = make_bucket(capacity=5)
    assert bucket.allow(cost=3) is True
    assert bucket.allow(cost=3) is False


def test_denied_request_spends_nothing():
    bucket, _ = make_bucket(capacity=5)
    assert bucket.allow(cost=3) is True
    assert bucket.allow(cost=3) is False
    assert bucket.allow(cost=2) is True


def test_default_cost_is_one():
    signature = inspect.signature(TokenBucket.allow)
    assert signature.parameters["cost"].default == 1
    bucket, _ = make_bucket(capacity=2)
    assert bucket.allow() is True
    assert bucket.allow() is True
    assert bucket.allow() is False


def test_remaining_reports_tokens_after_refill():
    bucket, clock = make_bucket(capacity=5, rate=1.0)
    assert bucket.allow(cost=4) is True
    assert bucket.remaining() == 1
    clock.advance(2.0)
    assert bucket.remaining() == 3


def test_remaining_capped_at_capacity():
    bucket, clock = make_bucket(capacity=5, rate=1.0)
    clock.advance(1000.0)
    assert bucket.remaining() == 5

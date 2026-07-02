from src.limiter import TokenBucket


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make_bucket(capacity=2, rate=1.0):
    clock = FakeClock()
    return TokenBucket(capacity=capacity, rate=rate, clock=clock), clock


def test_initial_burst_up_to_capacity():
    bucket, _ = make_bucket(capacity=3)
    assert [bucket.allow() for _ in range(4)] == [True, True, True, False]


def test_refill_grants_one_token_per_second():
    bucket, clock = make_bucket(capacity=1, rate=1.0)
    assert bucket.allow() is True
    assert bucket.allow() is False
    clock.advance(1.0)
    assert bucket.allow() is True


def test_idle_bucket_does_not_exceed_capacity():
    bucket, clock = make_bucket(capacity=2, rate=1.0)
    assert bucket.allow() is True
    assert bucket.allow() is True
    clock.advance(100.0)
    allowed = sum(1 for _ in range(5) if bucket.allow())
    assert allowed == 2, "an idle bucket must not accumulate beyond capacity"

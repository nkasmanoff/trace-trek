"""Hidden: grade the retry/backoff behavior against the config contract."""

import time

import pytest

from src.apiclient import config
from src.apiclient.client import ApiClient
from src.apiclient.errors import ApiError, TransientError


class FlakyTransport:
    def __init__(self, failures, error=TransientError):
        self.failures = failures
        self.error = error
        self.calls = 0
        self.call_times = []

    def send(self, request):
        self.calls += 1
        self.call_times.append(time.monotonic())
        if self.calls <= self.failures:
            raise self.error("boom")
        return {"status": 200, "payload": request.get("payload")}


def test_recovers_from_transient_failures():
    transport = FlakyTransport(failures=config.MAX_ATTEMPTS - 1)
    client = ApiClient(transport)
    response = client.send({"payload": "ping"})
    assert response["status"] == 200
    assert transport.calls == config.MAX_ATTEMPTS


def test_gives_up_after_max_attempts():
    transport = FlakyTransport(failures=10)
    client = ApiClient(transport)
    with pytest.raises(TransientError):
        client.send({"payload": "ping"})
    assert transport.calls == config.MAX_ATTEMPTS


def test_non_transient_error_not_retried():
    transport = FlakyTransport(failures=10, error=ApiError)
    client = ApiClient(transport)
    with pytest.raises(ApiError):
        client.send({"payload": "ping"})
    assert transport.calls == 1


def test_backoff_delays_double():
    transport = FlakyTransport(failures=2)
    client = ApiClient(transport)
    client.send({"payload": "ping"})
    gaps = [
        b - a
        for a, b in zip(transport.call_times, transport.call_times[1:])
    ]
    assert len(gaps) == 2
    assert gaps[0] >= config.BACKOFF_BASE * 0.9
    assert gaps[1] >= config.BACKOFF_BASE * 2 * 0.9


def test_success_path_has_no_delay():
    transport = FlakyTransport(failures=0)
    client = ApiClient(transport)
    start = time.monotonic()
    client.send({"payload": "ping"})
    assert time.monotonic() - start < config.BACKOFF_BASE

class ApiError(Exception):
    """Base error for API failures. Not retryable."""


class TransientError(ApiError):
    """A retryable failure (timeouts, 5xx, connection resets).

    Transient failures are the ONLY errors that may be retried.
    """

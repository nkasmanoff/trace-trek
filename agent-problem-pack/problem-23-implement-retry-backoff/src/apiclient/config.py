"""Client tuning constants.

MAX_ATTEMPTS is the TOTAL number of attempts for a request whose failures
are transient (first try included). BACKOFF_BASE is the delay in seconds
before the first retry; the delay DOUBLES after every subsequent retry
(exponential backoff): BACKOFF_BASE, 2 * BACKOFF_BASE, 4 * BACKOFF_BASE, ...
"""

MAX_ATTEMPTS = 3
BACKOFF_BASE = 0.05

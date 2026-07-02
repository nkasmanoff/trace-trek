"""Transports turn a request dict into a response dict.

A transport exposes send(request) and raises TransientError for retryable
failures and ApiError for permanent ones.
"""


class EchoTransport:
    """Test/local transport: echoes the request payload back."""

    def send(self, request):
        return {"status": 200, "payload": request.get("payload")}

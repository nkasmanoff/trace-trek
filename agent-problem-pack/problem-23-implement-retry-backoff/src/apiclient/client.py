"""ApiClient: the public entry point for issuing requests."""


class ApiClient:
    def __init__(self, transport):
        self._transport = transport

    def send(self, request):
        return self._transport.send(request)

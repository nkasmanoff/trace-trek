from src.apiclient.client import ApiClient
from src.apiclient.transport import EchoTransport


def test_send_returns_transport_response():
    client = ApiClient(EchoTransport())
    response = client.send({"payload": "ping"})
    assert response == {"status": 200, "payload": "ping"}

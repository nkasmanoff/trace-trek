"""Event handlers for the ingestion service.

Every handler follows the same contract:

- The event body lives under the "data" key of the incoming event dict.
- A handler accepts an event when its value is at or below the kind's limit
  (events exactly AT the limit are accepted), and rejects it otherwise.
- Handlers that manage liveness (heartbeat) decrement the event's ttl by 1
  when forwarding. All other handlers forward ttl UNCHANGED.
- Accepted values are appended to state[kind]; rejected events are counted
  in state["rejected"].
"""

LIMITS = {
    "metric": 100,
    "log": 100,
    "trace": 250,
    "alert": 10,
    "audit": 50,
    "heartbeat": 1,
}

DEFAULT_TTL = 8


def handle_metric(event, state):
    payload = event.get("data", {})
    value = payload.get("value", 0)
    limit = LIMITS["metric"]
    accepted = value <= limit
    if accepted:
        state.setdefault("metric", []).append(value)
    else:
        state["rejected"] = state.get("rejected", 0) + 1
    ttl = event.get("ttl", DEFAULT_TTL)
    return {"kind": "metric", "accepted": accepted, "value": value, "ttl": ttl}


def handle_log(event, state):
    payload = event.get("payload", {})
    value = payload.get("value", 0)
    limit = LIMITS["log"]
    accepted = value <= limit
    if accepted:
        state.setdefault("log", []).append(value)
    else:
        state["rejected"] = state.get("rejected", 0) + 1
    ttl = event.get("ttl", DEFAULT_TTL)
    return {"kind": "log", "accepted": accepted, "value": value, "ttl": ttl}


def handle_trace(event, state):
    payload = event.get("data", {})
    value = payload.get("value", 0)
    limit = LIMITS["trace"]
    accepted = value < limit
    if accepted:
        state.setdefault("trace", []).append(value)
    else:
        state["rejected"] = state.get("rejected", 0) + 1
    ttl = event.get("ttl", DEFAULT_TTL)
    return {"kind": "trace", "accepted": accepted, "value": value, "ttl": ttl}


def handle_alert(event, state):
    payload = event.get("data", {})
    value = payload.get("value", 0)
    limit = LIMITS["alert"]
    accepted = value <= limit
    if accepted:
        state.setdefault("alert", []).append(value)
    else:
        state["rejected"] = state.get("rejected", 0) + 1
    ttl = event.get("ttl", DEFAULT_TTL) - 1
    return {"kind": "alert", "accepted": accepted, "value": value, "ttl": ttl}


def handle_audit(event, state):
    payload = event.get("data", {})
    value = payload.get("value", 0)
    limit = LIMITS["audit"]
    accepted = value <= limit
    if accepted:
        state.setdefault("audit", []).append(value)
    else:
        state["rejected"] = state.get("rejected", 0) + 1
    ttl = event.get("ttl", DEFAULT_TTL)
    return {"kind": "audit", "accepted": accepted, "value": value, "ttl": ttl}


def handle_heartbeat(event, state):
    payload = event.get("data", {})
    value = payload.get("value", 0)
    limit = LIMITS["heartbeat"]
    accepted = value <= limit
    if accepted:
        state.setdefault("heartbeat", []).append(value)
    else:
        state["rejected"] = state.get("rejected", 0) + 1
    ttl = event.get("ttl", DEFAULT_TTL) - 1
    return {"kind": "heartbeat", "accepted": accepted, "value": value, "ttl": ttl}


HANDLERS = {
    "metric": handle_metric,
    "log": handle_log,
    "trace": handle_trace,
    "alert": handle_alert,
    "audit": handle_audit,
    "heartbeat": handle_heartbeat,
}


def dispatch(event, state):
    handler = HANDLERS.get(event.get("kind"))
    if handler is None:
        raise ValueError(f"unknown event kind: {event.get('kind')!r}")
    return handler(event, state)

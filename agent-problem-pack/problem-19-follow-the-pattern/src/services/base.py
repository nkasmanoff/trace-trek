"""Service framework.

A service subclasses BaseService and declares two class attributes:

- name:   the kebab-case public identifier, unique across the registry
- schema: a dict mapping REQUIRED payload keys to their expected types

Subclasses are registered automatically via __init_subclass__. Modules in
this package are imported by src/services/__init__.py, which is what makes
registration actually happen for a new service module.

run(payload) validates the payload against schema — a missing key or a value
of the wrong type raises ServiceInputError — then delegates to
_process(payload) and wraps its return value in the response envelope:

    {"ok": True, "service": <name>, "result": <_process return value>}

Subclasses implement _process(payload) only; they never build the envelope
or duplicate validation.
"""

from src.services.errors import ServiceError, ServiceInputError

_REGISTRY = {}


class BaseService:
    name = None
    schema = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not isinstance(cls.name, str) or not cls.name:
            raise ServiceError(f"{cls.__name__} must declare a name")
        if cls.name in _REGISTRY:
            raise ServiceError(f"duplicate service name: {cls.name}")
        _REGISTRY[cls.name] = cls

    def validate(self, payload):
        if not isinstance(payload, dict):
            raise ServiceInputError("payload must be a dict")
        for key, expected_type in self.schema.items():
            if key not in payload:
                raise ServiceInputError(f"missing required key: {key}")
            if not isinstance(payload[key], expected_type):
                raise ServiceInputError(
                    f"{key} must be {expected_type.__name__}, "
                    f"got {type(payload[key]).__name__}"
                )
        return payload

    def run(self, payload):
        self.validate(payload)
        return {"ok": True, "service": self.name, "result": self._process(payload)}

    def _process(self, payload):
        raise NotImplementedError


def get_service(name):
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise ServiceError(f"unknown service: {name}") from None


def registered_services():
    return sorted(_REGISTRY)

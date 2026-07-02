"""Stage registry.

Transforms register themselves under a short stage name at import time (see
src/flowkit/transforms/__init__.py). Pipelines look stages up by name.
"""

from src.flowkit.core.errors import UnknownStageError

_STAGES = {}


def register(name):
    def wrap(fn):
        _STAGES[name] = fn
        return fn
    return wrap


def get_stage(name):
    try:
        return _STAGES[name]
    except KeyError:
        raise UnknownStageError(f"unknown stage: {name}") from None


def registered_stages():
    return sorted(_STAGES)

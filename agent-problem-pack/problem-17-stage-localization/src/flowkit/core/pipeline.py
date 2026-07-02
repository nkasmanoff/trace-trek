"""Run a configured sequence of stages over a list of numbers."""

import src.flowkit.transforms  # noqa: F401  (import registers the stages)
from src.flowkit.config.schema import validate_config
from src.flowkit.core.registry import get_stage


class Pipeline:
    def __init__(self, config):
        self.config = validate_config(config)

    def run(self, values):
        data = list(values)
        for entry in self.config["stages"]:
            stage = get_stage(entry["stage"])
            data = stage(data, **entry.get("params", {}))
        return data

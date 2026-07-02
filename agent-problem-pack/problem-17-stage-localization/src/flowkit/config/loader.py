"""Load pipeline configs from JSON files."""

import json
from pathlib import Path

from src.flowkit.config.schema import validate_config


def load_config(path):
    text = Path(path).read_text(encoding="utf-8")
    return validate_config(json.loads(text))

import json
import os


def load_file(path):
    with open(path) as f:
        return json.load(f)


def load_env(prefix="APP_"):
    return {k[len(prefix):].lower(): v for k, v in os.environ.items() if k.startswith(prefix)}

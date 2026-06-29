from src import loader


class Config:
    DEFAULTS = {"host": "localhost", "port": 8080, "debug": False}

    def __init__(self, config_path=None):
        self._values = dict(self.DEFAULTS)
        self._values.update(loader.load_env())
        if config_path:
            self._values.update(loader.load_file(config_path))

    def get(self, key, default=None):
        return self._values.get(key, default)

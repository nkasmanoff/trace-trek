from src.services.base import BaseService


class WordCountService(BaseService):
    name = "word-count"
    schema = {"text": str}

    def _process(self, payload):
        words = payload["text"].split()
        return {"words": len(words)}

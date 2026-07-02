from src.services.base import BaseService


class DedupeService(BaseService):
    name = "dedupe"
    schema = {"items": list}

    def _process(self, payload):
        seen = set()
        kept = []
        for item in payload["items"]:
            if item not in seen:
                seen.add(item)
                kept.append(item)
        return {"items": kept}

"""File loaders.

Loaders are registered by file extension in EXT_LOADERS. A loader takes a
Path and returns a list of records (dicts). load_any() picks the loader for
the path's suffix and raises UnsupportedFormatError for unknown extensions.

Malformed content raises RecordParseError; for line-oriented formats the
error message includes the file path and the 1-based line number of the bad
line. Blank lines in line-oriented formats are skipped, not errors.
"""

import json
from pathlib import Path

from src.datakit.errors import RecordParseError, UnsupportedFormatError


def load_json_file(path):
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RecordParseError(f"{path}: {exc}") from None
    if isinstance(data, list):
        return data
    return [data]


EXT_LOADERS = {
    ".json": load_json_file,
}


def get_loader(path):
    suffix = Path(path).suffix.lower()
    try:
        return EXT_LOADERS[suffix]
    except KeyError:
        raise UnsupportedFormatError(
            f"no loader registered for {suffix!r} ({path})"
        ) from None


def load_any(path):
    return get_loader(path)(Path(path))

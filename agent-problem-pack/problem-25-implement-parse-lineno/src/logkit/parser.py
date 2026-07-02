"""Parser for the simple ``KEY=VALUE`` config format.

Rules:

- One ``KEY=VALUE`` assignment per line. Keys are non-empty and contain no
  whitespace; values may be empty and keep inner whitespace.
- Blank lines and lines starting with ``#`` (after stripping leading
  whitespace) are skipped.
- When a key repeats, the LAST assignment wins.
- Invalid lines (no ``=``, or an empty/whitespace-containing key) raise
  ``ParseError`` — see ``errors.py`` for the exact failure contract.
"""

from src.logkit.errors import ParseError


def parse(text):
    result = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ParseError(f"missing '=' in {line!r}")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or any(ch.isspace() for ch in key):
            raise ParseError(f"bad key in {line!r}")
        result[key] = value.strip()
    return result

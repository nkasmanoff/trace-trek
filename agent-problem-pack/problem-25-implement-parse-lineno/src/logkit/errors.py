"""Errors raised by logkit.

Contract for parse failures (the parser must follow this exactly):

- ``ParseError(reason, lineno)`` — ``lineno`` is the 1-BASED line number of
  the offending line in the original input, counting every line including
  blank lines and comments.
- ``str(error)`` must be exactly ``"line {lineno}: {reason}"``.
- The error object must expose the line number as an ``int`` attribute
  named ``lineno`` and the bare reason as an attribute named ``reason``.
- Parsing stops at the FIRST invalid line; later invalid lines are not
  reported.
"""


class ParseError(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason

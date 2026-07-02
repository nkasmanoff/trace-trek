class DataKitError(Exception):
    """Base error for datakit."""


class UnsupportedFormatError(DataKitError):
    """Raised when no loader is registered for a file's extension."""


class RecordParseError(DataKitError):
    """Raised when a record in a data file cannot be parsed.

    The message must include the file path, and the 1-based line number when
    the format is line-oriented.
    """

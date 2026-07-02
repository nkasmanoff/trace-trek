class FlowError(Exception):
    """Base error for flowkit."""


class ConfigError(FlowError):
    """Raised when a pipeline config is malformed."""


class UnknownStageError(FlowError):
    """Raised when a config references a stage that is not registered."""

class ServiceError(Exception):
    """Base error for the service framework."""


class ServiceInputError(ServiceError):
    """Raised when a payload fails schema validation."""

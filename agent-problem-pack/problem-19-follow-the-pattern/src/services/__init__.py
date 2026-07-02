"""Service package.

Importing this package imports every service module, which registers each
service with the registry in base.py. New services must be imported here or
they will never be registered.
"""

from src.services import dedupe_service, word_count_service  # noqa: F401
from src.services.base import get_service, registered_services  # noqa: F401

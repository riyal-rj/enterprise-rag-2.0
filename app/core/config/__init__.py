"""Public config API.

Prefer dependency injection with :func:`get_settings` in new code (it is
mockable in tests via ``get_settings.cache_clear()`` plus env patching, or
via framework-level DI overrides). The module-level ``settings`` instance is
kept for backward compatibility with call sites that used to do
``from config import settings``.
"""

from app.core.config.environment import Environment
from app.core.config.settings import Settings, get_settings

settings = get_settings()

__all__ = ["Environment", "Settings", "get_settings", "settings"]

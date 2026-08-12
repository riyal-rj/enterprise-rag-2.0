"""Public config API.

Use dependency injection with :func:`get_settings` (mockable in tests via
``get_settings.cache_clear()`` plus env patching, or via framework-level DI
overrides). There is deliberately no module-level ``settings`` instance:
constructing ``Settings`` eagerly at import time means importing *anything*
under ``app.core.config`` — even code that only needs one unrelated field —
fails hard the moment any setting is misconfigured, which previously broke
test collection for modules with no actual dependency on the broken field.
Every current call site already uses ``get_settings()``, so nothing needs
the eager instance.
"""

from app.core.config.environment import Environment
from app.core.config.settings import Settings, get_settings

__all__ = ["Environment", "Settings", "get_settings"]

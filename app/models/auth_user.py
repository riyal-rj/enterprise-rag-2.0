"""Identity decoded from a verified access token.

Distinct from :class:`app.models.user.User` (a persisted DB row): this is
the lightweight principal attached to a request after JWT verification, and
carries no password hash.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuthenticatedUser:
    """The caller identity derived from a verified bearer token."""

    username: str
    is_admin: bool

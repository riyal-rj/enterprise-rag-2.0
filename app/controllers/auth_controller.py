"""Auth controller: schema in, schema out.

Domain exceptions raised by :class:`AuthService` propagate unchanged; they
are translated to HTTP responses centrally by
:func:`app.core.exception_handlers.register_exception_handlers`, not here.
"""

from __future__ import annotations

from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse
from app.services.auth_service import AuthService


class AuthController:
    """Orchestrates :class:`AuthService` calls for the ``/auth`` routes."""

    def __init__(self, 
                 auth_service: AuthService) -> None:
        self._auth_service = auth_service

    def register(self, 
                 client_ip: str, 
                 payload: RegisterRequest) -> TokenResponse:
        token = self._auth_service.register(client_ip, 
                                            payload.username, 
                                            payload.password)
        return TokenResponse(token=token)

    def login(self, 
              client_ip: str, 
              payload: LoginRequest) -> TokenResponse:
        token = self._auth_service.login(client_ip, 
                                         payload.username, 
                                         payload.password)
        return TokenResponse(token=token)

"""``/auth`` HTTP routes.

Thin by design: parse the request, delegate to :class:`AuthController`,
return its response. No business logic and no exception handling lives
here — see :mod:`app.services.auth_service` and
:mod:`app.core.exception_handlers` respectively.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request, status

from app.api.deps import get_auth_controller
from app.controllers.auth_controller import AuthController
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse

router = APIRouter(prefix="/auth", tags=["auth"])


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(request: Request,
             payload: RegisterRequest,
             controller: AuthController = Depends(get_auth_controller),) -> TokenResponse:
    
    return controller.register(_client_ip(request), 
                               payload)


@router.post("/login", response_model=TokenResponse)
def login(request: Request,
          payload: LoginRequest,
          controller: AuthController = Depends(get_auth_controller),
          ) -> TokenResponse:
    
    return controller.login(_client_ip(request), 
                            payload)

"""Registration and login business logic.

Depends only on the ``UserRepository``/``PasswordHasher``/``TokenIssuer``/
``RateLimiter`` protocols (constructor injection), so it can be unit tested
with fakes — no FastAPI app or live Postgres/Redis required.
"""

from __future__ import annotations

import logging

from app.core.config.security import RateLimitSettings
from app.core.exceptions import InvalidCredentialsError, RateLimitExceededError
from app.core.security.passwords import PasswordHasher
from app.core.security.rate_limiter import RateLimiter
from app.core.security.tokens import TokenIssuer
from app.repositories.user_repository import UserRepository

logger = logging.getLogger(__name__)

_REGISTER_ROUTE = "/auth/register"
_LOGIN_ROUTE = "/auth/login"
_ONE_HOUR_SECONDS = 3_600
_ONE_MINUTE_SECONDS = 60


class AuthService:
    """Registration and login use cases."""

    def __init__(
        self,
        user_repository: UserRepository,
        password_hasher: PasswordHasher,
        token_issuer: TokenIssuer,
        rate_limiter: RateLimiter,
        rate_limits: RateLimitSettings,
    ) -> None:
        self._user_repository = user_repository
        self._password_hasher = password_hasher
        self._token_issuer = token_issuer
        self._rate_limiter = rate_limiter
        self._rate_limits = rate_limits

    def register(self, client_ip: str, username: str, password: str) -> str:
        """Create a new user and return an access token.

        Raises:
            RateLimitExceededError: if ``client_ip`` exceeded the register
                rate limit.
            UserAlreadyExistsError: if ``username`` is already taken.
        """
        self._enforce_rate_limit(
            client_ip,
            _REGISTER_ROUTE,
            self._rate_limits.auth_register_rate_limit_per_hour,
            _ONE_HOUR_SECONDS,
        )
        password_hash = self._password_hasher.hash(password)
        user = self._user_repository.create(username, password_hash)
        logger.info("auth.register.success", extra={"username": username})
        return self._token_issuer.issue(user.username, is_admin=user.is_admin)

    def login(self, client_ip: str, username: str, password: str) -> str:
        """Validate credentials and return an access token.

        Raises:
            RateLimitExceededError: if ``client_ip`` exceeded the login rate
                limit.
            InvalidCredentialsError: if the username/password don't match.
        """
        self._enforce_rate_limit(
            client_ip,
            _LOGIN_ROUTE,
            self._rate_limits.auth_login_rate_limit_per_min,
            _ONE_MINUTE_SECONDS,
        )
        user = self._user_repository.get_by_username(username)
        if user is None or not self._password_hasher.verify(password, user.password_hash):
            logger.warning("auth.login.invalid_credentials", extra={"username": username})
            raise InvalidCredentialsError()

        logger.info("auth.login.success", extra={"username": username})
        return self._token_issuer.issue(user.username, is_admin=user.is_admin)

    def _enforce_rate_limit(self, 
                            client_ip: str, 
                            route: str, 
                            limit: int, 
                            window_seconds: int) -> None:
        
        allowed, _, _ = self._rate_limiter.is_allowed(client_ip,
                                                       route, 
                                                       limit, 
                                                       window_seconds)
        if not allowed:
            logger.warning(
                "auth.rate_limited", 
                extra={
                    "client_ip": client_ip, 
                    "route": route
                    }
            )
            raise RateLimitExceededError(route)

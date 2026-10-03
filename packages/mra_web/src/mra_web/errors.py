"""
Uniform error envelope for the Market Regime Analysis API.

Every error response, whatever raised it, has the :class:`ErrorResponse` shape::

    {"error_code": "...", "message": "...", "details": {...}, "timestamp": "<ISO 8601>"}
"""

import logging
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from slowapi.errors import RateLimitExceeded
from starlette.exceptions import HTTPException as StarletteHTTPException

from mra_web.models import ErrorResponse
from mra_web.utils import to_jsonable

logger = logging.getLogger(__name__)

# Client errors are reported by HTTP status; a few have stable names.
_ERROR_CODES = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    422: "VALIDATION_ERROR",
    429: "RATE_LIMITED",
    500: "INTERNAL_SERVER_ERROR",
    503: "SERVICE_UNAVAILABLE",
    504: "TIMEOUT",
}


def error_code_for(status_code: int) -> str:
    """Stable error code for an HTTP status."""
    return _ERROR_CODES.get(status_code, f"HTTP_{status_code}")


def error_response(
    status_code: int,
    message: str,
    error_code: str | None = None,
    details: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Build a JSON error response in the standard envelope."""
    body = ErrorResponse(
        error_code=error_code or error_code_for(status_code),
        message=message,
        details=to_jsonable(details or {}),
    )
    return JSONResponse(
        status_code=status_code, content=body.model_dump(mode="json"), headers=headers
    )


def _envelope_from_detail(status_code: int, detail: Any) -> ErrorResponse:
    """Normalize an ``HTTPException.detail`` (str, dict, or ErrorResponse dump)."""
    if isinstance(detail, dict):
        try:
            return ErrorResponse.model_validate(detail)
        except ValidationError:
            message = str(detail.get("message") or HTTPStatus(status_code).phrase)
            return ErrorResponse(error_code=error_code_for(status_code), message=message)
    if detail is None or detail == "":
        detail = HTTPStatus(status_code).phrase
    return ErrorResponse(error_code=error_code_for(status_code), message=str(detail))


async def http_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """FastAPI and Starlette HTTP exceptions (including 404/405 from routing)."""
    assert isinstance(exc, StarletteHTTPException)
    body = _envelope_from_detail(exc.status_code, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=body.model_dump(mode="json"),
        headers=getattr(exc, "headers", None),
    )


async def validation_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Request validation errors: field locations and messages, never input values."""
    assert isinstance(exc, RequestValidationError)
    errors = [
        {
            "loc": [str(part) for part in err.get("loc", ())],
            "msg": str(err.get("msg", "")),
            "type": str(err.get("type", "")),
        }
        for err in exc.errors()
    ]
    return error_response(
        status.HTTP_422_UNPROCESSABLE_CONTENT,
        "Request validation failed",
        details={"errors": errors},
    )


async def rate_limit_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """slowapi ``RateLimitExceeded`` (for routes using ``@limiter.limit``)."""
    return error_response(
        status.HTTP_429_TOO_MANY_REQUESTS,
        "Rate limit exceeded",
        details={"limit": str(getattr(exc, "detail", ""))},
        headers={"Retry-After": "60"},
    )


async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Anything else: log server-side, return a generic 500."""
    logger.error("Unhandled exception", exc_info=exc)
    return error_response(status.HTTP_500_INTERNAL_SERVER_ERROR, "An unexpected error occurred")


def install_error_handlers(app: FastAPI) -> None:
    """Register the uniform error envelope on an app."""
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(RateLimitExceeded, rate_limit_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

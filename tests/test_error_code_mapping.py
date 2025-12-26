from __future__ import annotations

import json

from fastapi import HTTPException
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request

from app.api.error_handlers import (
    _resolve_error_code,
    http_exception_handler,
    validation_exception_handler,
)
from app.api.schemas import ErrorCode


def _dummy_request() -> Request:
    scope = {"type": "http", "method": "GET", "path": "/", "headers": []}
    return Request(scope)


def _load_body(response) -> dict:
    return json.loads(response.body.decode("utf-8"))


def test_resolve_error_code() -> None:
    assert _resolve_error_code(401, "authorization header missing") == ErrorCode.AUTH_MISSING_TOKEN
    assert _resolve_error_code(401, "invalid token header") == ErrorCode.AUTH_INVALID_TOKEN
    assert _resolve_error_code(429, "daily usage limit exceeded") == ErrorCode.USAGE_LIMIT_EXCEEDED
    assert _resolve_error_code(503, "timeout") == ErrorCode.UPSTREAM_TIMEOUT
    assert _resolve_error_code(504, "timed out") == ErrorCode.UPSTREAM_TIMEOUT
    assert _resolve_error_code(503, "rate limit backend unavailable") == ErrorCode.UPSTREAM_ERROR
    assert _resolve_error_code(400, "질문을 입력해주세요.") == ErrorCode.VALIDATION_ERROR


def test_http_exception_handler() -> None:
    exc = HTTPException(status_code=401, detail="authorization header missing")
    response = http_exception_handler(_dummy_request(), exc)
    body = _load_body(response)
    assert body["status"] == "error"
    assert body["error_code"] == ErrorCode.AUTH_MISSING_TOKEN.value


def test_validation_exception_handler() -> None:
    exc = RequestValidationError([])
    response = validation_exception_handler(_dummy_request(), exc)
    body = _load_body(response)
    assert body["status"] == "error"
    assert body["error_code"] == ErrorCode.VALIDATION_ERROR.value

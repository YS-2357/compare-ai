"""표준 에러 응답 핸들러."""

from __future__ import annotations

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.schemas import ErrorCode, ErrorResponse
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _resolve_error_code(status_code: int, detail: str) -> ErrorCode:
    """HTTP 상태와 메시지를 기반으로 에러 코드를 결정한다."""

    normalized = detail.lower()
    if status_code == 401:
        if "missing" in normalized:
            return ErrorCode.AUTH_MISSING_TOKEN
        return ErrorCode.AUTH_INVALID_TOKEN
    if status_code == 429:
        return ErrorCode.USAGE_LIMIT_EXCEEDED
    if status_code in (502, 503, 504):
        if "timeout" in normalized or "timed out" in normalized:
            return ErrorCode.UPSTREAM_TIMEOUT
        return ErrorCode.UPSTREAM_ERROR
    if status_code == 400:
        return ErrorCode.VALIDATION_ERROR
    return ErrorCode.UNKNOWN_ERROR


def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    """HTTPException을 표준 응답으로 변환한다."""

    detail = str(exc.detail)
    error_code = _resolve_error_code(exc.status_code, detail)
    payload = ErrorResponse(
        status="error",
        error_code=error_code,
        detail=detail,
    )
    logger.warning("http_exception_handler:응답 status=%s error_code=%s detail=%s", exc.status_code, error_code, detail)
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """요청 검증 오류를 표준 응답으로 변환한다."""

    detail = "요청 검증 실패"
    payload = ErrorResponse(
        status="error",
        error_code=ErrorCode.VALIDATION_ERROR,
        detail=detail,
    )
    logger.warning("validation_exception_handler:응답 errors=%s", exc.errors())
    return JSONResponse(status_code=400, content=payload.model_dump())


def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    """미처리 예외를 표준 응답으로 변환한다."""

    detail = "알 수 없는 오류"
    payload = ErrorResponse(
        status="error",
        error_code=ErrorCode.UNKNOWN_ERROR,
        detail=detail,
    )
    logger.error("unhandled_exception_handler:응답 err=%s", exc)
    return JSONResponse(status_code=500, content=payload.model_dump())

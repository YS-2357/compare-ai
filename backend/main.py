"""FastAPI 애플리케이션 팩토리."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .api import router as api_router
from .api.error_handlers import (
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from .utils.config import Settings, get_settings


FASTAPI_DESCRIPTION = (
    "LangGraph 기반 Compare-AI 백엔드 API입니다. 처음 보는 분도 이해하기 쉽게 기본 개념과 경로를 설명합니다.\n\n"
    "핵심 개념:\n"
    "- **NDJSON 스트림**: `/api/ask`는 한 줄씩 JSON이 오는 방식입니다. `event`가 `partial`이면 진행 중, `summary`이면 최종 요약입니다.\n"
    "- **모델 오버라이드**: 요청 본문 `models` 필드로 공급자별 기본 모델을 덮어쓸 수 있습니다.\n"
    "- **공급자 ON/OFF**: `active_providers`로 실제 호출할 벤더를 제한할 수 있습니다.\n"
    "- **로컬 우선 실행**: 기본 경로에서는 인증/사용량 제한 없이 로컬 테스트를 우선합니다.\n\n"
    "주요 엔드포인트:\n"
    "- `/health` (GET): 서비스가 살아있는지 단순 확인.\n"
    "- `/api/ask` (POST): LangGraph 워크플로우 스트리밍. 질문/히스토리/모델 오버라이드를 보내면 모델별 답변이 순서대로 흘러옵니다.\n"
    "Swagger UI(`/docs`)와 ReDoc(`/redoc`)에서 요청/응답 예시, 스키마, 오류 포맷을 확인하세요."
)

TAGS_METADATA = [
    {"name": "system", "description": "헬스 체크 및 공통 시스템 엔드포인트"},
    {"name": "questions", "description": "LangGraph 질의 처리 스트리밍 API"},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI startup/shutdown에서 공용 리소스를 정리한다."""

    try:
        yield
    finally:
        return


def create_app(settings: Settings | None = None) -> FastAPI:
    """FastAPI 애플리케이션을 구성해 반환한다.

    Args:
        settings: 외부에서 주입할 `Settings` 인스턴스. 생략 시 `.env`/환경변수를 읽어 생성한다.

    Returns:
        FastAPI: 라우터와 미들웨어가 등록된 FastAPI 인스턴스.
    """

    settings = settings or get_settings()
    app = FastAPI(
        title=(settings.fastapi_title or "").strip('"'),
        version=settings.fastapi_version,
        description=FASTAPI_DESCRIPTION,
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
    app.state.settings = settings
    return app


app = create_app()

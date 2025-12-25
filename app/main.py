"""FastAPI 애플리케이션 팩토리."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import router as api_router
from .auth.supabase import shutdown_auth_client
from .rate_limit.upstash import shutdown_rate_limiter
from .utils.config import Settings, get_settings


FASTAPI_DESCRIPTION = (
    "LangGraph 기반 Compare-AI 백엔드 API입니다. 처음 보는 분도 이해하기 쉽게 기본 개념과 경로를 설명합니다.\n\n"
    "핵심 개념:\n"
    "- **토큰 방식 인증**: `Authorization: Bearer <JWT>` 헤더를 넣어야 합니다. 로그인/회원가입으로 토큰을 먼저 받습니다.\n"
    "- **NDJSON 스트림**: `/api/ask`는 한 줄씩 JSON이 오는 방식입니다. `type`이 `partial`이면 진행 중, `summary`이면 최종 요약입니다.\n"
    "- **모델 오버라이드**: 요청 본문 `models` 필드로 공급자별 기본 모델을 덮어쓸 수 있습니다.\n"
    "- **공급자 ON/OFF**: `active_providers`로 실제 호출할 벤더를 제한할 수 있습니다.\n"
    "- **사용량 헤더**: 응답 헤더 `X-Usage-Limit`, `X-Usage-Remaining`에 남은 호출 횟수가 담깁니다.\n\n"
    "주요 엔드포인트:\n"
    "- `/health` (GET): 서비스가 살아있는지 단순 확인.\n"
    "- `/auth/register`, `/auth/login` (POST): 이메일/비밀번호 기반 회원관리(Supabase).\n"
    "- `/api/ask` (POST): LangGraph 워크플로우 스트리밍. 질문/히스토리/모델 오버라이드를 보내면 모델별 답변이 순서대로 흘러옵니다.\n"
    "- `/api/prompt-eval` (POST): 공통 프롬프트로 여러 모델을 호출하고, 벤더별 최신 모델이 교차 평가한 점수/근거를 JSON 스트림으로 반환합니다.\n"
    "- `/usage` (GET): 오늘 남은 호출 횟수 조회(관리자는 `null`).\n\n"
    "제약:\n"
    "- Cohere `command-a-reasoning-08-2025`는 텍스트 파이프라인과 호환되지 않아 채팅 UI 목록에서 제외됩니다.\n\n"
    "Swagger UI(`/docs`)와 ReDoc(`/redoc`)에서 요청/응답 예시, 스키마, 오류 포맷을 확인하세요."
)

TAGS_METADATA = [
    {"name": "system", "description": "헬스 체크 및 공통 시스템 엔드포인트"},
    {"name": "auth", "description": "Supabase 기반 회원가입/로그인 API"},
    {"name": "questions", "description": "LangGraph 질의 처리 스트리밍 API"},
    {"name": "usage", "description": "일일 사용량 및 제한 정보"},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI startup/shutdown에서 공용 리소스를 정리한다."""

    try:
        yield
    finally:
        await shutdown_auth_client()
        await shutdown_rate_limiter()


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
    app.state.settings = settings
    return app


app = create_app()

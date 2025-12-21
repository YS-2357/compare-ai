"""회원가입/로그인 엔드포인트."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.auth import get_auth_client
from app.api.schemas import LoginRequest, LoginResponse, RegisterRequest, RegisterResponse
from app.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/register", response_model=RegisterResponse)
async def register(payload: RegisterRequest) -> RegisterResponse:
    """이메일+비밀번호 회원가입을 수행한다.

    - 입력: `email`, `password`.
    - 처리: Supabase Auth REST `/signup` 호출.
    - 결과: 사용자 ID, 이메일, 인증 메일 발송 시각(`confirmation_sent_at`)을 반환한다.
    """

    logger.debug("auth/register:시작 email=%s", payload.email)
    client = get_auth_client()
    try:
        result = await client.signup(payload.email, payload.password)
    except HTTPException as exc:
        logger.warning("회원가입 실패: %s", exc.detail)
        raise

    user = result.get("user") or {}
    logger.info("auth/register:성공 user_id=%s email=%s", user.get("id"), user.get("email"))
    return RegisterResponse(
        id=user.get("id"),
        email=user.get("email"),
        confirmation_sent_at=result.get("confirmation_sent_at"),
    )


@router.post("/login", response_model=LoginResponse)
async def login(payload: LoginRequest) -> LoginResponse:
    """이메일+비밀번호 로그인 후 Supabase 토큰을 반환한다.

    - 입력: `email`, `password`.
    - 처리: Supabase Auth REST `token?grant_type=password` 호출.
    - 결과: `access_token`, `token_type`(bearer), `expires_in`, `refresh_token`, `user` 정보를 반환한다.
    - 클라이언트는 `Authorization: Bearer <access_token>` 헤더로 API를 호출한다.
    """

    logger.debug("auth/login:시작 email=%s", payload.email)
    client = get_auth_client()
    try:
        result = await client.signin(payload.email, payload.password)
    except HTTPException as exc:
        logger.warning("로그인 실패: %s", exc.detail)
        raise

    resp = LoginResponse(
        access_token=result.get("access_token"),
        token_type=result.get("token_type", "bearer"),
        expires_in=result.get("expires_in"),
        refresh_token=result.get("refresh_token"),
        user=result.get("user"),
    )
    logger.info("auth/login:성공 email=%s expires_in=%s", payload.email, result.get("expires_in"))
    return resp


__all__ = ["router"]

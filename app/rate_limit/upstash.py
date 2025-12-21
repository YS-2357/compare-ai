"""Upstash Redis를 이용한 사용자별 일일 사용량 제한 클라이언트."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from fastapi import HTTPException, status

from app.config import get_settings
from app.logger import get_logger

logger = get_logger(__name__)


class UpstashClient:
    """간단한 Upstash REST 클라이언트."""

    def __init__(self, url: str, token: str, timeout: float = 5.0) -> None:
        self.url = url.rstrip("/")
        self.token = token
        self._client = httpx.AsyncClient(timeout=timeout)
        self._lock = asyncio.Lock()

    async def incr_with_expiry(self, key: str, ttl_seconds: int) -> int:
        """INCR 후 키에 만료를 설정한다."""

        logger.debug("Upstash:incr_with_expiry 시작 key=%s ttl=%s", key, ttl_seconds)
        payload: list[list[Any]] = [["INCR", key], ["EXPIRE", key, ttl_seconds]]
        async with self._lock:
            resp = await self._client.post(
                f"{self.url}/pipeline",
                headers={"Authorization": f"Bearer {self.token}"},
                json=payload,
            )
        if resp.status_code >= 400:
            logger.error("Upstash:incr_with_expiry 실패 status=%s", resp.status_code)
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="rate limit backend error")

        data = resp.json()
        # Pipeline 결과는 [{"result": int}, {"result": int}] 형태
        try:
            value = int(data[0]["result"])
            logger.info("Upstash:incr_with_expiry 성공 key=%s value=%s", key, value)
            return value
        except Exception as exc:  # pragma: no cover
            logger.error("Upstash:incr_with_expiry 파싱 실패 key=%s err=%s", key, exc)
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="rate limit parse error") from exc

    async def get(self, key: str) -> int:
        """현재 카운트를 조회한다. 존재하지 않으면 0."""

        logger.debug("Upstash:get 시작 key=%s", key)
        payload: list[list[Any]] = [["GET", key]]
        async with self._lock:
            resp = await self._client.post(
                f"{self.url}/pipeline",
                headers={"Authorization": f"Bearer {self.token}"},
                json=payload,
            )
        if resp.status_code >= 400:
            logger.error("Upstash:get 실패 status=%s", resp.status_code)
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="rate limit backend error")
        data = resp.json()
        try:
            value = data[0]["result"]
            result = int(value) if value is not None else 0
            logger.debug("Upstash:get 종료 key=%s value=%s", key, result)
            return result
        except Exception as exc:  # pragma: no cover
            logger.error("Upstash:get 파싱 실패 key=%s err=%s", key, exc)
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="rate limit parse error") from exc

    async def aclose(self) -> None:
        """HTTP 클라이언트를 종료한다."""

        logger.debug("Upstash:aclose 시작")
        await self._client.aclose()
        logger.debug("Upstash:aclose 종료")


_client: UpstashClient | None = None


def get_rate_limiter() -> UpstashClient:
    """환경변수를 기반으로 Upstash 클라이언트를 생성/재사용한다."""

    global _client
    if _client is not None:
        logger.debug("get_rate_limiter:캐시 사용")
        return _client

    settings = get_settings()
    if not settings.upstash_redis_url or not settings.upstash_redis_token:
        raise RuntimeError("Upstash Redis 설정이 없습니다.")
    _client = UpstashClient(
        settings.upstash_redis_url,
        settings.upstash_redis_token,
        timeout=settings.upstash_http_timeout,
    )
    logger.info("get_rate_limiter:생성 완료 url=%s", settings.upstash_redis_url)
    return _client


async def shutdown_rate_limiter() -> None:
    """FastAPI lifespan에서 Upstash 클라이언트를 정리한다."""

    global _client
    if _client is None:
        return
    try:
        logger.debug("shutdown_rate_limiter:시작")
        await _client.aclose()
    finally:
        _client = None
        logger.info("shutdown_rate_limiter:완료")


__all__ = ["UpstashClient", "get_rate_limiter", "shutdown_rate_limiter"]

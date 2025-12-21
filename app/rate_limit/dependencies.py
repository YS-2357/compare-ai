"""레이트 리밋 관련 FastAPI dependency."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, status

from app.utils.logger import get_logger
from .upstash import get_rate_limiter

logger = get_logger(__name__)


def _seconds_until_midnight_utc() -> int:
    logger.debug("seconds_until_midnight:시작")
    now = datetime.now(timezone.utc)
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    result = int((tomorrow - now).total_seconds())
    logger.debug("seconds_until_midnight:종료 secs=%s", result)
    return result


async def enforce_daily_limit(user_id: str, limit: int) -> int:
    """사용자별 일일 호출 제한을 적용한다."""

    logger.debug("enforce_daily_limit:시작 user=%s limit=%s", user_id, limit)
    try:
        client = get_rate_limiter()
        key = f"usage:{user_id}:{datetime.now(timezone.utc).date().isoformat()}"
        count = await client.incr_with_expiry(key, _seconds_until_midnight_utc())
        if count > limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="daily usage limit exceeded",
            )
        remaining = max(0, limit - count)
        logger.info("enforce_daily_limit:성공 user=%s remaining=%s", user_id, remaining)
        return remaining
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("레이트리밋 백엔드 오류: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="rate limit backend unavailable",
        )


__all__ = ["enforce_daily_limit"]

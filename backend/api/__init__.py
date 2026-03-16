"""API 라우터 패키지."""

from fastapi import APIRouter

from .routes import router as public_router

router = APIRouter()
router.include_router(public_router)

__all__ = ["router"]

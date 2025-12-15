from __future__ import annotations

import sys
from pathlib import Path

import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# 리포 루트를 모듈 경로에 추가해 `app` 임포트가 보장되도록 함
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.main import app  # noqa: E402


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    """ASGI 클라이언트를 제공한다."""

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

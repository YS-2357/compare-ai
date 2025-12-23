"""프롬프트 파일 로딩 유틸리티."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _prompt_root() -> Path:
    settings = get_settings()
    root = Path(settings.prompt_root)
    if not root.is_absolute():
        repo_root = Path(__file__).resolve().parents[3]
        root = (repo_root / root).resolve()
    return root


@lru_cache(maxsize=128)
def load_prompt(name: str, version: str) -> str:
    """docs/prompt에서 프롬프트 파일을 읽는다."""

    path = _prompt_root() / f"{name}@{version}.md"
    if not path.exists():
        raise FileNotFoundError(f"prompt file not found: {path}")
    content = path.read_text(encoding="utf-8")
    logger.info("prompt_loaded name=%s version=%s path=%s", name, version, path)
    return content


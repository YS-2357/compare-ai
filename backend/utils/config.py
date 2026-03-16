"""애플리케이션 설정 모듈."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """환경 변수를 통해 주입되는 기본 설정."""

    fastapi_host: str = "127.0.0.1"
    fastapi_port: int = 8000
    frontend_port: int = 5173
    frontend_host: str = "127.0.0.1"
    frontend_dir: str = "frontend"

    # 대화 컨텍스트 제어
    max_context_messages: int = 10  # 최근 메시지 유지 개수 (약 5턴: user/assistant 합산)

    env: Literal["local", "test", "prod"] = "local"
    fastapi_title: str = "Compare-AI Backend API"
    fastapi_version: str = "0.1.0"
    langsmith_project: str = "Compare-AI-BE"
    model_openai: str = "gpt-4.1-mini"
    model_gemini: str = "gemini-2.5-flash-lite"
    model_anthropic: str = "claude-3-5-haiku-20241022"
    model_upstage: str = "solar-mini"
    model_perplexity: str = "sonar"
    model_mistral: str = "mistral-small-latest"
    model_groq: str = "llama-3.3-70b-versatile"
    model_cohere: str = "command-r-08-2024"
    model_deepseek: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    prompt_root: str = "docs/prompt"
    prompt_chat_compare_version: str = "1.0.0"

    @staticmethod
    def from_env() -> Settings:
        """환경 변수와 `.env` 값을 기반으로 Settings를 생성한다.

        Returns:
            Settings: 현재 실행 환경에 맞춘 설정 인스턴스.
        """

        return Settings(
            fastapi_host=os.getenv("FASTAPI_HOST", "127.0.0.1"),
            fastapi_port=int(os.getenv("FASTAPI_PORT", "8000")),
            frontend_port=int(os.getenv("FRONTEND_PORT", "5173")),
            frontend_host=os.getenv("FRONTEND_HOST", "127.0.0.1"),
            frontend_dir=os.getenv("FRONTEND_DIR", "frontend"),
            env=os.getenv("APP_ENV", "local"),  # type: ignore[assignment]
            fastapi_title=os.getenv("FASTAPI_TITLE", "API LangGraph Test"),
            fastapi_version=os.getenv("FASTAPI_VERSION", "0.1.0"),
            langsmith_project=os.getenv("LANGSMITH_PROJECT", "Compare-AI-BE"),
            model_openai=os.getenv("MODEL_OPENAI", "gpt-4.1-mini"),
            model_gemini=os.getenv("MODEL_GEMINI", "gemini-2.5-flash-lite"),
            model_anthropic=os.getenv("MODEL_ANTHROPIC", "claude-3-5-haiku-20241022"),
            model_upstage=os.getenv("MODEL_UPSTAGE", "solar-mini"),
            model_perplexity=os.getenv("MODEL_PERPLEXITY", "sonar"),
            model_mistral=os.getenv("MODEL_MISTRAL", "mistral-small-latest"),
            model_groq=os.getenv("MODEL_GROQ", "llama-3.3-70b-versatile"),
            model_cohere=os.getenv("MODEL_COHERE", "command-r-08-2024"),
            model_deepseek=os.getenv("MODEL_DEEPSEEK", "deepseek-chat"),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            prompt_root="docs/prompt",
            prompt_chat_compare_version="1.0.0",
            max_context_messages=int(os.getenv("MAX_CONTEXT_MESSAGES", "10")),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """전역적으로 재사용 가능한 Settings 인스턴스를 반환한다.

    Returns:
        Settings: 캐싱된 설정 인스턴스.
    """

    return Settings.from_env()

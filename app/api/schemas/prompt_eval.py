"""프롬프트 평가 요청 스키마."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PromptEvalRequest(BaseModel):
    """프롬프트 비교/평가 요청 스키마."""

    question: str = Field(..., description="사용자 질문")
    prompt: str | None = Field(
        None,
        description="모든 모델에 공통으로 적용할 프롬프트 (미지정 시 기본 프롬프트 사용)",
    )
    models: list[str] | None = Field(
        None,
        description="평가 대상 모델 라벨 목록 (미지정 시 기본 8개)",
        examples=[["OpenAI", "Gemini", "Anthropic"]],
    )


__all__ = ["PromptEvalRequest"]

"""프롬프트 평가 결과 모델."""

from pydantic import BaseModel, Field


class Score(BaseModel):
    """단일 답변 평가 결과."""

    id: str = Field(..., description="익명 응답 ID")
    accuracy: float | None = Field(default=None, description="정확성(0~10)")
    completeness: float | None = Field(default=None, description="완전성(0~10)")
    clarity: float | None = Field(default=None, description="명료성(0~10)")
    score: float | None = Field(default=None, description="가중치 적용 전 점수(옵션)")
    rank: int | None = Field(default=None, description="순위 (없으면 백엔드에서 산출)")
    rationale: str = Field(..., description="짧은 근거")


class ScoreList(BaseModel):
    """평가 모델 출력 스키마."""

    scores: list[Score]

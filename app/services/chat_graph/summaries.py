"""응답 및 히스토리 요약 유틸리티."""

from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.logger import get_logger

logger = get_logger(__name__)
SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    "Summarize the following answer in Korean, in 2 sentences or fewer (under 400 characters).\n\n"
    "Answer:\n{answer}\n"
)
SUMMARY_PARSER = StrOutputParser()


def preview_text(text: str, limit: int = 80) -> str:
    """긴 문자열을 로그에 표시하기 위한 요약 버전으로 변환한다."""

    compact = " ".join(text.split())
    return compact[:limit] + ("…" if len(compact) > limit else "")


async def summarize_content(llm: Any, content: str, label: str) -> str:
    """모델 응답을 짧게 요약한다."""

    try:
        chain = SUMMARY_PROMPT | llm | SUMMARY_PARSER
        return await chain.ainvoke({"answer": content})
    except Exception as exc:
        logger.warning("%s 요약 실패: %s", label, exc)
        return content[:200]


__all__ = ["preview_text", "summarize_content"]

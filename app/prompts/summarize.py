"""LCEL 기반 요약 프롬프트 정의."""

from langchain_core.prompts import ChatPromptTemplate

SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    "다음 답변을 2문장 이하(400자 이내)로 매우 간결하게 요약하세요.\n\n"
    "답변:\n{answer}\n"
)

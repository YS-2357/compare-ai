"""LangGraph 프롬프트/메시지 유틸."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.utils.logger import get_logger
from app.utils.config import get_settings
from app.services.shared import load_prompt
from .summaries import preview_text

logger = get_logger(__name__)
settings_cache = get_settings()


class Answer(BaseModel):
    """LCEL 체인 출력 스키마."""

    content: str = Field(..., description="짧게 요약된 답변")
    source: str | None = Field(default=None, description="출처 URL 또는 참고 정보(옵션)")


def message_to_text(message: Any) -> str | None:
    """여러 형태의 메시지를 role: content 문자열로 변환한다."""

    logger.debug("message_to_text:시작 type=%s", type(message).__name__)
    if isinstance(message, (list, tuple)) and len(message) == 2:
        role, content = message
        result = f"{role}: {content}"
        logger.debug("message_to_text:종료 preview=%s", preview_text(result))
        return result
    if isinstance(message, dict):
        result = f"{message.get('role')}: {message.get('content')}"
        logger.debug("message_to_text:종료 preview=%s", preview_text(result))
        return result
    if isinstance(message, BaseMessage):
        role = getattr(message, "type", message.__class__.__name__)
        content = message.content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                else:
                    parts.append(str(item))
            content = " ".join([p for p in parts if p])
        result = f"{role}: {content}"
        logger.debug("message_to_text:종료 preview=%s", preview_text(result))
        return result
    logger.debug("message_to_text:종료 preview=None")
    return None


def build_chat_prompt() -> ChatPromptTemplate:
    logger.debug("build_chat_prompt:시작")
    parser = PydanticOutputParser(pydantic_object=Answer)
    instructions = parser.get_format_instructions()
    settings = get_settings()
    version = settings.prompt_chat_graph_version
    system_template = load_prompt("chat_graph_system", version)
    try:
        escaped = instructions.replace("{", "{{").replace("}", "}}")
        system = system_template.format(format_instructions=escaped)
    except Exception as exc:
        logger.warning("build_chat_prompt:템플릿 포맷 실패 err=%s", exc)
        system = system_template + "\n" + instructions
    logger.info("chat_graph_prompt_version=%s", version)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("user", "{question}"),
        ]
    ).partial(format_instructions=instructions)
    logger.debug("build_chat_prompt:종료")
    return prompt


def render_chat_history(state: Any, label: str, max_messages: int = 10) -> str:
    """모델별로 최근 히스토리와 요약을 합쳐 프롬프트용 문자열을 만든다."""

    user_msgs = state.get("user_messages") or []
    model_histories = state.get("model_messages") or {}
    model_msgs = model_histories.get(label, []) or []
    summary_present = bool((state.get("model_summaries") or {}).get(label))
    logger.info(
        "history/render label=%s user_msgs=%d model_msgs=%d summary=%s",
        label,
        len(user_msgs),
        len(model_msgs),
        summary_present,
    )

    lines: list[str] = []
    summaries = state.get("model_summaries") or {}
    summary = summaries.get(label)
    if summary:
        lines.append(f"[이전 요약] {summary}")

    combined: list[str] = []
    max_len = max(len(user_msgs), len(model_msgs))
    for idx in range(max_len):
        if idx < len(user_msgs):
            msg_text = message_to_text(user_msgs[idx])
            if msg_text:
                combined.append(msg_text)
        if idx < len(model_msgs):
            msg_text = message_to_text(model_msgs[idx])
            if msg_text:
                combined.append(msg_text)

    lines.extend(combined[-max_messages:])

    history_text = "\n".join(lines)
    logger.info(
        "history/rendered label=%s lines=%d chars=%d preview=%s",
        label,
        len(lines),
        len(history_text),
        preview_text(history_text),
    )
    logger.debug("render_chat_history:종료 label=%s", label)
    return history_text


def build_chat_prompt_input(state: Any, label: str) -> str:
    """
    대화 이력과 현재 질문을 분리해 전달한다.
    - history: 모델별 인터리브된 최근 히스토리
    - question: 현재 질문
    """

    max_context = settings_cache.max_context_messages
    logger.debug("build_chat_prompt_input:시작 label=%s max_context=%d", label, max_context)
    history_text = render_chat_history(state, label, max_messages=max_context)
    user_msgs = state.get("user_messages") or []
    current_question = ""
    if user_msgs:
        for msg in reversed(user_msgs):
            if isinstance(msg, (list, tuple)) and len(msg) == 2 and msg[0] == "user":
                current_question = msg[1]
                break
            if isinstance(msg, dict) and msg.get("role") == "user":
                current_question = msg.get("content", "")
                break
    if history_text.strip():
        prompt = (
            "[Conversation History]\n"
            f"{history_text}\n\n"
            "[Current Question]\n"
            f"{current_question}\n\n"
            "If the wording is ambiguous, prefer the most recent topic or flow. Respond only in Korean."
        )
        mode = "with_history"
    else:
        prompt = (
            "[Current Question]\n"
            f"{current_question}\n\n"
            "This is the first turn; there is no prior conversation. Answer clearly and only in Korean."
        )
        mode = "first_turn"
    logger.info(
        "prompt/built label=%s turn=%s mode=%s chars=%d preview=%s",
        label,
        state.get("turn") or 1,
        mode,
        len(prompt),
        preview_text(prompt),
    )
    logger.debug("build_chat_prompt_input:종료 label=%s mode=%s", label, mode)
    return prompt


__all__ = ["Answer", "message_to_text", "build_chat_prompt", "render_chat_history", "build_chat_prompt_input"]

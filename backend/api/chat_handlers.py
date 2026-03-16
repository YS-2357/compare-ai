"""채팅/평가 라우트 내부 헬퍼."""

from __future__ import annotations

import time
from typing import Any

from backend.services.chat_compare.providers import resolve_active_nodes


def _status_code(status: Any) -> int | None:
    """상태 객체에서 숫자 HTTP 코드를 추출한다."""

    if isinstance(status, dict):
        value = status.get("status")
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.isdigit():
            return int(value)
    if isinstance(status, int):
        return status
    return None


def _is_success_status(status: Any) -> bool:
    """상태가 성공(2xx)인지 판별한다."""

    code = _status_code(status)
    return code is not None and 200 <= code < 300


def resolve_chat_execution_options(
    *,
    active_providers: list[str] | None,
    models: dict[str, str] | None,
) -> tuple[list[str] | None, dict[str, str] | None]:
    """채팅 실행 옵션을 정규화한다."""

    active_nodes = resolve_active_nodes(active_providers)
    return (active_nodes or None), (models or None)


def create_stream_context(question: str) -> dict[str, Any]:
    """스트림 요약 생성을 위한 누적 상태를 초기화한다."""

    return {
        "answers": {},
        "api_status": {},
        "durations_ms": {},
        "sources": {},
        "response_meta": {},
        "error_by_model": {},
        "messages": [{"role": "user", "content": question}],
        "seen_messages": {("user", question)},
        "completion_order": [],
        "errors": [],
        "start_time": time.perf_counter(),
    }


def extend_messages(stream_state: dict[str, Any], new_messages: list[dict[str, str]] | None) -> None:
    """중복 없이 스트림 메시지를 합친다."""

    seen_messages = stream_state["seen_messages"]
    messages = stream_state["messages"]
    for message in new_messages or []:
        role = str(message.get("role"))
        content = str(message.get("content"))
        key = (role, content)
        if key in seen_messages:
            continue
        seen_messages.add(key)
        messages.append({"role": role, "content": content})


def accumulate_chat_event(stream_state: dict[str, Any], event: dict[str, Any]) -> None:
    """partial/error 이벤트를 누적해 summary 생성 준비를 한다."""

    event_type = event.get("event") or event.get("type", "partial")
    if event_type == "partial":
        model = event.get("model")
        if model:
            completion_order = stream_state["completion_order"]
            if model not in completion_order:
                completion_order.append(model)
            stream_state["answers"][model] = event.get("answer")
            status = event.get("status")
            if status:
                stream_state["api_status"][model] = status
            if event.get("source") is not None:
                stream_state["sources"][model] = event.get("source")
            if event.get("response_meta") is not None:
                stream_state["response_meta"][model] = event.get("response_meta")
            elapsed_ms = event.get("elapsed_ms")
            if elapsed_ms is not None:
                stream_state["durations_ms"][model] = int(elapsed_ms)
            if _is_success_status(status):
                stream_state["error_by_model"].pop(model, None)
            elif status:
                stream_state["error_by_model"][model] = {
                    "detail": (status.get("detail") if isinstance(status, dict) else None) or event.get("answer"),
                    "error_code": event.get("error_code") or "MODEL_ERROR",
                    "node": event.get("node"),
                    "model": model,
                }
        extend_messages(stream_state, event.get("messages"))
        return

    if event_type == "error":
        error_item = {
            "detail": event.get("detail"),
            "error_code": event.get("error_code"),
            "node": event.get("node"),
            "model": event.get("model"),
        }
        stream_state["errors"].append(error_item)
        model = event.get("model")
        if model:
            stream_state["error_by_model"][model] = error_item


def build_chat_summary(
    stream_state: dict[str, Any],
    *,
    question: str,
    usage_limit: int,
    usage_remaining: int | None,
    model_overrides: dict[str, str] | None,
) -> dict[str, Any]:
    """누적 상태에서 최종 summary 이벤트를 조립한다."""

    completion_order = stream_state["completion_order"]
    answers = stream_state["answers"]
    api_status = stream_state["api_status"]
    success_models = [model for model in completion_order if _is_success_status(api_status.get(model))]
    error_models = [
        model
        for model in completion_order
        if model not in success_models and (model in stream_state["error_by_model"] or model in api_status)
    ]
    combined_errors = list(stream_state["errors"])
    for error_item in stream_state["error_by_model"].values():
        if error_item not in combined_errors:
            combined_errors.append(error_item)
    primary_model = next((model for model in completion_order if answers.get(model)), None)
    primary_answer = (
        {
            "model": primary_model,
            "answer": answers.get(primary_model),
            "status": api_status.get(primary_model),
        }
        if primary_model
        else None
    )
    return {
        "event": "summary",
        "status": "ok",
        "elapsed_ms": int((time.perf_counter() - stream_state["start_time"]) * 1000),
        "result": {
            "question": question,
            "answers": answers,
            "api_status": api_status,
            "durations_ms": stream_state["durations_ms"],
            "sources": stream_state["sources"],
            "response_meta": stream_state["response_meta"],
            "messages": stream_state["messages"],
            "order": completion_order,
            "primary_model": primary_model,
            "primary_answer": primary_answer,
            "success_models": success_models,
            "success_count": len(success_models),
            "error_models": error_models,
            "error_count": len(combined_errors),
            "errors": combined_errors,
            "usage_limit": usage_limit,
            "usage_remaining": usage_remaining,
            "model_overrides": model_overrides or {},
        },
    }


__all__ = [
    "accumulate_chat_event",
    "build_chat_summary",
    "create_stream_context",
    "resolve_chat_execution_options",
]

"""API 라우터와 엔드포인트 정의."""

from __future__ import annotations

import json
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.api.chat_handlers import (
    accumulate_chat_event,
    build_chat_summary,
    create_stream_context,
    resolve_chat_execution_options,
)
from backend.api.schemas import AskRequest
from backend.services import stream_chat
from backend.utils.logger import get_logger

router = APIRouter()
logger = get_logger(__name__)


def _preview(text: str, limit: int = 80) -> str:
    """로그 출력을 위해 문자열을 요약한다."""

    compact = " ".join(text.split())
    return compact[:limit] + ("…" if len(compact) > limit else "")


@router.get("/health")
async def health():
    """서비스 가용성을 확인하는 헬스 체크.

    - 인증/쿼리 파라미터가 필요 없는 가장 단순한 경로.
    - 응답은 `{"status": "ok"}` 형태의 JSON 한 건이다.
    """

    logger.debug("health:시작")
    resp = {"status": "ok"}
    logger.info("health:성공 응답=%s", resp)
    return resp


@router.post(
    "/api/ask",
    responses={
        200: {
            "description": "NDJSON 스트림 (partial/summary) 반환",
            "content": {
                "application/json": {
                    "example": {
                        "event": "partial",
                        "model": "OpenAI",
                        "answer": "...",
                        "status": {"status": 200, "detail": "stop"},
                        "elapsed_ms": 1234,
                    }
                }
            },
        },
    },
    summary="LangGraph 스트리밍 질의 (NDJSON)",
)
async def ask_question(payload: AskRequest):
    """LangGraph 워크플로우를 NDJSON 스트림으로 실행한다.

    요청 본문:
    - `question`(str): 필수 질문.
    - `history`(list[{"role","content"}]): 이전 대화 히스토리(없으면 새 대화로 처리). 현재 질문은 서버가 별도 분리해 `[Current Question]`에 넣는다.
    - `models`(dict): 공급자별 기본 모델을 덮어쓸 때 사용(예: `{"openai": "gpt-4.1-mini"}`).
    - `active_providers`(list[str]): 활성화된 공급자 목록(없으면 전체 사용).
    - Cohere `command-a-reasoning-08-2025`는 텍스트 파이프라인과 호환되지 않아 채팅 UI 목록에서 제외됨.

    응답 스트림(한 줄씩 JSON):
    - `event="partial"`: 모델별 진행 중 결과. `model`, `answer`, `elapsed_ms`, `status`(LLM 응답 상태), `source`(출처), `response_meta`(모델/토큰/종료 사유 등) 포함.
    - `event="error"`: 특정 모델/노드 오류. `detail`, `model`, `node`, `status`, `error_code` 포함.
    - `event="summary"`: 전체 완료 메타. `answers`(모델별 최종 답변), `order`(완료 순서), `api_status`, `durations_ms`, `sources`, `response_meta`, `messages`, `errors`, `usage_limit`, `usage_remaining` 포함.

    """

    logger.debug("ask_question:시작 question=%s", _preview(payload.question))
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")

    history = payload.history or []

    logger.info("질문 수신: %s", _preview(question))

    active_nodes, model_overrides = resolve_chat_execution_options(
        active_providers=payload.active_providers,
        models=payload.models,
    )

    async def response_stream():
        stream_state = create_stream_context(question)
        try:
            async for event in stream_chat(
                question,
                history=history,
                model_overrides=model_overrides,
                active_models=active_nodes if active_nodes else None,
            ):
                accumulate_chat_event(stream_state, event)
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as exc:  # pragma: no cover
            error_event = {
                "event": "error",
                "error_code": "UNKNOWN_ERROR",
                "detail": str(exc),
                "status": "error",
                "node": None,
                "model": None,
                "elapsed_ms": int((time.perf_counter() - stream_state["start_time"]) * 1000),
            }
            accumulate_chat_event(stream_state, error_event)
            logger.error("응답 스트림 처리 중 오류: %s", exc)
            yield json.dumps(error_event, ensure_ascii=False) + "\n"
        finally:
            summary = build_chat_summary(
                stream_state,
                question=question,
                usage_limit=0,
                usage_remaining=None,
                model_overrides=model_overrides,
            )
            logger.info(
                "요약 응답 전송 - 성공 모델 수: %d, 오류 수: %d",
                summary["result"]["success_count"],
                summary["result"]["error_count"],
            )
            logger.debug("ask_question:종료")
            yield json.dumps(summary, ensure_ascii=False) + "\n"

    return StreamingResponse(response_stream(), media_type="application/json")

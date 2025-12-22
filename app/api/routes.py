"""API 라우터와 엔드포인트 정의."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.deps import AuthenticatedUser, enforce_daily_limit, get_current_user, get_settings
from app.api.schemas import AskRequest, PromptEvalRequest
from app.utils.logger import get_logger
from app.services import stream_chat
from app.services.chat_graph import DEFAULT_MAX_TURNS
from app.services.prompt_eval import stream_prompt_eval
from app.rate_limit.router import router as usage_router

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
                        "type": "partial",
                        "model": "OpenAI",
                        "answer": "...",
                        "status": {"status": 200, "detail": "stop"},
                        "elapsed_ms": 1234,
                    }
                }
            },
        },
        401: {"description": "인증 필요 (Authorization: Bearer <JWT>)"},
        503: {"description": "레이트리밋 백엔드 오류"},
    },
    summary="LangGraph 스트리밍 질의 (NDJSON)",
)
async def ask_question(payload: AskRequest, user: AuthenticatedUser = Depends(get_current_user)):
    """LangGraph 워크플로우를 NDJSON 스트림으로 실행한다.

    요청 본문:
    - `question`(str): 필수 질문.
    - `history`(list[{"role","content"}]): 이전 대화 히스토리(없으면 새 대화로 처리).
    - `turn`/`max_turns`: 멀티턴 제한 제어(관리자는 우회).
    - `models`(dict): 공급자별 기본 모델을 덮어쓸 때 사용(예: `{"openai": "gpt-4.1-mini"}`).

    응답 스트림(한 줄씩 JSON):
    - `type="partial"`: 모델별 진행 중 결과. `model`, `answer`, `elapsed_ms`, `status`(LLM 응답 상태), `source`(출처), `response_meta`(모델/토큰/종료 사유 등) 포함.
    - `type="error"`: 특정 모델/노드 오류. `message`, `model`, `node`, `status` 포함.
    - `type="summary"`: 전체 완료 메타. `answers`(모델별 최종 답변), `order`(완료 순서), `api_status`, `durations_ms`, `sources`, `response_meta`, `messages`, `errors`, `usage_limit`, `usage_remaining` 포함.

    헤더:
    - `X-Usage-Limit`, `X-Usage-Remaining`: 남은 일일 호출 횟수(관리자는 null).
    """

    logger.debug("ask_question:시작 question=%s", _preview(payload.question))
    settings = get_settings()
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")

    turn = payload.turn or 1
    max_turns = payload.max_turns or DEFAULT_MAX_TURNS
    history = payload.history or []
    if turn < 1:
        turn = 1

    logger.info("질문 수신: %s", _preview(question))

    usage_remaining: int | None = None
    if not user.get("bypass"):
        usage_remaining = await enforce_daily_limit(user["sub"], settings.daily_usage_limit)

    model_overrides = payload.models or None
    bypass_limits = bool(user.get("bypass"))

    async def response_stream():
        answers = {}
        api_status = {}
        durations_ms: dict[str, int] = {}
        sources: dict[str, str | None] = {}
        response_meta: dict[str, dict[str, Any] | None] = {}
        messages = [{"role": "user", "content": question}]
        seen_messages = {("user", question)}
        completion_order: list[str] = []
        errors: list[dict[str, str | None]] = []

        def _extend_messages(new_messages: list[dict[str, str]] | None):
            for message in new_messages or []:
                role = str(message.get("role"))
                content = str(message.get("content"))
                key = (role, content)
                if key in seen_messages:
                    continue
                seen_messages.add(key)
                messages.append({"role": role, "content": content})

        try:
            async for event in stream_chat(
                question,
                turn=turn,
                max_turns=max_turns,
                history=history,
                model_overrides=model_overrides,
                bypass_turn_limit=bypass_limits,
            ):
                event_type = event.get("type", "partial")
                if event_type == "partial":
                    model = event.get("model")
                    if model:
                        if model not in completion_order:
                            completion_order.append(model)
                        answers[model] = event.get("answer")
                        status = event.get("status")
                        if status:
                            api_status[model] = status
                        if event.get("source") is not None:
                            sources[model] = event.get("source")
                        if event.get("response_meta") is not None:
                            response_meta[model] = event.get("response_meta")
                        elapsed_ms = event.get("elapsed_ms")
                        if elapsed_ms is not None:
                            durations_ms[model] = int(elapsed_ms)
                        logger.debug("부분 응답 누적: %s", model)
                    _extend_messages(event.get("messages"))
                elif event_type == "error":
                    logger.warning(
                        "스트림 오류 이벤트 수신 (node=%s, model=%s): %s",
                        event.get("node"),
                        event.get("model"),
                        event.get("message"),
                    )
                    errors.append(
                        {
                            "message": event.get("message"),
                            "node": event.get("node"),
                            "model": event.get("model"),
                        }
                    )
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as exc:  # pragma: no cover
            error_event = {"type": "error", "message": str(exc), "node": None, "model": None}
            errors.append(
                {
                    "message": str(exc),
                    "node": None,
                    "model": None,
                }
            )
            logger.error("응답 스트림 처리 중 오류: %s", exc)
            yield json.dumps(error_event, ensure_ascii=False) + "\n"
        finally:
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
            summary = {
                "type": "summary",
                "result": {
                    "question": question,
                    "answers": answers,
                    "api_status": api_status,
                    "durations_ms": durations_ms,
                    "sources": sources,
                    "response_meta": response_meta,
                    "messages": messages,
                    "order": completion_order,
                    "primary_model": primary_model,
                    "primary_answer": primary_answer,
                    "errors": errors,
                    "turn": turn,
                    "max_turns": max_turns,
                    "usage_limit": settings.daily_usage_limit,
                    "usage_remaining": usage_remaining,
                    "model_overrides": model_overrides or {},
                },
            }
            logger.info("요약 응답 전송 - 완료 모델 수: %d, 오류 수: %d", len(answers), len(errors))
            logger.debug("ask_question:종료 turn=%s max_turns=%s", turn, max_turns)
            yield json.dumps(summary, ensure_ascii=False) + "\n"

    headers = {}
    if usage_remaining is not None:
        headers["X-Usage-Limit"] = str(settings.daily_usage_limit)
        headers["X-Usage-Remaining"] = str(usage_remaining)
    return StreamingResponse(response_stream(), media_type="application/json", headers=headers)


@router.post(
    "/api/prompt-eval",
    responses={
        200: {
            "description": "NDJSON 스트림 (partial/summary) 반환",
            "content": {
                "application/json": {
                    "example": {
                        "type": "summary",
                        "result": {
                            "scores": [
                                {"model": "OpenAI", "score": 9.5, "rank": 1, "rationale": "응답 완결성/정확도 우수", "status": {"status": 200, "detail": "stop"}}
                            ],
                            "avg_score": 8.7,
                            "evaluations": [
                                {
                                    "evaluator": "OpenAI",
                                    "model": "gpt-4.1-mini",
                                    "scores": [{"target": "Gemini", "score": 8, "rationale": "간결/정확"}],
                                    "status": {"status": 200, "detail": "stop"},
                                    "elapsed_ms": 3500,
                                }
                            ],
                        },
                    },
                }
            },
        }
    },
    summary="프롬프트 평가 스트리밍 질의",
    description=(
        "공통 프롬프트로 여러 모델의 답변을 생성한 뒤, 벤더별 최신 모델이 교차 평가하여 점수/근거를 NDJSON 스트림으로 반환합니다.\n\n"
        "응답 이벤트:\n"
        "- `type=\"partial\"` & `phase=\"generation\"`: 모델별 원본 답변이 도착할 때 발생. `model`, `answer`, `status`, `elapsed_ms`, `response_meta` 포함.\n"
        "- `type=\"partial\"` & `phase=\"evaluation\"`: 평가자가 각 타깃 모델을 채점할 때 발생. `evaluator`, `target_model`, `score`, `rationale`, `status`, `elapsed_ms` 포함.\n"
        "- `type=\"summary\"`: 모든 평가가 끝난 후 최종 점수 표(`scores`), 평가자별 원본 점수/근거(`evaluations`), 평균점수(`avg_score`), 모델별 `response_meta`를 포함.\n"
        "- `type=\"error\"`: 처리 중 오류.\n\n"
        "옵션:\n"
        "- `reference_answer`: 모범 답변 예시를 넣으면 평가 프롬프트에 참고용으로 포함(없으면 기본 루브릭으로 평가).\n"
        "- `model_overrides`: 공급자별 모델 오버라이드(예: `{ \"openai\": \"gpt-4.1-mini\" }`).\n\n"
        "헤더: `X-Usage-Limit`, `X-Usage-Remaining`에 남은 일일 호출 수가 담깁니다."
    ),
)
async def prompt_eval(payload: PromptEvalRequest, user: AuthenticatedUser = Depends(get_current_user)):
    """공통 프롬프트로 여러 모델을 호출하고, 최신 평가자들이 교차 평가해 점수를 스트리밍한다."""

    logger.debug("prompt_eval:시작 question=%s", _preview(payload.question))
    settings = get_settings()
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="질문을 입력해주세요.")

    usage_remaining: int | None = None
    if not user.get("bypass"):
        usage_remaining = await enforce_daily_limit(user["sub"], settings.daily_usage_limit)

    async def response_stream():
        try:
            async for event in stream_prompt_eval(
                question,
                prompt=payload.prompt,
                active_models=payload.models,
                model_overrides=payload.model_overrides,
                reference_answer=payload.reference_answer,
            ):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as exc:  # pragma: no cover
            logger.error("프롬프트 평가 스트림 오류: %s", exc)
            error_event = {"type": "error", "message": str(exc), "node": None, "model": None}
            yield json.dumps(error_event, ensure_ascii=False) + "\n"
        finally:
            logger.debug("prompt_eval:종료 question=%s", _preview(question))

    headers = {}
    if usage_remaining is not None:
        headers["X-Usage-Limit"] = str(settings.daily_usage_limit)
        headers["X-Usage-Remaining"] = str(usage_remaining)

    return StreamingResponse(response_stream(), media_type="application/json", headers=headers)


# 사용량 조회 라우터 포함
router.include_router(usage_router)

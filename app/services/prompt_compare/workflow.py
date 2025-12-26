"""Prompt-level 비교용 스트리밍 유틸리티.

질문 + 공통 프롬프트로 각 모델을 병렬 호출하고,
각 벤더의 최신 모델들이 모든 응답을 블라인드 크로스 평가한 뒤
partial(모델별 응답)과 summary(평가 결과 테이블)를 스트리밍한다.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator

from app.utils.logger import get_logger
from app.services.shared.errors import build_status_from_error, build_status_from_response
from app.services.shared import MODEL_ALIASES
from .aggregator import aggregate_scores
from .evaluator import evaluate_answers
from .extractors import append_sources_block, extract_response_meta, extract_sources
from .clients import LABEL_TO_KEY, build_model_prompt, llm_factory

logger = get_logger(__name__)


def _normalize_content_to_text(content: Any) -> str:
    """LLM content를 문자열로 안전하게 변환한다."""

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif hasattr(item, "text"):
                parts.append(str(getattr(item, "text") or ""))
            elif hasattr(item, "content"):
                parts.append(str(getattr(item, "content") or ""))
            else:
                parts.append(str(item))
        return " ".join([p for p in parts if p])
    if hasattr(content, "text"):
        return str(getattr(content, "text") or "")
    if hasattr(content, "content"):
        return str(getattr(content, "content") or "")
    return str(content)


async def _call_single_model(label: str, prompt_text: str, model_name: str | None = None) -> dict[str, Any]:
    """단일 모델 호출 및 파싱 실패 대비."""

    logger.debug("_call_single_model:시작 label=%s prompt_preview=%s", label, prompt_text[:200])
    start = time.perf_counter()
    try:
        llm = llm_factory(label, model_name)
        response = await llm.ainvoke(prompt_text)
        status = build_status_from_response(response)
        response_meta = extract_response_meta(response)
        raw_text = _normalize_content_to_text(getattr(response, "content", response))
        content = raw_text.strip()
        if not content:
            content = str(response) if response is not None else ""
        if not content:
            content = "응답이 비어있습니다."
        logger.debug("_call_single_model:raw_response label=%s raw=%s", label, repr(response))
        sources = extract_sources(response)
        content_with_sources = append_sources_block(content, sources)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.info("%s 호출 성공 (elapsed_ms=%s)", label, elapsed_ms)
        logger.debug("_call_single_model:raw_response label=%s body=%s", label, str(content))
        return {
            "model": label,
            "answer": content,
            "answer_with_sources": content_with_sources,
            "status": status,
            "elapsed_ms": elapsed_ms,
            "source": sources[0] if sources else None,
            "response_meta": response_meta,
            "error": False,
        }
    except Exception as exc:
        status = build_status_from_error(exc)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        message = f"응답 실패: {status.get('detail') or exc}"
        logger.warning("%s 호출 실패: %s", label, message)
        logger.debug("_call_single_model:예외 raw_response label=%s status=%s", label, status)
        return {
            "model": label,
            "answer": message,
            "answer_with_sources": message,
            "status": status,
            "elapsed_ms": elapsed_ms,
            "source": None,
            "response_meta": None,
            "error": True,
        }


async def stream_prompt_eval(
    question: str,
    prompt: str | None = None,
    active_models: list[str] | None = None,
    model_overrides: dict[str, str] | None = None,
    reference_answer: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """모델별 프롬프트 적용 + 블라인드 평가를 스트리밍한다."""

    logger.debug("stream_prompt_eval:시작 question=%s", question[:50] if question else "")
    if not question or not question.strip():
        logger.error("stream_prompt_eval:질문 비어있음")
        raise ValueError("질문을 입력해주세요.")

    start_time = time.perf_counter()
    try:
        raw_models = active_models or [
            "OpenAI",
            "Gemini",
            "Anthropic",
            "Perplexity",
            "Upstage",
            "Mistral",
            "Groq",
            "Cohere",
            "DeepSeek",
        ]
        models: list[str] = []
        for m in raw_models:
            canonical = MODEL_ALIASES.get(m, m)
            if canonical not in models:
                models.append(canonical)
        logger.info("PromptCompare 실행: models=%s", ", ".join(models))

        prompt_text = build_model_prompt(question, prompt)
        overrides = model_overrides or {}
        tasks = []
        for label in models:
            key = LABEL_TO_KEY.get(label)
            override_model = overrides.get(key) if key else None
            tasks.append(asyncio.create_task(_call_single_model(label, prompt_text, override_model)))

        results: list[dict[str, Any]] = []
        start = time.perf_counter()
        for coro in asyncio.as_completed(tasks):
            res = await coro
            results.append(res)
            yield {
                "event": "partial",
                "phase": "generation",
                "model": res["model"],
                "answer": res["answer"],
                "status": res.get("status") or {},
                "source": res.get("source"),
                "response_meta": res.get("response_meta"),
                "elapsed_ms": res.get("elapsed_ms"),
            }

        # 평가 단계 (실패 응답 제외)
        success_results = [r for r in results if not r.get("error")]
        prompt_for_eval = prompt_text
        evaluation_tasks = []
        for evaluator_label in models:
            evaluation_tasks.append(
                asyncio.create_task(
                    evaluate_answers(question, prompt_for_eval, success_results, evaluator_label, reference_answer)
                )
            )
        evaluation_outputs: list[dict[str, Any]] = []
        evaluation_results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
        for evaluator_label, ev_res in zip(models, evaluation_results):
            if isinstance(ev_res, Exception):
                logger.warning("평가 호출 실패: evaluator=%s err=%s", evaluator_label, ev_res)
                evaluation_outputs.append(
                    {
                        "evaluator": evaluator_label,
                        "scores": [
                            {
                                "model": r["model"],
                                "id": f"resp_{i+1}",
                                "score": -1,
                                "rank": i + 1,
                                "rationale": "evaluation call failed",
                            }
                            for i, r in enumerate(results)
                        ],
                        "status": {"status": "error", "detail": str(ev_res), "model": None},
                    }
                )
            else:
                evaluation_outputs.append(ev_res)

        for ev in evaluation_outputs:
            yield {
                "event": "partial",
                "phase": "evaluation",
                "model": ev.get("evaluator"),
                "evaluator": ev.get("evaluator"),
                "scores": ev.get("scores"),
                "status": ev.get("status"),
                "elapsed_ms": ev.get("elapsed_ms"),
            }

        # 평가 실패/미응답 모델은 점수 -1로 보강
        failed_models = [r["model"] for r in results if r.get("error")]
        if failed_models:
            for ev in evaluation_outputs:
                scored_models = {sc.get("model") for sc in ev.get("scores", [])}
                for model in failed_models:
                    if model in scored_models:
                        continue
                    ev.setdefault("scores", []).append(
                        {
                            "model": model,
                            "id": f"resp_missing_{model}",
                            "score": -1,
                            "rank": None,
                            "rationale": "response failed",
                        }
                    )

        aggregated_scores = aggregate_scores(results, evaluation_outputs)
        numeric_scores = [
            s["score"] for s in aggregated_scores if isinstance(s.get("score"), (int, float)) and s["score"] >= 0
        ]
        avg_score = round(sum(numeric_scores) / len(numeric_scores), 2) if numeric_scores else None
        elapsed_total_ms = int((time.perf_counter() - start) * 1000)

        yield {
            "event": "summary",
            "status": "ok",
            "elapsed_ms": elapsed_total_ms,
            "result": {
                "question": question,
                "answers": {r["model"]: r.get("answer") for r in results},
                "response_meta": {r["model"]: r.get("response_meta") for r in results},
                "scores": aggregated_scores,
                "avg_score": avg_score,
                "evaluations": evaluation_outputs,
                "elapsed_ms": elapsed_total_ms,
            },
        }
    except Exception as exc:
        logger.error("PromptCompare 전체 실패: %s", exc)
        yield {
            "event": "error",
            "error_code": "UNKNOWN_ERROR",
            "detail": str(exc),
            "status": "error",
            "model": None,
            "node": "prompt_compare",
            "elapsed_ms": int((time.perf_counter() - start_time) * 1000),
        }
    finally:
        logger.debug("stream_prompt_eval:종료 question=%s", question[:50] if question else "")


__all__ = ["stream_prompt_eval"]

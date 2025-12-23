"""Prompt-level 비교용 스트리밍 유틸리티.

질문 + 공통 프롬프트로 각 모델을 병렬 호출하고,
각 벤더의 최신 모델들이 모든 응답을 블라인드 크로스 평가한 뒤
partial(모델별 응답)과 summary(평가 결과 테이블)를 스트리밍한다.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, AsyncIterator

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from app.utils.config import get_settings
from app.utils.logger import get_logger
from app.services.shared.errors import build_status_from_error, build_status_from_response
from app.services.shared.llm_registry import (
    ChatAnthropic,
    ChatCohere,
    ChatGoogleGenerativeAI,
    ChatGroq,
    ChatMistralAI,
    ChatOpenAI,
    ChatPerplexity,
    ChatUpstage,
)
from app.services.shared import LATEST_EVAL_MODELS, MODEL_ALIASES, load_prompt

logger = get_logger(__name__)
settings_cache = get_settings()


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


def _build_deepseek_llm(model_name: str, base_url: str) -> ChatOpenAI:
    """DeepSeek OpenAI 호환 클라이언트를 생성한다."""

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing")
    return ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url)


def _extract_sources(raw_response: Any) -> list[str]:
    """응답 메타에서 출처 URL을 최대한 추출한다."""

    sources: list[str] = []

    def _maybe_add(value: Any) -> None:
        if isinstance(value, str) and value.startswith("http"):
            sources.append(value)
        if isinstance(value, dict):
            for key in ("url", "source", "link", "href"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.startswith("http"):
                    sources.append(candidate)

    meta = getattr(raw_response, "response_metadata", None)
    if isinstance(meta, dict):
        for key in ("citations", "sources", "search_results"):
            items = meta.get(key)
            if isinstance(items, list):
                for item in items:
                    _maybe_add(item)

    additional_kwargs = getattr(raw_response, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        for key in ("citations", "sources", "search_results"):
            items = additional_kwargs.get(key)
            if isinstance(items, list):
                for item in items:
                    _maybe_add(item)

    raw_sources = getattr(raw_response, "citations", None) or getattr(raw_response, "sources", None)
    if isinstance(raw_sources, list):
        for item in raw_sources:
            _maybe_add(item)

    # 중복 제거 (순서 유지)
    seen = set()
    unique_sources = []
    for src in sources:
        if src in seen:
            continue
        seen.add(src)
        unique_sources.append(src)
    return unique_sources


def _extract_response_meta(response: Any) -> dict[str, Any]:
    """응답 본문 외에 표시할 메타 정보를 추출한다."""

    meta: dict[str, Any] = {}
    response_meta = getattr(response, "response_metadata", None)
    additional_kwargs = getattr(response, "additional_kwargs", None)
    usage_metadata = getattr(response, "usage_metadata", None)

    if isinstance(response_meta, dict):
        for key in ("model_name", "model", "model_provider"):
            if response_meta.get(key):
                meta["model_name"] = response_meta.get(key)
                break
        if response_meta.get("finish_reason"):
            meta["finish_reason"] = response_meta.get("finish_reason")
        if response_meta.get("stop_reason"):
            meta["stop_reason"] = response_meta.get("stop_reason")
        if response_meta.get("safety_ratings"):
            meta["safety_ratings"] = response_meta.get("safety_ratings")
        if response_meta.get("prompt_feedback"):
            meta["prompt_feedback"] = response_meta.get("prompt_feedback")
        token_usage = response_meta.get("token_usage")
        if isinstance(token_usage, dict):
            meta["token_usage"] = {
                "input_tokens": token_usage.get("prompt_tokens"),
                "output_tokens": token_usage.get("completion_tokens"),
                "total_tokens": token_usage.get("total_tokens"),
            }

    if isinstance(additional_kwargs, dict) and "refusal" in additional_kwargs:
        meta["refusal"] = additional_kwargs.get("refusal")

    if isinstance(usage_metadata, dict):
        meta.setdefault(
            "token_usage",
            {
                "input_tokens": usage_metadata.get("input_tokens"),
                "output_tokens": usage_metadata.get("output_tokens"),
                "total_tokens": usage_metadata.get("total_tokens"),
            },
        )

    sources = _extract_sources(response)
    if sources:
        meta["sources"] = sources

    return meta


def _llm_factory(label: str, model_name: str | None = None) -> Any:
    """모델 라벨에 맞는 LLM 팩토리를 반환한다."""

    logger.debug("_llm_factory:시작 label=%s", label)
    sc = settings_cache
    factories = {
        "OpenAI": lambda: ChatOpenAI(model=model_name or sc.model_openai),
        "Gemini": lambda: ChatGoogleGenerativeAI(model=model_name or sc.model_gemini, temperature=0),
        "Anthropic": lambda: ChatAnthropic(model=model_name or sc.model_anthropic),
        "Perplexity": lambda: _build_perplexity_llm(model_name or sc.model_perplexity),
        "Upstage": lambda: ChatUpstage(model=model_name or sc.model_upstage),
        "Mistral": lambda: ChatMistralAI(model=model_name or sc.model_mistral),
        "Groq": lambda: ChatGroq(model=model_name or sc.model_groq),
        "Cohere": lambda: ChatCohere(model=model_name or sc.model_cohere),
        "DeepSeek": lambda: _build_deepseek_llm(model_name or sc.model_deepseek, sc.deepseek_base_url),
    }
    if label not in factories:
        raise ValueError(f"지원하지 않는 모델 라벨: {label}")
    llm = factories[label]()
    logger.debug("_llm_factory:종료 label=%s", label)
    return llm


def _build_perplexity_llm(model_name: str) -> ChatPerplexity:
    """Perplexity 클라이언트를 생성한다."""

    api_key = os.getenv("PPLX_API_KEY")
    if not api_key:
        raise RuntimeError("PPLX_API_KEY is missing")
    return ChatPerplexity(model=model_name, pplx_api_key=api_key)


def _append_sources_block(content: str, sources: list[str] | None) -> str:
    """평가용 응답에 출처 섹션을 추가한다."""

    if "[Sources]" in content:
        return content
    if sources:
        lines = "\n".join(f"- {src}" for src in sources)
        return f"{content}\n\n[Sources]\n{lines}"
    return f"{content}\n\n[Sources]\n- 제공되지 않음"


def _select_eval_llm(active_labels: list[str]) -> tuple[Any, str]:
    """평가에 사용할 LLM과 모델명을 선택한다."""

    logger.debug("_select_eval_llm:시작 labels=%s", active_labels)
    sc = settings_cache
    for label in active_labels:
        if label == "OpenAI":
            model_name = LATEST_EVAL_MODELS.get("OpenAI") or sc.model_openai
            logger.info("_select_eval_llm:선택 evaluator=%s model=%s", label, model_name)
            return ChatOpenAI(model=model_name), model_name
        if label == "Gemini":
            model_name = LATEST_EVAL_MODELS.get("Gemini") or sc.model_gemini
            logger.info("_select_eval_llm:선택 evaluator=%s model=%s", label, model_name)
            return ChatGoogleGenerativeAI(model=model_name, temperature=0), model_name
        if label == "Anthropic":
            model_name = LATEST_EVAL_MODELS.get("Anthropic") or sc.model_anthropic
            logger.info("_select_eval_llm:선택 evaluator=%s model=%s", label, model_name)
            return ChatAnthropic(model=model_name), model_name
        if label == "Perplexity":
            model_name = LATEST_EVAL_MODELS.get("Perplexity") or sc.model_perplexity
            logger.info("_select_eval_llm:선택 evaluator=%s model=%s", label, model_name)
            return ChatPerplexity(model=model_name), model_name
        if label == "Upstage":
            model_name = LATEST_EVAL_MODELS.get("Upstage") or sc.model_upstage
            logger.info("_select_eval_llm:선택 evaluator=%s model=%s", label, model_name)
            return ChatUpstage(model=model_name), model_name
        if label == "Mistral":
            model_name = LATEST_EVAL_MODELS.get("Mistral") or sc.model_mistral
            logger.info("_select_eval_llm:선택 evaluator=%s model=%s", label, model_name)
            return ChatMistralAI(model=model_name), model_name
        if label == "Groq":
            model_name = LATEST_EVAL_MODELS.get("Groq") or sc.model_groq
            logger.info("_select_eval_llm:선택 evaluator=%s model=%s", label, model_name)
            return ChatGroq(model=model_name), model_name
        if label == "Cohere":
            model_name = LATEST_EVAL_MODELS.get("Cohere") or sc.model_cohere
            logger.info("_select_eval_llm:선택 evaluator=%s model=%s", label, model_name)
            return ChatCohere(model=model_name), model_name
        if label == "DeepSeek":
            model_name = LATEST_EVAL_MODELS.get("DeepSeek") or sc.model_deepseek
            logger.info("_select_eval_llm:선택 evaluator=%s model=%s", label, model_name)
            return _build_deepseek_llm(model_name, sc.deepseek_base_url), model_name

    # fallback: OpenAI 기본
    model_name = LATEST_EVAL_MODELS.get("OpenAI") or sc.model_openai
    logger.info("_select_eval_llm:기본 선택 evaluator=OpenAI model=%s", model_name)
    return ChatOpenAI(model=model_name), model_name


DEFAULT_PROMPT = "[Question]\n{question}\n\n답변은 한국어로 작성하세요."

LABEL_TO_KEY = {
    "OpenAI": "openai",
    "Gemini": "gemini",
    "Anthropic": "anthropic",
    "Perplexity": "perplexity",
    "Upstage": "upstage",
    "Mistral": "mistral",
    "Groq": "groq",
    "Cohere": "cohere",
    "DeepSeek": "deepseek",
}


def _build_model_prompt(question: str, prompt: str | None) -> str:
    """모든 모델에 동일하게 적용할 프롬프트를 생성한다."""

    logger.debug("_build_model_prompt:시작")
    base = prompt or DEFAULT_PROMPT
    try:
        result = base.format(question=question)
        logger.debug("_build_model_prompt:종료 format 성공")
        return result
    except Exception:
        # 포맷 실패 시 안전하게 합치기
        fallback = f"{base}\n\n[Question]\n{question}"
        logger.warning("_build_model_prompt:포맷 실패, fallback 사용")
        return fallback


async def _call_single_model(label: str, prompt_text: str, model_name: str | None = None) -> dict[str, Any]:
    """단일 모델 호출 및 파싱 실패 대비."""

    logger.debug("_call_single_model:시작 label=%s prompt_preview=%s", label, prompt_text[:200])
    start = time.perf_counter()
    try:
        llm = _llm_factory(label, model_name)
        response = await llm.ainvoke(prompt_text)
        status = build_status_from_response(response)
        response_meta = _extract_response_meta(response)
        content = getattr(response, "content", None) or str(response)
        logger.debug("_call_single_model:raw_response label=%s raw=%s", label, repr(response))
        sources = _extract_sources(response)
        content_with_sources = _append_sources_block(content, sources)
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


def _build_eval_prompt(
    question: str,
    prompt_text: str,
    anonymized: list[tuple[str, str]],
    reference: str | None,
) -> ChatPromptTemplate:
    """블라인드 평가 프롬프트를 생성한다."""

    logger.debug("_build_eval_prompt:question preview=%s reference=%s", question[:200], bool(reference))
    examples = "\n".join([f"ID {anon_id}:\n{content}\n" for anon_id, content in anonymized])
    rubric_line = (
        "Use the reference answer to check factual alignment; if the wording differs but facts match, allow it."
        if reference
        else "Grade by accuracy, completeness, and clarity; do not penalize stylistic differences."
    )
    settings = get_settings()
    version = settings.prompt_eval_version
    system_template = load_prompt("prompt_eval_system", version)
    user_template = load_prompt("prompt_eval_user", version)
    reference_block = f"\n\n[Reference Answer]\n{reference}" if reference else ""
    parser = PydanticOutputParser(pydantic_object=ScoreList)
    instructions_raw = parser.get_format_instructions()
    instructions = instructions_raw.replace("{", "{{").replace("}", "}}")
    safe_system_template = system_template.replace("{", "{{").replace("}", "}}")
    safe_system_template = safe_system_template.replace("{{rubric_line}}", "{rubric_line}")
    safe_system_template = safe_system_template.replace("{{format_instructions}}", "{format_instructions}")
    system = safe_system_template.format(
        rubric_line=rubric_line,
        format_instructions=instructions,
    )
    user = user_template.format(
        question=question,
        prompt_text=prompt_text,
        answers=examples,
        reference_block=reference_block,
    )
    logger.info("prompt_eval_version=%s", version)
    template = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("user", user),
        ]
    )
    return template


async def _evaluate_answers(
    question: str,
    prompt_text: str,
    results: list[dict[str, Any]],
    evaluator_label: str,
    reference: str | None,
) -> dict[str, Any]:
    """단일 평가 모델로 모든 응답을 블라인드 평가."""

    logger.debug("_evaluate_answers:시작 evaluator=%s answers=%d", evaluator_label, len(results))
    # 익명 ID 매핑
    anonymized = [(f"resp_{i+1}", r.get("answer_with_sources") or r.get("answer", "")) for i, r in enumerate(results)]
    id_to_model = {f"resp_{i+1}": r["model"] for i, r in enumerate(results)}

    prompt = _build_eval_prompt(question, prompt_text, anonymized, reference)
    parser = PydanticOutputParser(pydantic_object=ScoreList)
    start_eval = time.perf_counter()
    try:
        eval_llm, eval_model_name = _select_eval_llm([evaluator_label])
    except Exception as exc:
        logger.warning("평가자 생성 실패: evaluator=%s err=%s", evaluator_label, exc)
        return {
            "evaluator": evaluator_label,
            "scores": [
                {
                    "model": r["model"],
                    "id": f"resp_{i+1}",
                    "score": -1,
                    "rank": i + 1,
                    "rationale": "evaluator init failed",
                }
                for i, r in enumerate(results)
            ],
            "status": {"status": "error", "detail": str(exc), "model": None},
            "elapsed_ms": int((time.perf_counter() - start_eval) * 1000),
        }

    chain = prompt | eval_llm
    try:
        response = await chain.ainvoke({})
        logger.debug("_evaluate_answers:raw_response evaluator=%s raw=%s", evaluator_label, repr(response))
        raw = response.content if hasattr(response, "content") else response
        if isinstance(raw, list):
            # LangChain 일부 드라이버가 list[{"type":"text","text":...}] 형식으로 반환 가능
            raw_text = " ".join(
                [str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in raw]
            )
        else:
            raw_text = str(raw) if raw is not None else ""
        parsed: ScoreList = parser.parse(raw_text)
        scores = []
        for sc in parsed.scores:
            acc = sc.accuracy
            comp = sc.completeness
            clar = sc.clarity
            if all(isinstance(v, (int, float)) for v in (acc, comp, clar)):
                weighted = round(float(acc) * 0.5 + float(comp) * 0.3 + float(clar) * 0.2, 2)
            elif isinstance(sc.score, (int, float)):
                weighted = float(sc.score)
            else:
                weighted = -1
            logger.debug(
                "가중치 점수 계산 evaluator=%s id=%s acc=%s comp=%s clar=%s weighted=%s",
                evaluator_label,
                sc.id,
                acc,
                comp,
                clar,
                weighted,
            )
            scores.append(
                {
                    "model": id_to_model.get(sc.id, sc.id),
                    "id": sc.id,
                    "score": weighted,
                    "rank": sc.rank,
                    "rationale": sc.rationale,
                    "accuracy": acc,
                    "completeness": comp,
                    "clarity": clar,
                }
            )
        elapsed_ms = int((time.perf_counter() - start_eval) * 1000)
        logger.info("평가 성공: evaluator=%s model=%s elapsed_ms=%s", evaluator_label, eval_model_name, elapsed_ms)
        logger.debug("_evaluate_answers:raw_response evaluator=%s body=%s", evaluator_label, raw_text)
        return {
            "evaluator": evaluator_label,
            "scores": scores,
            "status": {"status": 200, "detail": "success", "model": eval_model_name},
            "elapsed_ms": elapsed_ms,
        }
    except Exception as exc:
        logger.warning("평가 파싱 실패: %s", exc)
        fallback_scores = [
            {
                "model": r["model"],
                "id": f"resp_{i+1}",
                "score": -1,
                "rank": i + 1,
                "rationale": "parse failed",
            }
            for i, r in enumerate(results)
        ]
        elapsed_ms = int((time.perf_counter() - start_eval) * 1000)
        logger.debug("_evaluate_answers:예외 raw_response evaluator=%s error=%s", evaluator_label, exc)
        return {
            "evaluator": evaluator_label,
            "scores": fallback_scores,
            "status": {"status": "error", "detail": str(exc), "model": eval_model_name},
            "elapsed_ms": elapsed_ms,
        }


def _build_score_table(scores: list[dict[str, Any]]) -> str:
    """Markdown 테이블을 생성한다."""

    logger.debug("_build_score_table:시작 rows=%d", len(scores))
    header = "| Model | Score | Rank | Rationale |\n|---|---|---|---|"
    rows = [
        f"| {s.get('model','-')} | {s.get('score','-')} | {s.get('rank','-')} | {s.get('rationale','-')} |"
        for s in scores
    ]
    table = "\n".join([header] + rows)
    logger.debug("_build_score_table:종료")
    return table


def _aggregate_scores(results: list[dict[str, Any]], evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """여러 평가자의 점수를 모델별로 집계."""

    logger.debug("_aggregate_scores:시작 results=%d evaluations=%d", len(results), len(evaluations))
    try:
        targets = [r["model"] for r in results]
        aggregated: list[dict[str, Any]] = []
        for target in targets:
            collected_scores = []
            rationales = []
            for ev in evaluations:
                evaluator = ev.get("evaluator")
                for sc in ev.get("scores", []):
                    if sc.get("model") == target:
                        collected_scores.append(sc.get("score"))
                        if sc.get("rationale"):
                            rationales.append(f"[{evaluator}] {sc['rationale']}")
            valid_scores = [s for s in collected_scores if isinstance(s, (int, float)) and s >= 0]
            avg = round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else None
            logger.debug("집계 점수 target=%s scores=%s avg=%s", target, valid_scores, avg)
            aggregated.append(
                {
                    "model": target,
                    "score": avg if avg is not None else -1,
                    "raw_scores": collected_scores,
                    "rationale": "\n".join(rationales) if rationales else None,
                }
            )

        # 점수 내림차순 정렬 후 동점 허용 랭크 계산
        sorted_items = sorted(aggregated, key=lambda x: x["score"] if isinstance(x.get("score"), (int, float)) else -1, reverse=True)
        # dense rank
        last_score = None
        current_rank = 0
        for item in sorted_items:
            score_val = item.get("score")
            if score_val != last_score:
                current_rank += 1
                last_score = score_val
            item["rank"] = current_rank
        logger.debug("_aggregate_scores:종료 aggregated=%d", len(sorted_items))
        return sorted_items
    except Exception as exc:
        logger.warning("집계 실패: %s", exc)
        return [
            {
                "model": r["model"],
                "score": -1,
                "raw_scores": [],
                "rationale": "aggregation failed",
                "rank": idx + 1,
            }
            for idx, r in enumerate(results)
        ]


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
        logger.info("PromptEval 실행: models=%s", ", ".join(models))

        prompt_text = _build_model_prompt(question, prompt)
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
                "type": "partial",
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
                    _evaluate_answers(question, prompt_for_eval, success_results, evaluator_label, reference_answer)
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

        aggregated_scores = _aggregate_scores(results, evaluation_outputs)
        numeric_scores = [
            s["score"] for s in aggregated_scores if isinstance(s.get("score"), (int, float)) and s["score"] >= 0
        ]
        avg_score = round(sum(numeric_scores) / len(numeric_scores), 2) if numeric_scores else None
        elapsed_total_ms = int((time.perf_counter() - start) * 1000)

        yield {
            "type": "summary",
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
        logger.error("PromptEval 전체 실패: %s", exc)
        yield {"type": "error", "message": str(exc), "model": None, "node": "prompt_eval"}
    finally:
        logger.debug("stream_prompt_eval:종료 question=%s", question[:50] if question else "")


__all__ = ["stream_prompt_eval"]

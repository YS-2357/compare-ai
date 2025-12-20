"""Prompt-level 비교용 스트리밍 유틸리티.

질문 + 공통 프롬프트로 각 모델을 병렬 호출하고,
각 벤더의 최신 모델들이 모든 응답을 블라인드 크로스 평가한 뒤
partial(모델별 응답)과 summary(평가 결과 테이블)를 스트리밍한다.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from app.config import get_settings
from app.logger import get_logger
from app.services.chat_graph.errors import build_status_from_error, build_status_from_response
from app.services.chat_graph.llm_registry import (
    ChatAnthropic,
    ChatCohere,
    ChatGoogleGenerativeAI,
    ChatGroq,
    ChatMistralAI,
    ChatOpenAI,
    ChatPerplexity,
    ChatUpstage,
)
from app.services.shared import LATEST_EVAL_MODELS, MODEL_ALIASES

logger = get_logger(__name__)
settings_cache = get_settings()


class Score(BaseModel):
    """단일 답변 평가 결과."""

    id: str = Field(..., description="익명 응답 ID")
    score: float = Field(..., description="0~10 점수")
    rank: int = Field(..., description="순위 (1이 최고)")
    rationale: str = Field(..., description="짧은 근거")


class ScoreList(BaseModel):
    """평가 모델 출력 스키마."""

    scores: list[Score]


def _llm_factory(label: str) -> Any:
    """모델 라벨에 맞는 LLM 팩토리를 반환한다."""

    sc = settings_cache
    factories = {
        "OpenAI": lambda: ChatOpenAI(model=sc.model_openai),
        "Gemini": lambda: ChatGoogleGenerativeAI(model=sc.model_gemini, temperature=0),
        "Anthropic": lambda: ChatAnthropic(model=sc.model_anthropic),
        "Perplexity": lambda: ChatPerplexity(model=sc.model_perplexity),
        "Upstage": lambda: ChatUpstage(model=sc.model_upstage),
        "Mistral": lambda: ChatMistralAI(model=sc.model_mistral),
        "Groq": lambda: ChatGroq(model=sc.model_groq),
        "Cohere": lambda: ChatCohere(model=sc.model_cohere),
    }
    if label not in factories:
        raise ValueError(f"지원하지 않는 모델 라벨: {label}")
    return factories[label]()


def _select_eval_llm(active_labels: list[str]) -> tuple[Any, str]:
    """평가에 사용할 LLM과 모델명을 선택한다."""

    sc = settings_cache
    for label in active_labels:
        if label == "OpenAI":
            model_name = LATEST_EVAL_MODELS.get("OpenAI") or sc.model_openai
            return ChatOpenAI(model=model_name), model_name
        if label == "Gemini":
            model_name = LATEST_EVAL_MODELS.get("Gemini") or sc.model_gemini
            return ChatGoogleGenerativeAI(model=model_name, temperature=0), model_name
        if label == "Anthropic":
            model_name = LATEST_EVAL_MODELS.get("Anthropic") or sc.model_anthropic
            return ChatAnthropic(model=model_name), model_name
        if label == "Perplexity":
            model_name = LATEST_EVAL_MODELS.get("Perplexity") or sc.model_perplexity
            return ChatPerplexity(model=model_name), model_name
        if label == "Upstage":
            model_name = LATEST_EVAL_MODELS.get("Upstage") or sc.model_upstage
            return ChatUpstage(model=model_name), model_name
        if label == "Mistral":
            model_name = LATEST_EVAL_MODELS.get("Mistral") or sc.model_mistral
            return ChatMistralAI(model=model_name), model_name
        if label == "Groq":
            model_name = LATEST_EVAL_MODELS.get("Groq") or sc.model_groq
            return ChatGroq(model=model_name), model_name
        if label == "Cohere":
            model_name = LATEST_EVAL_MODELS.get("Cohere") or sc.model_cohere
            return ChatCohere(model=model_name), model_name

    # fallback: OpenAI 기본
    model_name = LATEST_EVAL_MODELS.get("OpenAI") or sc.model_openai
    return ChatOpenAI(model=model_name), model_name


DEFAULT_PROMPT = "[Question]\n{question}\n\n답변은 한국어로 작성하세요."


def _build_model_prompt(question: str, prompt: str | None) -> str:
    """모든 모델에 동일하게 적용할 프롬프트를 생성한다."""

    base = prompt or DEFAULT_PROMPT
    try:
        return base.format(question=question)
    except Exception:
        # 포맷 실패 시 안전하게 합치기
        return f"{base}\n\n[Question]\n{question}"


async def _call_single_model(label: str, prompt_text: str) -> dict[str, Any]:
    """단일 모델 호출 및 파싱 실패 대비."""

    start = time.perf_counter()
    try:
        llm = _llm_factory(label)
        response = await llm.ainvoke(prompt_text)
        status = build_status_from_response(response)
        content = getattr(response, "content", None) or str(response)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        logger.debug("%s 호출 성공 (elapsed_ms=%s)", label, elapsed_ms)
        return {
            "model": label,
            "answer": content,
            "status": status,
            "elapsed_ms": elapsed_ms,
            "source": None,
            "error": False,
        }
    except Exception as exc:
        status = build_status_from_error(exc)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        message = f"응답 실패: {status.get('detail') or exc}"
        logger.warning("%s 호출 실패: %s", label, message)
        return {
            "model": label,
            "answer": message,
            "status": status,
            "elapsed_ms": elapsed_ms,
            "source": None,
            "error": True,
        }


def _build_eval_prompt(question: str, anonymized: list[tuple[str, str]]) -> ChatPromptTemplate:
    """블라인드 평가 프롬프트를 생성한다."""

    examples = "\n".join([f"ID {anon_id}:\n{content}\n" for anon_id, content in anonymized])
    system = (
        "You are grading multiple anonymous answers to the same question.\n"
        "All answers are in Korean. Do NOT guess the original model/provider.\n"
        "Score each answer 0-10, rank them (1 is best), and give a brief rationale.\n"
        "Return JSON only with the provided schema."
    )
    user = f"Question:\n{question}\n\nAnswers:\n{examples}"
    parser = PydanticOutputParser(pydantic_object=ScoreList)
    instructions_raw = parser.get_format_instructions()
    instructions = instructions_raw.replace("{", "{{").replace("}", "}}")
    template = ChatPromptTemplate.from_messages(
        [
            ("system", system + "\n" + instructions),
            ("user", user),
        ]
    )
    return template


async def _evaluate_answers(
    question: str, results: list[dict[str, Any]], evaluator_label: str
) -> dict[str, Any]:
    """단일 평가 모델로 모든 응답을 블라인드 평가."""

    # 익명 ID 매핑
    anonymized = [(f"resp_{i+1}", r.get("answer", "")) for i, r in enumerate(results)]
    id_to_model = {f"resp_{i+1}": r["model"] for i, r in enumerate(results)}

    prompt = _build_eval_prompt(question, anonymized)
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
        raw = response.content if hasattr(response, "content") else response
        if isinstance(raw, list):
            # LangChain 일부 드라이버가 list[{"type":"text","text":...}] 형식으로 반환 가능
            raw_text = " ".join(
                [str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in raw]
            )
        else:
            raw_text = str(raw) if raw is not None else ""
        parsed: ScoreList = parser.parse(raw_text)
        scores = [
            {
                "model": id_to_model.get(sc.id, sc.id),
                "id": sc.id,
                "score": sc.score,
                "rank": sc.rank,
                "rationale": sc.rationale,
            }
            for sc in parsed.scores
        ]
        elapsed_ms = int((time.perf_counter() - start_eval) * 1000)
        logger.info("평가 성공: evaluator=%s model=%s elapsed_ms=%s", evaluator_label, eval_model_name, elapsed_ms)
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
        return {
            "evaluator": evaluator_label,
            "scores": fallback_scores,
            "status": {"status": "error", "detail": str(exc), "model": eval_model_name},
            "elapsed_ms": elapsed_ms,
        }


def _build_score_table(scores: list[dict[str, Any]]) -> str:
    """Markdown 테이블을 생성한다."""

    header = "| Model | Score | Rank | Rationale |\n|---|---|---|---|"
    rows = [
        f"| {s.get('model','-')} | {s.get('score','-')} | {s.get('rank','-')} | {s.get('rationale','-')} |"
        for s in scores
    ]
    return "\n".join([header] + rows)


def _aggregate_scores(results: list[dict[str, Any]], evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """여러 평가자의 점수를 모델별로 집계."""

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
            aggregated.append(
                {
                    "model": target,
                    "score": avg if avg is not None else -1,
                    "raw_scores": collected_scores,
                    "rationale": "\n".join(rationales) if rationales else None,
                }
            )

        # 순위 계산 (내림차순)
        sorted_items = sorted(
            aggregated,
            key=lambda x: x["score"] if isinstance(x.get("score"), (int, float)) else -1,
            reverse=True,
        )
        for idx, item in enumerate(sorted_items, start=1):
            item["rank"] = idx
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
) -> AsyncIterator[dict[str, Any]]:
    """모델별 프롬프트 적용 + 블라인드 평가를 스트리밍한다."""

    if not question or not question.strip():
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
        ]
        models: list[str] = []
        for m in raw_models:
            canonical = MODEL_ALIASES.get(m, m)
            if canonical not in models:
                models.append(canonical)
        logger.info("PromptEval 실행: models=%s", ", ".join(models))

        tasks = []
        for label in models:
            prompt_text = _build_model_prompt(question, prompt)
            tasks.append(asyncio.create_task(_call_single_model(label, prompt_text)))

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
                "elapsed_ms": res.get("elapsed_ms"),
            }

        # 평가 단계
        evaluation_tasks = [
            asyncio.create_task(_evaluate_answers(question, results, evaluator_label))
            for evaluator_label in models
        ]
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
                "scores": aggregated_scores,
                "avg_score": avg_score,
                "evaluations": evaluation_outputs,
                "elapsed_ms": elapsed_total_ms,
            },
        }
    except Exception as exc:
        logger.error("PromptEval 전체 실패: %s", exc)
        yield {"type": "error", "message": str(exc), "model": None, "node": "prompt_eval"}


__all__ = ["stream_prompt_eval"]

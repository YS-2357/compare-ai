"""평가 프롬프트 생성 및 실행."""

from __future__ import annotations

import time
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.utils.config import get_settings
from app.utils.logger import get_logger
from app.services.shared import load_prompt
from .schemas import ScoreList
from .clients import select_eval_llm

logger = get_logger(__name__)


def build_eval_prompt(
    question: str,
    prompt_text: str,
    anonymized: list[tuple[str, str]],
    reference: str | None,
) -> ChatPromptTemplate:
    """블라인드 평가 프롬프트를 생성한다."""

    logger.debug("build_eval_prompt:question preview=%s reference=%s", question[:200], bool(reference))
    examples = "\n".join([f"ID {anon_id}:\n{content}\n" for anon_id, content in anonymized])
    rubric_line = (
        "Use the reference answer to check factual alignment; if the wording differs but facts match, allow it."
        if reference
        else "Grade by accuracy, completeness, and clarity; do not penalize stylistic differences."
    )
    settings = get_settings()
    version = settings.prompt_compare_version
    system_template = load_prompt("prompt_compare_system", version)
    user_template = load_prompt("prompt_compare_user", version)
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
    logger.info("prompt_compare_version=%s", version)
    template = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("user", user),
        ]
    )
    return template


async def evaluate_answers(
    question: str,
    prompt_text: str,
    results: list[dict[str, Any]],
    evaluator_label: str,
    reference: str | None,
) -> dict[str, Any]:
    """단일 평가 모델로 모든 응답을 블라인드 평가."""

    logger.debug("evaluate_answers:시작 evaluator=%s answers=%d", evaluator_label, len(results))
    anonymized = [(f"resp_{i+1}", r.get("answer_with_sources") or r.get("answer", "")) for i, r in enumerate(results)]
    id_to_model = {f"resp_{i+1}": r["model"] for i, r in enumerate(results)}

    prompt = build_eval_prompt(question, prompt_text, anonymized, reference)
    parser = PydanticOutputParser(pydantic_object=ScoreList)
    start_eval = time.perf_counter()
    try:
        eval_llm, eval_model_name = select_eval_llm([evaluator_label])
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
        logger.debug("evaluate_answers:raw_response evaluator=%s raw=%s", evaluator_label, repr(response))
        raw = response.content if hasattr(response, "content") else response
        if isinstance(raw, list):
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
        logger.debug("evaluate_answers:raw_response evaluator=%s body=%s", evaluator_label, raw_text)
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
        logger.debug("evaluate_answers:예외 raw_response evaluator=%s error=%s", evaluator_label, exc)
        return {
            "evaluator": evaluator_label,
            "scores": fallback_scores,
            "status": {"status": "error", "detail": str(exc), "model": eval_model_name},
            "elapsed_ms": elapsed_ms,
        }

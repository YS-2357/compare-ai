"""프롬프트 평가용 LLM 생성/선택 유틸."""

from __future__ import annotations

import os
from typing import Any

from app.utils.config import get_settings
from app.utils.logger import get_logger
from app.services.shared import LATEST_EVAL_MODELS
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

logger = get_logger(__name__)
settings_cache = get_settings()


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


def build_deepseek_llm(model_name: str, base_url: str) -> ChatOpenAI:
    """DeepSeek OpenAI 호환 클라이언트를 생성한다."""

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing")
    return ChatOpenAI(model=model_name, api_key=api_key, base_url=base_url)


def build_perplexity_llm(model_name: str) -> ChatPerplexity:
    """Perplexity 클라이언트를 생성한다."""

    api_key = os.getenv("PPLX_API_KEY")
    if not api_key:
        raise RuntimeError("PPLX_API_KEY is missing")
    return ChatPerplexity(model=model_name, pplx_api_key=api_key)


def llm_factory(label: str, model_name: str | None = None) -> Any:
    """모델 라벨에 맞는 LLM 팩토리를 반환한다."""

    sc = settings_cache
    factories = {
        "OpenAI": lambda: ChatOpenAI(model=model_name or sc.model_openai),
        "Gemini": lambda: ChatGoogleGenerativeAI(model=model_name or sc.model_gemini, temperature=0),
        "Anthropic": lambda: ChatAnthropic(model=model_name or sc.model_anthropic),
        "Perplexity": lambda: build_perplexity_llm(model_name or sc.model_perplexity),
        "Upstage": lambda: ChatUpstage(model=model_name or sc.model_upstage),
        "Mistral": lambda: ChatMistralAI(model=model_name or sc.model_mistral),
        "Groq": lambda: ChatGroq(model=model_name or sc.model_groq),
        "Cohere": lambda: ChatCohere(model=model_name or sc.model_cohere),
        "DeepSeek": lambda: build_deepseek_llm(model_name or sc.model_deepseek, sc.deepseek_base_url),
    }
    if label not in factories:
        raise ValueError(f"지원하지 않는 모델 라벨: {label}")
    llm = factories[label]()
    return llm


def select_eval_llm(active_labels: list[str]) -> tuple[Any, str]:
    """평가에 사용할 LLM과 모델명을 선택한다."""

    logger.debug("select_eval_llm:시작 labels=%s", active_labels)
    sc = settings_cache
    for label in active_labels:
        if label == "OpenAI":
            model_name = LATEST_EVAL_MODELS.get("OpenAI") or sc.model_openai
            logger.info("select_eval_llm:선택 evaluator=%s model=%s", label, model_name)
            return ChatOpenAI(model=model_name), model_name
        if label == "Gemini":
            model_name = LATEST_EVAL_MODELS.get("Gemini") or sc.model_gemini
            logger.info("select_eval_llm:선택 evaluator=%s model=%s", label, model_name)
            return ChatGoogleGenerativeAI(model=model_name, temperature=0), model_name
        if label == "Anthropic":
            model_name = LATEST_EVAL_MODELS.get("Anthropic") or sc.model_anthropic
            logger.info("select_eval_llm:선택 evaluator=%s model=%s", label, model_name)
            return ChatAnthropic(model=model_name), model_name
        if label == "Perplexity":
            model_name = LATEST_EVAL_MODELS.get("Perplexity") or sc.model_perplexity
            logger.info("select_eval_llm:선택 evaluator=%s model=%s", label, model_name)
            return ChatPerplexity(model=model_name), model_name
        if label == "Upstage":
            model_name = LATEST_EVAL_MODELS.get("Upstage") or sc.model_upstage
            logger.info("select_eval_llm:선택 evaluator=%s model=%s", label, model_name)
            return ChatUpstage(model=model_name), model_name
        if label == "Mistral":
            model_name = LATEST_EVAL_MODELS.get("Mistral") or sc.model_mistral
            logger.info("select_eval_llm:선택 evaluator=%s model=%s", label, model_name)
            return ChatMistralAI(model=model_name), model_name
        if label == "Groq":
            model_name = LATEST_EVAL_MODELS.get("Groq") or sc.model_groq
            logger.info("select_eval_llm:선택 evaluator=%s model=%s", label, model_name)
            return ChatGroq(model=model_name), model_name
        if label == "Cohere":
            model_name = LATEST_EVAL_MODELS.get("Cohere") or sc.model_cohere
            logger.info("select_eval_llm:선택 evaluator=%s model=%s", label, model_name)
            return ChatCohere(model=model_name), model_name
        if label == "DeepSeek":
            model_name = LATEST_EVAL_MODELS.get("DeepSeek") or sc.model_deepseek
            logger.info("select_eval_llm:선택 evaluator=%s model=%s", label, model_name)
            return build_deepseek_llm(model_name, sc.deepseek_base_url), model_name

    model_name = LATEST_EVAL_MODELS.get("OpenAI") or sc.model_openai
    logger.info("select_eval_llm:기본 선택 evaluator=OpenAI model=%s", model_name)
    return ChatOpenAI(model=model_name), model_name


def build_model_prompt(question: str, prompt: str | None) -> str:
    """모든 모델에 동일하게 적용할 프롬프트를 생성한다."""

    base = prompt or DEFAULT_PROMPT
    try:
        result = base.format(question=question)
        return result
    except Exception:
        fallback = f"{base}\n\n[Question]\n{question}"
        logger.warning("build_model_prompt:포맷 실패, fallback 사용")
        return fallback

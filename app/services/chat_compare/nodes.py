"""LangGraph 노드 정의 및 LLM 호출 래퍼."""

from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph.message import add_messages
from langgraph.types import Send

from app.utils.config import get_settings
from app.utils.logger import get_logger
from app.services.shared.errors import build_status_from_error, build_status_from_response
from .prompts import Answer, build_chat_prompt, build_chat_prompt_input, message_to_text, render_chat_history

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
from .summaries import preview_text, summarize_content

logger = get_logger(__name__)
settings_cache = get_settings()
DEFAULT_MAX_TURNS = settings_cache.max_turns_default
MAX_CONTEXT_MESSAGES = settings_cache.max_context_messages  # 최근 메시지 유지 개수 (약 5턴: user/assistant 합산)


def merge_dicts(existing: dict | None, new: dict | None) -> dict:
    """LangGraph 상태 병합 시 딕셔너리를 병합한다."""

    merged: dict = dict(existing or {})
    merged.update(new or {})
    return merged


def merge_model_messages(existing: dict | None, new: dict | None) -> dict:
    """모델별 메시지 딕셔너리를 병합한다."""

    merged: dict[str, list] = dict(existing or {})
    for model, messages in (new or {}).items():
        merged[model] = add_messages(merged.get(model, []), messages or [])
    return merged


def resolve_model_name(state: GraphState, key: str, default: str) -> str:
    logger.debug("resolve_model_name:시작 key=%s", key)
    overrides = state.get("model_overrides") or {}
    override = overrides.get(key)
    resolved = override or default
    logger.info("resolve_model_name:결정 key=%s resolved=%s override=%s", key, resolved, bool(override))
    return resolved


class GraphState(TypedDict, total=False):
    """LangGraph 실행 시 공유되는 상태 정의."""

    max_turns: Annotated[int | None, "최대 턴 수"]
    turn: Annotated[int | None, "현재 턴 인덱스"]
    active_models: Annotated[list[str] | None, "활성화된 모델 목록"]

    # 공통 유저 발화 + 모델별 히스토리/요약
    user_messages: Annotated[list, add_messages]
    model_messages: Annotated[dict[str, list], merge_model_messages]
    model_summaries: Annotated[dict[str, str] | None, merge_dicts]
    model_overrides: Annotated[dict[str, str] | None, merge_dicts]

    # 호출 메타(필요 시 사용)
    raw_responses: Annotated[dict[str, str] | None, merge_dicts]
    raw_sources: Annotated[dict[str, str | None] | None, merge_dicts]
    response_meta: Annotated[dict[str, dict[str, Any] | None] | None, merge_dicts]
    api_status: Annotated[dict[str, Any] | None, merge_dicts]


def default_active_models() -> list[str]:
    models = list(NODE_CONFIG.keys())
    logger.debug("default_active_models:반환 count=%d", len(models))
    return models


def _model_label(node_name: str) -> str:
    meta = NODE_CONFIG.get(node_name)
    label = meta["label"] if meta else node_name
    logger.debug("model_label:node=%s label=%s", node_name, label)
    return label


async def call_model_common(
    label: str,
    state: GraphState,
    llm_factory: Callable[[], Any],
    *,
    message_transform: Callable[[str, str | None], str] | None = None,
) -> GraphState:
    """모델 호출 공통 루틴."""

    prompt_input = build_chat_prompt_input(state, label)
    logger.debug(
        "call_model_common:시작 label=%s turn=%s max_turns=%s models=%s",
        label,
        state.get("turn"),
        state.get("max_turns"),
        state.get("active_models"),
    )
    logger.debug("call_model_common:prompt_input preview=%s", preview_text(prompt_input))
    try:
        llm = llm_factory()
        content, source, status, response_meta = await invoke_parsed(llm, prompt_input, label)
        msg_payload = message_transform(content, source) if message_transform else content
        updated_msgs, summaries = await maybe_summarize_history(
            llm, state, label, [("assistant", msg_payload)], state.get("turn")
        )
        logger.debug("call_model_common:원본응답 label=%s body=%s", label, str(content))
        logger.info("call_model_common:성공 label=%s status=%s", label, status.get("status") if isinstance(status, dict) else status)
        return GraphState(
            model_messages={label: updated_msgs},
            model_summaries=summaries,
            api_status={label: status},
            raw_responses={label: content},
            raw_sources={label: source},
            response_meta={label: response_meta},
        )
    except Exception as exc:
        status = build_status_from_error(exc)
        logger.warning("call_model_common:실패 label=%s error=%s", label, exc)
        logger.debug("call_model_common:원본응답 실패 label=%s status=%s", label, status)
        error_msg = f"응답 실패: {status.get('detail') or exc}"
        return GraphState(
            api_status={label: status},
            model_messages={label: [format_response_message(label, error_msg)]},
            raw_responses={label: error_msg},
            raw_sources={label: None},
            response_meta={label: None},
        )
    finally:
        logger.debug("call_model_common:종료 label=%s", label)


async def maybe_summarize_history(
    llm: Any, state: GraphState, label: str, new_messages: list, turn: int | None
) -> tuple[list, dict[str, str]]:
    """
    모델별 히스토리를 갱신하고 필요 시 요약한다.
    - turn >= 2일 때 최근 MAX_CONTEXT_MESSAGES(기본 10) 기준으로 요약
    - 요약은 model_summaries[label]에 1개만 유지하고, 히스토리는 최근 메시지만 보존
    """

    logger.debug("maybe_summarize_history:시작 label=%s turn=%s", label, turn)
    existing = (state.get("model_messages") or {}).get(label, [])
    updated = add_messages(existing, new_messages)
    summaries = state.get("model_summaries") or {}

    # 최근 대화 10개만 살펴 요약(멀티턴 시)
    recent_messages = updated[-MAX_CONTEXT_MESSAGES:]
    should_summarize = (turn or 1) >= 2 and len(recent_messages) >= 3
    if should_summarize:
        text_lines = []
        for msg in recent_messages:
            msg_text = message_to_text(msg)
            if msg_text:
                text_lines.append(msg_text)
        history_text = "\n".join(text_lines)
        summary = await summarize_content(llm, history_text, label)
        summaries[label] = summary
        logger.info(
            "history/summarize label=%s turn=%s msgs_before=%d msgs_after=%d summary_len=%d preview=%s",
            label,
            turn,
            len(existing),
            len(recent_messages),
            len(summary),
            preview_text(summary),
        )
    # 컨텍스트 부풀림을 막기 위해 최근 메시지만 유지
    logger.debug("maybe_summarize_history:종료 label=%s msgs=%d summaries=%d", label, len(recent_messages), len(summaries))
    return recent_messages, summaries


def build_chat_prompt_input(state: GraphState, label: str) -> str:
    """
    대화 이력과 현재 질문을 분리해 전달한다.
    - history: 모델별 인터리브된 최근 히스토리
    - question: 최근 user 메시지(현재 질문)
    """
    user_messages = list(state.get("user_messages") or [])
    history_user_messages = list(user_messages)
    for idx in range(len(history_user_messages) - 1, -1, -1):
        msg = history_user_messages[idx]
        if isinstance(msg, (list, tuple)) and len(msg) == 2 and msg[0] == "user":
            del history_user_messages[idx]
            break
        if isinstance(msg, dict) and msg.get("role") == "user":
            del history_user_messages[idx]
            break
        if isinstance(msg, BaseMessage) and getattr(msg, "type", None) in ("human", "user"):
            del history_user_messages[idx]
            break
    history_state = dict(state)
    history_state["user_messages"] = history_user_messages
    history_text = render_chat_history(history_state, label, max_messages=MAX_CONTEXT_MESSAGES)
    current_question = ""
    logger.debug("build_chat_prompt_input:시작 label=%s user_msgs=%d", label, len(user_messages))
    for message in reversed(user_messages):
        # user 역할의 최신 메시지가 곧 현재 질문이다.
        if isinstance(message, (list, tuple)) and len(message) == 2 and message[0] == "user":
            current_question = str(message[1])
            break
        if isinstance(message, dict) and message.get("role") == "user":
            current_question = str(message.get("content", ""))
            break
        if isinstance(message, BaseMessage) and getattr(message, "type", None) in ("human", "user"):
            content = message.content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(str(item.get("text") or ""))
                    else:
                        parts.append(str(item))
                current_question = " ".join([p for p in parts if p])
            else:
                current_question = str(content)
            break
    if history_text.strip():
        prompt_text = (
            "[Conversation History]\n"
            f"{history_text}\n\n"
            "[Current Question]\n"
            f"{current_question}\n\n"
            "If anything is ambiguous, prefer the most recent topic or flow. Respond only in Korean."
        )
        mode = "with_history"
    else:
        prompt_text = (
            "[Current Question]\n"
            f"{current_question}\n\n"
            "This is the first turn; there is no prior conversation. "
            "Do not mention or reference any previous conversation; ask for clarification only if needed. "
            "Respond only in Korean."
        )
        mode = "first_turn"
    logger.debug("build_chat_prompt_input:종료 label=%s mode=%s question_preview=%s", label, mode, preview_text(current_question))
    return prompt_text


def _extract_source(extras: dict[str, Any] | None) -> str | None:
    """추가 메타에서 출처 URL을 추출한다."""

    logger.debug("extract_source:시작 extras=%s", bool(extras))
    def _maybe_url(value: Any) -> str | None:
        if isinstance(value, str):
            match = re.search(r"https?://\S+", value)
            if match:
                return match.group(0).rstrip(".,);]")
        if isinstance(value, dict):
            for key in ("url", "source", "link", "href"):
                candidate = value.get(key)
                if candidate:
                    found = _maybe_url(candidate)
                    if found:
                        return found
        return None

    if not extras:
        return None
    citations = extras.get("citations")
    if isinstance(citations, list):
        for item in citations:
            url = _maybe_url(item)
            if url:
                logger.info("extract_source:citations hit url=%s", url)
                return url
    search_results = extras.get("search_results")
    if isinstance(search_results, list):
        for item in search_results:
            url = _maybe_url(item)
            if url:
                logger.info("extract_source:search_results hit url=%s", url)
                return url
    sources = extras.get("sources")
    if isinstance(sources, list):
        for item in sources:
            url = _maybe_url(item)
            if url:
                logger.info("extract_source:sources hit url=%s", url)
                return url
    logger.debug("extract_source:종료 url=None")
    return None


def _extract_sources_list(response: Any) -> list[str]:
    """응답 객체에서 출처 URL 목록을 추출한다."""

    sources: list[str] = []

    def _maybe_add(value: Any) -> None:
        if isinstance(value, str) and value.startswith("http"):
            sources.append(value)
            return
        if isinstance(value, dict):
            for key in ("url", "source", "link", "href"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.startswith("http"):
                    sources.append(candidate)

    response_meta = getattr(response, "response_metadata", None)
    if isinstance(response_meta, dict):
        for key in ("citations", "sources", "search_results"):
            items = response_meta.get(key)
            if isinstance(items, list):
                for item in items:
                    _maybe_add(item)

    additional_kwargs = getattr(response, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        for key in ("citations", "sources", "search_results"):
            items = additional_kwargs.get(key)
            if isinstance(items, list):
                for item in items:
                    _maybe_add(item)

    raw_sources = getattr(response, "citations", None) or getattr(response, "sources", None)
    if isinstance(raw_sources, list):
        for item in raw_sources:
            _maybe_add(item)

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

    sources = _extract_sources_list(response)
    if sources:
        meta["sources"] = sources

    return meta


def _parse_answer_json(raw_text: str) -> Answer | None:
    """원문에서 JSON 블록을 찾아 Answer로 파싱한다."""

    if not raw_text:
        return None
    candidates = []
    if raw_text.strip().startswith("{") and raw_text.strip().endswith("}"):
        candidates.append(raw_text.strip())
    candidates.extend(re.findall(r"\{.*\}", raw_text, flags=re.DOTALL))
    for candidate in reversed(candidates):
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict) and ("content" in parsed or "source" in parsed):
            try:
                return Answer(**parsed)
            except Exception:
                continue
    return None


async def invoke_parsed(llm: Any, prompt_input: str, label: str) -> tuple[str, str | None, dict[str, Any], dict[str, Any]]:
    """LLM을 한 번 호출한 뒤 파서를 적용하고, 실패하면 원문을 그대로 사용한다."""

    logger.debug("invoke_parsed:시작 label=%s", label)
    parser = PydanticOutputParser(pydantic_object=Answer)
    prompt = build_chat_prompt()
    chain = prompt | llm
    response = await chain.ainvoke({"question": prompt_input})
    response_meta = _extract_response_meta(response)
    sources_list = response_meta.get("sources") if isinstance(response_meta, dict) else None
    status = build_status_from_response(response)
    raw_text = response.content if hasattr(response, "content") else str(response)
    if raw_text is None:
        raw_text = ""
    try:
        parsed: Answer = parser.parse(raw_text)
        content = parsed.content or raw_text
        source = parsed.source or _extract_source(getattr(parsed, "model_extra", None))
        if not source:
            source = _extract_source(getattr(response, "response_metadata", None))
        if not source and isinstance(sources_list, list) and sources_list:
            source = sources_list[0]
        logger.info("invoke_parsed:파싱 성공 label=%s status=%s source=%s", label, status.get("status"), source)
    except Exception:
        fallback = _parse_answer_json(raw_text)
        if fallback:
            content = fallback.content or raw_text
            source = fallback.source
            if not source:
                source = _extract_source(getattr(response, "response_metadata", None))
            if not source and isinstance(sources_list, list) and sources_list:
                source = sources_list[0]
            logger.info("invoke_parsed:JSON 추출 성공 label=%s", label)
        else:
            content = raw_text
            source = _extract_source(getattr(response, "response_metadata", None))
            if not source and isinstance(sources_list, list) and sources_list:
                source = sources_list[0]
            logger.warning("invoke_parsed:파싱 실패 label=%s 원문사용", label)
    logger.debug("invoke_parsed:종료 label=%s content_preview=%s", label, preview_text(content))
    return content, source, status, response_meta


def format_response_message(label: str, payload: Any) -> tuple[str, str]:
    """메시지 로그에 저장할 간단한 (role, content) 튜플을 생성한다."""

    logger.debug("format_response_message:생성 label=%s", label)
    return ("assistant", f"[{label}] {payload}")


def init_question(state: GraphState) -> GraphState:
    """그래프 초기 상태를 검증하고 기본 메시지를 설정한다."""

    logger.debug("init_question:시작")
    max_turns = state.get("max_turns") or DEFAULT_MAX_TURNS
    active_models = state.get("active_models") or list(NODE_CONFIG.keys())
    # 유저/모델 메시지 초기화
    user_messages = state.get("user_messages") or []
    if not user_messages:
        raise ValueError("질문이 비어 있습니다.")
    model_messages = state.get("model_messages") or {}
    model_summaries = state.get("model_summaries") or {}
    turn_value = state.get("turn") or 1

    last_user = next((msg for msg in reversed(user_messages) if isinstance(msg, (list, tuple, dict))), None)
    preview_target = ""
    if isinstance(last_user, (list, tuple)) and len(last_user) == 2:
        preview_target = last_user[1]
    elif isinstance(last_user, dict):
        preview_target = last_user.get("content", "")
    logger.debug("질문 초기화: %s", preview_text(preview_target))
    result = GraphState(
        max_turns=max_turns,
        turn=turn_value,
        active_models=active_models,
        raw_responses=state.get("raw_responses") or {},
        raw_sources=state.get("raw_sources") or {},
        response_meta=state.get("response_meta") or {},
        api_status=state.get("api_status") or {},
        user_messages=user_messages,
        model_messages=model_messages,
        model_summaries=model_summaries,
        model_overrides=state.get("model_overrides") or {},
    )
    logger.debug("init_question:종료 turn=%s max_turns=%s active_models=%d", turn_value, max_turns, len(active_models))
    return result


async def invoke_llm_async(llm: Any, question: str) -> Any:
    """주어진 LLM에서 비동기 호출을 수행한다."""

    logger.debug("invoke_llm_async:시작 question_preview=%s", preview_text(question))
    if hasattr(llm, "ainvoke"):
        result = await llm.ainvoke(question)
        logger.debug("invoke_llm_async:종료 mode=ainvoke")
        return result
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, llm.invoke, question)
    logger.debug("invoke_llm_async:종료 mode=executor")
    return result


async def call_openai(state: GraphState) -> GraphState:
    """OpenAI 모델을 호출하고 응답/상태를 반환한다."""

    settings = get_settings()
    model_name = resolve_model_name(state, "openai", settings.model_openai)
    logger.debug("call_openai:시작 model=%s", model_name)
    llm_factory = lambda: ChatOpenAI(model=model_name)
    result = await call_model_common("OpenAI", state, llm_factory)
    logger.debug("call_openai:종료")
    return result


async def call_gemini(state: GraphState) -> GraphState:
    """Google Gemini 모델을 호출한다."""

    settings = get_settings()
    model_name = resolve_model_name(state, "gemini", settings.model_gemini)
    logger.debug("call_gemini:시작 model=%s", model_name)
    llm_factory = lambda: ChatGoogleGenerativeAI(model=model_name, temperature=0)
    result = await call_model_common("Gemini", state, llm_factory)
    logger.debug("call_gemini:종료")
    return result


async def call_anthropic(state: GraphState) -> GraphState:
    """Anthropic Claude 모델을 호출한다."""

    settings = get_settings()
    model_name = resolve_model_name(state, "anthropic", settings.model_anthropic)
    logger.debug("call_anthropic:시작 model=%s", model_name)
    llm_factory = lambda: ChatAnthropic(model=model_name, temperature=0)
    result = await call_model_common("Anthropic", state, llm_factory)
    logger.debug("call_anthropic:종료")
    return result


async def call_upstage(state: GraphState) -> GraphState:
    """Upstage Solar 모델을 호출한다."""

    settings = get_settings()
    model_name = resolve_model_name(state, "upstage", settings.model_upstage)
    logger.debug("call_upstage:시작 model=%s", model_name)
    llm_factory = lambda: ChatUpstage(model=model_name)
    result = await call_model_common("Upstage", state, llm_factory)
    logger.debug("call_upstage:종료")
    return result


async def call_perplexity(state: GraphState) -> GraphState:
    """Perplexity Sonar 모델을 호출한다."""

    def llm_factory() -> Any:
        pplx_api_key = os.getenv("PPLX_API_KEY")
        if not pplx_api_key:
            raise RuntimeError("PPLX_API_KEY is missing")
        settings = get_settings()
        model_name = resolve_model_name(state, "perplexity", settings.model_perplexity)
        return ChatPerplexity(temperature=0, model=model_name, pplx_api_key=pplx_api_key)

    msg_transform = lambda content, source: content if not source else f"{content} (src: {source})"
    logger.debug("call_perplexity:시작")
    result = await call_model_common("Perplexity", state, llm_factory, message_transform=msg_transform)
    logger.debug("call_perplexity:종료")
    return result


async def call_mistral(state: GraphState) -> GraphState:
    """Mistral AI 모델을 호출한다."""

    def llm_factory() -> Any:
        if ChatMistralAI is None:
            raise RuntimeError("langchain-mistralai 패키지가 설치되어 있지 않습니다.")
        settings = get_settings()
        model_name = resolve_model_name(state, "mistral", settings.model_mistral)
        return ChatMistralAI(model=model_name, temperature=0)

    logger.debug("call_mistral:시작")
    result = await call_model_common("Mistral", state, llm_factory)
    logger.debug("call_mistral:종료")
    return result


async def call_groq(state: GraphState) -> GraphState:
    """Groq 기반 모델을 호출한다."""

    def llm_factory() -> Any:
        if ChatGroq is None:
            raise RuntimeError("langchain-groq 패키지가 설치되어 있지 않습니다.")
        settings = get_settings()
        model_name = resolve_model_name(state, "groq", settings.model_groq)
        return ChatGroq(model=model_name, temperature=0)

    logger.debug("call_groq:시작")
    result = await call_model_common("Groq", state, llm_factory)
    logger.debug("call_groq:종료")
    return result


async def call_cohere(state: GraphState) -> GraphState:
    """Cohere Command 모델을 호출한다."""

    def llm_factory() -> Any:
        if ChatCohere is None:
            raise RuntimeError("langchain-cohere 패키지가 설치되어 있지 않습니다.")
        settings = get_settings()
        model_name = resolve_model_name(state, "cohere", settings.model_cohere)
        return ChatCohere(model=model_name, temperature=0)

    logger.debug("call_cohere:시작")
    result = await call_model_common("Cohere", state, llm_factory)
    logger.debug("call_cohere:종료")
    return result


async def call_deepseek(state: GraphState) -> GraphState:
    """DeepSeek 모델을 호출한다."""

    def llm_factory() -> Any:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is missing")
        settings = get_settings()
        model_name = resolve_model_name(state, "deepseek", settings.model_deepseek)
        return ChatOpenAI(model=model_name, api_key=api_key, base_url=settings.deepseek_base_url)

    logger.debug("call_deepseek:시작")
    result = await call_model_common("DeepSeek", state, llm_factory)
    logger.debug("call_deepseek:종료")
    return result


NODE_CONFIG: dict[str, dict[str, str]] = {
    "call_openai": {"label": "OpenAI", "answer_key": "openai_answer", "status_key": "openai_status"},
    "call_gemini": {"label": "Gemini", "answer_key": "gemini_answer", "status_key": "gemini_status"},
    "call_anthropic": {"label": "Anthropic", "answer_key": "anthropic_answer", "status_key": "anthropic_status"},
    "call_perplexity": {"label": "Perplexity", "answer_key": "perplexity_answer", "status_key": "perplexity_status"},
    "call_upstage": {"label": "Upstage", "answer_key": "upstage_answer", "status_key": "upstage_status"},
    "call_mistral": {"label": "Mistral", "answer_key": "mistral_answer", "status_key": "mistral_status"},
    "call_groq": {"label": "Groq", "answer_key": "groq_answer", "status_key": "groq_status"},
    "call_cohere": {"label": "Cohere", "answer_key": "cohere_answer", "status_key": "cohere_status"},
    "call_deepseek": {"label": "DeepSeek", "answer_key": "deepseek_answer", "status_key": "deepseek_status"},
}


def dispatch_llm_calls(state: GraphState) -> list[Send]:
    """Send API를 활용해 각 LLM 노드를 동시에 실행할 태스크 목록을 생성한다."""

    logger.debug("dispatch_llm_calls:시작")
    user_messages = state.get("user_messages") or []
    if not user_messages:
        raise ValueError("질문이 비어 있습니다.")
    active_models = state.get("active_models") or default_active_models()
    preview_target = ""
    last_user = next((msg for msg in reversed(user_messages) if isinstance(msg, (list, tuple, dict))), None)
    if isinstance(last_user, (list, tuple)) and len(last_user) == 2:
        preview_target = last_user[1]
    elif isinstance(last_user, dict):
        preview_target = last_user.get("content", "")
    logger.info("LLM fan-out 실행: %s | 질문: %s", ", ".join(active_models), preview_text(preview_target))
    sends = [Send(node_name, state) for node_name in active_models]
    logger.debug("dispatch_llm_calls:종료 tasks=%d", len(sends))
    return sends


__all__ = [
    "GraphState",
    "NODE_CONFIG",
    "DEFAULT_MAX_TURNS",
    "dispatch_llm_calls",
    "init_question",
    "invoke_llm_async",
    "preview_text",
    "call_openai",
    "call_gemini",
    "call_anthropic",
    "call_upstage",
    "call_perplexity",
    "call_mistral",
    "call_groq",
    "call_cohere",
    "call_deepseek",
    "format_response_message",
    "resolve_model_name",
]

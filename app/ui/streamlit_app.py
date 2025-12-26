"""Streamlit UI 엔트리포인트."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv
from app.utils.logger import get_logger

load_dotenv()
logger = get_logger(__name__)

FASTAPI_URL_FILE = Path(__file__).resolve().parents[2] / ".fastapi_url"
DEFAULT_FASTAPI_BASE = FASTAPI_URL_FILE.read_text().strip() if FASTAPI_URL_FILE.exists() else ""
MODEL_OPTIONS: dict[str, dict[str, Any]] = {
    "openai": {
        "label": "OpenAI",
        "env": "MODEL_OPENAI",
        "choices": [
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "gpt-4o",
            "gpt-4o-mini",
            "chatgpt-4o-latest",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5-pro",
            "gpt-5-chat-latest",
            "gpt-5.1",
            "gpt-5.2",
            "gpt-5.2-pro",
            "gpt-5.2-chat-latest",
            "o1",
            "o1-pro",
            "o3",
            "o3-mini",
            "o4-mini",
        ],
    },
    "gemini": {
        "label": "Google Gemini",
        "env": "MODEL_GEMINI",
        "choices": [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-2.0-flash-001",
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash-lite-001",
            "gemini-2.5-flash-lite",
        ],
    },
    "anthropic": {
        "label": "Anthropic Claude",
        "env": "MODEL_ANTHROPIC",
        "choices": [
            "claude-3-haiku-20240307",
            "claude-3-5-haiku-20241022",
            "claude-3-7-sonnet-20250219",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-20250514",
            "claude-opus-4-1-20250805",
            "claude-opus-4-5-20251101",
            "claude-sonnet-4-20250514",
            "claude-sonnet-4-5-20250929",
        ],
    },
    "upstage": {
        "label": "Upstage Solar",
        "env": "MODEL_UPSTAGE",
        "choices": [
            "solar-mini",
            "solar-mini-250422",
            "solar-pro",
            "solar-pro-250422",
            "solar-pro2",
            "solar-pro2-250909",
            "solar-pro2-250909-correction",
            "solar-pro2-251215",
            "solar-pro2-nightly",
            "solar-pro2-sagemaker",
            "syn-pro",
            "syn-pro-251021",
            "tripod",
            "tripod-test",
        ],
    },
    "perplexity": {
        "label": "Perplexity Sonar",
        "env": "MODEL_PERPLEXITY",
        "choices": ["sonar", "sonar-pro", "sonar-reasoning-pro", "sonar-deep-research"],
    },
    "mistral": {
        "label": "Mistral",
        "env": "MODEL_MISTRAL",
        "choices": [
            "codestral-2411-rc5",
            "codestral-2412",
            "codestral-2501",
            "codestral-2508",
            "codestral-latest",
            "devstral-2512",
            "devstral-latest",
            "devstral-medium-2507",
            "devstral-medium-latest",
            "devstral-small-2507",
            "devstral-small-latest",
            "labs-devstral-small-2512",
            "labs-mistral-small-creative",
            "magistral-medium-2509",
            "magistral-medium-latest",
            "magistral-small-2509",
            "magistral-small-latest",
            "ministral-14b-2512",
            "ministral-14b-latest",
            "ministral-3b-2410",
            "ministral-3b-2512",
            "ministral-3b-latest",
            "ministral-8b-2410",
            "ministral-8b-2512",
            "ministral-8b-latest",
            "mistral-large-2411",
            "mistral-large-2512",
            "mistral-large-latest",
            "mistral-large-pixtral-2411",
            "mistral-medium",
            "mistral-medium-2505",
            "mistral-medium-2508",
            "mistral-medium-latest",
            "mistral-small-2501",
            "mistral-small-2506",
            "mistral-small-latest",
            "mistral-tiny",
            "mistral-tiny-2312",
            "mistral-tiny-2407",
            "mistral-tiny-latest",
            "mistral-vibe-cli-latest",
            "open-mistral-7b",
            "open-mistral-nemo",
            "open-mistral-nemo-2407",
            "pixtral-12b",
            "pixtral-12b-2409",
            "pixtral-12b-latest",
            "pixtral-large-2411",
            "pixtral-large-latest",
        ],
    },
    "groq": {
        "label": "Groq",
        "env": "MODEL_GROQ",
        "choices": [
            "allam-2-7b",
            "groq/compound",
            "groq/compound-mini",
            "llama-3.1-8b-instant",
            "llama-3.3-70b-versatile",
            "meta-llama/llama-4-maverick-17b-128e-instruct",
            "meta-llama/llama-4-scout-17b-16e-instruct",
            "moonshotai/kimi-k2-instruct",
            "moonshotai/kimi-k2-instruct-0905",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3-32b",
        ],
    },
    "cohere": {
        "label": "Cohere",
        "env": "MODEL_COHERE",
        "choices": [
            "c4ai-aya-expanse-32b",
            "c4ai-aya-expanse-8b",
            "c4ai-aya-vision-32b",
            "c4ai-aya-vision-8b",
            "command-a-vision-07-2025",
            "command-r-08-2024",
            "command-r7b-12-2024",
            "command-r7b-arabic-02-2025",
        ],
    },
    "deepseek": {
        "label": "DeepSeek",
        "env": "MODEL_DEEPSEEK",
        "choices": ["deepseek-chat", "deepseek-reasoner"],
    },
}

st.set_page_config(page_title="Compare-AI", page_icon="🤖", layout="wide")


def _default_model(provider: str) -> str:
    meta = MODEL_OPTIONS[provider]
    env_value = os.getenv(meta["env"])
    if env_value:
        _log_model_default_if_changed(provider, env_value, "환경변수")
        return env_value
    provider_defaults = {
        "openai": "gpt-4.1-mini",
        "gemini": "gemini-2.5-flash-lite",
        "anthropic": "claude-3-5-haiku-20241022",
        "upstage": "solar-mini",
        "perplexity": "sonar",
        "mistral": "mistral-small-latest",
        "groq": "llama-3.3-70b-versatile",
        "cohere": "command-r-08-2024",
        "deepseek": "deepseek-chat",
    }
    preferred = provider_defaults.get(provider)
    if preferred and preferred in meta["choices"]:
        _log_model_default_if_changed(provider, preferred, "기본")
        return preferred
    # 가벼운/저렴한 모델을 기본값으로 선택(리스트에 없으면 첫 번째)
    cheap_candidates = [
        "gpt-4.1-mini",
        "gpt-5-nano",
        "gemini-2.5-flash-lite",
        "claude-3-haiku",
        "solar-mini",
        "sonar-small",
        "mistral-small-3.2",
        "ministral-3-8b",
        "llama-3.3-70b-versatile",
        "command-light",
        "deepseek-chat",
    ]
    for candidate in cheap_candidates:
        if candidate in meta["choices"]:
            _log_model_default_if_changed(provider, candidate, "저가형")
            return candidate
    chosen = meta["choices"][0]
    _log_model_default_if_changed(provider, chosen, "기본")
    return chosen


def _log_model_default_if_changed(provider: str, model: str, source: str) -> None:
    """모델 기본값이 바뀔 때만 로그를 남긴다."""

    cache = st.session_state.setdefault("default_model_cache", {})
    last = cache.get(provider)
    if last == model:
        return
    cache[provider] = model
    logger.info("_default_model:선택 provider=%s model=%s source=%s", provider, model, source)


def _ensure_model_selections() -> None:
    if not st.session_state.get("_log_model_selections_once"):
        logger.debug("_ensure_model_selections:시작")
        st.session_state["_log_model_selections_once"] = True
    defaults = {key: _default_model(key) for key in MODEL_OPTIONS}
    selections = st.session_state.get("model_selections") or {}
    locked = st.session_state.get("model_selections_locked") or {}
    enabled = st.session_state.get("model_enabled") or {}
    merged = {}
    for key, default in defaults.items():
        if locked.get(key):
            merged[key] = selections.get(key, default)
        else:
            merged[key] = default
        enabled.setdefault(key, True)
    st.session_state["model_selections"] = merged
    st.session_state["model_selections_locked"] = locked
    st.session_state["model_enabled"] = enabled
    if not st.session_state.get("_log_model_selections_done"):
        logger.debug("_ensure_model_selections:종료 selections=%s", merged)
        st.session_state["_log_model_selections_done"] = True


def _render_model_selector() -> None:
    if not st.session_state.get("_log_model_selector_once"):
        logger.debug("_render_model_selector:시작 model_options=%s", list(MODEL_OPTIONS.keys()))
        st.session_state["_log_model_selector_once"] = True
    _ensure_model_selections()
    st.subheader("모델 선택")
    for key, meta in MODEL_OPTIONS.items():
        options = list(meta["choices"])
        current = st.session_state["model_selections"].get(key, _default_model(key))
        if current not in options:
            options = [current] + options
        index = options.index(current) if current in options else 0
        col_toggle, col_select = st.columns([1, 3])
        enabled = col_toggle.toggle(
            "ON",
            value=st.session_state["model_enabled"].get(key, True),
            key=f"model_toggle_{key}",
        )
        st.session_state["model_enabled"][key] = enabled
        selection = col_select.selectbox(
            f"{meta['label']} 모델",
            options,
            index=index,
            key=f"model_select_{key}",
        )
        if selection != current:
            st.session_state.setdefault("model_selections_locked", {})[key] = True
        st.session_state["model_selections"][key] = selection
    if not st.session_state.get("_log_model_selector_done"):
        logger.debug("_render_model_selector:종료 selections=%s", st.session_state.get("model_selections"))
        st.session_state["_log_model_selector_done"] = True


def _load_base_url() -> str:
    if not st.session_state.get("_log_load_base_url_once"):
        logger.debug("_load_base_url:시작")
        st.session_state["_log_load_base_url_once"] = True
    saved = (
        st.session_state.get("fastapi_base_url")
        or DEFAULT_FASTAPI_BASE
        or os.getenv("FASTAPI_URL", "")
        or st.secrets.get("FASTAPI_URL", "")
    )
    if saved.endswith("/api/ask"):
        base = saved.rsplit("/api/ask", 1)[0]
    else:
        base = saved
    last = st.session_state.get("_log_load_base_url_value")
    if last != base:
        logger.debug("_load_base_url:종료 base=%s", base)
        st.session_state["_log_load_base_url_value"] = base
    return base


def _get_usage_limit() -> str:
    value = os.getenv("DAILY_USAGE_LIMIT") or "3"
    logger.debug("_get_usage_limit:limit=%s", value)
    return value


def _usage_limit_int() -> int:
    try:
        result = int(_get_usage_limit())
        logger.debug("_usage_limit_int:성공 value=%s", result)
        return result
    except Exception:
        logger.warning("_usage_limit_int:변환 실패, 기본값 3 사용")
        return 3


def _sync_usage_from_headers(resp: requests.Response) -> None:
    """서버 헤더의 사용량 정보를 세션에 반영한다."""

    limit = resp.headers.get("X-Usage-Limit")
    remaining = resp.headers.get("X-Usage-Remaining")
    if limit is not None and limit.isdigit():
        st.session_state["usage_limit"] = int(limit)
    if remaining is not None and remaining.isdigit():
        st.session_state["usage_remaining"] = int(remaining)
    logger.debug(
        "_sync_usage_from_headers:limit=%s remaining=%s", st.session_state.get("usage_limit"), st.session_state.get("usage_remaining")
    )


def _build_history_payload(chat_log: list[dict[str, Any]]) -> list[dict[str, str]]:
    """기존 대화 로그를 LangGraph history 페이로드로 변환한다."""

    logger.debug("_build_history_payload:시작 entries=%d", len(chat_log or []))
    history_payload: list[dict[str, str]] = []
    for entry in chat_log or []:
        q = entry.get("question")
        if q:
            history_payload.append({"role": "user", "content": q})
        model_answers: dict[str, str] = {}
        for model, ans in (entry.get("answers") or {}).items():
            if ans:
                model_answers[model] = ans
        if not model_answers:
            for ev in entry.get("events") or []:
                model = ev.get("model")
                ans = ev.get("answer")
                if model and ans and model not in model_answers:
                    model_answers[model] = ans
        for model, ans in model_answers.items():
            history_payload.append({"role": "assistant", "model": model, "content": ans})
    logger.debug("_build_history_payload:종료 payload_len=%d", len(history_payload))
    return history_payload


def _update_usage_after_response(resp: requests.Response, *, admin_mode: bool) -> None:
    """응답 이후 사용량 카운터를 갱신한다."""

    logger.debug("_update_usage_after_response:시작 admin_mode=%s status=%s", admin_mode, resp.status_code)
    if admin_mode:
        st.session_state["usage_remaining"] = None
        return
    if resp.status_code == 429:
        st.session_state["usage_remaining"] = 0
    elif resp.ok:
        if "X-Usage-Remaining" not in resp.headers:
            new_value = max(0, st.session_state.get("usage_remaining", _usage_limit_int()) - 1)
            st.session_state["usage_remaining"] = new_value
    logger.debug(
        "_update_usage_after_response:종료 usage_remaining=%s", st.session_state.get("usage_remaining")
    )


def _append_chat_log_entry(
    question: str,
    answers: dict[str, str],
    sources: dict[str, str | None],
    events: list[dict[str, Any]],
) -> None:
    """대화 로그에 새 엔트리를 추가한다."""

    logger.debug("_append_chat_log_entry:시작 question=%s answers=%d events=%d", question, len(answers), len(events))
    st.session_state["chat_log"].append(
        {
            "question": question,
            "answers": answers,
            "sources": sources,
            "events": events,
        }
    )
    logger.debug("_append_chat_log_entry:종료 total=%d", len(st.session_state["chat_log"]))


def _status_to_emoji(status_val: Any) -> str:
    """상태 코드/문자열을 이모지로 변환한다."""

    code = None
    if isinstance(status_val, dict):
        code = status_val.get("status")
    elif isinstance(status_val, (int, str)):
        code = status_val

    if isinstance(code, str):
        code_lower = code.lower()
        if code_lower.isdigit():
            code = int(code_lower)
        elif "error" in code_lower or "fail" in code_lower or "exception" in code_lower:
            return "❌"
        elif "timeout" in code_lower or "rate" in code_lower:
            return "⚠️"
        elif "ok" in code_lower or "success" in code_lower:
            return "✅"

    try:
        code_int = int(code) if code is not None else None
    except Exception:
        code_int = None
    if code_int is None:
        return "❔"
    if code_int >= 500:
        return "❌"
    if code_int >= 400:
        return "⚠️"
    return "✅"


def _is_error_status(status_val: Any) -> bool:
    """상태 코드/문자열이 오류인지 판별한다."""

    code = None
    if isinstance(status_val, dict):
        code = status_val.get("status")
    elif isinstance(status_val, (int, str)):
        code = status_val

    if isinstance(code, str):
        lower = code.lower()
        if lower.isdigit():
            code = int(lower)
        elif "error" in lower or "fail" in lower or "exception" in lower:
            return True
        elif "timeout" in lower or "rate" in lower:
            return True

    try:
        code_int = int(code) if code is not None else None
    except Exception:
        code_int = None
    if code_int is None:
        return False
    result = code_int >= 400
    return result


def _format_response_meta(meta: dict[str, Any] | None) -> str | None:
    """응답 메타를 한 줄 요약으로 만든다."""

    if not meta:
        return None
    parts: list[str] = []
    model_name = meta.get("model_name")
    if model_name:
        parts.append(f"모델: {model_name}")
    finish_reason = meta.get("finish_reason") or meta.get("stop_reason")
    if finish_reason:
        parts.append(f"종료: {finish_reason}")
    refusal = meta.get("refusal")
    if refusal:
        parts.append(f"거부: {refusal}")
    prompt_feedback = meta.get("prompt_feedback")
    if isinstance(prompt_feedback, dict):
        block_reason = prompt_feedback.get("block_reason")
        if block_reason is not None:
            parts.append(f"안전피드백: {block_reason}")
    token_usage = meta.get("token_usage")
    if isinstance(token_usage, dict):
        input_tokens = token_usage.get("input_tokens")
        output_tokens = token_usage.get("output_tokens")
        total_tokens = token_usage.get("total_tokens")
        if any(v is not None for v in (input_tokens, output_tokens, total_tokens)):
            parts.append(f"토큰: {input_tokens}/{output_tokens}/{total_tokens}")
    if not parts:
        return None
    return " | ".join(parts)


def _render_sources_from_meta(meta: dict[str, Any] | None) -> bool:
    """메타에 포함된 출처를 표시하고, 표시 여부를 반환한다."""

    if not meta:
        return False
    sources = meta.get("sources")
    if not isinstance(sources, list) or not sources:
        return False
    st.caption("출처:")
    st.markdown("\n".join(f"- {src}" for src in sources))
    return True


def _render_auth_section(base_url: str) -> None:
    """로그인/회원가입 UI를 렌더링한다."""

    if not st.session_state.get("_log_auth_section_once"):
        logger.debug("_render_auth_section:시작 base_url=%s", base_url)
        st.session_state["_log_auth_section_once"] = True
    st.header("로그인 또는 회원가입")
    email = st.text_input("이메일")
    password = st.text_input("비밀번호", type="password")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("회원가입"):
            if not base_url:
                st.error("FastAPI Base URL을 입력하세요.")
            elif not email or not password:
                st.warning("이메일/비밀번호를 입력하세요.")
            else:
                try:
                    resp = requests.post(
                        f"{base_url}/auth/register",
                        json={"email": email, "password": password},
                        timeout=15,
                    )
                    data = resp.json()
                    st.write(f"회원가입 상태: {resp.status_code}")
                    if resp.ok:
                        st.success("회원가입이 완료되었습니다.")
                        st.json(data)
                    else:
                        st.error(data.get("detail") or "회원가입에 실패했습니다.")
                except Exception as exc:
                    logger.error("_render_auth_section:회원가입 실패 email=%s err=%s", email, exc)
                    st.error("회원가입 요청 중 오류가 발생했습니다.")
    with col2:
        if st.button("로그인"):
            if not base_url:
                st.error("FastAPI Base URL을 입력하세요.")
            elif not email or not password:
                st.warning("이메일/비밀번호를 입력하세요.")
            else:
                try:
                    resp = requests.post(
                        f"{base_url}/auth/login",
                        json={"email": email, "password": password},
                        timeout=15,
                    )
                    data = resp.json()
                    st.write(f"로그인 상태: {resp.status_code}")
                    if resp.ok and data.get("access_token"):
                        token = f"{data.get('token_type', 'bearer')} {data['access_token']}"
                        st.session_state["auth_token"] = token
                        st.session_state["auth_user"] = data.get("user")
                        st.success("로그인 성공: 토큰 저장 완료")
                        st.rerun()
                    elif not resp.ok:
                        st.error(data.get("detail") or "로그인에 실패했습니다.")
                    st.json(data)
                except Exception as exc:
                    logger.error("_render_auth_section:로그인 실패 email=%s err=%s", email, exc)
                    st.error("로그인 요청 중 오류가 발생했습니다.")
    if not st.session_state.get("_log_auth_section_done"):
        logger.debug("_render_auth_section:종료")
        st.session_state["_log_auth_section_done"] = True
    st.stop()


def _render_chat_history(chat_log: list[dict[str, Any]]) -> None:
    """기존 대화 로그를 챗봇 형식으로 표시한다."""

    logger.debug("_render_chat_history:시작 entries=%d", len(chat_log or []))
    if not chat_log:
        st.info("아직 대화가 없습니다. 질문을 입력해보세요.")
        return

    for item in chat_log:
        with st.chat_message("user"):
            st.write(item.get("question"))
        answers = item.get("answers") or {}
        sources = item.get("sources") or {}
        events = item.get("events") or []
        # 모델별 상태/시간 메타 구성
        event_meta: dict[str, dict[str, Any]] = {}
        for ev in events:
            model = ev.get("model")
            if not model:
                continue
            event_meta[model] = {
                "status": ev.get("status"),
                "elapsed_ms": ev.get("elapsed_ms"),
                "response_meta": ev.get("response_meta"),
            }
        with st.chat_message("assistant"):
            if answers:
                for model, answer in answers.items():
                    meta = event_meta.get(model) or {}
                    status = meta.get("status")
                    elapsed_ms = meta.get("elapsed_ms")
                    response_meta = meta.get("response_meta")
                    emoji = _status_to_emoji(status)
                    elapsed_txt = f"{elapsed_ms/1000:.1f}s" if elapsed_ms is not None else "-"
                    st.markdown(f"{emoji} **{model}** ⏱️ {elapsed_txt}")
                    st.write(answer)
                    meta_text = _format_response_meta(response_meta)
                    if meta_text:
                        st.caption(meta_text)
                    src = sources.get(model)
                    if not _render_sources_from_meta(response_meta):
                        if model == "Perplexity":
                            st.caption(f"출처: {src or '제공되지 않음'}")
                        elif src:
                            st.caption(f"출처: {src}")
            elif events:
                st.caption("응답 스트림")
                for ev in events:
                    model = ev.get("model") or "unknown"
                    ans = ev.get("answer")
                    src = ev.get("source")
                    status = ev.get("status") or {}
                    elapsed = ev.get("elapsed_ms")
                    response_meta = ev.get("response_meta")
                    elapsed_txt = f"{elapsed/1000:.1f}s" if elapsed is not None else "-"
                    emoji = _status_to_emoji(status)
                    st.write(f"{emoji} [{model}] {ans}")
                    st.caption(f"⏱️ {elapsed_txt}")
                    meta_text = _format_response_meta(response_meta)
                    if meta_text:
                        st.caption(meta_text)
                    if not _render_sources_from_meta(response_meta):
                        if model == "Perplexity":
                            st.caption(f"출처: {src or '제공되지 않음'}")
                        elif src:
                            st.caption(f"출처: {src}")
    logger.debug("_render_chat_history:종료")


def _render_connection_status(base_url: str) -> None:
    """API 연결 상태를 간단히 표시한다."""

    if not st.session_state.get("_log_connection_status_once"):
        logger.debug("_render_connection_status:시작 base_url=%s", base_url)
        st.session_state["_log_connection_status_once"] = True
    status_box = st.empty()
    if not base_url:
        status_box.warning("FastAPI URL을 입력하세요.")
        return
    with st.spinner("API 연결 확인 중..."):
        try:
            resp = requests.get(f"{base_url}/health", timeout=5)
            if resp.ok:
                status_box.success("✅ API 연결됨")
            else:
                status_box.error(f"❌ API 응답 오류 ({resp.status_code})")
        except Exception as exc:  # pragma: no cover - UI 통신 예외
            status_box.error(f"❌ 연결 실패: {exc}")
            logger.error("_render_connection_status:실패 err=%s", exc)
    if not st.session_state.get("_log_connection_status_done"):
        logger.debug("_render_connection_status:종료")
        st.session_state["_log_connection_status_done"] = True


def _handle_logout() -> None:
    """로그아웃 처리."""

    logger.debug("_handle_logout:시작")
    st.session_state.pop("auth_token", None)
    st.session_state.pop("auth_user", None)
    st.session_state.pop("usage_remaining", None)
    st.session_state.pop("usage_bypass", None)
    st.session_state.pop("usage_fetched", None)
    st.session_state.pop("chat_log", None)
    st.rerun()


def _send_question(
    question: str,
    ask_url: str,
    headers: dict[str, str],
    turn_value: int,
    history_payload: list[dict[str, str]],
    model_overrides: dict[str, str] | None = None,
    active_providers: list[str] | None = None,
) -> None:
    """질문을 전송하고 응답을 세션에 반영한다."""

    logger.debug("_send_question:시작 question=%s turn=%s", question, turn_value)
    payload: dict[str, Any] = {"question": question, "turn": turn_value, "history": history_payload}
    if model_overrides:
        payload["models"] = {k: v for k, v in model_overrides.items() if v}
    if active_providers:
        payload["active_providers"] = active_providers
    resp = requests.post(ask_url, headers=headers, json=payload, stream=True, timeout=60)
    _sync_usage_from_headers(resp)
    if resp.status_code >= 400:
        try:
            data = resp.json()
            st.error(data.get("detail") or "요청 처리에 실패했습니다.")
        except Exception:
            st.error("요청 처리에 실패했습니다.")
        return

    live_key = f"live_{turn_value}"
    placeholders = {}
    events_acc: list[dict[str, Any]] = []
    answers_acc: dict[str, str] = {}
    sources_acc: dict[str, str | None] = {}

    for line in resp.iter_lines():
        if not line:
            continue
        try:
            parsed = json.loads(line.decode("utf-8"))
        except Exception:
            parsed = line
        if not isinstance(parsed, dict):
            continue
        event_type = parsed.get("event") or parsed.get("type", "partial")
        if event_type == "partial":
            if parsed.get("phase") == "evaluation":
                events_acc.append(
                    {
                        "evaluator": parsed.get("evaluator"),
                        "scores": parsed.get("scores"),
                        "status": parsed.get("status"),
                        "elapsed_ms": parsed.get("elapsed_ms"),
                    }
                )
                logger.debug(
                    "_send_prompt_eval:평가 이벤트 evaluator=%s elapsed_ms=%s",
                    parsed.get("evaluator"),
                    parsed.get("elapsed_ms"),
                )
                continue
            model = parsed.get("model")
            if not model:
                continue
            status = parsed.get("status") or {}
            answer = parsed.get("answer")
            source = parsed.get("source")
            answers_acc[model] = answer
            sources_acc[model] = source
            events_acc.append(
                {
                    "model": model,
                    "answer": answer,
                    "source": source,
                    "status": status,
                    "elapsed_ms": parsed.get("elapsed_ms"),
                    "response_meta": parsed.get("response_meta"),
                }
            )
            if model not in placeholders:
                placeholders[model] = st.empty()
            slot = placeholders[model]
            with slot.container():
                elapsed = parsed.get("elapsed_ms")
                elapsed_txt = f"{elapsed/1000:.1f}s" if elapsed is not None else "-"
                emoji = _status_to_emoji(status)
                st.markdown(f"{emoji} **{model}** ⏱️ {elapsed_txt}")
                if _is_error_status(status):
                    st.error(answer or "응답이 실패했습니다.")
                else:
                    st.write(answer)
                meta_text = _format_response_meta(parsed.get("response_meta"))
                if meta_text:
                    st.caption(meta_text)
                src = source
                if not _render_sources_from_meta(parsed.get("response_meta")):
                    if model == "Perplexity":
                        st.caption(f"출처: {src or '제공되지 않음'}")
                    elif src:
                        st.caption(f"출처: {src}")
        elif event_type == "error":
            model = parsed.get("model") or "unknown"
            message = parsed.get("detail") or parsed.get("message") or "에러가 발생했습니다."
            status = parsed.get("status") or "error"
            elapsed = parsed.get("elapsed_ms")
            elapsed_txt = f"{elapsed/1000:.1f}s" if elapsed is not None else "-"
            emoji = _status_to_emoji(status or "error")
            events_acc.append(
                {
                    "model": model,
                    "answer": message,
                    "source": None,
                    "status": status or "error",
                    "elapsed_ms": elapsed,
                    "response_meta": parsed.get("response_meta"),
                    "error_code": parsed.get("error_code"),
                }
            )
            if model not in placeholders:
                placeholders[model] = st.empty()
            slot = placeholders[model]
            with slot.container():
                st.markdown(f"{emoji} **{model}** ⏱️ {elapsed_txt}")
                st.error(message)
        elif event_type == "summary":
            result = parsed.get("result") or {}
            answers_acc = result.get("answers") or answers_acc
            sources_acc = result.get("sources") or sources_acc
            turn = result.get("turn", turn_value)
            max_turns = result.get("max_turns")
            usage_remaining = result.get("usage_remaining")
            if usage_remaining is not None:
                st.session_state["usage_remaining"] = usage_remaining
            _append_chat_log_entry(question, answers_acc, sources_acc, events_acc)
            _update_usage_after_response(resp, admin_mode=st.session_state.get("usage_bypass"))
            logger.info("_send_question:요약 수신 turn=%s max_turns=%s answers=%d", turn, max_turns, len(answers_acc))
            st.rerun()
            return

    # 요약이 안 왔을 때도 기록만 남김
    _append_chat_log_entry(question, answers_acc, sources_acc, events_acc)
    _update_usage_after_response(resp, admin_mode=st.session_state.get("usage_bypass"))
    logger.warning("_send_question:요약 미수신, 기록만 저장")
    st.rerun()


def _send_prompt_eval(
    question: str,
    eval_url: str,
    headers: dict[str, str],
    prompt_payload: str | None,
    active_models: list[str],
) -> None:
    """프롬프트 평가 요청을 전송하고 스트림 응답을 표시한다."""

    logger.debug("_send_prompt_eval:시작 question=%s models=%s", question, active_models)
    payload: dict[str, Any] = {"question": question, "models": active_models}
    model_overrides = st.session_state.get("model_selections") or {}
    if model_overrides:
        payload["model_overrides"] = model_overrides
    if prompt_payload:
        payload["prompt"] = prompt_payload
    reference_answer = st.session_state.get("prompt_eval_reference") or ""
    if reference_answer.strip():
        payload["reference_answer"] = reference_answer.strip()

    resp = requests.post(eval_url, headers=headers, json=payload, stream=True, timeout=120)
    if resp.status_code >= 400:
        try:
            data = resp.json()
            st.error(data.get("detail") or "요청 처리에 실패했습니다.")
        except Exception:
            st.error("요청 처리에 실패했습니다.")
        return
    placeholders: dict[str, Any] = {}
    events_acc: list[dict[str, Any]] = []
    summary_data: dict[str, Any] | None = None

    for line in resp.iter_lines():
        if not line:
            continue
        try:
            parsed = json.loads(line.decode("utf-8"))
        except Exception:
            parsed = line
        if not isinstance(parsed, dict):
            continue
        event_type = parsed.get("event") or parsed.get("type", "partial")
        if event_type == "partial":
            model = parsed.get("model")
            if not model:
                continue
            status = parsed.get("status") or {}
            answer = parsed.get("answer")
            elapsed = parsed.get("elapsed_ms")
            elapsed_txt = f"{elapsed/1000:.1f}s" if elapsed is not None else "-"
            emoji = _status_to_emoji(status)
            events_acc.append(
                {
                    "model": model,
                    "answer": answer,
                    "source": None,
                    "status": status,
                    "elapsed_ms": elapsed,
                    "response_meta": parsed.get("response_meta"),
                }
            )
            if model not in placeholders:
                placeholders[model] = st.empty()
            with placeholders[model].container():
                st.markdown(f"{emoji} **{model}** ⏱️ {elapsed_txt}")
                if _is_error_status(status):
                    st.error(answer or "응답이 실패했습니다.")
                else:
                    st.write(answer)
                meta_text = _format_response_meta(parsed.get("response_meta"))
                if meta_text:
                    st.caption(meta_text)
                _render_sources_from_meta(parsed.get("response_meta"))
        elif event_type == "error":
            message = parsed.get("detail") or parsed.get("message") or "에러가 발생했습니다."
            st.error(message)
            logger.error("_send_prompt_eval:에러 이벤트 model=%s message=%s", parsed.get("model"), message)
            events_acc.append(
                {
                    "model": parsed.get("model") or "unknown",
                    "answer": message,
                    "source": None,
                    "status": parsed.get("status") or "error",
                    "elapsed_ms": parsed.get("elapsed_ms"),
                    "response_meta": parsed.get("response_meta"),
                    "error_code": parsed.get("error_code"),
                }
            )
        elif event_type == "summary":
            summary_data = parsed.get("result") or {}
            scores = summary_data.get("scores") or []
            avg_score = summary_data.get("avg_score")
            logger.info("_send_prompt_eval:요약 수신 scores=%d avg=%s", len(scores), avg_score)
            st.subheader("🏁 평가 결과")
            if avg_score is not None:
                st.markdown(f"✨ **평균 점수:** {avg_score}")
            if scores:
                evaluations = summary_data.get("evaluations") or []
                # 모델별 평가자 점수/근거를 매핑
                per_model: dict[str, list[dict[str, Any]]] = {}
                for ev in evaluations:
                    evaluator = ev.get("evaluator")
                    status = ev.get("status") or {}
                    evaluator_model = ""
                    if isinstance(status, dict):
                        evaluator_model = status.get("model") or ""
                    for sc in ev.get("scores", []):
                        target = sc.get("model")
                        if not target:
                            continue
                        per_model.setdefault(target, []).append(
                            {
                                "evaluator": evaluator,
                                "evaluator_model": evaluator_model,
                                "accuracy": sc.get("accuracy"),
                                "completeness": sc.get("completeness"),
                                "clarity": sc.get("clarity"),
                                "score": sc.get("score"),
                                "rationale": sc.get("rationale"),
                            }
                        )
                # 순위/점수 요약 표
                sorted_scores = sorted(scores, key=lambda x: x.get("rank") or 999)
                st.markdown("**순위/점수 요약**")
                summary_rows = []
                for s in sorted_scores:
                    model = s.get("model")
                    rank = s.get("rank")
                    avg_raw = s.get("score")
                    avg = f"{avg_raw:.2f}" if isinstance(avg_raw, (int, float)) else avg_raw
                    eval_items = per_model.get(model, [])
                    score_list = ", ".join(
                        (
                            f"{item.get('evaluator')}: {item.get('score'):.2f}"
                            if isinstance(item.get("score"), (int, float))
                            else f"{item.get('evaluator')}: {item.get('score')}"
                        )
                        for item in eval_items
                        if item.get("score") is not None
                    )
                    summary_rows.append(
                        {
                            "순위": rank,
                            "모델": model,
                            "평균점수": avg,
                            "받은 점수": score_list,
                        }
                    )
                if summary_rows:
                    st.table(summary_rows)

                # 모델별 상세(평가자 근거 분리)
                st.markdown("**모델별 상세 근거**")
                lines = []
                for s in sorted_scores:
                    model = s.get("model")
                    rank = s.get("rank")
                    avg = s.get("score")
                    st.markdown(f"### {rank}위 · {model} (평균점수: {avg})")
                    eval_items = per_model.get(model, [])
                    if eval_items:
                        # 평가자별 점수/근거 테이블
                        rationale_rows = []
                        for item in eval_items:
                            rationale_rows.append(
                                {
                                    "평가자": item.get("evaluator"),
                                    "평가 모델": item.get("evaluator_model"),
                                    "정확성": (
                                        f"{item.get('accuracy'):.2f}"
                                        if isinstance(item.get("accuracy"), (int, float))
                                        else item.get("accuracy")
                                    ),
                                    "완전성": (
                                        f"{item.get('completeness'):.2f}"
                                        if isinstance(item.get("completeness"), (int, float))
                                        else item.get("completeness")
                                    ),
                                    "명료성": (
                                        f"{item.get('clarity'):.2f}"
                                        if isinstance(item.get("clarity"), (int, float))
                                        else item.get("clarity")
                                    ),
                                    "가중치 점수": (
                                        f"{item.get('score'):.2f}"
                                        if isinstance(item.get("score"), (int, float))
                                        else item.get("score")
                                    ),
                                    "근거": item.get("rationale") or "",
                                }
                            )
                        st.table(rationale_rows)
                    elif s.get("rationale"):
                        st.caption(f"- 근거: {s.get('rationale')}")
                    lines.append(f"{rank}위 | {model} | 평균점수={avg}")
                    for item in eval_items:
                        lines.append(f"  {item.get('evaluator')}: {item.get('score')} | 근거: {item.get('rationale') or ''}")
                st.markdown("📋 복사용 텍스트")
                st.code("\n".join(lines), language="text")
                st.download_button(
                    "결과 JSON 다운로드",
                    data=json.dumps(summary_data, ensure_ascii=False, indent=2),
                    file_name="prompt_eval_result.json",
                    mime="application/json",
                )
            evaluations = summary_data.get("evaluations") or []
            if evaluations:
                with st.expander("🧠 평가자별 원본 점수/근거 보기", expanded=False):
                    for ev in evaluations:
                        status = ev.get("status") or {}
                        status_str = ""
                        if isinstance(status, dict):
                            status_str = str(status.get("status") or status)
                        elif status is not None:
                            status_str = str(status)
                        emoji = _status_to_emoji(status)
                        st.markdown(
                            f"{emoji} **평가자:** {ev.get('evaluator')} | 상태: {status_str} | 모델: "
                            f"{(status.get('model') if isinstance(status, dict) else '')}"
                        )
                        score_list = ev.get("scores") or []
                        for sc in score_list:
                            st.write(f"- 대상: {sc.get('model')} | 점수: {sc.get('score')} | 순위: {sc.get('rank')}")
                            if sc.get("rationale"):
                                st.caption(f"  근거: {sc.get('rationale')}")
                        if ev.get("elapsed_ms") is not None:
                            st.caption(f"소요 시간: {ev.get('elapsed_ms')} ms")
        elif event_type == "usage":
            # 사용량 메타는 표시만 건너뜀
            continue

    if summary_data:
        # 간단한 로그 저장
        st.session_state.setdefault("prompt_eval_log", []).append(
            {
                "question": question,
                "events": events_acc,
                "summary": summary_data,
            }
        )
    logger.debug("_send_prompt_eval:종료")


def main() -> None:
    logger.debug("streamlit_main:시작")
    st.title("Compare-AI")
    st.caption("여러 LLM 중 내 질문에 가장 잘 답하는 모델을 찾아보세요.")

    with st.sidebar:
        base_url = _load_base_url().rstrip("/")
        st.session_state["fastapi_base_url"] = base_url
        st.text_input("FastAPI URL", value=base_url or "환경변수/파일로 설정하세요", disabled=True)
        if not base_url:
            st.error("FASTAPI_URL 환경변수나 .fastapi_url 파일로 백엔드 주소를 설정하세요.")
            st.stop()
        _render_connection_status(base_url)
        _render_model_selector()

    ask_url = f"{base_url}/api/ask" if base_url else ""
    eval_url = f"{base_url}/api/prompt-eval" if base_url else ""

    # 인증/회원가입 뷰
    if not st.session_state.get("auth_token"):
        _render_auth_section(base_url)

    # 세션 기본값
    if "usage_remaining" not in st.session_state:
        st.session_state["usage_remaining"] = _usage_limit_int()
    if "usage_bypass" not in st.session_state:
        st.session_state["usage_bypass"] = False
    if "chat_log" not in st.session_state:
        st.session_state["chat_log"] = []
    if "prompt_eval_log" not in st.session_state:
        st.session_state["prompt_eval_log"] = []

    # 로그인 후 최초 1회 사용량 조회
    if st.session_state.get("auth_token") and "usage_fetched" not in st.session_state:
        usage_url = f"{base_url}/usage"
        try:
            resp = requests.get(usage_url, headers={"Authorization": st.session_state["auth_token"]}, timeout=5)
            data = resp.json()
            if resp.ok:
                remaining_val = data.get("remaining")
                if data.get("bypass"):
                    st.session_state["usage_remaining"] = None
                    st.session_state["usage_bypass"] = True
                elif isinstance(remaining_val, int):
                    st.session_state["usage_remaining"] = remaining_val
                    st.session_state["usage_bypass"] = False
            st.session_state["usage_fetched"] = True
        except Exception:
            st.session_state["usage_fetched"] = True
    if user := st.session_state.get("auth_user"):
        st.caption(f"로그인됨: {user.get('email')}")
    if st.session_state.get("usage_bypass"):
        st.caption("관리자 권한 활성화 (일일 제한 없음)")
    remaining = st.session_state.get("usage_remaining")
    if remaining is None:
        st.success("남은 일일 사용 횟수: 무제한 (관리자 모드)")
    elif remaining == 0:
        st.error("남은 일일 사용 횟수: **0회** (관리자 우회 필요)")
    else:
        st.info(f"남은 일일 사용 횟수: **{remaining}회** (관리자 우회 시 제한 없음)")
    if st.button("로그아웃"):
        _handle_logout()

    logger.debug("streamlit_main:종료")

    tab_compare, tab_prompt = st.tabs(["모델 비교", "프롬프트 평가"])

    with tab_compare:
        st.header("대화")
        _render_chat_history(st.session_state["chat_log"])
        show_chat_graph = st.toggle("그래프 보기 (Chat Compare)", value=False)

        question = st.chat_input("질문을 입력하세요...")

        if question:
            if not ask_url:
                st.error("FastAPI Base URL을 설정해주세요.")
                return
            headers = {"Content-Type": "application/json"}
            if token := st.session_state.get("auth_token"):
                headers["Authorization"] = token
            history_payload = _build_history_payload(st.session_state.get("chat_log", []))
            model_overrides = st.session_state.get("model_selections")
            enabled_map = st.session_state.get("model_enabled") or {}
            active_providers = [k for k, v in enabled_map.items() if v]
            if not active_providers:
                st.error("활성화된 모델이 없습니다. 사이드바에서 모델을 켜주세요.")
                return
            turn_value = len(st.session_state.get("chat_log", [])) + 1

            with st.spinner("모델 비교 중..."):
                try:
                    _send_question(
                        question,
                        ask_url,
                        headers,
                        turn_value,
                        history_payload,
                        model_overrides=model_overrides,
                        active_providers=active_providers,
                    )
                except Exception as exc:  # pragma: no cover - UI 예외
                    st.error(f"요청 실패: {exc}")
        if show_chat_graph:
            st.subheader("Chat Compare")
            chat_dot = """
            digraph G {
              rankdir=LR;
              Q [label="User Question", shape=box];
              INIT [label="init_question", shape=box];
              OAI [label="call_openai"];
              GEM [label="call_gemini"];
              ANT [label="call_anthropic"];
              PPLX [label="call_perplexity"];
              UPS [label="call_upstage"];
              MIS [label="call_mistral"];
              GRQ [label="call_groq"];
              COH [label="call_cohere"];
              DS [label="call_deepseek"];
              END1 [label="END", shape=Msquare];
              Q -> INIT;
              INIT -> OAI [label="fan-out"];
              INIT -> GEM;
              INIT -> ANT;
              INIT -> PPLX;
              INIT -> UPS;
              INIT -> MIS;
              INIT -> GRQ;
              INIT -> COH;
              INIT -> DS;
              OAI -> END1;
              GEM -> END1;
              ANT -> END1;
              PPLX -> END1;
              UPS -> END1;
              MIS -> END1;
              GRQ -> END1;
              COH -> END1;
              DS -> END1;
            }
            """
            st.graphviz_chart(chat_dot)

    with tab_prompt:
        st.header("프롬프트 평가")
        st.write("모델별 프롬프트를 다르게 적용해 응답을 받고, 고정 평가모델로 블라인드 평가합니다.")
        show_eval_graph = st.toggle("그래프 보기 (Prompt Compare)", value=False)
        if not eval_url:
            st.error("FastAPI Base URL을 설정해주세요.")
            return
        headers = {"Content-Type": "application/json"}
        if token := st.session_state.get("auth_token"):
            headers["Authorization"] = token

        active_models = st.multiselect(
            "평가할 모델 선택",
            options=list(MODEL_OPTIONS.keys()),
            default=list(MODEL_OPTIONS.keys()),
        )
        question_eval = st.text_area("질문", placeholder="비교할 질문을 입력하세요", height=100)

        default_prompt = """[Question]
{question}

답변은 한국어로 작성하세요."""
        st.markdown("공통 프롬프트 (미입력 시 기본값 사용)")
        prompt_val = st.text_area(
            "프롬프트",
            key="prompt_common",
            value=default_prompt,
            placeholder="[Question]\n{question}\n\n답변은 한국어로 작성하세요.",
            height=120,
        )
        st.markdown("선택사항: 모범 답변 예시")
        st.text_area(
            "[선택] 모범답변 예시",
            key="prompt_eval_reference",
            placeholder="예시 모범답변을 입력하세요.",
            height=120,
        )

        prompt_payload = prompt_val.strip()
        prompt_missing_question = bool(prompt_payload) and "{question}" not in prompt_payload
        if prompt_missing_question:
            st.warning("프롬프트에는 `{question}` 변수가 반드시 포함되어야 합니다.")

        if st.button(
            "프롬프트 평가 실행",
            disabled=not question_eval or not active_models or prompt_missing_question,
        ):
            with st.spinner("프롬프트 평가 실행 중..."):
                try:
                    active_labels = [MODEL_OPTIONS[k]["label"] for k in active_models if k in MODEL_OPTIONS]
                    prompt_payload = prompt_payload or None
                    _send_prompt_eval(question_eval, eval_url, headers, prompt_payload, active_labels)
                except Exception as exc:  # pragma: no cover
                    st.error(f"요청 실패: {exc}")
        if show_eval_graph:
            st.subheader("Prompt Compare")
            eval_dot = """
            digraph G {
              rankdir=LR;
              Q2 [label="Question + Prompt", shape=box];
              OAI_G [label="Generate OpenAI"];
              GEM_G [label="Generate Gemini"];
              ANT_G [label="Generate Anthropic"];
              UPS_G [label="Generate Upstage"];
              PPLX_G [label="Generate Perplexity"];
              MIS_G [label="Generate Mistral"];
              GRQ_G [label="Generate Groq"];
              COH_G [label="Generate Cohere"];
              DS_G [label="Generate DeepSeek"];

              Q2 -> OAI_G;
              Q2 -> GEM_G;
              Q2 -> ANT_G;
              Q2 -> UPS_G;
              Q2 -> PPLX_G;
              Q2 -> MIS_G;
              Q2 -> GRQ_G;
              Q2 -> COH_G;
              Q2 -> DS_G;

              OAI_E [label="Eval by OpenAI (latest)"];
              GEM_E [label="Eval by Gemini (latest)"];
              ANT_E [label="Eval by Anthropic (latest)"];
              UPS_E [label="Eval by Upstage (latest)"];
              PPLX_E [label="Eval by Perplexity (latest)"];
              MIS_E [label="Eval by Mistral (latest)"];
              GRQ_E [label="Eval by Groq (latest)"];
              COH_E [label="Eval by Cohere (latest)"];
              DS_E [label="Eval by DeepSeek (latest)"];

              OAI_G -> OAI_E;
              GEM_G -> GEM_E;
              ANT_G -> ANT_E;
              UPS_G -> UPS_E;
              PPLX_G -> PPLX_E;
              MIS_G -> MIS_E;
              GRQ_G -> GRQ_E;
              COH_G -> COH_E;
              DS_G -> DS_E;

              SUM [label="Summary", shape=Msquare];
              OAI_E -> SUM;
              GEM_E -> SUM;
              ANT_E -> SUM;
              UPS_E -> SUM;
              PPLX_E -> SUM;
              MIS_E -> SUM;
              GRQ_E -> SUM;
              COH_E -> SUM;
              DS_E -> SUM;
            }
            """
            st.graphviz_chart(eval_dot)


if __name__ == "__main__":
    main()

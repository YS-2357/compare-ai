"""Streamlit UI 엔트리포인트."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

FASTAPI_URL_FILE = Path(__file__).resolve().parents[2] / ".fastapi_url"
DEFAULT_FASTAPI_BASE = FASTAPI_URL_FILE.read_text().strip() if FASTAPI_URL_FILE.exists() else ""

st.set_page_config(page_title="Compare-AI", page_icon="🤖", layout="wide")


def _load_base_url() -> str:
    saved = (
        st.session_state.get("fastapi_base_url")
        or DEFAULT_FASTAPI_BASE
        or os.getenv("FASTAPI_URL", "")
        or st.secrets.get("FASTAPI_URL", "")
    )
    if saved.endswith("/api/ask"):
        return saved.rsplit("/api/ask", 1)[0]
    return saved


def _get_admin_token() -> str:
    # 환경변수에 있어도 UI에 노출되지 않도록 세션 값만 사용
    return st.session_state.get("admin_token", "")


def _get_usage_limit() -> str:
    return os.getenv("DAILY_USAGE_LIMIT") or "3"


def _usage_limit_int() -> int:
    try:
        return int(_get_usage_limit())
    except Exception:
        return 3


def _get_admin_env_token() -> str:
    return os.getenv("ADMIN_BYPASS_TOKEN", "") or st.secrets.get("ADMIN_BYPASS_TOKEN", "")


def _sync_usage_from_headers(resp: requests.Response) -> None:
    """서버 헤더의 사용량 정보를 세션에 반영한다."""

    limit = resp.headers.get("X-Usage-Limit")
    remaining = resp.headers.get("X-Usage-Remaining")
    if limit is not None and limit.isdigit():
        st.session_state["usage_limit"] = int(limit)
    if remaining is not None and remaining.isdigit():
        st.session_state["usage_remaining"] = int(remaining)


def _build_history_payload(chat_log: list[dict[str, Any]]) -> list[dict[str, str]]:
    """기존 대화 로그를 LangGraph history 페이로드로 변환한다."""

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
    return history_payload


def _update_usage_after_response(resp: requests.Response, *, use_admin_bypass: bool) -> None:
    """응답 이후 사용량 카운터를 갱신한다."""

    if resp.status_code == 429:
        st.session_state["usage_remaining"] = 0
    elif resp.ok and not use_admin_bypass:
        if "X-Usage-Remaining" not in resp.headers:
            new_value = max(0, st.session_state.get("usage_remaining", _usage_limit_int()) - 1)
            st.session_state["usage_remaining"] = new_value


def _append_chat_log_entry(
    question: str,
    answers: dict[str, str],
    sources: dict[str, str | None],
    events: list[dict[str, Any]],
) -> None:
    """대화 로그에 새 엔트리를 추가한다."""

    st.session_state["chat_log"].append(
        {
            "question": question,
            "answers": answers,
            "sources": sources,
            "events": events,
        }
    )


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


def _render_auth_section(base_url: str) -> None:
    """로그인/회원가입 UI를 렌더링한다."""

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
                    st.write(f"회원가입 상태: {resp.status_code}")
                    st.json(resp.json())
                except Exception as exc:
                    st.error(f"회원가입 실패: {exc}")
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
                    st.json(data)
                except Exception as exc:
                    st.error(f"로그인 실패: {exc}")
    st.stop()


def _render_chat_history(chat_log: list[dict[str, Any]]) -> None:
    """기존 대화 로그를 챗봇 형식으로 표시한다."""

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
            }
        with st.chat_message("assistant"):
            if answers:
                for model, answer in answers.items():
                    meta = event_meta.get(model) or {}
                    status = meta.get("status")
                    elapsed_ms = meta.get("elapsed_ms")
                    emoji = _status_to_emoji(status)
                    elapsed_txt = f"{elapsed_ms/1000:.1f}s" if elapsed_ms is not None else "-"
                    st.markdown(f"{emoji} **{model}** ⏱️ {elapsed_txt}")
                    st.write(answer)
                    src = sources.get(model)
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
                    elapsed_txt = f"{elapsed/1000:.1f}s" if elapsed is not None else "-"
                    emoji = _status_to_emoji(status)
                    st.write(f"{emoji} [{model}] {ans}")
                    st.caption(f"⏱️ {elapsed_txt}")
                    if model == "Perplexity":
                        st.caption(f"출처: {src or '제공되지 않음'}")
                    elif src:
                        st.caption(f"출처: {src}")


def _render_connection_status(base_url: str) -> None:
    """API 연결 상태를 간단히 표시한다."""

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


def _handle_logout() -> None:
    """로그아웃 처리."""

    st.session_state.pop("auth_token", None)
    st.session_state.pop("auth_user", None)
    st.session_state.pop("usage_remaining", None)
    st.session_state.pop("chat_log", None)
    st.session_state.pop("use_admin_bypass", None)
    st.session_state.pop("admin_token", None)
    st.rerun()


def _send_question(
    question: str, ask_url: str, headers: dict[str, str], turn_value: int, history_payload: list[dict[str, str]]
) -> None:
    """질문을 전송하고 응답을 세션에 반영한다."""

    payload = {"question": question, "turn": turn_value, "history": history_payload}
    resp = requests.post(ask_url, headers=headers, json=payload, stream=True, timeout=60)
    _sync_usage_from_headers(resp)

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
        event_type = parsed.get("type", "partial")
        if event_type == "partial":
            model = parsed.get("model")
            if not model:
                continue
            answers_acc[model] = parsed.get("answer")
            sources_acc[model] = parsed.get("source")
            events_acc.append(
                {
                    "model": model,
                    "answer": parsed.get("answer"),
                    "source": parsed.get("source"),
                    "status": parsed.get("status"),
                    "elapsed_ms": parsed.get("elapsed_ms"),
                }
            )
            if model not in placeholders:
                placeholders[model] = st.empty()
            slot = placeholders[model]
            with slot.container():
                status = parsed.get("status") or {}
                elapsed = parsed.get("elapsed_ms")
                elapsed_txt = f"{elapsed/1000:.1f}s" if elapsed is not None else "-"
                emoji = _status_to_emoji(status)
                st.markdown(f"{emoji} **{model}** ⏱️ {elapsed_txt}")
                st.write(parsed.get("answer"))
                src = parsed.get("source")
                if model == "Perplexity":
                    st.caption(f"출처: {src or '제공되지 않음'}")
                elif src:
                    st.caption(f"출처: {src}")
        elif event_type == "error":
            model = parsed.get("model") or "unknown"
            message = parsed.get("message") or "에러가 발생했습니다."
            status = parsed.get("status") or {}
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
            _update_usage_after_response(resp, use_admin_bypass=st.session_state.get("use_admin_bypass"))
            st.rerun()
            return

    # 요약이 안 왔을 때도 기록만 남김
    _append_chat_log_entry(question, answers_acc, sources_acc, events_acc)
    _update_usage_after_response(resp, use_admin_bypass=st.session_state.get("use_admin_bypass"))
    st.rerun()


def main() -> None:
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

        st.subheader("관리자 우회 토큰 (선택)")
        admin_token = st.text_input("x-admin-bypass", value=_get_admin_token(), type="password")
        st.session_state["admin_token"] = admin_token
        use_admin = st.checkbox("우회 토큰 사용", value=False, help="체크 시 인증/레이트리밋 우회")
        st.session_state["use_admin_bypass"] = use_admin

    ask_url = f"{base_url}/api/ask" if base_url else ""

    # 인증/회원가입 뷰
    if not st.session_state.get("auth_token"):
        _render_auth_section(base_url)

    # 로그인 후 질문 뷰 (챗봇 형식)
    st.header("대화")
    if "usage_remaining" not in st.session_state:
        st.session_state["usage_remaining"] = _usage_limit_int()
    if "chat_log" not in st.session_state:
        st.session_state["chat_log"] = []
    # 우회 토글이 남아있지 않도록 기본값 보정
    if "use_admin_bypass" not in st.session_state:
        st.session_state["use_admin_bypass"] = False
    # 로그인 후 최초 1회 사용량 조회
    if st.session_state.get("auth_token") and "usage_fetched" not in st.session_state:
        usage_url = f"{base_url}/usage"
        try:
            resp = requests.get(usage_url, headers={"Authorization": st.session_state["auth_token"]}, timeout=5)
            data = resp.json()
            if resp.ok:
                remaining_val = data.get("remaining")
                if remaining_val is None and data.get("bypass"):
                    st.session_state["usage_remaining"] = _usage_limit_int()
                elif isinstance(remaining_val, int):
                    st.session_state["usage_remaining"] = remaining_val
            st.session_state["usage_fetched"] = True
        except Exception:
            st.session_state["usage_fetched"] = True
    if user := st.session_state.get("auth_user"):
        st.caption(f"로그인됨: {user.get('email')}")
    remaining = st.session_state.get("usage_remaining", _usage_limit_int())
    if remaining == 0:
        st.error("남은 일일 사용 횟수: **0회** (관리자 우회 시 제한 없음)")
    else:
        st.info(f"남은 일일 사용 횟수: **{remaining}회** (관리자 우회 시 제한 없음)")
    if st.button("로그아웃"):
        _handle_logout()

    _render_chat_history(st.session_state["chat_log"])

    question = st.chat_input("질문을 입력하세요...")

    if question:
        if not ask_url:
            st.error("FastAPI Base URL을 설정해주세요.")
            return
        headers = {"Content-Type": "application/json"}
        if token := st.session_state.get("auth_token"):
            headers["Authorization"] = token
        history_payload = _build_history_payload(st.session_state.get("chat_log", []))
        turn_value = len(st.session_state.get("chat_log", [])) + 1

        # 관리자 우회 토큰 검증: 환경/secret에 설정된 값과 일치할 때만 헤더 추가
        admin_env = _get_admin_env_token()
        admin_input = st.session_state.get("admin_token")
        allow_admin = False
        if st.session_state.get("use_admin_bypass"):
            if admin_env and admin_input and admin_input == admin_env:
                allow_admin = True
            elif admin_input:
                st.warning("관리자 우회 토큰이 일치하지 않아 일반 요청으로 진행합니다.")
            elif not admin_env:
                st.warning("서버에 관리자 우회 토큰이 설정되지 않아 우회할 수 없습니다.")
        if allow_admin:
            headers["x-admin-bypass"] = admin_input
        with st.spinner("모델 비교 중..."):
            try:
                _send_question(question, ask_url, headers, turn_value, history_payload)
            except Exception as exc:  # pragma: no cover - UI 예외
                st.error(f"요청 실패: {exc}")


if __name__ == "__main__":
    main()

"""Streamlit UI 엔트리포인트."""

from __future__ import annotations

import json
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

FASTAPI_URL_FILE = Path(__file__).resolve().parents[2] / ".fastapi_url"
DEFAULT_FASTAPI_URL = FASTAPI_URL_FILE.read_text().strip() if FASTAPI_URL_FILE.exists() else ""


def _get_fastapi_url() -> str:
    return st.session_state.get("fastapi_url") or DEFAULT_FASTAPI_URL or st.secrets.get("FASTAPI_URL", "")


def main() -> None:
    st.title("API LangGraph Test")
    st.caption("멀티 LLM 비교 (FastAPI + LangGraph)")

    with st.sidebar:
        st.header("Backend 설정")
        fastapi_url = st.text_input("FastAPI /api/ask URL", value=_get_fastapi_url())
        st.session_state["fastapi_url"] = fastapi_url
        st.caption("예: http://127.0.0.1:8000/api/ask")

    question = st.text_area("질문을 입력하세요", height=120)
    if st.button("질문하기"):
        if not question.strip():
            st.warning("질문을 입력해주세요.")
            return
        if not fastapi_url:
            st.error("FastAPI URL을 설정해주세요.")
            return
        with st.spinner("질문 보내는 중..."):
            try:
                resp = requests.post(
                    fastapi_url,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps({"question": question}),
                    stream=True,
                    timeout=60,
                )
                st.write(f"Status: {resp.status_code}")
                st.write("응답 스트림:")
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        parsed = json.loads(line.decode("utf-8"))
                    except Exception:
                        parsed = line
                    st.write(parsed)
            except Exception as exc:  # pragma: no cover - UI 예외
                st.error(f"요청 실패: {exc}")


if __name__ == "__main__":
    main()

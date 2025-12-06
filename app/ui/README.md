# app/ui/

> 최종 업데이트: 2025-12-03 — 스트리밍 즉시 표시(모델별 순서), `/usage` 조회로 초기 사용량 반영, 프롬프트 영어 지시(응답/요약은 한국어)

- Streamlit 로컬/개발용 UI (별도 Render 서비스로도 사용 가능).
- 로그인/회원가입 → 질문 → 스트림(partial/summary) 표시, 모델별 답변과 source 표시.
- 로그인 후 `/usage`를 호출해 실제 남은 횟수를 즉시 표시(0회면 경고), 이후 사용량 헤더(`X-Usage-*`)를 읽어 카운터를 동기화.
- 사이드바에서 OpenAI/Gemini/Claude 등 LLM별 모델을 선택할 수 있으며, 선택값은 `/api/ask` 요청의 `models` 필드로 전달되어 LangGraph 실행에 반영됩니다.
- 멀티턴: 이전 질문/모델 응답을 `history`로 백엔드에 보내 모델별 히스토리를 유지합니다.

실행 (로컬):
```bash
streamlit run app/ui/streamlit_app.py
```

환경변수:
- `FASTAPI_URL` (예: https://<render-be>.onrender.com)
- `STREAMLIT_SERVER_HEADLESS=true` (Render 등 배포 시)

# app/

> 최종 업데이트: 2025-12-03 — LangGraph 멀티턴 상태/요약 및 UI 히스토리 전송 리팩터링 반영

- FastAPI 애플리케이션 루트 패키지.
- `config.py`: 환경변수 로딩/Settings.
- `main.py`: FastAPI 앱 팩토리.
- `api/`: 엔드포인트(/health, /api/ask, /auth/*), 스키마, Depends.
- `auth/`: Supabase JWT 검증/클라이언트.
- `rate_limit/`: Upstash 클라이언트, 일일 한도 Depends.
- `services/langgraph/`: LangGraph 워크플로우/노드/LLM 설정.
- `ui/`: 로컬용 Streamlit UI(별도 서비스로도 사용 가능).

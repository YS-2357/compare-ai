# app/

> 최종 업데이트: 2025-12-18 — 프롬프트 평가 스트림 추가(`/api/prompt-eval`), 평가 결과 JSON 반환 및 에러 내성 강화

- FastAPI 애플리케이션 루트 패키지.
- `config.py`: 환경변수 로딩/Settings.
- `main.py`: FastAPI 앱 팩토리.
- `api/`: 엔드포인트(/health, /api/ask, /api/prompt-eval, /auth/*), 스키마, Depends.
- `auth/`: Supabase JWT 검증/클라이언트.
- `rate_limit/`: Upstash 클라이언트, 일일 한도 Depends.
- `services/langgraph/`: LangGraph 워크플로우/노드/LLM 설정.
- `services/prompt_eval.py`: 모델 병렬 호출 + 각 벤더 최신 모델로 블라인드 교차 평가 스트리밍.
- `ui/`: 로컬용 Streamlit UI(별도 서비스로도 사용 가능).

# app/

> 최종 업데이트: 2025-12-20 — `services/chat_graph` 리네임, 프롬프트 평가 병렬화/런타임 오류 수정, Streamlit 평가 테이블 렌더링 개선

- FastAPI 애플리케이션 루트 패키지.
- `utils/config.py`: 환경변수 로딩/Settings.
- `utils/logger.py`: 콘솔 로거 설정.
- `main.py`: FastAPI 앱 팩토리.
- `api/`: 엔드포인트(/health, /api/ask, /api/prompt-eval, /auth/*), 스키마, Depends.
- `auth/`: Supabase JWT 검증/클라이언트.
- `rate_limit/`: Upstash 클라이언트, 일일 한도 Depends.
- `services/chat_graph/`: LangGraph 워크플로우/노드/LLM 설정.
- `services/prompt_eval/runner.py`: 모델 병렬 호출 + 각 벤더 최신 모델로 블라인드 교차 평가 스트리밍.
- `ui/`: 로컬용 Streamlit UI(별도 서비스로도 사용 가능).

# app/

> 최종 업데이트: 2025-12-22 — shared 모듈 분리, 프롬프트 평가 모델 오버라이드, DeepSeek 추가

- FastAPI 애플리케이션 루트 패키지.
- `utils/config.py`: 환경변수 로딩/Settings(DeepSeek 포함).
- `utils/logger.py`: 콘솔/파일 로거 설정.
- `main.py`: FastAPI 앱 팩토리.
- `api/`: 엔드포인트(/health, /api/ask, /api/prompt-eval, /auth/*), 스키마, Depends.
- `auth/`: Supabase JWT 검증/클라이언트.
- `rate_limit/`: Upstash 클라이언트, 일일 한도 Depends.
- `services/chat_compare/`: LangGraph 워크플로우/노드/LLM 설정.
- `services/prompt_compare/workflow.py`: 모델 병렬 호출 + 각 벤더 최신 모델로 블라인드 교차 평가 스트리밍(생성 모델 오버라이드 지원).
- `services/shared/`: 공통 LLM 레지스트리/에러 헬퍼/모델 매핑.
- `ui/`: 로컬용 Streamlit UI(별도 서비스로도 사용 가능).

# app/api/

> 최종 업데이트: 2025-12-22 · 프롬프트 평가 모델 오버라이드, 응답 메타/출처 분리, 히스토리/현재 질문 분리 문서화
> 추가: `/api/prompt-eval`에 `model_overrides`로 생성 모델을 지정, 평가 모델은 최신 고정 모델 사용

- `routes.py`: `/health`, `/api/ask`, `/api/prompt-eval`, `/usage` 엔드포인트. 모델 덮어쓰기/관리자 우회 전달, 사용량 헤더 스트리밍, 프롬프트 평가 NDJSON 제공.
- `auth_routes.py`: `/auth/register`, `/auth/login` (Supabase Auth REST 연동).
- `deps.py`: 공용 Depends(현재 사용자, 설정, 레이트리밋) — JWT 클레임과 `ADMIN_EMAIL` 비교.
- `schemas/`: 요청/응답 Pydantic 모델(`ask.py`, `auth.py`, `prompt_eval.py` 등).

비고:
- Authorization: Bearer <token> 필수. `ADMIN_EMAIL` 계정은 `remaining=null`, 턴 제한 우회.
- `/api/ask` payload에 `models` 매핑을 넣으면 공급자별 기본 모델을 덮어씀. 현재 질문은 서버가 별도 분리해 `[Current Question]`으로 전달.
- `/api/prompt-eval`은 `model_overrides`로 생성 모델명을 덮어쓸 수 있음.
- Upstash 연결 시 응답 헤더에 `X-Usage-Limit` / `X-Usage-Remaining` 포함.


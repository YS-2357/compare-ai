# app/api/

> 최종 업데이트: 2025-12-03 — `/api/ask` 멀티턴 history 전달/스트림 이벤트 정리 반영

- `routes.py`: `/health`, `/api/ask` 스트림 엔드포인트.
- `auth_routes.py`: `/auth/register`, `/auth/login` (Supabase Auth REST).
- `deps.py`: 공용 Depends (현재 사용자, 설정, 레이트리밋).
- `schemas/`: 요청/응답 Pydantic 모델 (`ask.py`, `auth.py`).

사용 시 주의:
- `Authorization: Bearer <token>` 헤더 필요(관리자 우회 시 `x-admin-bypass`).
- `/api/ask` 응답 헤더에 `X-Usage-Limit`, `X-Usage-Remaining` 포함(Upstash 정상 시, 기본 한도 3회).

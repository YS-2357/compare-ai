# app/api/schemas/

> 최종 업데이트: 2025-12-20 — 프롬프트 평가 병렬화 후 응답 필드(status 문자열화 등) 반영 필요사항 확인

- FastAPI 엔드포인트에서 사용하는 요청/응답 데이터 모델 모음.
- `ask.py`: 질문 본문(`question`), 턴 제어(`turn`, `max_turns`), 대화 히스토리(`history`), 모델 오버라이드(`models`)를 담는 `AskRequest`.
- `auth.py`: 회원가입/로그인 요청(`RegisterRequest`, `LoginRequest`)과 Supabase 토큰 응답 래퍼(`LoginResponse`, `RegisterResponse`).
- API 페이로드 규칙(루트 README의 `/api/ask`, `/auth/*`)과 동기화된 상태를 유지한다.

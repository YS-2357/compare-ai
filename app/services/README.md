# app/services/

> 최종 업데이트: 2025-12-18 — 프롬프트 평가 스트리머 추가 및 에러 내성 강화

- 애플리케이션 서비스 계층 패키지. API 라우터와 분리된 도메인 로직을 모은다.
- `chat_graph/`(구 langgraph): 멀티 LLM 병렬 실행 그래프, 노드/헬퍼/레지스트리/요약 유틸 집합.
- `prompt_eval/`: 공통 프롬프트로 모델 호출 후, 선택 벤더 최신 모델들이 교차 평가하여 점수를 스트리밍(`runner.py`, `__init__.py`).
- `__init__.py`: `stream_graph`, `stream_prompt_eval`을 export.
- 새로운 서비스 모듈은 이 디렉터리 아래에 추가하고, 외부에서는 공개된 진입점만 사용한다.

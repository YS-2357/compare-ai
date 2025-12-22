# app/services/

> 최종 업데이트: 2025-12-22 — `shared` 공통 모듈 분리(llm_registry/errors), 프롬프트 평가 모델 오버라이드 적용

- 애플리케이션 서비스 계층 패키지. API 라우터와 분리된 도메인 로직을 모은다.
- `chat_graph/`(구 langgraph): 멀티 LLM 병렬 실행 그래프, 노드/헬퍼/요약 유틸 집합.
- `prompt_eval/`: 공통 프롬프트로 모델 호출 후, 선택 모델의 응답을 최신 평가 모델이 교차 평가하여 점수를 스트리밍(`runner.py`, `__init__.py`). 생성 단계는 `model_overrides`로 모델명을 덮어쓸 수 있음.
- `shared/`: 공통 유틸(LLM 레지스트리, 에러 헬퍼, 모델 매핑).
- `__init__.py`: `stream_chat`, `stream_prompt_eval`을 export.
- 새로운 서비스 모듈은 이 디렉터리 아래에 추가하고, 외부에서는 공개된 진입점만 사용한다.

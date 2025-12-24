# app/services/chat_graph/

> 최종 업데이트: 2025-12-22 — 현재 질문/히스토리 분리 보강, 공통 에러/레지스트리 모듈 분리

- `workflow.py`: 그래프 생성/컴파일, 이벤트 스트림, 모델 오버라이드/턴 제한 처리.
- `nodes.py`: call_* 노드(LLM 호출 + 파싱), 상태 병합/요약, 현재 질문 분리 처리.
- `helpers.py`: 프롬프트 빌더/출력 파서, 히스토리/요약 렌더.
- `../shared/llm_registry.py`: LLM 클라이언트/UUID 도우미.
- `../shared/errors.py`: 공통 에러/상태 변환 헬퍼.
- `summaries.py`: 요약 유틸.
- `__init__.py`: `stream_chat`, `DEFAULT_MAX_TURNS`, `build_chat_workflow` 노출.

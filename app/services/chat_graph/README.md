# app/services/chat_graph/

> LangGraph 기반 채팅/비교 워크플로우 (stream_graph 등)

- `workflow.py`: 그래프 생성/컴파일, 이벤트 스트림, 모델 오버라이드/턴 제한 처리.
- `nodes.py`: call_* 노드(LLM 호출 + 파싱), 상태 병합/요약.
- `helpers.py`: 프롬프트 빌더/출력 파서, 히스토리/요약 렌더.
- `llm_registry.py`: LLM 클라이언트/UUID 도우미.
- `summaries.py`: 요약 유틸.
- `__init__.py`: `stream_graph`, `DEFAULT_MAX_TURNS`, `build_workflow` 노출.

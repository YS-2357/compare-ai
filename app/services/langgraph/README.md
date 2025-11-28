# app/services/langgraph/

- LangGraph 워크플로우 패키지.
- `workflow.py`: 그래프 컴파일, 이벤트 스트림.
- `nodes.py`: 각 모델 call_* 노드 (LCEL 체인 + Pydantic parser), 모델명은 `config.py`의 `MODEL_*` 사용.
- `llm_registry.py`: LLM 클라이언트/환경 설정.
- `summaries.py`: 요약 유틸.
- `__init__.py`: `stream_graph`, `DEFAULT_MAX_TURNS` 노출.

이벤트 필드:
- `answer`, `model`, `status`, `source`(있을 때), `messages`, `type`, `turn`, `elapsed_ms`.
- summary에 `answers`, `api_status`, `sources`, `usage_limit/remaining` 포함.


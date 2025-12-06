# app/services/langgraph/

- LangGraph workflow package.
- workflow.py: builds/compiles the graph, streams events, accepts model_overrides and ypass_turn_limit.
- 
odes.py: call_* nodes (LCEL chain + Pydantic parser) honour per-provider overrides via MODEL_* defaults.
- helpers.py: prompt builder / structured parser (long-form Korean responses).
- llm_registry.py: LLM clients + LangSmith UUID helper.
- summaries.py: answer summarisation utilities.
- __init__.py: exports stream_graph, DEFAULT_MAX_TURNS 등.

Event payloads:
- partial: model, nswer, status, source?, messages, 	urn, elapsed_ms.
- summary: nswers, pi_status, sources, durations_ms, messages, order, primary_model, usage_limit/remaining, model_overrides 등.


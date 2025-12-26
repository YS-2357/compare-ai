from __future__ import annotations


def _validate_event(event: dict) -> None:
    event_type = event.get("event")
    assert event_type in {"partial", "summary", "error"}
    if event_type == "partial":
        assert event.get("model")
        assert "status" in event
        assert "elapsed_ms" in event
        if event.get("phase") == "evaluation":
            assert event.get("scores") is not None
            assert event.get("evaluator") or event.get("model")
    elif event_type == "summary":
        assert event.get("result") is not None
        assert "status" in event
        assert "elapsed_ms" in event
    elif event_type == "error":
        assert event.get("error_code")
        assert event.get("detail")
        assert event.get("status") == "error"


def test_streaming_event_schema_samples() -> None:
    _validate_event(
        {
            "event": "partial",
            "model": "OpenAI",
            "status": {"status": 200, "detail": "stop"},
            "elapsed_ms": 120,
        }
    )
    _validate_event(
        {
            "event": "partial",
            "phase": "evaluation",
            "evaluator": "OpenAI",
            "scores": [{"model": "Gemini", "score": 0.8}],
            "status": {"status": 200, "detail": "stop"},
            "elapsed_ms": 980,
            "model": "OpenAI",
        }
    )
    _validate_event(
        {
            "event": "summary",
            "status": "ok",
            "elapsed_ms": 1200,
            "result": {"scores": [], "avg_score": None},
        }
    )
    _validate_event(
        {
            "event": "error",
            "error_code": "UNKNOWN_ERROR",
            "detail": "failed",
            "status": "error",
            "elapsed_ms": 5,
        }
    )

"""응답 메타/출처 추출 유틸리티."""

from __future__ import annotations

from typing import Any


def extract_sources(raw_response: Any) -> list[str]:
    """응답 메타에서 출처 URL을 최대한 추출한다."""

    sources: list[str] = []

    def _maybe_add(value: Any) -> None:
        if isinstance(value, str) and value.startswith("http"):
            sources.append(value)
        if isinstance(value, dict):
            for key in ("url", "source", "link", "href"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate.startswith("http"):
                    sources.append(candidate)

    meta = getattr(raw_response, "response_metadata", None)
    if isinstance(meta, dict):
        for key in ("citations", "sources", "search_results"):
            items = meta.get(key)
            if isinstance(items, list):
                for item in items:
                    _maybe_add(item)

    additional_kwargs = getattr(raw_response, "additional_kwargs", None)
    if isinstance(additional_kwargs, dict):
        for key in ("citations", "sources", "search_results"):
            items = additional_kwargs.get(key)
            if isinstance(items, list):
                for item in items:
                    _maybe_add(item)

    raw_sources = getattr(raw_response, "citations", None) or getattr(raw_response, "sources", None)
    if isinstance(raw_sources, list):
        for item in raw_sources:
            _maybe_add(item)

    # 중복 제거 (순서 유지)
    seen = set()
    unique_sources = []
    for src in sources:
        if src in seen:
            continue
        seen.add(src)
        unique_sources.append(src)
    return unique_sources


def extract_response_meta(response: Any) -> dict[str, Any]:
    """응답 본문 외에 표시할 메타 정보를 추출한다."""

    meta: dict[str, Any] = {}
    response_meta = getattr(response, "response_metadata", None)
    additional_kwargs = getattr(response, "additional_kwargs", None)
    usage_metadata = getattr(response, "usage_metadata", None)

    if isinstance(response_meta, dict):
        for key in ("model_name", "model", "model_provider"):
            if response_meta.get(key):
                meta["model_name"] = response_meta.get(key)
                break
        if response_meta.get("finish_reason"):
            meta["finish_reason"] = response_meta.get("finish_reason")
        if response_meta.get("stop_reason"):
            meta["stop_reason"] = response_meta.get("stop_reason")
        if response_meta.get("safety_ratings"):
            meta["safety_ratings"] = response_meta.get("safety_ratings")
        if response_meta.get("prompt_feedback"):
            meta["prompt_feedback"] = response_meta.get("prompt_feedback")
        token_usage = response_meta.get("token_usage")
        if isinstance(token_usage, dict):
            meta["token_usage"] = {
                "input_tokens": token_usage.get("prompt_tokens"),
                "output_tokens": token_usage.get("completion_tokens"),
                "total_tokens": token_usage.get("total_tokens"),
            }

    if isinstance(additional_kwargs, dict) and "refusal" in additional_kwargs:
        meta["refusal"] = additional_kwargs.get("refusal")

    if isinstance(usage_metadata, dict):
        meta.setdefault(
            "token_usage",
            {
                "input_tokens": usage_metadata.get("input_tokens"),
                "output_tokens": usage_metadata.get("output_tokens"),
                "total_tokens": usage_metadata.get("total_tokens"),
            },
        )

    sources = extract_sources(response)
    if sources:
        meta["sources"] = sources

    return meta


def append_sources_block(content: str, sources: list[str] | None) -> str:
    """평가용 응답에 출처 섹션을 추가한다."""

    if "[Sources]" in content:
        return content
    if sources:
        lines = "\n".join(f"- {src}" for src in sources)
        return f"{content}\n\n[Sources]\n{lines}"
    return f"{content}\n\n[Sources]\n- 제공되지 않음"

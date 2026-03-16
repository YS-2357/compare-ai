"""채팅 비교 공급자 메타데이터."""

from __future__ import annotations

from collections.abc import Iterable


PROVIDER_SPECS: tuple[dict[str, str], ...] = (
    {"provider": "openai", "node": "call_openai", "label": "OpenAI"},
    {"provider": "gemini", "node": "call_gemini", "label": "Gemini"},
    {"provider": "anthropic", "node": "call_anthropic", "label": "Anthropic"},
    {"provider": "perplexity", "node": "call_perplexity", "label": "Perplexity"},
    {"provider": "upstage", "node": "call_upstage", "label": "Upstage"},
    {"provider": "mistral", "node": "call_mistral", "label": "Mistral"},
    {"provider": "groq", "node": "call_groq", "label": "Groq"},
    {"provider": "cohere", "node": "call_cohere", "label": "Cohere"},
    {"provider": "deepseek", "node": "call_deepseek", "label": "DeepSeek"},
)

NODE_CONFIG: dict[str, dict[str, str]] = {
    spec["node"]: {
        "provider": spec["provider"],
        "label": spec["label"],
        "answer_key": f"{spec['provider']}_answer",
        "status_key": f"{spec['provider']}_status",
    }
    for spec in PROVIDER_SPECS
}
PROVIDER_TO_NODE: dict[str, str] = {spec["provider"]: spec["node"] for spec in PROVIDER_SPECS}


def default_active_nodes() -> list[str]:
    """기본 활성 노드 목록을 반환한다."""

    return list(NODE_CONFIG.keys())


def resolve_active_nodes(providers: Iterable[str] | None) -> list[str]:
    """공급자 목록을 노드 목록으로 변환한다."""

    if not providers:
        return default_active_nodes()
    return [PROVIDER_TO_NODE[provider] for provider in providers if provider in PROVIDER_TO_NODE]


__all__ = ["NODE_CONFIG", "PROVIDER_SPECS", "PROVIDER_TO_NODE", "default_active_nodes", "resolve_active_nodes"]

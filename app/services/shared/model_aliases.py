"""모델 별칭 및 최신 모델 매핑."""

LATEST_EVAL_MODELS: dict[str, str] = {
    "OpenAI": "gpt-5-mini",
    "Gemini": "gemini-2.5-flash",
    "Anthropic": "claude-sonnet-4-5-20250929",
    "Perplexity": "sonar-reasoning-pro",
    "Upstage": "solar-pro2-251215",
    "Mistral": "mistral-medium-latest",
    "Groq": "llama-3.3-70b-versatile",
    "Cohere": "command-r-08-2024",
    "DeepSeek": "deepseek-reasoner",
}

MODEL_ALIASES = {
    "OpenAI": "OpenAI",
    "Google Gemini": "Gemini",
    "Gemini": "Gemini",
    "Anthropic Claude": "Anthropic",
    "Anthropic": "Anthropic",
    "Perplexity Sonar": "Perplexity",
    "Perplexity": "Perplexity",
    "Upstage Solar": "Upstage",
    "Upstage": "Upstage",
    "Mistral": "Mistral",
    "Groq": "Groq",
    "Cohere": "Cohere",
    "DeepSeek": "DeepSeek",
}

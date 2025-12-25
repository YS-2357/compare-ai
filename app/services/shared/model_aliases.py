"""모델 별칭 및 최신 모델 매핑."""

LATEST_EVAL_MODELS: dict[str, str] = {
    "OpenAI": "gpt-5.2-pro",
    "Gemini": "gemini-3-flash-preview",
    "Anthropic": "claude-sonnet-4-5-20250929",
    "Perplexity": "sonar",
    "Upstage": "solar-mini",
    "Mistral": "mistral-large-latest",
    "Groq": "llama-3.3-70b-versatile",
    "Cohere": "command-a-03-2025",
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

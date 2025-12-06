# app/rate_limit/

> Last update: 2025-12-05 · Upstash client timeout + graceful shutdown

- Upstash Redis based daily usage limiter.
- upstash.py: async client (INCR + EXPIRE pipeline) with configurable HTTP timeout and shutdown hook.
- dependencies.py: enforce_daily_limit (propagates HTTP errors, logs backend failures).

Environment:
- UPSTASH_REDIS_URL / UPSTASH_REDIS_TOKEN`r
- DAILY_USAGE_LIMIT (default 3)
- UPSTASH_HTTP_TIMEOUT (seconds, default 5)

Notes:
- When Upstash is reachable, counters are stored remotely; otherwise FastAPI returns 503 to surface backend issues.


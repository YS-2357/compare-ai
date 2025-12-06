# app/api/

> Last update: 2025-12-05 · model overrides and admin email bypass

- outes.py: /health, /api/ask, /usage endpoints stream usage data and forward models plus bypass flags into LangGraph.
- uth_routes.py: /auth/register, /auth/login (Supabase Auth REST).
- deps.py: shared Depends (current user, settings, rate limit) comparing ADMIN_EMAIL with JWT claims.
- schemas/: request/response Pydantic models (sk.py, uth.py).

Notes:
- Authorization: Bearer <token> required; the ADMIN_EMAIL account sees emaining = null and turn-limit bypass.
- Add a models mapping to /api/ask payloads to override provider models.
- Responses include X-Usage-Limit / X-Usage-Remaining headers when Upstash is reachable.


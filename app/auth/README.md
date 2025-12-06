# app/auth

> Last update: 2025-12-05 · Admin email bypass + Supabase timeout cleanup

- Supabase JWT verification + Auth REST client utilities.
- `dependencies.py`: `get_current_user` (validates Bearer JWT and flags `ADMIN_EMAIL` users as bypass).
- `supabase.py`: JWKS cache/verification + signup/signin client with configurable timeouts and shutdown hook.

Environment:
- `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_JWKS_URL` (default `https://<project>.supabase.co/auth/v1/.well-known/jwks.json`)
- `SUPABASE_JWT_AUD` (default `authenticated`)
- `SUPABASE_HTTP_TIMEOUT`, `SUPABASE_JWKS_CACHE_TTL`
- `ADMIN_EMAIL` (optional, enables limitless usage/turn bypass for that account)

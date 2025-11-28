# app/auth/

- Supabase JWT 검증 및 Auth REST 클라이언트.
- `dependencies.py`: `get_current_user` (Bearer 검증, admin bypass 지원).
- `supabase.py`: JWKS 캐시/검증기, Auth signUp/signIn 클라이언트.

환경변수:
- `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_JWKS_URL` 권장: `https://<project>.supabase.co/auth/v1/.well-known/jwks.json`
- `SUPABASE_JWT_AUD` (기본 authenticated)
- `ADMIN_BYPASS_TOKEN` (옵션)


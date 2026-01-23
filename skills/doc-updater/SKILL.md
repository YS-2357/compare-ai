---
name: doc-updater
description: Update compare-ai documentation after code or behavior changes. Use when syncing README.md files across folders, updating Swagger/FastAPI descriptions, or refreshing docs indices/changelogs to match current behavior.
---

# Doc Updater (compare-ai)

Update repository docs so they reflect current behavior and APIs. Keep edits concise and aligned with existing tone.

## Scope
- **README.md updates**: root `README.md`, `app/README.md`, `app/api/README.md`, `app/rate_limit/README.md`, `docs/README.md`, and other folder-level `README.md` files.
- **Swagger/FastAPI descriptions**: `app/main.py` (`FASTAPI_DESCRIPTION`) and route docstrings/`responses` blocks in `app/api/routes.py`.
- **Docs indices/logs**: `docs/changelog/`, `docs/development/` when user requests logging.

## Workflow
1) Find target docs:
   - `rg --files -g 'README.md'`
   - `rg -n "FASTAPI_DESCRIPTION|summary=|description=" app/api app/main.py`
2) Identify behavior changes to reflect:
   - endpoints, request/response fields, headers, stream event schema, usage limits, model overrides.
3) Update content with minimal wording changes:
   - prefer existing structure; add short examples if needed.
4) Align Swagger description to actual stream schema:
   - `event` keys, `error_code`, `phase`, `usage` headers, etc.
5) If requested, add log entries:
   - `docs/changelog/YYYY-MM-DD.md`
   - `docs/development/YYYY-MM-DD.md`
   - update both indices.

## Style Rules
- Keep edits under 5–10 lines per doc unless a full section is outdated.
- Use repository terminology (chat_compare, prompt_compare, model_overrides, usage limits).
- Examples should match actual payload keys (e.g., `event`, `error_code`, `elapsed_ms`).

## Example Targets
- `app/main.py`: update `FASTAPI_DESCRIPTION` for stream fields and usage headers.
- `app/api/routes.py`: ensure `/api/ask` and `/api/prompt-eval` docs match current response schema.
- `docs/README.md`: update “Quick Links” and guide summaries.

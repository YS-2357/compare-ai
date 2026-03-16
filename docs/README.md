# Docs Overview (Updated: 2026-03-16)

This folder keeps project history, development notes, guides, and prompt source files.

| Section | Description | Link |
| --- | --- | --- |
| Changelog | Dated change history | [changelog/README.md](changelog/README.md) |
| Development | Dated implementation notes | [development/README.md](development/README.md) |
| Roadmap | Medium-term planning notes | [development/roadmap-2025.md](development/roadmap-2025.md) |
| Guides | Repo conventions and reference docs | [guides/README.md](guides/README.md) |
| Prompts | Prompt source files used by the backend | [prompt/README.md](prompt/README.md) |

## Quick Links
- Latest changelog: [2026-03-16](changelog/2026-03-16.md)
- Latest development note: [2026-03-16](development/2026-03-16.md)
- Active runtime: local-first chat comparison with `backend/` + React frontend

## Current Architecture Notes
- `chat_compare` is the active product flow.
- `prompt_compare` remains only in historical documentation; it is no longer part of the active runtime.
- The repository now uses `main.py` as the single public entrypoint, `backend/` as the backend root, and `frontend/` as the React UI.

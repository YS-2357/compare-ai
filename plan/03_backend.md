# Backend Refactor Notes

## Scope
- Backend stack remains Python + FastAPI + LangGraph/LangChain.
- The backend rename from `app/` to `backend/` is part of the planned refactor.
- Chat compare is the mandatory preserved capability.

## Refactor Focus
- Replace `app/` references with `backend/` in code, startup wiring, and docs.
- Clarify boundaries between:
  - API layer
  - chat compare workflow
  - shared provider/model utilities
- Reduce oversized modules only if behavior remains preserved.

## Interface Rules
- Preserve `/health` and `/api/ask`.
- Preserve streaming event types: `partial`, `summary`, `error`.
- Preserve model override and active provider behavior.

## Success Criteria
- `backend/` is the active import root.
- Chat compare is still available after the rename.
- Hosted auth/rate-limit code is removed from the active backend tree.
- Public route behavior stays compatible unless explicitly approved for change.
- Backend folder documentation explains the final boundaries clearly.

# Backend Workflow Refactor

## Purpose
- Track the refactor order for backend Python files.
- Preserve chat compare while restructuring the backend.
- Prevent ad hoc file moves without a defined dependency order.

## Planned Refactor Order
1. Entry and app factory
- Root `main.py`
- backend app factory / startup module
- any remaining launcher helper kept behind `main.py`

2. API surface
- backend route registration
- request/response schema modules
- route handlers for health and chat

3. Chat compare flow
- chat workflow wiring
- node/provider mapping
- prompt/history shaping
- stream event accumulation and summary behavior

4. Shared and support modules
- config
- logger
- shared provider/model helpers

## File-Level Rules
- Refactor in dependency order, not arbitrary folder order.
- Rename imports to `backend.*` only after the target module path is ready.
- Do not break chat compare during backend cleanup.

## Success Criteria
- There is a documented order for backend Python refactoring.
- Chat compare remains preserved through the workflow refactor.
- The backend rename to `backend/` can be executed without uncontrolled import churn.

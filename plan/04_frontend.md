# Frontend Refactor Notes

## Scope
- Frontend structure is still part of the approved-plan discussion.
- The frontend must continue to support the chat compare flow.
- Frontend planning must follow the single-entry `main.py` decision and the `backend/` rename.

## Required Alignment
- Frontend startup must not compete with `main.py` as a public entrypoint.
- Frontend API usage must track the preserved backend contracts.
- Any legacy UI path should be documented as legacy or removed from the active architecture.

## Success Criteria
- Frontend role is clearly documented in the final structure.
- Frontend supports chat compare in the approved scope.
- Frontend docs match the actual runtime path and folder structure.

# Refactor Overview

## Goal
- Refactor the project only after approval, with planning tracked under `plan/`.
- Change the active backend root from `app/` to `backend/`.
- Replace the current startup flow so the project has one explicit user-facing entrypoint centered on `main.py`.
- Preserve the existing chat feature during the refactor.
- Require every kept folder to have a `README.md` that matches the latest actual contents.

## Current Direction
- `main.py` must be reconsidered as the single entrypoint and may delegate to a thin launcher layer if approved.
- `app/` should become `backend/`.
- `tests/` is considered low value and should not drive the architecture.
- `scripts/` is also under review and should either be folded into the single-entry flow or removed.
- Planning must be revised before any more implementation work.

## Non-Negotiable Invariants
- Chat compare behavior must remain available.
- Backend API behavior should stay compatible unless a change is explicitly approved.
- Folder-level documentation must match the final approved structure.

## Success Criteria
- The approved plan clearly defines how `main.py` becomes the single entrypoint.
- The approved plan clearly defines how `app/` becomes `backend/`, including import and doc updates.
- The approved plan includes a backend workflow refactor document in `plan/05_workflow.md`.
- The approved plan defines what happens to `tests/` and `scripts/` instead of leaving them ambiguous.
- The approved plan requires `README.md` coverage for every kept folder.

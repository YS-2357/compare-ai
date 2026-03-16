# Target Folder Structure

## Target Shape
- Root `main.py`: the only user-facing startup entrypoint.
- `backend/`: renamed backend root replacing `app/`.
- `frontend/`: frontend application root if the React path is kept.
- `plan/`: planning and phased refactor documentation.
- `docs/`: historical and supporting documents.

## Backend Structure
- `backend/api`: route layer and request/response schemas.
- `backend/services`: chat compare and shared model/provider logic.
- `backend/utils`: configuration, logging, and support utilities.

## Folder Decisions To Lock
- `tests/`: either remove entirely or mark as non-core and exclude from the refactor target.
- `scripts/`: either remove and fold startup logic into `main.py`, or keep only as an internal implementation detail behind `main.py`.
- hosted-only integrations should be removed from the active tree instead of retained as optional modules.

## README Rule
- Every kept folder must contain a `README.md`.
- Each README must describe the folder’s current purpose, key files, and boundaries.
- No README should describe removed or inactive structure as if it were current.

## Success Criteria
- The approved structure names `backend/` as the active backend root instead of `app/`.
- The approved structure defines whether `scripts/` remains or is removed.
- The approved structure defines whether `tests/` remains or is removed.
- The approved structure includes README ownership rules for all kept folders.

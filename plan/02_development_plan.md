# Development Plan

## Phase 1: Planning Reset
- Revise every plan file before further code changes.
- Add `plan/05_workflow.md` for backend Python file refactor order and dependency flow.
- Freeze implementation until the revised plan is approved.

### Success Criteria
- All plan files reflect the current intended direction.
- `05_workflow.md` exists and covers backend file-level workflow.
- No additional code refactor happens before approval.

## Phase 2: Entry Point And Runtime
- Redefine `main.py` as the single public entrypoint.
- Decide whether startup orchestration lives directly in `main.py` or in one internal launcher module called only by `main.py`.
- Remove or reduce `scripts/` if it duplicates the single-entry model.

### Success Criteria
- Users only need one startup entrypoint: `main.py`.
- Startup ownership is not split across multiple competing entry files.
- The final design clearly explains what happens to `scripts/`.

## Phase 3: Backend Rename And Preservation
- Rename `app/` to `backend/`.
- Update imports, startup wiring, docs, and folder READMEs.
- Preserve chat compare behavior while moving code.

### Success Criteria
- `backend/` fully replaces `app/` in active code and docs.
- Chat compare still works.
- No undocumented mixed `app/` and `backend/` structure remains.

## Phase 4: Documentation And Cleanup
- Add or update `README.md` in every kept folder.
- Update root `README.md` to reflect the final approved structure only.
- Decide whether `tests/` is removed, ignored, or minimally retained.

### Success Criteria
- Every kept folder has a current `README.md`.
- Root `README.md` matches the actual repo structure.
- `tests/` and `scripts/` have explicit final status.

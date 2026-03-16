# Compare-AI

Local-first LLM comparison app with a single public entrypoint in [`main.py`](/home/user/compare-ai/main.py).

## Current Structure
- [`backend/`](/home/user/compare-ai/backend): FastAPI + LangGraph backend
- [`frontend/`](/home/user/compare-ai/frontend): React frontend
- [`plan/`](/home/user/compare-ai/plan): approved refactor planning docs
- [`docs/`](/home/user/compare-ai/docs): historical notes and supporting guides

## Core Features
- Chat comparison across multiple providers
- NDJSON streaming responses from the backend
- Local runtime without requiring Supabase, Upstash, or Render in the default path

## Run
```bash
pip install -r requirements.txt
cd frontend && npm install && cd ..
python3 main.py
```

### Modes
- Full local app: `python3 main.py`
- Backend only: `APP_MODE=api python3 main.py`

### Default Local URLs
- Backend: `http://127.0.0.1:8000`
- Frontend: `http://127.0.0.1:5173`

## Backend Contracts Preserved
- `GET /health`
- `POST /api/ask`

## Notes
- `backend/` replaced the old `app/` package as the active backend root.
- `scripts/` and `tests/` are no longer part of the active structure.
- Prompt evaluation has been removed from the active runtime.
- Folder-level READMEs are expected to match the current contents of each kept directory.

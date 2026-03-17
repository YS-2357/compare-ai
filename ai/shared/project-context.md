# Project Context

- Project: Compare-AI
- Goal: compare multiple LLM/provider responses in one local-first application
- Entry point: `main.py`
- Backend: FastAPI + LangGraph in `backend/`
- Frontend: React app in `frontend/`
- Default local flow should work without cloud-only infrastructure
- Preserved API endpoints: `GET /health`, `POST /api/ask`

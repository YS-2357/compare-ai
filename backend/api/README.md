# backend/api/

FastAPI-facing API layer.

## Contents
- [`routes.py`](/home/user/compare-ai/backend/api/routes.py): health and chat endpoints
- [`chat_handlers.py`](/home/user/compare-ai/backend/api/chat_handlers.py): stream accumulation and summary helpers
- [`deps.py`](/home/user/compare-ai/backend/api/deps.py): shared FastAPI dependencies
- [`error_handlers.py`](/home/user/compare-ai/backend/api/error_handlers.py): API error mapping
- [`schemas/`](/home/user/compare-ai/backend/api/schemas): request and response models

## Boundary
- This layer should stay thin.
- Business logic belongs in `backend/services/`.

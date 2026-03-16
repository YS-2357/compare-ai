# backend/

Active backend package for Compare-AI.

## Contents
- [`api/`](/home/user/compare-ai/backend/api): routes, dependencies, schemas, error handlers
- [`services/`](/home/user/compare-ai/backend/services): chat compare and shared provider logic
- [`utils/`](/home/user/compare-ai/backend/utils): config and logging
- [`main.py`](/home/user/compare-ai/backend/main.py): FastAPI app factory

## Boundary
- This folder owns backend runtime behavior and API contracts.
- Chat compare is the only active feature flow here.

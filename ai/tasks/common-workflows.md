# Common Workflows

## Local Run
```bash
pip install -r requirements.txt
cd frontend && npm install && cd ..
python3 main.py
```

## Backend Only
```bash
APP_MODE=api python3 main.py
```

## Change Rules
- Preserve `GET /health` and `POST /api/ask` unless the task explicitly changes the contract.
- Prefer documenting provider-comparison behavior changes in repo docs when they affect users.

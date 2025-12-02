# scripts/

> 최종 업데이트: 2025-12-03 — FastAPI/Streamlit 동시 실행 스크립트 안내 유지

- `run_app.py`: FastAPI(Uvicorn) + Streamlit을 함께 실행하는 개발용 스크립트.
- `python main.py` 실행 시 기본으로 run_app을 호출하며, `APP_MODE=api` 설정 시 FastAPI만 단독 실행.

환경변수:
- `FASTAPI_HOST`, `PORT` (Render 등에서 주입)
- `STREAMLIT_SERVER_PORT`, `STREAMLIT_SERVER_HEADLESS`

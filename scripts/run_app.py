"""FastAPI와 Streamlit(로컬용)을 함께 실행하는 CLI 스크립트."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import uvicorn

from app.config import get_settings
from app.logger import get_logger
from app.main import app as fastapi_app

logger = get_logger(__name__)
BASE_DIR = Path(__file__).resolve().parents[1]
STREAMLIT_SCRIPT = BASE_DIR / "app" / "ui" / "streamlit_app.py"
FASTAPI_URL_FILE = BASE_DIR / ".fastapi_url"


def _run_fastapi(host: str, port: int) -> None:
    """별도 스레드에서 FastAPI(Uvicorn) 서버를 실행한다.

    Args:
        host: FastAPI 서버가 바인딩할 호스트/IP.
        port: FastAPI 서버 포트.

    Returns:
        None
    """
    logger.debug("_run_fastapi:시작 host=%s port=%s", host, port)
    logger.info("FastAPI 서버 시작: http://%s:%s", host, port)
    config = uvicorn.Config(fastapi_app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    server.run()
    logger.debug("_run_fastapi:종료 host=%s port=%s", host, port)


def _ensure_streamlit_config() -> None:
    """Streamlit 실행에 필요한 최소 설정 파일을 생성한다."""

    logger.debug("_ensure_streamlit_config:시작")
    config_dir = Path.home() / ".streamlit"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.toml"
    if not config_file.exists():
        config_file.write_text("[browser]\ngatherUsageStats = false\n")
        logger.info("_ensure_streamlit_config:config.toml 생성")
    logger.debug("_ensure_streamlit_config:종료")


def _run_streamlit(port: int, env: dict[str, str]) -> None:
    """지정된 포트에서 Streamlit 앱을 실행한다."""

    logger.debug("_run_streamlit:시작 port=%s", port)
    _ensure_streamlit_config()
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(STREAMLIT_SCRIPT),
        "--server.port",
        str(port),
        "--server.headless=true",
    ]
    logger.info("Streamlit 실행: 포트 %s", port)
    subprocess.run(cmd, check=False, env=env)
    logger.debug("_run_streamlit:종료 port=%s", port)


def main() -> None:
    """FastAPI(Uvicorn)와 Streamlit(로컬용)을 부트스트랩한다."""

    logger.debug("run_app main:시작")
    settings = get_settings()
    host = os.getenv("FASTAPI_HOST", settings.fastapi_host)
    fastapi_port = int(os.getenv("PORT") or settings.fastapi_port)
    streamlit_port = int(os.getenv("STREAMLIT_SERVER_PORT") or settings.streamlit_port)

    fastapi_url = f"http://{host}:{fastapi_port}/api/ask"

    # FastAPI URL을 파일과 환경변수에 저장 (Streamlit에서 사용)
    try:
        FASTAPI_URL_FILE.write_text(fastapi_url)
    except OSError:
        pass
    os.environ["FASTAPI_URL"] = fastapi_url
    os.environ["STREAMLIT_SERVER_PORT"] = str(streamlit_port)
    os.environ["STREAMLIT_SERVER_HEADLESS"] = str(settings.streamlit_headless).lower()

    env = os.environ.copy()

    api_thread = threading.Thread(
        target=_run_fastapi,
        args=(host, fastapi_port),
        name="fastapi-thread",
        daemon=True,
    )
    api_thread.start()

    logger.info("FastAPI: http://%s:%s", host, fastapi_port)
    logger.info("Streamlit: http://%s:%s", host, streamlit_port)
    logger.info("FastAPI URL이 설정되었습니다: %s", fastapi_url)

    try:
        _run_streamlit(streamlit_port, env)
    except KeyboardInterrupt:
        logger.warning("사용자 중단 감지, Streamlit 종료")
    finally:
        logger.debug("run_app main:종료")


if __name__ == "__main__":
    main()

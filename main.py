"""단일 로컬 엔트리포인트."""

from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import uvicorn

from backend.utils.config import get_settings
from backend.main import app as fastapi_app

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"


def _port_in_use(host: str, port: int) -> bool:
    """호스트/포트가 이미 사용 중인지 확인한다."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def _http_available(url: str) -> bool:
    """로컬 HTTP 엔드포인트가 응답하는지 확인한다."""

    try:
        with urllib.request.urlopen(url, timeout=0.5) as response:
            return 200 <= getattr(response, "status", 0) < 500
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False


def _wait_for_backend(api_base_url: str, timeout_seconds: float = 5.0) -> bool:
    """새로 띄운 백엔드가 헬스 체크에 응답할 때까지 대기한다."""

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _http_available(f"{api_base_url}/health"):
            return True
        time.sleep(0.1)
    return False


def _run_fastapi(host: str, port: int) -> None:
    """별도 스레드에서 FastAPI 서버를 실행한다."""

    config = uvicorn.Config(fastapi_app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    server.run()


def _run_frontend(host: str, port: int, api_base_url: str, env: dict[str, str]) -> None:
    """React 개발 서버를 실행한다."""

    if not FRONTEND_DIR.exists():
        raise FileNotFoundError(f"frontend directory not found: {FRONTEND_DIR}")
    command = [
        "npm",
        "run",
        "dev",
        "--",
        "--host",
        host,
        "--port",
        str(port),
    ]
    run_env = env | {"VITE_API_BASE_URL": api_base_url}
    subprocess.run(command, check=False, env=run_env, cwd=FRONTEND_DIR)


def main() -> None:
    """`main.py` 하나로 백엔드 또는 전체 로컬 앱을 실행한다."""

    mode = os.getenv("APP_MODE", "").lower()
    settings = get_settings()
    host = os.getenv("FASTAPI_HOST", settings.fastapi_host)
    port = int(os.getenv("PORT") or settings.fastapi_port)

    if mode == "api":
        uvicorn.run(fastapi_app, host=host, port=port, log_level="info")
        return

    frontend_host = os.getenv("FRONTEND_HOST", settings.frontend_host)
    frontend_port = int(os.getenv("FRONTEND_PORT") or settings.frontend_port)
    api_base_url = f"http://{host}:{port}"
    frontend_url = f"http://{frontend_host}:{frontend_port}"
    env = os.environ.copy()
    backend_reused = False
    frontend_reused = False

    if _port_in_use(host, port):
        if _http_available(f"{api_base_url}/health"):
            backend_reused = True
            print(f"Backend already running at {api_base_url}; reusing existing process.")
        else:
            raise SystemExit(
                f"Port {port} is already in use on {host}, but it does not look like Compare-AI backend."
            )

    api_thread: threading.Thread | None = None
    if not backend_reused:
        api_thread = threading.Thread(
            target=_run_fastapi,
            args=(host, port),
            name="fastapi-thread",
            daemon=True,
        )
        api_thread.start()
        if not _wait_for_backend(api_base_url):
            raise SystemExit(f"Backend failed to start on {api_base_url}.")

    if _port_in_use(frontend_host, frontend_port):
        if _http_available(frontend_url):
            frontend_reused = True
            print(f"Frontend already running at {frontend_url}; reusing existing process.")
            print(f"Open {frontend_url}")
        else:
            raise SystemExit(
                f"Port {frontend_port} is already in use on {frontend_host}, but it does not look like Compare-AI frontend."
            )

    if frontend_reused:
        if backend_reused:
            return
        if api_thread is not None:
            api_thread.join()
        return

    _run_frontend(frontend_host, frontend_port, api_base_url, env)


if __name__ == "__main__":
    main()

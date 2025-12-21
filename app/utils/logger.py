"""프로젝트 전역에서 재사용할 이모지 기반 로거 유틸리티."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

LEVEL_EMOJI: dict[int, str] = {
    logging.DEBUG: "🛠️",
    logging.INFO: "✅",
    logging.WARNING: "⚠️",
    logging.ERROR: "❌",
    logging.CRITICAL: "💥",
}


class _Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    TIME = "\033[90m"
    NAME = "\033[94m"
    DEBUG = "\033[36m"  # Cyan
    INFO = "\033[92m"  # Green
    WARNING = "\033[93m"  # Yellow
    ERROR = "\033[91m"  # Red
    CRITICAL = "\033[95m"  # Magenta


LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: _Colors.DEBUG,
    logging.INFO: _Colors.INFO,
    logging.WARNING: _Colors.WARNING,
    logging.ERROR: _Colors.ERROR,
    logging.CRITICAL: _Colors.CRITICAL,
}


class EmojiFormatter(logging.Formatter):
    """로그 레코드에 로그 레벨에 따른 이모지를 추가한다."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        emoji = LEVEL_EMOJI.get(record.levelno, "")
        level_color = LEVEL_COLORS.get(record.levelno, _Colors.RESET)
        time_str = self.formatTime(record, self.datefmt)
        level_name = f"{_Colors.BOLD}{level_color}{record.levelname}{_Colors.RESET}"
        logger_name = f"{_Colors.NAME}{record.name}{_Colors.RESET}"
        message = record.getMessage()
        # 예외가 있으면 기본 포맷터가 붙이도록 그대로 둠
        formatted = f"{emoji} [{level_name}] {_Colors.TIME}{time_str}{_Colors.RESET} {logger_name} - {message}"
        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)
        return formatted


def get_logger(name: str) -> logging.Logger:
    """표준 출력으로 기록하는 프로젝트 전용 로거를 반환한다."""

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        formatter = EmojiFormatter(datefmt="%H:%M:%S")
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 파일 핸들러 추가 (루트/logs/날짜.log)
        logs_dir = Path(__file__).resolve().parents[2] / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_path = logs_dir / f"{date.today().isoformat()}.log"
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.propagate = False
    return logger

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

_CONFIGURED = False


class RedactingFilter(logging.Filter):
    """Redact known secret-bearing values from log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True

        api_key = os.getenv("OLLAMA_API_KEY", "").strip()
        if api_key:
            message = message.replace(api_key, "***REDACTED***")

        # Keep common bearer-token forms from being accidentally logged.
        message = re.sub(
            r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+",
            r"\1***REDACTED***",
            message,
        )
        record.msg = message
        record.args = ()
        return True


class _LineBufferingHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        try:
            self.flush()
        except Exception:
            pass


def _parse_level(value: str | None) -> int:
    name = (value or "INFO").strip().upper()
    level = getattr(logging, name, None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid LLMWIKI_LOG_LEVEL={value!r}; expected DEBUG/INFO/WARNING/ERROR/CRITICAL")
    return level


def configure_logging(
    *,
    component: str = "app",
    log_file: Path | None = None,
    level: str | None = None,
) -> logging.Logger:
    """Configure root logging once for both console and optional file output."""

    global _CONFIGURED

    root = logging.getLogger()
    root_level = _parse_level(level or os.getenv("LLMWIKI_LOG_LEVEL", "INFO"))
    root.setLevel(root_level)

    if _CONFIGURED:
        return logging.getLogger(component)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    redactor = RedactingFilter()

    console = _LineBufferingHandler(sys.stdout)
    console.setLevel(root_level)
    console.setFormatter(formatter)
    console.addFilter(redactor)
    root.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(root_level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(redactor)
        root.addHandler(file_handler)

    # Make noisy third-party libraries less overwhelming unless DEBUG was requested.
    if root_level > logging.DEBUG:
        for noisy_name in ("httpx", "httpcore", "urllib3", "PIL", "rapidocr"):
            logging.getLogger(noisy_name).setLevel(logging.WARNING)

    _CONFIGURED = True
    return logging.getLogger(component)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_environment(logger: logging.Logger, keys: tuple[str, ...] = ("OLLAMA_HOST", "OLLAMA_API_KEY")) -> None:
    """Log relevant environment configuration without exposing secrets."""
    for key in keys:
        value = os.getenv(key)
        if key == "OLLAMA_API_KEY":
            logger.info("%s=%s", key, "SET" if value else "NOT_SET")
        else:
            logger.info("%s=%s", key, value or "<not set>")

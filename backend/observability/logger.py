import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from backend.config import settings


_LOGGER_NAME = "sandevistan-rag"


def _configure_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)

    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(handler)
    logger.setLevel(settings.log_level.upper())
    logger.propagate = False

    return logger


logger = _configure_logger()


def _safe_json_default(value: Any) -> str:
    return str(value)


def log_event(
    event: str,
    level: str = "INFO",
    request_id: str | None = None,
    **fields: Any,
) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level.upper(),
        "event": event,
        "request_id": request_id,
        "service": settings.app_name,
        "environment": settings.app_env,
        **fields,
    }

    message = json.dumps(
        payload,
        ensure_ascii=False,
        default=_safe_json_default,
    )

    log_level = level.upper()

    if log_level == "ERROR":
        logger.error(message)
    elif log_level == "WARNING":
        logger.warning(message)
    else:
        logger.info(message)

import random
import threading
import time
from typing import Any

from backend.config import settings
from backend.observability.logger import log_event


_CHAT_CALL_LOCK = threading.Lock()
_LAST_CHAT_CALL_MONOTONIC = 0.0


def wait_for_chat_turn(
    request_id: str | None,
    deployment: str,
    purpose: str,
) -> None:
    global _LAST_CHAT_CALL_MONOTONIC

    min_interval = max(float(settings.openai_chat_min_interval_seconds), 0.0)

    if min_interval <= 0:
        return

    with _CHAT_CALL_LOCK:
        now = time.monotonic()
        elapsed = now - _LAST_CHAT_CALL_MONOTONIC
        wait_seconds = min_interval - elapsed

        if wait_seconds > 0:
            log_event(
                event="openai_chat_burst_throttle_wait_started",
                request_id=request_id,
                deployment=deployment,
                purpose=purpose,
                waitSeconds=round(wait_seconds, 3),
                minIntervalSeconds=min_interval,
            )
            time.sleep(wait_seconds)
            log_event(
                event="openai_chat_burst_throttle_wait_completed",
                request_id=request_id,
                deployment=deployment,
                purpose=purpose,
                waitedSeconds=round(wait_seconds, 3),
            )

        _LAST_CHAT_CALL_MONOTONIC = time.monotonic()


def get_retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers: Any = getattr(response, "headers", None)

    if not headers:
        return None

    retry_after_ms = headers.get("retry-after-ms")
    if retry_after_ms:
        try:
            return max(float(retry_after_ms) / 1000.0, 0.0)
        except ValueError:
            pass

    retry_after = headers.get("retry-after")
    if retry_after:
        try:
            return max(float(retry_after), 0.0)
        except ValueError:
            pass

    return None


def get_rate_limit_headers(exc: Exception) -> dict[str, str | None]:
    response = getattr(exc, "response", None)
    headers: Any = getattr(response, "headers", None)

    if not headers:
        return {}

    interesting_headers = [
        "retry-after",
        "retry-after-ms",
        "x-ratelimit-limit-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
    ]

    return {
        header: headers.get(header)
        for header in interesting_headers
        if headers.get(header) is not None
    }


def compute_retry_sleep_seconds(exc: Exception, attempt: int) -> float:
    retry_after = get_retry_after_seconds(exc)

    if retry_after is not None:
        return retry_after

    base = max(float(settings.openai_chat_retry_base_seconds), 0.0)
    jitter = max(float(settings.openai_chat_retry_jitter_seconds), 0.0)

    exponential = base * (2 ** max(attempt - 1, 0))
    return exponential + random.uniform(0, jitter)

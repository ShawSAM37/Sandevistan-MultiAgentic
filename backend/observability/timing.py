import time
from contextlib import contextmanager
from typing import Iterator


def now_ms() -> int:
    return int(time.time() * 1000)


@contextmanager
def elapsed_timer() -> Iterator[dict[str, int]]:
    start = now_ms()
    data = {"startMs": start, "elapsedMs": 0}

    try:
        yield data
    finally:
        data["elapsedMs"] = now_ms() - start

from __future__ import annotations

import base64
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests


MANIFEST_LOCAL_PATH_ENV = "SANDEVISTAN_IMAGE_MANIFEST_LOCAL_PATH"
MANIFEST_URL_ENV = "SANDEVISTAN_IMAGE_MANIFEST_URL"
MANIFEST_URL_B64_ENV = "SANDEVISTAN_IMAGE_MANIFEST_URL_B64"


def _load_jsonl_text(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        cleaned = line.strip()
        if not cleaned:
            continue

        try:
            row = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid image manifest JSONL row at line {line_number}: {exc}") from exc

        if isinstance(row, dict):
            rows.append(row)

    return rows


def _load_manifest_from_local_path(path_value: str) -> list[dict[str, Any]]:
    path = Path(path_value).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Image manifest file does not exist: {path}")

    return _load_jsonl_text(path.read_text(encoding="utf-8"))


def _load_manifest_from_url(url: str) -> list[dict[str, Any]]:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return _load_jsonl_text(response.text)


@lru_cache(maxsize=1)
def load_image_manifest() -> list[dict[str, Any]]:
    local_path = os.getenv(MANIFEST_LOCAL_PATH_ENV, "").strip()
    manifest_url_b64 = os.getenv(MANIFEST_URL_B64_ENV, "").strip()
    manifest_url = os.getenv(MANIFEST_URL_ENV, "").strip()

    if local_path:
        return _load_manifest_from_local_path(local_path)

    if manifest_url_b64:
        decoded_url = base64.b64decode(manifest_url_b64).decode("utf-8")
        return _load_manifest_from_url(decoded_url)

    if manifest_url:
        return _load_manifest_from_url(manifest_url)

    default_path = Path("image-manifest.jsonl")
    if default_path.exists():
        return _load_manifest_from_local_path(str(default_path))

    return []


@lru_cache(maxsize=1)
def load_image_manifest_index() -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}

    for row in load_image_manifest():
        file_name = row.get("fileName")
        if not file_name:
            continue

        index.setdefault(str(file_name).lower(), []).append(row)

    return index


def clear_image_manifest_cache() -> None:
    load_image_manifest.cache_clear()
    load_image_manifest_index.cache_clear()

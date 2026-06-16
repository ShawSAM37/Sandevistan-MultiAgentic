from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests


MANIFEST_LOCAL_PATH_ENV = "SANDEVISTAN_IMAGE_MANIFEST_LOCAL_PATH"
MANIFEST_URL_ENV = "SANDEVISTAN_IMAGE_MANIFEST_URL"


def _load_jsonl_text(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        cleaned = line.strip()
        if not cleaned:
            continue

        try:
            row = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL manifest row at line {line_number}: {exc}") from exc

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
    """Load image manifest from local path or URL.

    Resolution order:
    1. SANDEVISTAN_IMAGE_MANIFEST_LOCAL_PATH
    2. SANDEVISTAN_IMAGE_MANIFEST_URL
    3. ./image-manifest.jsonl if present

    This loader intentionally has no dependency on Azure SDKs.
    For deployed debug testing, use a read-only SAS URL in
    SANDEVISTAN_IMAGE_MANIFEST_URL.
    """
    local_path = os.getenv(MANIFEST_LOCAL_PATH_ENV, "").strip()
    manifest_url = os.getenv(MANIFEST_URL_ENV, "").strip()

    if local_path:
        return _load_manifest_from_local_path(local_path)

    if manifest_url:
        return _load_manifest_from_url(manifest_url)

    default_path = Path("image-manifest.jsonl")
    if default_path.exists():
        return _load_manifest_from_local_path(str(default_path))

    return []


@lru_cache(maxsize=1)
def load_image_manifest_index() -> dict[str, list[dict[str, Any]]]:
    """Return manifest rows indexed by lower-cased fileName."""
    index: dict[str, list[dict[str, Any]]] = {}

    for row in load_image_manifest():
        file_name = row.get("fileName")
        if not file_name:
            continue

        key = str(file_name).lower()
        index.setdefault(key, []).append(row)

    return index


def clear_image_manifest_cache() -> None:
    load_image_manifest.cache_clear()
    load_image_manifest_index.cache_clear()

from __future__ import annotations

import base64
import os
from typing import Iterator
from urllib.parse import quote

import requests


IMAGE_CONTAINER_URL_ENV = "SANDEVISTAN_IMAGE_CONTAINER_URL"
IMAGE_CONTAINER_URL_B64_ENV = "SANDEVISTAN_IMAGE_CONTAINER_URL_B64"

ALLOWED_BLOB_PREFIX = "latest-manuals_png_only/"
ALLOWED_EXTENSION = ".png"


class ImageBlobValidationError(ValueError):
    pass


class ImageBlobNotFoundError(FileNotFoundError):
    pass


def _load_container_url() -> str:
    url_b64 = os.getenv(IMAGE_CONTAINER_URL_B64_ENV, "").strip()
    url = os.getenv(IMAGE_CONTAINER_URL_ENV, "").strip()

    if url_b64:
        return base64.b64decode(url_b64).decode("utf-8").strip()

    if url:
        return url

    raise RuntimeError(
        f"Missing image container URL. Set {IMAGE_CONTAINER_URL_B64_ENV} or {IMAGE_CONTAINER_URL_ENV}."
    )


def _split_container_url(container_url: str) -> tuple[str, str]:
    if "?" not in container_url:
        raise RuntimeError("Image container URL must include a SAS query string.")

    base_url, sas = container_url.split("?", 1)
    return base_url.rstrip("/"), sas.lstrip("?")


def validate_blob_name(blob_name: str) -> str:
    cleaned = (blob_name or "").strip().replace("\\", "/")

    if not cleaned:
        raise ImageBlobValidationError("blobName is required.")

    if ".." in cleaned:
        raise ImageBlobValidationError("blobName must not contain '..'.")

    if cleaned.startswith("/") or cleaned.startswith("http://") or cleaned.startswith("https://"):
        raise ImageBlobValidationError("blobName must be a relative blob path.")

    if not cleaned.startswith(ALLOWED_BLOB_PREFIX):
        raise ImageBlobValidationError(
            f"blobName must start with {ALLOWED_BLOB_PREFIX!r}."
        )

    if not cleaned.lower().endswith(ALLOWED_EXTENSION):
        raise ImageBlobValidationError("Only .png images are supported.")

    return cleaned


def build_blob_sas_url(blob_name: str) -> str:
    safe_blob_name = validate_blob_name(blob_name)

    container_url = _load_container_url()
    base_url, sas = _split_container_url(container_url)

    encoded_blob_name = quote(safe_blob_name, safe="/")
    return f"{base_url}/{encoded_blob_name}?{sas}"


def open_png_blob_stream(blob_name: str) -> requests.Response:
    url = build_blob_sas_url(blob_name)

    response = requests.get(url, stream=True, timeout=30)

    if response.status_code == 404:
        response.close()
        raise ImageBlobNotFoundError(f"Image blob not found: {blob_name}")

    try:
        response.raise_for_status()
    except Exception:
        response.close()
        raise

    return response


def iter_response_content(response: requests.Response, chunk_size: int = 1024 * 64) -> Iterator[bytes]:
    try:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                yield chunk
    finally:
        response.close()

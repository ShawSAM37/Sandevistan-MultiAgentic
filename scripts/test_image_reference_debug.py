from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests


DEFAULT_QUERY = (
    "In the Emergency stop button section under Complementary protective measures, "
    "what are the emergency stop buttons used for, how do you reset one, and where "
    "are the 8 emergency stop buttons located? for DR416i"
)

EXPECTED_LOCATION_IMAGE = "GUID-AU178D6052-24DE-43C2-B8B6-7C6867A2E0-low.png"
EXPECTED_WARNING_IMAGE = "GUID-AUBD0174F2-F88B-4359-A7EB-9861579281-low.png"


def count_items(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, list):
        return len(value)
    return 1


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def pass_check(message: str) -> None:
    print(f"PASS: {message}")


def request_debug_answer(
    *,
    backend_url: str,
    query: str,
    timeout: int,
    verify_tls: bool,
) -> dict[str, Any]:
    url = backend_url.rstrip("/") + "/debug/graph-answer"

    response = requests.post(
        url,
        json={"query": query},
        timeout=timeout,
        verify=verify_tls,
    )
    response.raise_for_status()
    return response.json()


def download_rendered_image(
    *,
    backend_url: str,
    render_url: str,
    output_path: Path,
    timeout: int,
    verify_tls: bool,
) -> int:
    full_url = urljoin(backend_url.rstrip("/") + "/", render_url.lstrip("/"))

    response = requests.get(
        full_url,
        timeout=timeout,
        verify=verify_tls,
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "image/png" not in content_type.lower():
        fail(f"Expected image/png content-type, got: {content_type}")

    output_path.write_bytes(response.content)
    return len(response.content)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Regression test for debug image reference retrieval and rendering."
    )
    parser.add_argument(
        "--backend-url",
        default=os.getenv("BACKEND_URL"),
        help="Backend URL. Defaults to BACKEND_URL env var.",
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Query to test image retrieval.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification.",
    )
    parser.add_argument(
        "--out",
        default="test-image-reference-debug-render.png",
        help="Output path for rendered image.",
    )

    args = parser.parse_args()

    if not args.backend_url:
        fail("Missing --backend-url or BACKEND_URL env var.")

    verify_tls = not args.insecure

    if args.insecure:
        print("WARNING: TLS certificate verification is disabled for this regression run.")

    print("Backend URL:", args.backend_url)
    print("Query      :", args.query)
    print()

    start = time.perf_counter()

    result = request_debug_answer(
        backend_url=args.backend_url,
        query=args.query,
        timeout=args.timeout,
        verify_tls=verify_tls,
    )

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    image_debug = result.get("imageReferenceDebug") or {}
    image_refs = result.get("imageReferences") or []
    image_errors = result.get("imageReferenceErrors") or []

    print("answerFound              :", result.get("answerFound"))
    print("candidateImageRefCount   :", count_items(result.get("candidateImageReferences")))
    print("finalImageRefCount       :", count_items(image_refs))
    print("imageErrorCount          :", count_items(image_errors))
    print("elapsedMs                :", elapsed_ms)
    print("imageReferenceDebug      :", image_debug)
    print()

    if image_errors:
        fail(f"Expected no imageReferenceErrors, got: {image_errors}")

    if not result.get("answerFound"):
        fail("Expected answerFound=True.")

    if image_debug.get("resolvedImageCount", 0) <= 0:
        fail(f"Expected resolvedImageCount > 0, got: {image_debug.get('resolvedImageCount')}")

    if image_debug.get("selectedImageCount", 0) <= 0:
        fail(f"Expected selectedImageCount > 0, got: {image_debug.get('selectedImageCount')}")

    if not image_refs:
        fail("Expected imageReferences to contain at least one selected image.")

    first_image = image_refs[0]
    first_file = first_image.get("fileName")

    print("First selected image:", first_file)

    if first_file != EXPECTED_LOCATION_IMAGE:
        fail(
            "Expected location diagram to rank first. "
            f"Expected {EXPECTED_LOCATION_IMAGE}, got {first_file}."
        )

    pass_check("Location diagram ranked first.")

    all_files = {image.get("fileName") for image in image_refs}

    if EXPECTED_WARNING_IMAGE not in all_files:
        print(
            "NOTE: Warning image was not selected. This is acceptable if only top diagram is returned."
        )
    else:
        pass_check("Warning image is present as secondary/supporting image.")

    render_url = first_image.get("renderUrl")

    if not render_url:
        fail("First selected image does not include renderUrl.")

    output_path = Path(args.out)

    image_size = download_rendered_image(
        backend_url=args.backend_url,
        render_url=render_url,
        output_path=output_path,
        timeout=args.timeout,
        verify_tls=verify_tls,
    )

    if image_size <= 0:
        fail("Rendered image file is empty.")

    pass_check(f"Rendered image downloaded: {output_path} ({image_size} bytes)")

    print()
    print("Image debug regression passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

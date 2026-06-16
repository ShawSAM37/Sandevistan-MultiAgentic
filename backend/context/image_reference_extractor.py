from __future__ import annotations

import os
import re
from typing import Any


GUID_IMAGE_REFERENCE_PATTERN = re.compile(
    r"""
    (?P<rawReference>
        (?:
            (?:\.{1,2}[\\/])?
            (?:
                [A-Za-z0-9_\- .]+[\\/]
            )*
        )?
        (?P<fileName>
            GUID-[A-Za-z0-9]+
            (?:-[A-Za-z0-9]+)*
            \.(?:png|jpg|jpeg|svg|gif)
        )
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


def normalize_raw_image_reference(value: str) -> str:
    """Normalize an image reference found in text."""
    if value is None:
        return ""

    cleaned = str(value).strip()
    cleaned = cleaned.strip("\"'`()[]{}<>")
    cleaned = cleaned.rstrip(".,;:")

    return cleaned.replace("\\", "/")


def normalize_image_file_name(value: str) -> str:
    """Return a normalized image file name from a raw image reference."""
    raw = normalize_raw_image_reference(value)
    return os.path.basename(raw.replace("\\", "/"))


def extract_guid_image_references_from_text(text: str | None) -> list[dict[str, str]]:
    """Extract GUID-style image references from a text block.

    Returns simple dictionaries:
    {
        "imageId": "...",
        "fileName": "...",
        "rawReference": "..."
    }
    """
    if not text:
        return []

    results: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for match in GUID_IMAGE_REFERENCE_PATTERN.finditer(str(text)):
        raw_reference = normalize_raw_image_reference(match.group("rawReference"))
        file_name = normalize_image_file_name(match.group("fileName") or raw_reference)

        if not file_name:
            continue

        dedupe_key = (raw_reference.lower(), file_name.lower())
        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)

        results.append(
            {
                "imageId": file_name,
                "fileName": file_name,
                "rawReference": raw_reference,
            }
        )

    return results


def _citation_by_path(citations: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}

    for citation in citations or []:
        citation_path = citation.get("citationPath")
        if citation_path:
            by_path[str(citation_path)] = citation

    return by_path


def _document_field(document: dict[str, Any], field_name: str) -> Any:
    if not document:
        return None

    return document.get(field_name)


def _text_for_image_extraction(document: dict[str, Any]) -> str:
    parts: list[str] = []

    for field_name in ("content", "title", "citationPath"):
        value = _document_field(document, field_name)
        if value:
            parts.append(str(value))

    return "\n".join(parts)


def _build_image_reference(
    base_reference: dict[str, str],
    document: dict[str, Any],
    citation: dict[str, Any] | None,
) -> dict[str, Any]:
    citation = citation or {}

    citation_path = (
        citation.get("citationPath")
        or _document_field(document, "citationPath")
    )

    return {
        "imageId": base_reference.get("imageId"),
        "fileName": base_reference.get("fileName"),
        "rawReference": base_reference.get("rawReference"),
        "citationId": citation.get("citationId"),
        "citationPath": citation_path,
        "title": citation.get("title") or _document_field(document, "title"),
        "machine": citation.get("machine") or _document_field(document, "machine"),
        "baseMachine": citation.get("baseMachine") or _document_field(document, "baseMachine"),
        "serialNumber": citation.get("serialNumber") or _document_field(document, "serialNumber"),
        "manualType": citation.get("manualType") or _document_field(document, "manualType"),
        "usedInAnswer": False,
        "source": "context_document",
    }


def extract_image_references_from_documents(
    documents: list[dict[str, Any]] | None,
    citations: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Extract candidate image references from retrieved/context documents.

    This does not decide whether the image was used in the final answer.
    It only attaches image filenames to their source citation/document metadata.
    """
    if not documents:
        return []

    citations_by_path = _citation_by_path(citations)
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for document in documents:
        if not isinstance(document, dict):
            continue

        citation_path = _document_field(document, "citationPath")
        citation = citations_by_path.get(str(citation_path)) if citation_path else None

        for base_reference in extract_guid_image_references_from_text(
            _text_for_image_extraction(document)
        ):
            image_reference = _build_image_reference(
                base_reference=base_reference,
                document=document,
                citation=citation,
            )

            dedupe_key = (
                str(image_reference.get("citationPath") or ""),
                str(image_reference.get("fileName") or "").lower(),
            )

            if dedupe_key in seen:
                continue

            seen.add(dedupe_key)
            results.append(image_reference)

    return results


def filter_image_references_for_used_citations(
    candidate_image_references: list[dict[str, Any]] | None,
    used_citation_paths: list[str] | None,
) -> list[dict[str, Any]]:
    """Return image references whose citationPath was used in the final answer."""
    if not candidate_image_references or not used_citation_paths:
        return []

    used_paths = {str(path) for path in used_citation_paths if path}
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for reference in candidate_image_references:
        citation_path = reference.get("citationPath")

        if not citation_path or str(citation_path) not in used_paths:
            continue

        filtered_reference = dict(reference)
        filtered_reference["usedInAnswer"] = True
        filtered_reference["source"] = "used_citation"

        dedupe_key = (
            str(filtered_reference.get("citationPath") or ""),
            str(filtered_reference.get("fileName") or "").lower(),
        )

        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)
        results.append(filtered_reference)

    return results

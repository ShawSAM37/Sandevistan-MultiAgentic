from __future__ import annotations

import os
import re
from typing import Any


# Matches GUID-style image filenames and lightweight relative paths containing them.
#
# Examples:
# - GUID-AT6A185541-2DC0-48A2-B264-1CB51788AE-low.png
# - ./images/GUID-AT6A185541-2DC0-48A2-B264-1CB51788AE-low.png
# - ../Images/GUID-AT6A185541-2DC0-48A2-B264-1CB51788AE-high.jpg
#
# Intentionally focused on GUID images for MVP to avoid extracting unrelated icons/assets.
GUID_IMAGE_REFERENCE_PATTERN = re.compile(
    r"""
    (?P<rawReference>
        (?:
            (?:\.{1,2}/)?
            (?:
                [A-Za-z0-9_\- .]+/
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


def normalize_image_file_name(value: str) -> str:
    """Return the normalized file name from an image reference."""

    cleaned = (value or "").strip().replace("\\", "/")
    return os.path.basename(cleaned)


def normalize_raw_image_reference(value: str) -> str:
    """Normalize slashes and trim punctuation often found around HTML/text refs."""

    cleaned = (value or "").strip()

    cleaned = cleaned.strip("\"'`()[]{}<>")
    cleaned = cleaned.replace("\\", "/")

    return cleaned


def extract_guid_image_references_from_text(text: str | None) -> list[dict[str, str]]:
    """Extract GUID-style image references from one text block.

    Returns dictionaries with:
    - rawReference
    - fileName
    - imageId
    """

    if not text:
        return []

    references: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for match in GUID_IMAGE_REFERENCE_PATTERN.finditer(text):
        raw_reference = normalize_raw_image_reference(match.group("rawReference"))
        file_name = normalize_image_file_name(match.group("fileName"))

        if not file_name:
            continue

        key = (raw_reference.lower(), file_name.lower())

        if key in seen:
            continue

        seen.add(key)

        references.append(
            {
                "imageId": file_name,
                "fileName": file_name,
                "rawReference": raw_reference,
            }
        )

    return references


def _citation_by_path(citations: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}

    for citation in citations or []:
        citation_path = citation.get("citationPath")

        if citation_path:
            by_path[str(citation_path)] = citation

    return by_path


def _document_field(document: dict[str, Any], field_name: str) -> Any:
    if field_name in document:
        return document.get(field_name)

    # Some search wrappers may nest the original document.
    source = document.get("source") or document.get("document") or {}

    if isinstance(source, dict):
        return source.get(field_name)

    return None


def _text_for_image_extraction(document: dict[str, Any]) -> str:
    parts: list[str] = []

    for field_name in ["content", "title", "citationPath"]:
        value = _document_field(document, field_name)

        if value:
            parts.append(str(value))

    return "\n".join(parts)


def _build_image_reference(
    base_reference: dict[str, str],
    document: dict[str, Any],
    citation: dict[str, Any] | None,
) -> dict[str, Any]:
    citation_path = (
        (citation or {}).get("citationPath")
        or _document_field(document, "citationPath")
    )

    return {
        "imageId": base_reference["imageId"],
        "fileName": base_reference["fileName"],
        "rawReference": base_reference.get("rawReference"),
        "citationId": (citation or {}).get("citationId"),
        "citationPath": citation_path,
        "title": (citation or {}).get("title") or _document_field(document, "title"),
        "machine": (citation or {}).get("machine") or _document_field(document, "machine"),
        "baseMachine": (citation or {}).get("baseMachine") or _document_field(document, "baseMachine"),
        "serialNumber": (citation or {}).get("serialNumber") or _document_field(document, "serialNumber"),
        "manualType": (citation or {}).get("manualType") or _document_field(document, "manualType"),
        "usedInAnswer": False,
        "source": "context_document",
    }


def extract_image_references_from_documents(
    documents: list[dict[str, Any]] | None,
    citations: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Extract candidate image references from context/retrieved documents.

    This returns a candidate pool. Final relevance filtering should happen later
    using final_used_citation_paths.
    """

    citation_lookup = _citation_by_path(citations)
    image_references: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None]] = set()

    for document in documents or []:
        if not isinstance(document, dict):
            continue

        citation_path = _document_field(document, "citationPath")
        citation = citation_lookup.get(str(citation_path)) if citation_path else None

        text = _text_for_image_extraction(document)
        extracted = extract_guid_image_references_from_text(text)

        for base_reference in extracted:
            image_reference = _build_image_reference(
                base_reference=base_reference,
                document=document,
                citation=citation,
            )

            key = (
                str(image_reference.get("fileName", "")).lower(),
                image_reference.get("citationPath"),
            )

            if key in seen:
                continue

            seen.add(key)
            image_references.append(image_reference)

    return image_references


def filter_image_references_for_used_citations(
    candidate_image_references: list[dict[str, Any]] | None,
    used_citation_paths: list[str] | None,
) -> list[dict[str, Any]]:
    """Return only image references whose citationPath was used in the final answer."""

    used_paths = {str(path) for path in used_citation_paths or [] if path}

    if not used_paths:
        return []

    filtered: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None]] = set()

    for image_reference in candidate_image_references or []:
        citation_path = image_reference.get("citationPath")

        if citation_path not in used_paths:
            continue

        updated_reference = dict(image_reference)
        updated_reference["usedInAnswer"] = True
        updated_reference["source"] = "used_citation"

        key = (
            str(updated_reference.get("fileName", "")).lower(),
            updated_reference.get("citationPath"),
        )

        if key in seen:
            continue

        seen.add(key)
        filtered.append(updated_reference)

    return filtered

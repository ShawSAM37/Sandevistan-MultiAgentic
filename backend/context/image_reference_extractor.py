from __future__ import annotations

import os
import re
from typing import Any


PNG_IMAGE_REFERENCE_PATTERN = re.compile(
    r"""
    (?P<rawReference>
        (?:
            (?:\.{1,2}[\\/])?
            (?:
                [A-Za-z0-9_\- .%]+[\\/]
            )*
        )?
        (?P<fileName>
            GUID-[A-Za-z0-9]+
            (?:-[A-Za-z0-9]+)*
            -low\.png
        )
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


def normalize_raw_image_reference(value: str | None) -> str:
    if value is None:
        return ""

    cleaned = str(value).strip()
    cleaned = cleaned.strip("\"'`()[]{}<>")
    cleaned = cleaned.rstrip(".,;:")
    return cleaned.replace("\\", "/")


def normalize_image_file_name(value: str | None) -> str:
    raw = normalize_raw_image_reference(value)
    return os.path.basename(raw.replace("\\", "/"))


def _compact_text(value: str) -> str:
    return " ".join(str(value or "").split())


def _nearby_text(text: str, start: int, end: int, window_chars: int = 700) -> str:
    left = max(0, start - window_chars)
    right = min(len(text), end + window_chars)
    return _compact_text(text[left:right])


def _before_text(text: str, start: int, window_chars: int = 450) -> str:
    left = max(0, start - window_chars)
    return _compact_text(text[left:start])


def _after_text(text: str, end: int, window_chars: int = 700) -> str:
    right = min(len(text), end + window_chars)
    return _compact_text(text[end:right])


def extract_png_image_references_from_text(
    text: str | None,
    *,
    include_nearby_text: bool = True,
    nearby_window_chars: int = 700,
) -> list[dict[str, Any]]:
    if not text:
        return []

    source_text = str(text)
    matches = list(PNG_IMAGE_REFERENCE_PATTERN.finditer(source_text))

    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for zero_index, match in enumerate(matches):
        image_index = zero_index + 1
        raw_reference = normalize_raw_image_reference(match.group("rawReference"))
        file_name = normalize_image_file_name(match.group("fileName") or raw_reference)

        if not file_name.lower().endswith(".png"):
            continue

        key = (raw_reference.lower(), file_name.lower())
        if key in seen:
            continue

        seen.add(key)

        previous_end = matches[zero_index - 1].end() if zero_index > 0 else 0
        next_start = matches[zero_index + 1].start() if zero_index + 1 < len(matches) else len(source_text)

        # Segment before/after is bounded by neighboring images.
        # This prevents the first warning icon from inheriting the second diagram's table/list context.
        before_segment = source_text[max(previous_end, match.start() - nearby_window_chars):match.start()]
        after_segment = source_text[match.end():min(next_start, match.end() + nearby_window_chars)]

        segment_left = max(previous_end, match.start() - nearby_window_chars)
        segment_right = min(next_start, match.end() + nearby_window_chars)
        nearby_segment = source_text[segment_left:segment_right]

        item: dict[str, Any] = {
            "imageId": file_name,
            "fileName": file_name,
            "rawReference": raw_reference,
            "imageIndexInChunk": image_index,
        }

        if include_nearby_text:
            item["nearbyText"] = _compact_text(nearby_segment)
            item["textBeforeImage"] = _compact_text(before_segment)
            item["textAfterImage"] = _compact_text(after_segment)

        results.append(item)

    return results

def _citation_by_path(citations: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}

    for citation in citations or []:
        citation_path = citation.get("citationPath")
        if citation_path:
            by_path[str(citation_path)] = citation

    return by_path


def _document_text_for_extraction(document: dict[str, Any]) -> str:
    parts: list[str] = []

    for field_name in ("content", "title", "citationPath", "filepath", "fileName"):
        value = document.get(field_name)
        if value:
            parts.append(str(value))

    return "\n".join(parts)


def _build_reference(
    base_reference: dict[str, Any],
    document: dict[str, Any],
    citation: dict[str, Any] | None,
) -> dict[str, Any]:
    citation = citation or {}

    citation_path = citation.get("citationPath") or document.get("citationPath")

    return {
        "imageId": base_reference.get("imageId"),
        "fileName": base_reference.get("fileName"),
        "rawReference": base_reference.get("rawReference"),
        "imageIndexInChunk": base_reference.get("imageIndexInChunk"),
        "nearbyText": base_reference.get("nearbyText", ""),
        "textBeforeImage": base_reference.get("textBeforeImage", ""),
        "textAfterImage": base_reference.get("textAfterImage", ""),
        "citationId": citation.get("citationId"),
        "citationPath": citation_path,
        "title": citation.get("title") or document.get("title"),
        "machine": citation.get("machine") or document.get("machine"),
        "baseMachine": citation.get("baseMachine") or document.get("baseMachine"),
        "serialNumber": citation.get("serialNumber") or document.get("serialNumber"),
        "manualType": citation.get("manualType") or document.get("manualType"),
        "usedInAnswer": False,
        "source": "context_document",
    }


def extract_image_references_from_documents(
    documents: list[dict[str, Any]] | None,
    citations: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not documents:
        return []

    citations_by_path = _citation_by_path(citations)
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for document in documents:
        if not isinstance(document, dict):
            continue

        citation_path = document.get("citationPath")
        citation = citations_by_path.get(str(citation_path)) if citation_path else None

        for base_reference in extract_png_image_references_from_text(
            _document_text_for_extraction(document)
        ):
            reference = _build_reference(base_reference, document, citation)

            dedupe_key = (
                str(reference.get("citationPath") or ""),
                str(reference.get("fileName") or "").lower(),
            )

            if dedupe_key in seen:
                continue

            seen.add(dedupe_key)
            results.append(reference)

    return results


def filter_image_references_for_used_citations(
    candidate_image_references: list[dict[str, Any]] | None,
    used_citation_paths: list[str] | None,
) -> list[dict[str, Any]]:
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


def extract_image_references_from_context_text(
    context: str | None,
    citations: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not context:
        return []

    context_text = str(context)
    citations = citations or []

    if not citations:
        synthetic_doc = {
            "content": context_text,
            "citationPath": None,
            "title": None,
        }
        return extract_image_references_from_documents([synthetic_doc], [])

    positioned: list[tuple[int, dict[str, Any]]] = []

    for citation in citations:
        citation_path = citation.get("citationPath")
        if not citation_path:
            continue

        position = context_text.find(str(citation_path))
        if position >= 0:
            positioned.append((position, citation))

    positioned.sort(key=lambda item: item[0])

    synthetic_documents: list[dict[str, Any]] = []

    if positioned:
        for index, (start, citation) in enumerate(positioned):
            end = positioned[index + 1][0] if index + 1 < len(positioned) else len(context_text)
            section_text = context_text[start:end]

            synthetic_documents.append(
                {
                    "content": section_text,
                    "title": citation.get("title"),
                    "citationPath": citation.get("citationPath"),
                    "machine": citation.get("machine"),
                    "baseMachine": citation.get("baseMachine"),
                    "serialNumber": citation.get("serialNumber"),
                    "manualType": citation.get("manualType"),
                }
            )

        return extract_image_references_from_documents(
            synthetic_documents,
            citations,
        )

    if len(citations) == 1:
        citation = citations[0]
        synthetic_doc = {
            "content": context_text,
            "title": citation.get("title"),
            "citationPath": citation.get("citationPath"),
            "machine": citation.get("machine"),
            "baseMachine": citation.get("baseMachine"),
            "serialNumber": citation.get("serialNumber"),
            "manualType": citation.get("manualType"),
        }

        return extract_image_references_from_documents(
            [synthetic_doc],
            citations,
        )

    return []


def extract_image_references_from_single_context_for_citation(
    context: str | None,
    citation: dict[str, Any],
) -> list[dict[str, Any]]:
    if not context:
        return []

    synthetic_doc = {
        "content": str(context),
        "title": citation.get("title"),
        "citationPath": citation.get("citationPath"),
        "machine": citation.get("machine"),
        "baseMachine": citation.get("baseMachine"),
        "serialNumber": citation.get("serialNumber"),
        "manualType": citation.get("manualType"),
    }

    return extract_image_references_from_documents(
        documents=[synthetic_doc],
        citations=[citation],
    )

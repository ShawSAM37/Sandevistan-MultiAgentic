from typing import Any


def build_citation_from_document(
    document: dict[str, Any],
    citation_id: int,
) -> dict[str, Any]:
    return {
        "citationId": citation_id,
        "id": document.get("id"),
        "title": document.get("title"),
        "citationPath": document.get("citationPath"),
        "machine": document.get("machine"),
        "baseMachine": document.get("baseMachine"),
        "serialNumber": document.get("serialNumber"),
        "manualType": document.get("manualType"),
    }


def citation_key(document: dict[str, Any]) -> str:
    citation_path = document.get("citationPath")
    if citation_path:
        return f"citationPath::{citation_path}"

    document_id = document.get("id")
    if document_id:
        return f"id::{document_id}"

    title = document.get("title") or ""
    machine = document.get("machine") or ""
    manual_type = document.get("manualType") or ""

    return f"fallback::{title}::{machine}::{manual_type}"


def build_deduplicated_citations(
    documents: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    citations: list[dict[str, Any]] = []
    citation_id_by_key: dict[str, int] = {}

    for document in documents:
        key = citation_key(document)

        if key in citation_id_by_key:
            continue

        citation_id = len(citations) + 1
        citation_id_by_key[key] = citation_id
        citations.append(build_citation_from_document(document, citation_id))

    return citations, citation_id_by_key

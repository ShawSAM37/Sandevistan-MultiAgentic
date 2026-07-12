from typing import Any

from backend.config import settings
from backend.context.citation_builder import build_deduplicated_citations, citation_key
from backend.observability.logger import log_event
from backend.observability.timing import elapsed_timer


def safe_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False

    return text[:max_chars].rstrip() + "\n\n[TRUNCATED]", True


def build_context_block(
    document: dict[str, Any],
    citation_id: int,
    max_chars_per_document: int,
) -> tuple[str, bool, int]:
    content = safe_text(document.get("content"))
    truncated_content, was_truncated = truncate_text(content, max_chars_per_document)

    title = safe_text(document.get("title")) or "Untitled"
    machine = safe_text(document.get("machine")) or "Unknown"
    base_machine = safe_text(document.get("baseMachine")) or "Unknown"
    serial_number = safe_text(document.get("serialNumber")) or "Unknown"
    manual_type = safe_text(document.get("manualType")) or "Unknown"
    citation_path = safe_text(document.get("citationPath")) or "Unavailable"
    score = document.get("score")
    reranker_score = document.get("rerankerScore")

    block = (
        f"[{citation_id}]\n"
        f"Title: {title}\n"
        f"Machine: {machine}\n"
        f"Base machine: {base_machine}\n"
        f"Serial number: {serial_number}\n"
        f"Manual type: {manual_type}\n"
        f"Citation: {citation_path}\n"
        f"Search score: {score}\n"
        f"Reranker score: {reranker_score}\n"
        f"Content:\n"
        f"{truncated_content}\n"
    )

    return block, was_truncated, len(content)


def build_context_from_documents(
    documents: list[dict[str, Any]],
    max_context_chars: int | None = None,
    max_chars_per_document: int | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    max_context_chars = max_context_chars or settings.max_context_chars
    max_chars_per_document = max_chars_per_document or settings.max_chars_per_document

    log_event(
        event="context_build_started",
        request_id=request_id,
        documentCount=len(documents),
        maxContextChars=max_context_chars,
        maxCharsPerDocument=max_chars_per_document,
    )

    with elapsed_timer() as timer:
        citations, citation_id_by_key = build_deduplicated_citations(documents)

        context_blocks: list[str] = []
        used_documents: list[dict[str, Any]] = []
        skipped_documents: list[dict[str, Any]] = []

        current_context_chars = 0

        for document in documents:
            key = citation_key(document)
            citation_id = citation_id_by_key[key]

            block, was_truncated, original_content_length = build_context_block(
                document=document,
                citation_id=citation_id,
                max_chars_per_document=max_chars_per_document,
            )

            projected_length = current_context_chars + len(block)

            if projected_length > max_context_chars:
                skipped_documents.append(
                    {
                        "id": document.get("id"),
                        "title": document.get("title"),
                        "citationPath": document.get("citationPath"),
                        "reason": "max_context_chars_exceeded",
                        "blockCharCount": len(block),
                    }
                )
                continue

            context_blocks.append(block)
            current_context_chars += len(block)

            used_documents.append(
                {
                    "id": document.get("id"),
                    "title": document.get("title"),
                    "citationPath": document.get("citationPath"),
                    "citationId": citation_id,
                    "manualType": document.get("manualType"),
                    "baseMachine": document.get("baseMachine"),
                    "serialNumber": document.get("serialNumber"),
                    "machine": document.get("machine"),
                    "score": document.get("score"),
                    "rerankerScore": document.get("rerankerScore"),
                    "contentLength": original_content_length,
                    "truncated": was_truncated,
                }
            )

        context = "\n\n---\n\n".join(context_blocks)

    result = {
        "context": context,
        "contextCharCount": len(context),
        "usedDocumentCount": len(used_documents),
        "skippedDocumentCount": len(skipped_documents),
        "citations": citations,
        "usedDocuments": used_documents,
        "skippedDocuments": skipped_documents,
        "latencyMs": timer["elapsedMs"],
    }

    log_event(
        event="context_build_completed",
        request_id=request_id,
        contextCharCount=result["contextCharCount"],
        usedDocumentCount=result["usedDocumentCount"],
        skippedDocumentCount=result["skippedDocumentCount"],
        citationCount=len(citations),
        latencyMs=timer["elapsedMs"],
    )

    return result

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from backend.context.image_reference_extractor import (
    extract_image_references_from_documents,
    filter_image_references_for_used_citations,
)
from backend.context.image_reference_resolver import resolve_image_references
from backend.context.image_reranker import rerank_image_references


def extract_candidate_images_from_chunks(
    *,
    documents: list[dict[str, Any]] | None,
    citations: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Extract PNG image candidates from raw retrieved/context documents."""
    return extract_image_references_from_documents(
        documents=documents or [],
        citations=citations or [],
    )


def retrieve_relevant_images_for_final_answer(
    *,
    question: str,
    final_answer: str,
    candidate_image_references: list[dict[str, Any]] | None,
    final_used_citation_paths: list[str] | None,
    max_images: int = 3,
) -> dict[str, Any]:
    """Filter, resolve, and rerank image candidates for the final answer."""
    debug: dict[str, Any] = {
        "candidateImageCountBeforeFilter": len(candidate_image_references or []),
        "finalUsedCitationPathCount": len(final_used_citation_paths or []),
        "usedCitationImageCount": 0,
        "resolvedImageCount": 0,
        "displayEligibleImageCount": 0,
        "selectedImageCount": 0,
        "selectionMode": "deterministic_filter_resolve_rerank",
    }

    errors: list[dict[str, Any]] = []

    try:
        used_citation_images = filter_image_references_for_used_citations(
            candidate_image_references=candidate_image_references or [],
            used_citation_paths=final_used_citation_paths or [],
        )

        debug["usedCitationImageCount"] = len(used_citation_images)

        resolved_images = resolve_image_references(used_citation_images)

        debug["resolvedImageCount"] = sum(
            1 for image in resolved_images if image.get("resolved")
        )

        reranked_images = rerank_image_references(
            question=question or "",
            final_answer=final_answer or "",
            image_references=resolved_images,
        )

        display_images = [
            image
            for image in reranked_images
            if image.get("displayEligible")
        ][:max_images]

        for image in display_images:
            blob_name = image.get("blobName")
            if blob_name:
                image["renderUrl"] = f"/images/render?blobName={quote(str(blob_name), safe='')}"

        debug["displayEligibleImageCount"] = sum(
            1 for image in reranked_images if image.get("displayEligible")
        )
        debug["selectedImageCount"] = len(display_images)

        return {
            "candidateImageReferences": reranked_images,
            "imageReferences": display_images,
            "imageReferenceDebug": debug,
            "imageReferenceErrors": errors,
        }

    except Exception as exc:
        errors.append(
            {
                "stage": "image_retrieval_agent",
                "message": str(exc),
                "recoverable": True,
            }
        )

        return {
            "candidateImageReferences": candidate_image_references or [],
            "imageReferences": [],
            "imageReferenceDebug": debug,
            "imageReferenceErrors": errors,
        }

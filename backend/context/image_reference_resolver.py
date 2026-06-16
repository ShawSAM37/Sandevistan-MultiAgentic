from __future__ import annotations

import re
from typing import Any

from backend.context.image_manifest_loader import load_image_manifest_index


_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+")


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _lower(value: Any) -> str:
    return _clean(value).lower()


def _tokens(value: Any) -> set[str]:
    return {
        match.group(0).lower()
        for match in _WORD_PATTERN.finditer(_clean(value))
    }


def _machine_hint(reference: dict[str, Any]) -> str:
    return " ".join(
        part
        for part in [
            _clean(reference.get("baseMachine")),
            _clean(reference.get("machine")),
            _clean(reference.get("citationPath")),
            _clean(reference.get("title")),
        ]
        if part
    )


def _score_manifest_match(
    reference: dict[str, Any],
    manifest_row: dict[str, Any],
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    ref_file = _lower(reference.get("fileName"))
    manifest_file = _lower(manifest_row.get("fileName"))

    if ref_file and ref_file == manifest_file:
        score += 0.50
        reasons.append("fileName exact match")

    product = _lower(manifest_row.get("product"))
    machine_hint = _lower(_machine_hint(reference))

    if product and product in machine_hint:
        score += 0.20
        reasons.append("manifest product matched citation/question context")

    ref_manual_type = _lower(reference.get("manualType"))
    manifest_manual_type = _lower(manifest_row.get("manualType"))

    if ref_manual_type and manifest_manual_type and ref_manual_type == manifest_manual_type:
        score += 0.12
        reasons.append("manualType matched citation metadata")

    citation_path = _lower(reference.get("citationPath"))
    manual_title = _lower(manifest_row.get("manualTitle"))

    if manual_title and manual_title in citation_path:
        score += 0.10
        reasons.append("manifest manual title appeared in citationPath")

    citation_tokens = _tokens(reference.get("citationPath")) | _tokens(reference.get("title"))
    manifest_tokens = _tokens(manifest_row.get("manualTitle")) | _tokens(manifest_row.get("manualFolder"))
    overlap = citation_tokens & manifest_tokens

    if overlap:
        score += min(0.08, len(overlap) * 0.01)
        reasons.append(f"citation/manual token overlap: {', '.join(sorted(list(overlap))[:6])}")

    return min(score, 1.0), reasons


def _resolve_one_image_reference(reference: dict[str, Any]) -> dict[str, Any]:
    file_name = _clean(reference.get("fileName"))

    resolved = dict(reference)
    resolved.setdefault("resolved", False)
    resolved.setdefault("displayEligible", False)
    resolved.setdefault("relevanceScore", 0.0)
    resolved.setdefault("relevance", "unresolved")
    resolved.setdefault("resolutionReason", [])

    if not file_name:
        resolved["resolutionReason"] = ["missing fileName"]
        return resolved

    manifest_index = load_image_manifest_index()
    candidates = manifest_index.get(file_name.lower(), [])

    resolved["manifestCandidateCount"] = len(candidates)

    if not candidates:
        resolved["resolutionReason"] = ["fileName not found in image manifest"]
        return resolved

    scored: list[tuple[float, list[str], dict[str, Any]]] = []

    for candidate in candidates:
        score, reasons = _score_manifest_match(reference, candidate)
        scored.append((score, reasons, candidate))

    scored.sort(key=lambda item: item[0], reverse=True)

    best_score, best_reasons, best_row = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else None

    ambiguous = second_score is not None and abs(best_score - second_score) < 0.03

    resolved["resolved"] = True
    resolved["blobContainer"] = best_row.get("blobContainer")
    resolved["blobName"] = best_row.get("blobName")
    resolved["manifestProduct"] = best_row.get("product")
    resolved["manifestManualType"] = best_row.get("manualType")
    resolved["manifestManualTitle"] = best_row.get("manualTitle")
    resolved["manifestLanguage"] = best_row.get("language")
    resolved["manifestCandidateCount"] = len(candidates)
    resolved["manifestBestScore"] = round(best_score, 3)
    resolved["manifestAmbiguous"] = ambiguous

    relevance_score = 0.50 if reference.get("usedInAnswer") else 0.0
    relevance_score += min(best_score * 0.50, 0.50)

    if ambiguous:
        relevance_score -= 0.20

    relevance_score = max(0.0, min(relevance_score, 1.0))

    if relevance_score >= 0.80 and not ambiguous:
        relevance = "strong_match"
    elif relevance_score >= 0.60 and not ambiguous:
        relevance = "context_matched"
    elif relevance_score >= 0.50:
        relevance = "citation_attached"
    else:
        relevance = "weak_or_ambiguous"

    resolved["relevanceScore"] = round(relevance_score, 3)
    resolved["relevance"] = relevance
    resolved["displayEligible"] = bool(
        resolved.get("resolved")
        and reference.get("usedInAnswer")
        and not ambiguous
        and relevance_score >= 0.60
    )

    resolution_reason = list(best_reasons)

    if ambiguous:
        resolution_reason.append("ambiguous manifest match: top candidates scored too closely")

    if not resolved["displayEligible"]:
        resolution_reason.append("not display eligible under current threshold")

    resolved["resolutionReason"] = resolution_reason

    return resolved


def resolve_image_references(
    image_references: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not image_references:
        return []

    return [_resolve_one_image_reference(reference) for reference in image_references]

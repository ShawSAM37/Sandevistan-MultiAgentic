from __future__ import annotations

import re
from typing import Any


_WORD_PATTERN = re.compile(r"[A-Za-z0-9]+")

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "onto", "then",
    "there", "their", "about", "manual", "section", "procedure", "steps", "step",
    "machine", "sandvik", "retrieved", "context", "answer", "citation", "how",
    "what", "where", "when", "why", "you", "your", "are", "is", "to", "of", "in",
    "on", "a", "an", "or", "as", "by", "be", "it", "guid", "latest", "dcs",
    "export", "html", "png", "low"
}

_WARNING_TERMS = {
    "warning", "hazard", "hazards", "danger", "caution", "alert", "injury",
    "death", "severe", "malfunctioning", "risk", "fatal", "protective"
}

_FIGURE_TERMS = {
    "figure", "illustration", "diagram", "location", "locations", "located",
    "refer", "shows", "shown", "equipped", "buttons", "button", "left", "right",
    "rear", "front", "ladder", "guardrail", "cabinet", "joystick", "mast",
    "deck", "frame", "access"
}

_LOCATION_QUESTION_TERMS = {
    "where", "located", "location", "locations", "8", "eight", "buttons",
    "button", "figure", "show", "shows"
}

_SAFETY_QUESTION_TERMS = {
    "warning", "hazard", "safety", "danger", "caution", "injury", "death"
}


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
        if len(match.group(0)) >= 3 and match.group(0).lower() not in _STOPWORDS
    }


def _has_any(text: str, terms: set[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _count_any(text: str, terms: set[str]) -> int:
    lowered = text.lower()
    return sum(1 for term in terms if term in lowered)


def _overlap_score(source: str, target: str, max_score: float) -> tuple[float, list[str]]:
    source_tokens = _tokens(source)
    target_tokens = _tokens(target)

    if not source_tokens or not target_tokens:
        return 0.0, []

    overlap = source_tokens & target_tokens
    if not overlap:
        return 0.0, []

    score = min(max_score, len(overlap) * 0.02)
    reasons = [f"text overlap: {', '.join(sorted(list(overlap))[:8])}"]
    return score, reasons


def _query_intent(question: str, final_answer: str) -> dict[str, bool]:
    combined = f"{question}\n{final_answer}".lower()

    asks_location = _has_any(combined, _LOCATION_QUESTION_TERMS)
    asks_safety = _has_any(combined, _SAFETY_QUESTION_TERMS)

    return {
        "asks_location": asks_location,
        "asks_safety": asks_safety,
    }


def _visual_context(reference: dict[str, Any]) -> str:
    return " ".join(
        [
            _clean(reference.get("nearbyText")),
            _clean(reference.get("textBeforeImage")),
            _clean(reference.get("textAfterImage")),
            _clean(reference.get("title")),
            _clean(reference.get("citationPath")),
        ]
    )


def _rerank_one(
    reference: dict[str, Any],
    *,
    question: str,
    final_answer: str,
) -> dict[str, Any]:
    item = dict(reference)

    score = 0.0
    reasons: list[str] = []

    if item.get("usedInAnswer"):
        score += 0.35
        reasons.append("image came from a final-used citation")

    if item.get("resolved"):
        score += 0.10
        reasons.append("image resolved in manifest")

    if item.get("blobName"):
        score += 0.05
        reasons.append("blobName available")

    if item.get("manifestAmbiguous"):
        score -= 0.20
        reasons.append("penalty: manifest match is ambiguous")

    base_machine = _lower(item.get("baseMachine"))
    manifest_product = _lower(item.get("manifestProduct"))

    if base_machine and manifest_product and base_machine == manifest_product:
        score += 0.10
        reasons.append("baseMachine matched manifest product")

    manual_type = _lower(item.get("manualType"))
    manifest_manual_type = _lower(item.get("manifestManualType"))

    if manual_type and manifest_manual_type and manual_type == manifest_manual_type:
        score += 0.10
        reasons.append("manualType matched manifest manualType")

    title = _clean(item.get("title"))
    citation_path = _clean(item.get("citationPath"))
    nearby_text = _clean(item.get("nearbyText"))
    before_text = _clean(item.get("textBeforeImage"))
    after_text = _clean(item.get("textAfterImage"))

    context_text = _visual_context(item)
    question_answer_text = " ".join([question, final_answer, title, citation_path])

    overlap_score, overlap_reasons = _overlap_score(
        source=context_text,
        target=question_answer_text,
        max_score=0.15,
    )
    score += overlap_score
    reasons.extend(overlap_reasons)

    intent = _query_intent(question, final_answer)

    figure_term_count = _count_any(context_text, _FIGURE_TERMS)
    warning_term_count = _count_any(context_text, _WARNING_TERMS)

    if figure_term_count:
        figure_boost = min(0.20, figure_term_count * 0.035)
        score += figure_boost
        reasons.append(f"figure/location context boost: {figure_boost:.2f}")

    if intent["asks_location"] and figure_term_count:
        score += 0.12
        reasons.append("query asks for locations and image context contains figure/location terms")

    if intent["asks_location"] and _lower(item.get("imageIndexInChunk")) not in {"", "1"}:
        score += 0.04
        reasons.append("query asks for locations and image is not the first icon-like image in chunk")

    # Warning icons are useful for safety questions, but secondary for location/diagram questions.
    if warning_term_count and not intent["asks_safety"]:
        score -= 0.23
        reasons.append("penalty: warning/hazard icon context but query is not primarily a safety-warning request")
    elif warning_term_count and intent["asks_safety"] and not intent["asks_location"]:
        score += 0.08
        reasons.append("safety warning context boost")

    # If a warning image occurs before a figure/location paragraph, keep it but make it secondary.
    if warning_term_count and intent["asks_location"] and figure_term_count < 2:
        score -= 0.08
        reasons.append("penalty: warning image is secondary to requested location figure")

    # Direct clues after an image are useful for manual diagrams.
    after_lower = after_text.lower()
    if "refer to the figure" in after_lower or "location of the buttons" in after_lower:
        score += 0.18
        reasons.append("strong boost: nearby text says to refer to figure for button locations")

    # If image is surrounded by hazard banner language, reduce it when another diagram is likely needed.
    nearby_lower = nearby_text.lower()
    if "emergency stops hazard" in nearby_lower and intent["asks_location"]:
        score -= 0.18
        reasons.append("penalty: emergency stops hazard banner image is secondary to location answer")

    score = max(0.0, min(score, 1.0))

    if score >= 0.80:
        relevance = "strong_match"
    elif score >= 0.60:
        relevance = "relevant"
    elif score >= 0.45:
        relevance = "weak"
    else:
        relevance = "reject"

    display_eligible = bool(
        item.get("usedInAnswer")
        and item.get("resolved")
        and not item.get("manifestAmbiguous")
        and score >= 0.60
    )

    item["relevanceScore"] = round(score, 3)
    item["relevance"] = relevance
    item["displayEligible"] = display_eligible
    item["rerankerReason"] = reasons

    return item


def rerank_image_references(
    *,
    question: str,
    final_answer: str,
    image_references: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not image_references:
        return []

    reranked = [
        _rerank_one(
            reference,
            question=question,
            final_answer=final_answer,
        )
        for reference in image_references
    ]

    reranked.sort(
        key=lambda item: (
            not bool(item.get("displayEligible")),
            -float(item.get("relevanceScore") or 0.0),
            int(item.get("imageIndexInChunk") or 999),
        )
    )

    return reranked

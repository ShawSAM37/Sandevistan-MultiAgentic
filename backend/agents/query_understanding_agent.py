from __future__ import annotations

import json
import re
import time
from typing import Any

from openai import APIError, APITimeoutError, BadRequestError, RateLimitError

from backend.agents.model_router import ModelRole, get_deployment_for_role
from backend.agents.openai_rate_limiter import (
    compute_retry_sleep_seconds,
    get_rate_limit_headers,
    wait_for_chat_turn,
)
from backend.config import settings
from backend.models import QueryDetectedEntities, QueryUnderstandingAgentResult
from backend.observability.logger import log_event
from backend.observability.timing import elapsed_timer
from backend.prompts.query_understanding_prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from backend.retrieval.azure_openai_client import get_azure_openai_client


def extract_json_object(text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("Model returned empty response.")

    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def build_recent_turns_json(recent_turns: list[dict[str, Any]] | None) -> str:
    if not recent_turns:
        return "[]"

    compact_turns = []

    for turn in recent_turns[-4:]:
        compact_turns.append(
            {
                "role": turn.get("role"),
                "content": str(turn.get("content", ""))[:1000],
            }
        )

    return json.dumps(compact_turns, ensure_ascii=False, indent=2)


MACHINE_PATTERN = re.compile(
    r"\b(?P<machine>(?:DR|DI|D)\s*-?\s*\d{2,4}\s*(?:i|I|KX|kx)?)\b",
    flags=re.IGNORECASE,
)

KNOWN_COMPONENT_PATTERNS = [
    "hydraulic tank breather filter",
    "hydraulic tank air filter",
    "breather filter",
    "air filter",
    "hydraulic filter",
]


def normalize_machine_name(raw_value: str) -> str:
    compact = re.sub(r"[\s\-]+", "", raw_value.strip())

    match = re.match(r"^(DR|DI)(\d{2,4})(i)?$", compact, flags=re.IGNORECASE)
    if match:
        prefix = match.group(1).upper()
        digits = match.group(2)
        suffix = "i" if match.group(3) else ""
        return f"{prefix}{digits}{suffix}"

    match = re.match(r"^(D)(\d{2,4})(KX)?$", compact, flags=re.IGNORECASE)
    if match:
        prefix = match.group(1).upper()
        digits = match.group(2)
        suffix = "KX" if match.group(3) else ""
        return f"{prefix}{digits}{suffix}"

    return compact


def extract_base_machine_from_text(text: str) -> str | None:
    if not text:
        return None

    match = MACHINE_PATTERN.search(text)
    if not match:
        return None

    return normalize_machine_name(match.group("machine"))


def extract_component_from_text(text: str) -> str | None:
    normalized = text.lower()

    for component in KNOWN_COMPONENT_PATTERNS:
        if component in normalized:
            return component

    return None


def append_keyword_once(keywords: list[str], value: str | None) -> list[str]:
    if not value:
        return keywords

    existing = {keyword.lower() for keyword in keywords}

    if value.lower() not in existing:
        keywords.append(value)

    return keywords


def apply_deterministic_query_understanding_fallback(
    question: str,
    result: QueryUnderstandingAgentResult,
) -> QueryUnderstandingAgentResult:
    """Fill obvious machine/component hints when the model misses them.

    Rules:
    - Do not override model-provided entities.
    - Do not override model-provided filters.
    - Only add baseMachine when the text clearly contains a machine-like token.
    - Set high confidence only for deterministic exact-looking machine extraction.
    """

    detected_base_machine = extract_base_machine_from_text(question)
    detected_component = extract_component_from_text(question)

    if detected_base_machine and not result.detectedEntities.baseMachine:
        result.detectedEntities.baseMachine = detected_base_machine

    if detected_component and not result.detectedEntities.component:
        result.detectedEntities.component = detected_component

    if detected_base_machine and "baseMachine" not in result.filters:
        result.filters["baseMachine"] = detected_base_machine
        result.filterConfidence["baseMachine"] = 0.95

    if detected_component:
        result.keywords = append_keyword_once(result.keywords, detected_component)

    if detected_base_machine:
        result.keywords = append_keyword_once(result.keywords, detected_base_machine)

    if not result.rewrittenQuery or not result.rewrittenQuery.strip():
        result.rewrittenQuery = question.strip()

    # If the rewritten query does not mention the deterministic component/machine,
    # append them softly to improve retrieval without inventing content.
    rewritten_lower = result.rewrittenQuery.lower()

    additions: list[str] = []

    if detected_base_machine and detected_base_machine.lower() not in rewritten_lower:
        additions.append(detected_base_machine)

    if detected_component and detected_component.lower() not in rewritten_lower:
        additions.append(detected_component)

    if additions:
        result.rewrittenQuery = f"{result.rewrittenQuery.strip()} {' '.join(additions)}".strip()

    return result


def query_understanding_fallback(question: str, reason: str) -> QueryUnderstandingAgentResult:
    cleaned_question = question.strip()

    result = QueryUnderstandingAgentResult(
        intent="unknown",
        confidence=0.0,
        rewrittenQuery=cleaned_question,
        keywords=[],
        detectedEntities=QueryDetectedEntities(),
        filters={},
        filterConfidence={},
        needsClarification=False,
        clarificationQuestion=None,
        reason=reason,
    )

    return apply_deterministic_query_understanding_fallback(
        question=question,
        result=result,
    )


def create_query_understanding_completion(
    client: Any,
    deployment: str,
    messages: list[dict[str, str]],
    request_id: str | None,
) -> Any:
    wait_for_chat_turn(
        request_id=request_id,
        deployment=deployment,
        purpose="query_understanding",
    )

    max_completion_tokens = int(getattr(settings, "guardrail_max_completion_tokens", 300))

    try:
        return client.chat.completions.create(
            model=deployment,
            messages=messages,
            response_format={"type": "json_object"},
            max_completion_tokens=max_completion_tokens,
        )
    except BadRequestError:
        wait_for_chat_turn(
            request_id=request_id,
            deployment=deployment,
            purpose="query_understanding_no_json_mode",
        )

        return client.chat.completions.create(
            model=deployment,
            messages=messages,
            max_completion_tokens=max_completion_tokens,
        )


def call_query_understanding_model_with_retries(
    client: Any,
    deployment: str,
    messages: list[dict[str, str]],
    request_id: str | None,
) -> Any:
    max_retries = max(int(settings.openai_chat_retry_count), 0)

    for attempt in range(1, max_retries + 2):
        try:
            log_event(
                event="query_understanding_model_call_started",
                request_id=request_id,
                deployment=deployment,
                attempt=attempt,
                maxAttempts=max_retries + 1,
            )

            return create_query_understanding_completion(
                client=client,
                deployment=deployment,
                messages=messages,
                request_id=request_id,
            )

        except RateLimitError as exc:
            log_event(
                event="query_understanding_rate_limited",
                level="WARNING",
                request_id=request_id,
                deployment=deployment,
                attempt=attempt,
                maxAttempts=max_retries + 1,
                rateLimitHeaders=get_rate_limit_headers(exc),
                error=str(exc),
            )

            if attempt > max_retries:
                raise

            sleep_seconds = compute_retry_sleep_seconds(exc, attempt)

            log_event(
                event="query_understanding_rate_limit_retry_wait_started",
                request_id=request_id,
                deployment=deployment,
                attempt=attempt,
                sleepSeconds=round(sleep_seconds, 3),
            )

            time.sleep(sleep_seconds)

            log_event(
                event="query_understanding_rate_limit_retry_wait_completed",
                request_id=request_id,
                deployment=deployment,
                attempt=attempt,
                sleptSeconds=round(sleep_seconds, 3),
            )

    raise RuntimeError("Query understanding retry loop exited unexpectedly.")


def understand_query(
    question: str,
    conversation_summary: str = "",
    recent_turns: list[dict[str, Any]] | None = None,
    request_id: str | None = None,
) -> QueryUnderstandingAgentResult:
    if not question or not question.strip():
        raise ValueError("Question is required.")

    deployment = get_deployment_for_role(ModelRole.PLANNER)

    log_event(
        event="query_understanding_started",
        request_id=request_id,
        deployment=deployment,
        questionCharCount=len(question),
        conversationSummaryCharCount=len(conversation_summary or ""),
        recentTurnCount=len(recent_turns or []),
    )

    with elapsed_timer() as timer:
        client = get_azure_openai_client()

        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=question.strip(),
            conversation_summary=(conversation_summary or "").strip(),
            recent_turns_json=build_recent_turns_json(recent_turns),
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = call_query_understanding_model_with_retries(
                client=client,
                deployment=deployment,
                messages=messages,
                request_id=request_id,
            )

            raw_text = response.choices[0].message.content or ""
            parsed = extract_json_object(raw_text)
            result = QueryUnderstandingAgentResult(**parsed)
            result = apply_deterministic_query_understanding_fallback(
                question=question,
                result=result,
            )

            log_event(
                event="query_understanding_completed",
                request_id=request_id,
                deployment=deployment,
                intent=result.intent,
                confidence=result.confidence,
                needsClarification=result.needsClarification,
                filters=result.filters,
                detectedEntities=result.detectedEntities.model_dump(),
                latencyMs=timer["elapsedMs"],
            )

            return result

        except RateLimitError as exc:
            log_event(
                event="query_understanding_rate_limit_fallback_returned",
                level="WARNING",
                request_id=request_id,
                deployment=deployment,
                rateLimitHeaders=get_rate_limit_headers(exc),
                latencyMs=timer["elapsedMs"],
            )
            return query_understanding_fallback(
                question=question,
                reason="Query understanding model was temporarily rate-limited; using original question without filters.",
            )

        except (APITimeoutError, APIError) as exc:
            log_event(
                event="query_understanding_api_error",
                level="ERROR",
                request_id=request_id,
                deployment=deployment,
                error=str(exc),
                latencyMs=timer["elapsedMs"],
            )
            return query_understanding_fallback(
                question=question,
                reason="Query understanding API call failed; using original question without filters.",
            )

        except Exception as exc:
            log_event(
                event="query_understanding_failed",
                level="ERROR",
                request_id=request_id,
                deployment=deployment,
                error=str(exc),
                latencyMs=timer["elapsedMs"],
            )
            return query_understanding_fallback(
                question=question,
                reason="Query understanding failed; using original question without filters.",
            )

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
from backend.models import GroundingCriticResult, RevisionResult
from backend.observability.logger import log_event
from backend.observability.timing import elapsed_timer
from backend.prompts.revision_prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
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


def build_citations_json(citations: list[dict[str, Any]]) -> str:
    compact_citations = [
        {
            "citationId": citation.get("citationId"),
            "id": citation.get("id"),
            "title": citation.get("title"),
            "citationPath": citation.get("citationPath"),
            "machine": citation.get("machine"),
            "manualType": citation.get("manualType"),
        }
        for citation in citations
    ]

    return json.dumps(compact_citations, ensure_ascii=False, indent=2)


def revision_failure_result(original_answer: str, reason: str) -> RevisionResult:
    return RevisionResult(
        revisedAnswer=original_answer,
        usedCitationPaths=[],
        revisionApplied=False,
        reason=reason,
        confidence=0.0,
    )


def create_revision_completion(
    client: Any,
    deployment: str,
    messages: list[dict[str, str]],
    request_id: str | None,
) -> Any:
    wait_for_chat_turn(
        request_id=request_id,
        deployment=deployment,
        purpose="revision_agent",
    )

    try:
        return client.chat.completions.create(
            model=deployment,
            messages=messages,
            response_format={"type": "json_object"},
            max_completion_tokens=settings.revision_max_completion_tokens,
        )
    except BadRequestError:
        wait_for_chat_turn(
            request_id=request_id,
            deployment=deployment,
            purpose="revision_agent_no_json_mode",
        )

        return client.chat.completions.create(
            model=deployment,
            messages=messages,
            max_completion_tokens=settings.revision_max_completion_tokens,
        )


def call_revision_model_with_retries(
    client: Any,
    deployment: str,
    messages: list[dict[str, str]],
    request_id: str | None,
) -> Any:
    max_retries = max(int(settings.openai_chat_retry_count), 0)

    for attempt in range(1, max_retries + 2):
        try:
            log_event(
                event="revision_model_call_started",
                request_id=request_id,
                deployment=deployment,
                attempt=attempt,
                maxAttempts=max_retries + 1,
                maxCompletionTokens=settings.revision_max_completion_tokens,
            )

            return create_revision_completion(
                client=client,
                deployment=deployment,
                messages=messages,
                request_id=request_id,
            )

        except RateLimitError as exc:
            headers = get_rate_limit_headers(exc)

            log_event(
                event="revision_rate_limited",
                level="WARNING",
                request_id=request_id,
                deployment=deployment,
                attempt=attempt,
                maxAttempts=max_retries + 1,
                rateLimitHeaders=headers,
                error=str(exc),
            )

            if attempt > max_retries:
                raise

            sleep_seconds = compute_retry_sleep_seconds(exc, attempt)

            log_event(
                event="revision_rate_limit_retry_wait_started",
                request_id=request_id,
                deployment=deployment,
                attempt=attempt,
                sleepSeconds=round(sleep_seconds, 3),
            )

            time.sleep(sleep_seconds)

            log_event(
                event="revision_rate_limit_retry_wait_completed",
                request_id=request_id,
                deployment=deployment,
                attempt=attempt,
                sleptSeconds=round(sleep_seconds, 3),
            )


def revise_answer(
    question: str,
    answer: str,
    grounding_result: GroundingCriticResult,
    context: str,
    citations: list[dict[str, Any]],
    request_id: str | None = None,
) -> RevisionResult:
    if not answer or not answer.strip():
        return revision_failure_result(
            original_answer="",
            reason="No original answer was provided for revision.",
        )

    if not context or not context.strip():
        return revision_failure_result(
            original_answer=answer,
            reason="No retrieved context was provided for revision.",
        )

    revision_deployment = get_deployment_for_role(ModelRole.ANSWER)

    log_event(
        event="revision_started",
        request_id=request_id,
        deployment=revision_deployment,
        answerCharCount=len(answer),
        contextCharCount=len(context),
        estimatedContextTokens=round(len(context) / 4),
        unsupportedClaimCount=len(grounding_result.unsupportedClaims),
        missingCitationCount=len(grounding_result.missingCitations),
        maxCompletionTokens=settings.revision_max_completion_tokens,
    )

    with elapsed_timer() as timer:
        client = get_azure_openai_client()

        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=question.strip(),
            answer=answer.strip(),
            grounding_json=json.dumps(
                grounding_result.model_dump(),
                ensure_ascii=False,
                indent=2,
            ),
            context=context,
            citations_json=build_citations_json(citations),
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = call_revision_model_with_retries(
                client=client,
                deployment=revision_deployment,
                messages=messages,
                request_id=request_id,
            )

            raw_text = response.choices[0].message.content or ""
            parsed = extract_json_object(raw_text)
            result = RevisionResult(**parsed)

            log_event(
                event="revision_completed",
                request_id=request_id,
                deployment=revision_deployment,
                revisionApplied=result.revisionApplied,
                confidence=result.confidence,
                usedCitationCount=len(result.usedCitationPaths),
                latencyMs=timer["elapsedMs"],
            )

            return result

        except RateLimitError as exc:
            log_event(
                event="revision_rate_limit_fallback_returned",
                level="WARNING",
                request_id=request_id,
                deployment=revision_deployment,
                rateLimitHeaders=get_rate_limit_headers(exc),
                latencyMs=timer["elapsedMs"],
            )

            return revision_failure_result(
                original_answer=answer,
                reason="Revision agent was temporarily rate-limited; original answer retained.",
            )

        except (APITimeoutError, APIError) as exc:
            log_event(
                event="revision_api_error",
                level="ERROR",
                request_id=request_id,
                deployment=revision_deployment,
                error=str(exc),
                latencyMs=timer["elapsedMs"],
            )

            return revision_failure_result(
                original_answer=answer,
                reason="Revision agent API call failed; original answer retained.",
            )

        except Exception as exc:
            log_event(
                event="revision_failed",
                level="ERROR",
                request_id=request_id,
                deployment=revision_deployment,
                error=str(exc),
                latencyMs=timer["elapsedMs"],
            )

            return revision_failure_result(
                original_answer=answer,
                reason="Revision agent failed; original answer retained.",
            )

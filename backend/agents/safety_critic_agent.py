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
from backend.models import SafetyCriticResult
from backend.observability.logger import log_event
from backend.observability.timing import elapsed_timer
from backend.prompts.safety_critic_prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
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


def safety_failure_result(reason: str) -> SafetyCriticResult:
    return SafetyCriticResult(
        safe=False,
        requiresRevision=True,
        safetyIssues=[],
        missingWarnings=[],
        unsafeOrUnsupportedInstructions=[],
        inventedSafetyCriticalDetails=[],
        reason=reason,
        confidence=0.0,
    )


def create_safety_completion(
    client: Any,
    deployment: str,
    messages: list[dict[str, str]],
    request_id: str | None,
) -> Any:
    wait_for_chat_turn(
        request_id=request_id,
        deployment=deployment,
        purpose="safety_critic",
    )

    try:
        return client.chat.completions.create(
            model=deployment,
            messages=messages,
            response_format={"type": "json_object"},
            max_completion_tokens=settings.critic_max_completion_tokens,
        )
    except BadRequestError:
        wait_for_chat_turn(
            request_id=request_id,
            deployment=deployment,
            purpose="safety_critic_no_json_mode",
        )

        return client.chat.completions.create(
            model=deployment,
            messages=messages,
            max_completion_tokens=settings.critic_max_completion_tokens,
        )


def call_safety_model_with_retries(
    client: Any,
    deployment: str,
    messages: list[dict[str, str]],
    request_id: str | None,
) -> Any:
    max_retries = max(int(settings.openai_chat_retry_count), 0)

    for attempt in range(1, max_retries + 2):
        try:
            log_event(
                event="safety_critic_model_call_started",
                request_id=request_id,
                deployment=deployment,
                attempt=attempt,
                maxAttempts=max_retries + 1,
                maxCompletionTokens=settings.critic_max_completion_tokens,
            )

            return create_safety_completion(
                client=client,
                deployment=deployment,
                messages=messages,
                request_id=request_id,
            )

        except RateLimitError as exc:
            headers = get_rate_limit_headers(exc)

            log_event(
                event="safety_critic_rate_limited",
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
                event="safety_critic_rate_limit_retry_wait_started",
                request_id=request_id,
                deployment=deployment,
                attempt=attempt,
                sleepSeconds=round(sleep_seconds, 3),
            )

            time.sleep(sleep_seconds)

            log_event(
                event="safety_critic_rate_limit_retry_wait_completed",
                request_id=request_id,
                deployment=deployment,
                attempt=attempt,
                sleptSeconds=round(sleep_seconds, 3),
            )


def evaluate_safety(
    question: str,
    answer: str,
    context: str,
    citations: list[dict[str, Any]],
    request_id: str | None = None,
) -> SafetyCriticResult:
    if not answer or not answer.strip():
        return safety_failure_result("No answer was provided for safety evaluation.")

    if not context or not context.strip():
        return safety_failure_result("No retrieved context was provided for safety evaluation.")

    critic_deployment = get_deployment_for_role(ModelRole.CRITIC)

    log_event(
        event="safety_critic_started",
        request_id=request_id,
        deployment=critic_deployment,
        answerCharCount=len(answer),
        contextCharCount=len(context),
        estimatedContextTokens=round(len(context) / 4),
        citationCount=len(citations),
        maxCompletionTokens=settings.critic_max_completion_tokens,
    )

    with elapsed_timer() as timer:
        client = get_azure_openai_client()

        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=question.strip(),
            answer=answer.strip(),
            context=context,
            citations_json=build_citations_json(citations),
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = call_safety_model_with_retries(
                client=client,
                deployment=critic_deployment,
                messages=messages,
                request_id=request_id,
            )

            raw_text = response.choices[0].message.content or ""
            parsed = extract_json_object(raw_text)
            result = SafetyCriticResult(**parsed)

            log_event(
                event="safety_critic_completed",
                request_id=request_id,
                deployment=critic_deployment,
                safe=result.safe,
                requiresRevision=result.requiresRevision,
                safetyIssueCount=len(result.safetyIssues),
                missingWarningCount=len(result.missingWarnings),
                unsafeInstructionCount=len(result.unsafeOrUnsupportedInstructions),
                inventedSafetyCriticalDetailCount=len(result.inventedSafetyCriticalDetails),
                confidence=result.confidence,
                latencyMs=timer["elapsedMs"],
            )

            return result

        except RateLimitError as exc:
            log_event(
                event="safety_critic_rate_limit_fallback_returned",
                level="WARNING",
                request_id=request_id,
                deployment=critic_deployment,
                rateLimitHeaders=get_rate_limit_headers(exc),
                latencyMs=timer["elapsedMs"],
            )

            return safety_failure_result(
                "Safety critic was temporarily rate-limited and the answer requires review."
            )

        except (APITimeoutError, APIError) as exc:
            log_event(
                event="safety_critic_api_error",
                level="ERROR",
                request_id=request_id,
                deployment=critic_deployment,
                error=str(exc),
                latencyMs=timer["elapsedMs"],
            )

            return safety_failure_result(
                "Safety critic API call failed and the answer requires review."
            )

        except Exception as exc:
            log_event(
                event="safety_critic_failed",
                level="ERROR",
                request_id=request_id,
                deployment=critic_deployment,
                error=str(exc),
                latencyMs=timer["elapsedMs"],
            )

            return safety_failure_result(
                "Safety critic failed and the answer requires review."
            )

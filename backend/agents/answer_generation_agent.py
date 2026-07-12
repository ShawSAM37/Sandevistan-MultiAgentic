import json
import re
import time
from typing import Any

from openai import APIError, APITimeoutError, BadRequestError, RateLimitError

from backend.agents.model_router import ModelRole, get_deployment_for_role, has_fallback_answer_deployment
from backend.agents.openai_rate_limiter import (
    compute_retry_sleep_seconds,
    get_rate_limit_headers,
    wait_for_chat_turn,
)
from backend.config import settings
from backend.models import AnswerGenerationResult
from backend.observability.logger import log_event
from backend.observability.timing import elapsed_timer
from backend.prompts.answer_generation_prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
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


def rate_limit_fallback() -> AnswerGenerationResult:
    return AnswerGenerationResult(
        answer=(
            "The answer generation model is temporarily rate-limited. "
            "Please retry shortly. Retrieval and context building completed, "
            "but the final answer could not be generated at this moment."
        ),
        usedCitationPaths=[],
        confidence=0.0,
        answerFound=False,
    )


def model_error_fallback(error_message: str) -> AnswerGenerationResult:
    return AnswerGenerationResult(
        answer=(
            "The answer generation step could not complete successfully. "
            "Please retry shortly or reduce the query scope."
        ),
        usedCitationPaths=[],
        confidence=0.0,
        answerFound=False,
    )


def create_answer_completion(
    client: Any,
    deployment: str,
    messages: list[dict[str, str]],
    request_id: str | None,
) -> Any:
    wait_for_chat_turn(
        request_id=request_id,
        deployment=deployment,
        purpose="answer_generation",
    )

    try:
        return client.chat.completions.create(
            model=deployment,
            messages=messages,
            response_format={"type": "json_object"},
            max_completion_tokens=settings.answer_max_completion_tokens,
        )
    except BadRequestError:
        wait_for_chat_turn(
            request_id=request_id,
            deployment=deployment,
            purpose="answer_generation_no_json_mode",
        )

        return client.chat.completions.create(
            model=deployment,
            messages=messages,
            max_completion_tokens=settings.answer_max_completion_tokens,
        )


def call_answer_model_with_retries(
    client: Any,
    deployment: str,
    messages: list[dict[str, str]],
    request_id: str | None,
) -> Any:
    max_retries = max(int(settings.openai_chat_retry_count), 0)

    for attempt in range(1, max_retries + 2):
        try:
            log_event(
                event="answer_generation_model_call_started",
                request_id=request_id,
                deployment=deployment,
                attempt=attempt,
                maxAttempts=max_retries + 1,
                maxCompletionTokens=settings.answer_max_completion_tokens,
            )

            return create_answer_completion(
                client=client,
                deployment=deployment,
                messages=messages,
                request_id=request_id,
            )

        except RateLimitError as exc:
            headers = get_rate_limit_headers(exc)

            log_event(
                event="answer_generation_rate_limited",
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
                event="answer_generation_rate_limit_retry_wait_started",
                request_id=request_id,
                deployment=deployment,
                attempt=attempt,
                sleepSeconds=round(sleep_seconds, 3),
            )

            time.sleep(sleep_seconds)

            log_event(
                event="answer_generation_rate_limit_retry_wait_completed",
                request_id=request_id,
                deployment=deployment,
                attempt=attempt,
                sleptSeconds=round(sleep_seconds, 3),
            )


def generate_answer_from_context(
    question: str,
    context: str,
    citations: list[dict[str, Any]],
    request_id: str | None = None,
) -> AnswerGenerationResult:
    if not question or not question.strip():
        raise ValueError("Question is required.")

    if not context or not context.strip():
        return AnswerGenerationResult(
            answer="I could not find enough information in the retrieved manual context to answer this question.",
            usedCitationPaths=[],
            confidence=0.0,
            answerFound=False,
        )

    answer_deployment = get_deployment_for_role(ModelRole.ANSWER)

    log_event(
        event="answer_generation_started",
        request_id=request_id,
        question=question,
        contextCharCount=len(context),
        estimatedContextTokens=round(len(context) / 4),
        citationCount=len(citations),
        deployment=answer_deployment,
        maxCompletionTokens=settings.answer_max_completion_tokens,
        minIntervalSeconds=settings.openai_chat_min_interval_seconds,
        retryCount=settings.openai_chat_retry_count,
    )

    with elapsed_timer() as timer:
        client = get_azure_openai_client()

        user_prompt = USER_PROMPT_TEMPLATE.format(
            question=question.strip(),
            context=context,
            citations_json=build_citations_json(citations),
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = call_answer_model_with_retries(
                client=client,
                deployment=answer_deployment,
                messages=messages,
                request_id=request_id,
            )

            raw_text = response.choices[0].message.content or ""
            parsed = extract_json_object(raw_text)

            result = AnswerGenerationResult(**parsed)

            log_event(
                event="answer_generation_completed",
                request_id=request_id,
                deployment=answer_deployment,
                answerFound=result.answerFound,
                confidence=result.confidence,
                usedCitationCount=len(result.usedCitationPaths),
                latencyMs=timer["elapsedMs"],
            )

            return result

        except RateLimitError as exc:
            if has_fallback_answer_deployment():
                fallback_deployment = get_deployment_for_role(ModelRole.FALLBACK_ANSWER)

                log_event(
                    event="answer_generation_trying_fallback_deployment",
                    level="WARNING",
                    request_id=request_id,
                    primaryDeployment=answer_deployment,
                    fallbackDeployment=fallback_deployment,
                    primaryRateLimitHeaders=get_rate_limit_headers(exc),
                )

                try:
                    response = call_answer_model_with_retries(
                        client=client,
                        deployment=fallback_deployment,
                        messages=messages,
                        request_id=request_id,
                    )

                    raw_text = response.choices[0].message.content or ""
                    parsed = extract_json_object(raw_text)
                    result = AnswerGenerationResult(**parsed)

                    log_event(
                        event="answer_generation_completed_with_fallback",
                        request_id=request_id,
                        deployment=fallback_deployment,
                        answerFound=result.answerFound,
                        confidence=result.confidence,
                        usedCitationCount=len(result.usedCitationPaths),
                        latencyMs=timer["elapsedMs"],
                    )

                    return result

                except RateLimitError as fallback_exc:
                    log_event(
                        event="answer_generation_fallback_rate_limited",
                        level="WARNING",
                        request_id=request_id,
                        fallbackDeployment=fallback_deployment,
                        fallbackRateLimitHeaders=get_rate_limit_headers(fallback_exc),
                        error=str(fallback_exc),
                        latencyMs=timer["elapsedMs"],
                    )

            log_event(
                event="answer_generation_rate_limit_fallback_returned",
                level="WARNING",
                request_id=request_id,
                deployment=answer_deployment,
                rateLimitHeaders=get_rate_limit_headers(exc),
                latencyMs=timer["elapsedMs"],
            )

            return rate_limit_fallback()

        except (APITimeoutError, APIError) as exc:
            log_event(
                event="answer_generation_api_error",
                level="ERROR",
                request_id=request_id,
                deployment=answer_deployment,
                error=str(exc),
                latencyMs=timer["elapsedMs"],
            )
            return model_error_fallback(str(exc))

        except Exception as exc:
            log_event(
                event="answer_generation_failed",
                level="ERROR",
                request_id=request_id,
                deployment=answer_deployment,
                error=str(exc),
                latencyMs=timer["elapsedMs"],
            )
            return model_error_fallback(str(exc))

from enum import StrEnum

from backend.config import settings


class ModelRole(StrEnum):
    ANSWER = "answer"
    GUARDRAIL = "guardrail"
    PLANNER = "planner"
    CRITIC = "critic"
    SUMMARIZER = "summarizer"
    FALLBACK_ANSWER = "fallback_answer"


def get_deployment_for_role(role: ModelRole) -> str:
    fallback_chat = settings.azure_openai_chat_deployment

    if role == ModelRole.ANSWER:
        deployment = settings.azure_openai_answer_deployment or fallback_chat
    elif role == ModelRole.GUARDRAIL:
        deployment = settings.azure_openai_guardrail_deployment or fallback_chat
    elif role == ModelRole.PLANNER:
        deployment = settings.azure_openai_planner_deployment or fallback_chat
    elif role == ModelRole.CRITIC:
        deployment = settings.azure_openai_critic_deployment or fallback_chat
    elif role == ModelRole.SUMMARIZER:
        deployment = settings.azure_openai_summarizer_deployment or fallback_chat
    elif role == ModelRole.FALLBACK_ANSWER:
        deployment = settings.azure_openai_fallback_chat_deployment
    else:
        deployment = fallback_chat

    if not deployment:
        raise ValueError(f"No Azure OpenAI deployment configured for role: {role}")

    return deployment


def has_fallback_answer_deployment() -> bool:
    return bool(settings.azure_openai_fallback_chat_deployment)

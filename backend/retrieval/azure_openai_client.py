from openai import AzureOpenAI

from backend.config import settings


def get_azure_openai_client() -> AzureOpenAI:
    if not settings.azure_openai_endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT is not configured.")

    if not settings.azure_openai_api_key:
        raise ValueError("AZURE_OPENAI_API_KEY is not configured.")

    return AzureOpenAI(
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
    )

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient

from backend.config import settings


def get_search_index_client() -> SearchIndexClient:
    if not settings.azure_search_endpoint:
        raise ValueError("AZURE_SEARCH_ENDPOINT is not configured.")

    if not settings.azure_search_admin_key:
        raise ValueError("AZURE_SEARCH_ADMIN_KEY is not configured.")

    return SearchIndexClient(
        endpoint=settings.azure_search_endpoint,
        credential=AzureKeyCredential(settings.azure_search_admin_key),
    )

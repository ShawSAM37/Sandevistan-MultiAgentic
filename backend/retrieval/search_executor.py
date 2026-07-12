import requests

from backend.config import settings
from backend.constants import RETRIEVABLE_FIELDS, VECTOR_FIELDS
from backend.observability.logger import log_event
from backend.observability.timing import elapsed_timer
from backend.retrieval.embedding_client import generate_query_embedding
from backend.retrieval.filters import build_filter_expression


def validate_vector_fields(vector_fields: list[str]) -> list[str]:
    if not vector_fields:
        return ["contentVector"]

    invalid = set(vector_fields) - set(VECTOR_FIELDS)
    if invalid:
        raise ValueError(f"Unsupported vector fields for V1: {sorted(invalid)}")

    return vector_fields


def normalize_search_result(
    raw: dict,
    search_mode: str,
    vector_fields_used: list[str],
) -> dict:
    document = {
        "id": raw.get("id"),
        "title": raw.get("title"),
        "content": raw.get("content"),
        "manualType": raw.get("manualType"),
        "baseMachine": raw.get("baseMachine"),
        "serialNumber": raw.get("serialNumber"),
        "machine": raw.get("machine"),
        "citationPath": raw.get("citationPath"),
        "score": raw.get("@search.score"),
        "rerankerScore": raw.get("@search.rerankerScore"),
        "searchMode": search_mode,
        "vectorFieldsUsed": vector_fields_used,
    }

    return document


def execute_search(
    query: str,
    search_mode: str = "hybrid",
    vector_fields: list[str] | None = None,
    filters: dict[str, str] | None = None,
    top: int = 10,
    k: int = 50,
    use_semantic_ranker: bool = False,
    request_id: str | None = None,
) -> dict:
    if search_mode not in {"keyword", "vector", "hybrid"}:
        raise ValueError(f"Unsupported search mode: {search_mode}")

    if not settings.azure_search_endpoint:
        raise ValueError("AZURE_SEARCH_ENDPOINT is not configured.")

    if not settings.azure_search_admin_key:
        raise ValueError("AZURE_SEARCH_ADMIN_KEY is not configured.")

    vector_fields_used = validate_vector_fields(vector_fields or ["contentVector"])
    filter_expression = build_filter_expression(filters)

    log_event(
        event="retrieval_started",
        request_id=request_id,
        query=query,
        searchMode=search_mode,
        vectorFields=vector_fields_used if search_mode in {"vector", "hybrid"} else [],
        filters=filters or {},
        filterExpression=filter_expression,
        top=top,
        k=k,
        useSemanticRanker=use_semantic_ranker,
    )

    with elapsed_timer() as timer:
        url = (
            f"{settings.azure_search_endpoint.rstrip('/')}"
            f"/indexes/{settings.azure_search_index_name}/docs/search"
            f"?api-version={settings.azure_search_api_version}"
        )

        headers = {
            "Content-Type": "application/json",
            "api-key": settings.azure_search_admin_key,
        }

        body: dict = {
            "top": top,
            "count": True,
            "select": ",".join(RETRIEVABLE_FIELDS),
        }

        if filter_expression:
            body["filter"] = filter_expression

        if search_mode in {"keyword", "hybrid"}:
            body["search"] = query

        if search_mode in {"vector", "hybrid"}:
            embedding = generate_query_embedding(query)

            body["vectorQueries"] = [
                {
                    "kind": "vector",
                    "vector": embedding,
                    "fields": vector_field,
                    "k": k,
                }
                for vector_field in vector_fields_used
            ]

        if use_semantic_ranker and search_mode in {"keyword", "hybrid"}:
            body["queryType"] = "semantic"
            body["semanticConfiguration"] = settings.azure_search_semantic_config

        response = requests.post(
            url,
            headers=headers,
            json=body,
            timeout=settings.request_timeout_seconds,
        )

        if response.status_code >= 400:
            log_event(
                event="retrieval_failed",
                level="ERROR",
                request_id=request_id,
                statusCode=response.status_code,
                responsePreview=response.text[:1000],
                elapsedMs=timer["elapsedMs"],
            )

            raise RuntimeError(
                "Azure AI Search query failed. "
                f"Status={response.status_code}. "
                f"Body={response.text[:2000]}"
            )

        payload = response.json()
        raw_results = payload.get("value", [])

        documents = [
            normalize_search_result(
                raw=result,
                search_mode=search_mode,
                vector_fields_used=vector_fields_used if search_mode in {"vector", "hybrid"} else [],
            )
            for result in raw_results
        ]

    result = {
        "requestId": request_id,
        "query": query,
        "searchMode": search_mode,
        "vectorFields": vector_fields_used if search_mode in {"vector", "hybrid"} else [],
        "filters": filters or {},
        "filterExpression": filter_expression,
        "top": top,
        "k": k,
        "useSemanticRanker": use_semantic_ranker,
        "count": payload.get("@odata.count"),
        "resultCount": len(documents),
        "latencyMs": timer["elapsedMs"],
        "documents": documents,
    }

    log_event(
        event="retrieval_completed",
        request_id=request_id,
        query=query,
        searchMode=search_mode,
        resultCount=len(documents),
        count=payload.get("@odata.count"),
        latencyMs=timer["elapsedMs"],
    )

    return result

from backend.config import settings
from backend.constants import APPROVED_INDEX_FIELDS, VECTOR_FIELDS
from backend.retrieval.azure_openai_client import get_azure_openai_client
from backend.retrieval.azure_search_client import get_search_index_client
from backend.retrieval.embedding_client import generate_query_embedding


def check_search_service() -> dict:
    try:
        client = get_search_index_client()

        index_names = [index.name for index in client.list_indexes()]
        index_exists = settings.azure_search_index_name in index_names

        details = {
            "endpointConfigured": bool(settings.azure_search_endpoint),
            "indexName": settings.azure_search_index_name,
            "indexExists": index_exists,
            "availableIndexCount": len(index_names),
        }

        if not index_exists:
            return {
                "status": "warning",
                "details": details,
                "message": "Search service is reachable, but target index does not exist yet.",
            }

        index = client.get_index(settings.azure_search_index_name)
        fields = {field.name: field for field in index.fields}

        approved_fields_present = [
            field_name for field_name in APPROVED_INDEX_FIELDS if field_name in fields
        ]

        vector_field_details = {}
        for vector_field in VECTOR_FIELDS:
            field = fields.get(vector_field)
            vector_field_details[vector_field] = {
                "exists": field is not None,
                "dimensions": getattr(field, "vector_search_dimensions", None)
                if field
                else None,
            }

        details.update(
            {
                "fieldCount": len(index.fields),
                "approvedFieldsPresent": approved_fields_present,
                "approvedFieldsPresentCount": len(approved_fields_present),
                "vectorFields": vector_field_details,
            }
        )

        return {
            "status": "ok",
            "details": details,
        }

    except Exception as exc:
        return {
            "status": "error",
            "details": {
                "endpointConfigured": bool(settings.azure_search_endpoint),
                "indexName": settings.azure_search_index_name,
            },
            "error": str(exc),
        }


def check_embedding_deployment() -> dict:
    try:
        embedding = generate_query_embedding("health check embedding test")

        return {
            "status": "ok",
            "details": {
                "deployment": settings.azure_openai_embedding_deployment,
                "dimensions": len(embedding),
                "expectedDimensions": settings.vector_dimensions,
            },
        }

    except Exception as exc:
        return {
            "status": "error",
            "details": {
                "deployment": settings.azure_openai_embedding_deployment,
                "expectedDimensions": settings.vector_dimensions,
            },
            "error": str(exc),
        }


def check_chat_deployment() -> dict:
    try:
        if not settings.azure_openai_chat_deployment:
            raise ValueError("AZURE_OPENAI_CHAT_DEPLOYMENT is not configured.")

        client = get_azure_openai_client()

        response = client.chat.completions.create(
            model=settings.azure_openai_chat_deployment,
            messages=[
                {
                    "role": "system",
                    "content": "You are a health check assistant. Reply with only OK.",
                },
                {
                    "role": "user",
                    "content": "Reply OK.",
                },
            ],
            max_completion_tokens=10,
        )

        message = response.choices[0].message.content or ""

        return {
            "status": "ok",
            "details": {
                "deployment": settings.azure_openai_chat_deployment,
                "responsePreview": message[:50],
            },
        }

    except Exception as exc:
        return {
            "status": "error",
            "details": {
                "deployment": settings.azure_openai_chat_deployment,
            },
            "error": str(exc),
        }


def run_deep_health_check() -> dict:
    search_check = check_search_service()
    embedding_check = check_embedding_deployment()
    chat_check = check_chat_deployment()

    checks = {
        "azureSearch": search_check,
        "azureOpenAIEmbedding": embedding_check,
        "azureOpenAIChat": chat_check,
    }

    has_error = any(check["status"] == "error" for check in checks.values())

    if has_error:
        overall_status = "error"
    else:
        overall_status = "ok"

    return {
        "status": overall_status,
        "checks": checks,
    }

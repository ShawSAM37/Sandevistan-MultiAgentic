from fastapi import FastAPI

from backend.config import settings
from backend.constants import APPROVED_INDEX_FIELDS, AZURE_SEARCH_INDEX_NAME
from backend.models import DebugRetrievalRequest
from backend.observability.logger import log_event
from backend.observability.request_context import new_request_id
from backend.observability.timing import elapsed_timer
from backend.retrieval.search_executor import execute_search
from backend.services.health_service import run_deep_health_check


app = FastAPI(
    title="Sandevistan Multi-Agentic RAG API",
    version="0.1.0",
)


@app.get("/")
async def root():
    return {
        "service": settings.app_name,
        "status": "ok",
        "phase": "phase_1_skeleton",
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.app_env,
        "indexName": AZURE_SEARCH_INDEX_NAME,
        "approvedFieldCount": len(APPROVED_INDEX_FIELDS),
        "approvedFields": APPROVED_INDEX_FIELDS,
        "configLoaded": True,
        "azureSearchConfigured": bool(settings.azure_search_endpoint),
        "azureOpenAIConfigured": bool(settings.azure_openai_endpoint),
    }


@app.get("/health/deep")
async def health_deep():
    return run_deep_health_check()


@app.post("/debug/retrieval")
async def debug_retrieval(request: DebugRetrievalRequest):
    request_id = new_request_id()

    log_event(
        event="debug_retrieval_request_received",
        request_id=request_id,
        query=request.query,
        searchMode=request.searchMode,
        filters=request.filters,
        top=request.top,
        k=request.k,
        vectorFields=request.vectorFields,
    )

    with elapsed_timer() as timer:
        result = execute_search(
            query=request.query,
            search_mode=request.searchMode,
            vector_fields=request.vectorFields,
            filters=request.filters,
            top=request.top,
            k=request.k,
            use_semantic_ranker=request.useSemanticRanker,
            request_id=request_id,
        )

        if not request.showContent:
            for document in result["documents"]:
                content = document.pop("content", None)
                if content:
                    document["contentPreview"] = content[:500]
                    document["contentLength"] = len(content)

    result["requestId"] = request_id
    result["endpointLatencyMs"] = timer["elapsedMs"]

    log_event(
        event="debug_retrieval_request_completed",
        request_id=request_id,
        resultCount=result.get("resultCount"),
        endpointLatencyMs=timer["elapsedMs"],
    )

    return result

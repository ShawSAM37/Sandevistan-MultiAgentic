from fastapi import FastAPI

from backend.config import settings
from backend.constants import APPROVED_INDEX_FIELDS, AZURE_SEARCH_INDEX_NAME
from backend.models import DebugRetrievalRequest
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
    result = execute_search(
        query=request.query,
        search_mode=request.searchMode,
        vector_fields=request.vectorFields,
        filters=request.filters,
        top=request.top,
        k=request.k,
        use_semantic_ranker=request.useSemanticRanker,
    )

    if not request.showContent:
        for document in result["documents"]:
            content = document.pop("content", None)
            if content:
                document["contentPreview"] = content[:500]
                document["contentLength"] = len(content)

    return result

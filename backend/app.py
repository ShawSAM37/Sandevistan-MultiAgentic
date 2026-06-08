from fastapi import FastAPI

from backend.agents.answer_generation_agent import generate_answer_from_context
from backend.agents.grounding_critic_agent import evaluate_grounding
from backend.agents.input_guardrail_agent import run_input_guardrail_agent
from backend.agents.revision_agent import revise_answer
from backend.config import settings
from backend.constants import APPROVED_INDEX_FIELDS, AZURE_SEARCH_INDEX_NAME
from backend.context.context_builder import build_context_from_documents
from backend.models import (
    DebugAnswerRequest,
    DebugContextRequest,
    DebugGroundingRequest,
    DebugGuardrailRequest,
    DebugRetrievalRequest,
    DebugRevisionRequest,
)
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


@app.post("/debug/guardrail")
async def debug_guardrail(request: DebugGuardrailRequest):
    request_id = new_request_id()

    result = run_input_guardrail_agent(
        question=request.question,
        request_id=request_id,
    )

    return {
        "requestId": request_id,
        **result.model_dump(),
    }


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


@app.post("/debug/context")
async def debug_context(request: DebugContextRequest):
    request_id = new_request_id()

    log_event(
        event="debug_context_request_received",
        request_id=request_id,
        query=request.query,
        searchMode=request.searchMode,
        filters=request.filters,
        top=request.top,
        k=request.k,
        vectorFields=request.vectorFields,
    )

    with elapsed_timer() as timer:
        retrieval_result = execute_search(
            query=request.query,
            search_mode=request.searchMode,
            vector_fields=request.vectorFields,
            filters=request.filters,
            top=request.top,
            k=request.k,
            use_semantic_ranker=request.useSemanticRanker,
            request_id=request_id,
        )

        context_result = build_context_from_documents(
            documents=retrieval_result["documents"],
            max_context_chars=request.maxContextChars,
            max_chars_per_document=request.maxCharsPerDocument,
            request_id=request_id,
        )

    response = {
        "requestId": request_id,
        "query": request.query,
        "searchMode": request.searchMode,
        "filters": request.filters,
        "retrieval": {
            "resultCount": retrieval_result["resultCount"],
            "count": retrieval_result["count"],
            "latencyMs": retrieval_result["latencyMs"],
            "documents": retrieval_result["documents"],
        },
        "context": context_result["context"],
        "contextCharCount": context_result["contextCharCount"],
        "usedDocumentCount": context_result["usedDocumentCount"],
        "skippedDocumentCount": context_result["skippedDocumentCount"],
        "citations": context_result["citations"],
        "usedDocuments": context_result["usedDocuments"],
        "skippedDocuments": context_result["skippedDocuments"],
        "endpointLatencyMs": timer["elapsedMs"],
    }

    log_event(
        event="debug_context_request_completed",
        request_id=request_id,
        retrievalResultCount=retrieval_result["resultCount"],
        contextCharCount=context_result["contextCharCount"],
        usedDocumentCount=context_result["usedDocumentCount"],
        skippedDocumentCount=context_result["skippedDocumentCount"],
        endpointLatencyMs=timer["elapsedMs"],
    )

    return response


@app.post("/debug/answer")
async def debug_answer(request: DebugAnswerRequest):
    request_id = new_request_id()

    log_event(
        event="debug_answer_request_received",
        request_id=request_id,
        query=request.query,
        searchMode=request.searchMode,
        filters=request.filters,
        top=request.top,
        k=request.k,
        vectorFields=request.vectorFields,
    )

    with elapsed_timer() as timer:
        guardrail_result = run_input_guardrail_agent(
            question=request.query,
            request_id=request_id,
        )

        if not guardrail_result.allowed:
            response = {
                "requestId": request_id,
                "query": request.query,
                "answer": "I cannot help with that request. Please ask a safe, manual-related question about Sandvik rotary equipment.",
                "answerFound": False,
                "confidence": 0.0,
                "usedCitationPaths": [],
                "citations": [],
                "usedDocuments": [],
                "guardrail": guardrail_result.model_dump(),
                "retrieval": None,
                "contextCharCount": 0,
                "usedDocumentCount": 0,
                "skippedDocumentCount": 0,
                "endpointLatencyMs": timer["elapsedMs"],
            }

            log_event(
                event="debug_answer_blocked_by_guardrail",
                request_id=request_id,
                riskLevel=guardrail_result.riskLevel,
                reason=guardrail_result.reason,
                endpointLatencyMs=timer["elapsedMs"],
            )

            return response

        retrieval_result = execute_search(
            query=guardrail_result.sanitizedQuestion or request.query,
            search_mode=request.searchMode,
            vector_fields=request.vectorFields,
            filters=request.filters,
            top=request.top,
            k=request.k,
            use_semantic_ranker=request.useSemanticRanker,
            request_id=request_id,
        )

        context_result = build_context_from_documents(
            documents=retrieval_result["documents"],
            max_context_chars=request.maxContextChars,
            max_chars_per_document=request.maxCharsPerDocument,
            request_id=request_id,
        )

        answer_result = generate_answer_from_context(
            question=guardrail_result.sanitizedQuestion or request.query,
            context=context_result["context"],
            citations=context_result["citations"],
            request_id=request_id,
        )

        if answer_result.answerFound:
            grounding_result = evaluate_grounding(
                question=guardrail_result.sanitizedQuestion or request.query,
                answer=answer_result.answer,
                context=context_result["context"],
                citations=context_result["citations"],
                request_id=request_id,
            )
        else:
            grounding_result = None

        revision_result = None
        revision_attempted = False
        revision_count = 0
        final_answer = answer_result.answer
        final_used_citation_paths = answer_result.usedCitationPaths
        final_confidence = answer_result.confidence

        if (
            answer_result.answerFound
            and grounding_result is not None
            and grounding_result.requiresRevision
            and settings.max_revision_count > 0
        ):
            revision_attempted = True
            revision_count = 1

            revision_result = revise_answer(
                question=guardrail_result.sanitizedQuestion or request.query,
                answer=answer_result.answer,
                grounding_result=grounding_result,
                context=context_result["context"],
                citations=context_result["citations"],
                request_id=request_id,
            )

            if revision_result.revisionApplied:
                final_answer = revision_result.revisedAnswer
                final_used_citation_paths = revision_result.usedCitationPaths
                final_confidence = revision_result.confidence

        elif answer_result.answerFound and grounding_result is not None:
            log_event(
                event="revision_skipped",
                request_id=request_id,
                reason="Grounding critic did not require revision or revision budget is zero.",
                grounded=grounding_result.grounded,
                requiresRevision=grounding_result.requiresRevision,
                maxRevisionCount=settings.max_revision_count,
            )

    response = {
        "requestId": request_id,
        "query": request.query,
        "sanitizedQuestion": guardrail_result.sanitizedQuestion,
        "guardrail": guardrail_result.model_dump(),
        "answer": answer_result.answer,
        "answerFound": answer_result.answerFound,
        "confidence": answer_result.confidence,
        "usedCitationPaths": answer_result.usedCitationPaths,
        "citations": context_result["citations"],
        "usedDocuments": context_result["usedDocuments"],
        "grounding": grounding_result.model_dump() if grounding_result else None,
        "revision": revision_result.model_dump() if revision_result else None,
        "revisionAttempted": revision_attempted,
        "revisionCount": revision_count,
        "finalAnswer": final_answer,
        "finalUsedCitationPaths": final_used_citation_paths,
        "finalConfidence": final_confidence,
        "retrieval": {
            "resultCount": retrieval_result["resultCount"],
            "count": retrieval_result["count"],
            "latencyMs": retrieval_result["latencyMs"],
            "searchMode": request.searchMode,
            "vectorFields": request.vectorFields,
            "filters": request.filters,
        },
        "contextCharCount": context_result["contextCharCount"],
        "usedDocumentCount": context_result["usedDocumentCount"],
        "skippedDocumentCount": context_result["skippedDocumentCount"],
        "endpointLatencyMs": timer["elapsedMs"],
    }

    if request.includeDebugContext:
        response["context"] = context_result["context"]

    log_event(
        event="debug_answer_request_completed",
        request_id=request_id,
        answerFound=answer_result.answerFound,
        confidence=answer_result.confidence,
        grounded=grounding_result.grounded if grounding_result else None,
        requiresRevision=grounding_result.requiresRevision if grounding_result else None,
        revisionAttempted=revision_attempted,
        revisionCount=revision_count,
        revisionApplied=revision_result.revisionApplied if revision_result else None,
        retrievalResultCount=retrieval_result["resultCount"],
        usedDocumentCount=context_result["usedDocumentCount"],
        endpointLatencyMs=timer["elapsedMs"],
    )

    return response


@app.post("/debug/grounding")
async def debug_grounding(request: DebugGroundingRequest):
    request_id = new_request_id()

    log_event(
        event="debug_grounding_request_received",
        request_id=request_id,
        query=request.query,
        searchMode=request.searchMode,
        filters=request.filters,
        top=request.top,
        k=request.k,
        vectorFields=request.vectorFields,
    )

    with elapsed_timer() as timer:
        guardrail_result = run_input_guardrail_agent(
            question=request.query,
            request_id=request_id,
        )

        if not guardrail_result.allowed:
            response = {
                "requestId": request_id,
                "query": request.query,
                "guardrail": guardrail_result.model_dump(),
                "answer": "I cannot help with that request. Please ask a safe, manual-related question about Sandvik rotary equipment.",
                "answerFound": False,
                "confidence": 0.0,
                "usedCitationPaths": [],
                "citations": [],
                "usedDocuments": [],
                "grounding": None,
                "retrieval": None,
                "contextCharCount": 0,
                "usedDocumentCount": 0,
                "skippedDocumentCount": 0,
                "endpointLatencyMs": timer["elapsedMs"],
            }

            log_event(
                event="debug_grounding_blocked_by_guardrail",
                request_id=request_id,
                riskLevel=guardrail_result.riskLevel,
                reason=guardrail_result.reason,
                endpointLatencyMs=timer["elapsedMs"],
            )

            return response

        retrieval_result = execute_search(
            query=guardrail_result.sanitizedQuestion or request.query,
            search_mode=request.searchMode,
            vector_fields=request.vectorFields,
            filters=request.filters,
            top=request.top,
            k=request.k,
            use_semantic_ranker=request.useSemanticRanker,
            request_id=request_id,
        )

        context_result = build_context_from_documents(
            documents=retrieval_result["documents"],
            max_context_chars=request.maxContextChars,
            max_chars_per_document=request.maxCharsPerDocument,
            request_id=request_id,
        )

        answer_result = generate_answer_from_context(
            question=guardrail_result.sanitizedQuestion or request.query,
            context=context_result["context"],
            citations=context_result["citations"],
            request_id=request_id,
        )

        if answer_result.answerFound:
            grounding_result = evaluate_grounding(
                question=guardrail_result.sanitizedQuestion or request.query,
                answer=answer_result.answer,
                context=context_result["context"],
                citations=context_result["citations"],
                request_id=request_id,
            )
        else:
            grounding_result = None

    response = {
        "requestId": request_id,
        "query": request.query,
        "sanitizedQuestion": guardrail_result.sanitizedQuestion,
        "guardrail": guardrail_result.model_dump(),
        "answer": answer_result.answer,
        "answerFound": answer_result.answerFound,
        "confidence": answer_result.confidence,
        "usedCitationPaths": answer_result.usedCitationPaths,
        "citations": context_result["citations"],
        "usedDocuments": context_result["usedDocuments"],
        "grounding": grounding_result.model_dump() if grounding_result else None,
        "retrieval": {
            "resultCount": retrieval_result["resultCount"],
            "count": retrieval_result["count"],
            "latencyMs": retrieval_result["latencyMs"],
            "searchMode": request.searchMode,
            "vectorFields": request.vectorFields,
            "filters": request.filters,
        },
        "contextCharCount": context_result["contextCharCount"],
        "usedDocumentCount": context_result["usedDocumentCount"],
        "skippedDocumentCount": context_result["skippedDocumentCount"],
        "endpointLatencyMs": timer["elapsedMs"],
    }

    if request.includeDebugContext:
        response["context"] = context_result["context"]

    log_event(
        event="debug_grounding_request_completed",
        request_id=request_id,
        answerFound=answer_result.answerFound,
        grounded=grounding_result.grounded if grounding_result else None,
        requiresRevision=grounding_result.requiresRevision if grounding_result else None,
        retrievalResultCount=retrieval_result["resultCount"],
        usedDocumentCount=context_result["usedDocumentCount"],
        endpointLatencyMs=timer["elapsedMs"],
    )

    return response




@app.post("/debug/revision")
async def debug_revision(request: DebugRevisionRequest):
    request_id = new_request_id()

    log_event(
        event="debug_revision_request_received",
        request_id=request_id,
        query=request.query,
        searchMode=request.searchMode,
        filters=request.filters,
        top=request.top,
        k=request.k,
        vectorFields=request.vectorFields,
    )

    with elapsed_timer() as timer:
        guardrail_result = run_input_guardrail_agent(
            question=request.query,
            request_id=request_id,
        )

        if not guardrail_result.allowed:
            response = {
                "requestId": request_id,
                "query": request.query,
                "sanitizedQuestion": guardrail_result.sanitizedQuestion,
                "guardrail": guardrail_result.model_dump(),
                "answer": "I cannot help with that request. Please ask a safe, manual-related question about Sandvik rotary equipment.",
                "answerFound": False,
                "confidence": 0.0,
                "usedCitationPaths": [],
                "citations": [],
                "usedDocuments": [],
                "grounding": None,
                "revision": None,
                "revisionAttempted": False,
                "revisionCount": 0,
                "finalAnswer": "I cannot help with that request. Please ask a safe, manual-related question about Sandvik rotary equipment.",
                "retrieval": None,
                "contextCharCount": 0,
                "usedDocumentCount": 0,
                "skippedDocumentCount": 0,
                "endpointLatencyMs": timer["elapsedMs"],
            }

            log_event(
                event="debug_revision_blocked_by_guardrail",
                request_id=request_id,
                riskLevel=guardrail_result.riskLevel,
                reason=guardrail_result.reason,
                endpointLatencyMs=timer["elapsedMs"],
            )

            return response

        retrieval_result = execute_search(
            query=guardrail_result.sanitizedQuestion or request.query,
            search_mode=request.searchMode,
            vector_fields=request.vectorFields,
            filters=request.filters,
            top=request.top,
            k=request.k,
            use_semantic_ranker=request.useSemanticRanker,
            request_id=request_id,
        )

        context_result = build_context_from_documents(
            documents=retrieval_result["documents"],
            max_context_chars=request.maxContextChars,
            max_chars_per_document=request.maxCharsPerDocument,
            request_id=request_id,
        )

        answer_result = generate_answer_from_context(
            question=guardrail_result.sanitizedQuestion or request.query,
            context=context_result["context"],
            citations=context_result["citations"],
            request_id=request_id,
        )

        if answer_result.answerFound:
            grounding_result = evaluate_grounding(
                question=guardrail_result.sanitizedQuestion or request.query,
                answer=answer_result.answer,
                context=context_result["context"],
                citations=context_result["citations"],
                request_id=request_id,
            )
        else:
            grounding_result = None

        revision_result = None
        revision_attempted = False
        revision_count = 0
        final_answer = answer_result.answer
        final_used_citation_paths = answer_result.usedCitationPaths
        final_confidence = answer_result.confidence

        if (
            answer_result.answerFound
            and grounding_result is not None
            and grounding_result.requiresRevision
            and settings.max_revision_count > 0
        ):
            revision_attempted = True
            revision_count = 1

            revision_result = revise_answer(
                question=guardrail_result.sanitizedQuestion or request.query,
                answer=answer_result.answer,
                grounding_result=grounding_result,
                context=context_result["context"],
                citations=context_result["citations"],
                request_id=request_id,
            )

            if revision_result.revisionApplied:
                final_answer = revision_result.revisedAnswer
                final_used_citation_paths = revision_result.usedCitationPaths
                final_confidence = revision_result.confidence

        elif answer_result.answerFound and grounding_result is not None:
            log_event(
                event="revision_skipped",
                request_id=request_id,
                reason="Grounding critic did not require revision or revision budget is zero.",
                grounded=grounding_result.grounded,
                requiresRevision=grounding_result.requiresRevision,
                maxRevisionCount=settings.max_revision_count,
            )

    response = {
        "requestId": request_id,
        "query": request.query,
        "sanitizedQuestion": guardrail_result.sanitizedQuestion,
        "guardrail": guardrail_result.model_dump(),
        "answer": answer_result.answer,
        "answerFound": answer_result.answerFound,
        "confidence": answer_result.confidence,
        "usedCitationPaths": answer_result.usedCitationPaths,
        "citations": context_result["citations"],
        "usedDocuments": context_result["usedDocuments"],
        "grounding": grounding_result.model_dump() if grounding_result else None,
        "revision": revision_result.model_dump() if revision_result else None,
        "revisionAttempted": revision_attempted,
        "revisionCount": revision_count,
        "finalAnswer": final_answer,
        "finalUsedCitationPaths": final_used_citation_paths,
        "finalConfidence": final_confidence,
        "retrieval": {
            "resultCount": retrieval_result["resultCount"],
            "count": retrieval_result["count"],
            "latencyMs": retrieval_result["latencyMs"],
            "searchMode": request.searchMode,
            "vectorFields": request.vectorFields,
            "filters": request.filters,
        },
        "contextCharCount": context_result["contextCharCount"],
        "usedDocumentCount": context_result["usedDocumentCount"],
        "skippedDocumentCount": context_result["skippedDocumentCount"],
        "endpointLatencyMs": timer["elapsedMs"],
    }

    if request.includeDebugContext:
        response["context"] = context_result["context"]

    log_event(
        event="debug_revision_request_completed",
        request_id=request_id,
        answerFound=answer_result.answerFound,
        grounded=grounding_result.grounded if grounding_result else None,
        requiresRevision=grounding_result.requiresRevision if grounding_result else None,
        revisionAttempted=revision_attempted,
        revisionCount=revision_count,
        revisionApplied=revision_result.revisionApplied if revision_result else None,
        retrievalResultCount=retrieval_result["resultCount"],
        usedDocumentCount=context_result["usedDocumentCount"],
        endpointLatencyMs=timer["elapsedMs"],
    )

    return response

from fastapi import FastAPI

from backend.agents.answer_generation_agent import generate_answer_from_context
from backend.agents.grounding_critic_agent import evaluate_grounding
from backend.agents.input_guardrail_agent import run_input_guardrail_agent
from backend.agents.revision_agent import revise_answer
from backend.agents.safety_critic_agent import evaluate_safety
from backend.config import settings
from backend.memory.factory import get_memory_repository
from backend.constants import APPROVED_INDEX_FIELDS, AZURE_SEARCH_INDEX_NAME
from backend.context.context_builder import build_context_from_documents
from backend.models import (
    ChatRequest,
    ClearChatMemoryRequest,
    ClearChatMemoryResponse,
    ChatResponse,
    DebugAnswerRequest,
    DebugContextRequest,
    DebugGroundingRequest,
    DebugGraphAnswerRequest,
    DebugGuardrailRequest,
    DebugRetrievalRequest,
    DebugRevisionRequest,
    DebugSafetyRequest,
    ProductionAskRequest,
    ProductionAskResponse,
)
from backend.observability.logger import log_event
from backend.observability.request_context import new_request_id
from backend.observability.timing import elapsed_timer
from backend.retrieval.search_executor import execute_search
from backend.graph.workflow import graph_state_to_debug_response, run_rag_graph
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

        if answer_result.answerFound:
            safety_result = evaluate_safety(
                question=guardrail_result.sanitizedQuestion or request.query,
                answer=final_answer,
                context=context_result["context"],
                citations=context_result["citations"],
                request_id=request_id,
            )
        else:
            safety_result = None

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
        "safety": safety_result.model_dump() if safety_result else None,
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
        safe=safety_result.safe if safety_result else None,
        safetyRequiresRevision=safety_result.requiresRevision if safety_result else None,
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



@app.post("/debug/safety")
async def debug_safety(request: DebugSafetyRequest):
    request_id = new_request_id()

    log_event(
        event="debug_safety_request_received",
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
                "safety": None,
                "revisionAttempted": False,
                "revisionCount": 0,
                "finalAnswer": "I cannot help with that request. Please ask a safe, manual-related question about Sandvik rotary equipment.",
                "finalUsedCitationPaths": [],
                "finalConfidence": 0.0,
                "retrieval": None,
                "contextCharCount": 0,
                "usedDocumentCount": 0,
                "skippedDocumentCount": 0,
                "endpointLatencyMs": timer["elapsedMs"],
            }

            log_event(
                event="debug_safety_blocked_by_guardrail",
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

        if answer_result.answerFound:
            safety_result = evaluate_safety(
                question=guardrail_result.sanitizedQuestion or request.query,
                answer=final_answer,
                context=context_result["context"],
                citations=context_result["citations"],
                request_id=request_id,
            )
        else:
            safety_result = None

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
        "safety": safety_result.model_dump() if safety_result else None,
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
        event="debug_safety_request_completed",
        request_id=request_id,
        answerFound=answer_result.answerFound,
        grounded=grounding_result.grounded if grounding_result else None,
        groundingRequiresRevision=grounding_result.requiresRevision if grounding_result else None,
        revisionAttempted=revision_attempted,
        revisionCount=revision_count,
        revisionApplied=revision_result.revisionApplied if revision_result else None,
        safe=safety_result.safe if safety_result else None,
        safetyRequiresRevision=safety_result.requiresRevision if safety_result else None,
        retrievalResultCount=retrieval_result["resultCount"],
        usedDocumentCount=context_result["usedDocumentCount"],
        endpointLatencyMs=timer["elapsedMs"],
    )

    return response



@app.post("/debug/graph-answer")
async def debug_graph_answer(request: DebugGraphAnswerRequest):
    request_id = new_request_id()

    log_event(
        event="debug_graph_answer_request_received",
        request_id=request_id,
        threadId=request.threadId,
        query=request.query,
        searchMode=request.searchMode,
        filters=request.filters,
        top=request.top,
        k=request.k,
        vectorFields=request.vectorFields,
        useSemanticRanker=request.useSemanticRanker,
    )

    with elapsed_timer() as timer:
        graph_state = run_rag_graph(
            request_id=request_id,
            question=request.query,
            thread_id=request.threadId,
            user_id=request.userId,
            search_mode=request.searchMode,
            vector_fields=request.vectorFields,
            filters=request.filters,
            top=request.top,
            k=request.k,
            use_semantic_ranker=request.useSemanticRanker,
            include_debug_context=request.includeDebugContext,
            max_context_chars=request.maxContextChars or settings.max_context_chars,
            max_chars_per_document=request.maxCharsPerDocument or settings.max_chars_per_document,
            answer_max_completion_tokens=settings.answer_max_completion_tokens,
            critic_max_completion_tokens=settings.critic_max_completion_tokens,
            revision_max_completion_tokens=settings.revision_max_completion_tokens,
            max_llm_calls=settings.max_llm_calls_per_request,
            max_revision_count=settings.max_revision_count,
            max_recent_turns=settings.max_recent_turns,
            conversation_summary_max_chars=settings.conversation_summary_max_chars,
        )

    response = graph_state_to_debug_response(graph_state)
    response["endpointLatencyMs"] = timer["elapsedMs"]

    log_event(
        event="debug_graph_answer_request_completed",
        request_id=request_id,
        threadId=graph_state.get("thread_id"),
        answerFound=graph_state.get("answer_found"),
        finalConfidence=graph_state.get("final_confidence"),
        llmCallsUsed=graph_state.get("budgets", {}).get("llmCallsUsed"),
        revisionCount=graph_state.get("budgets", {}).get("revisionCount"),
        traceStepCount=len(graph_state.get("trace_steps", [])),
        errorCount=len(graph_state.get("errors", [])),
        endpointLatencyMs=timer["elapsedMs"],
    )

    return response



@app.post("/ask", response_model=ProductionAskResponse)
async def ask(request: ProductionAskRequest):
    request_id = new_request_id()

    log_event(
        event="ask_request_received",
        request_id=request_id,
        threadId=request.threadId,
        question=request.question,
        searchMode=request.searchMode,
        filters=request.filters,
        top=request.top,
        k=request.k,
        vectorFields=request.vectorFields,
        useSemanticRanker=request.useSemanticRanker,
    )

    with elapsed_timer() as timer:
        graph_state = run_rag_graph(
            request_id=request_id,
            question=request.question,
            thread_id=request.threadId,
            user_id=request.userId,
            search_mode=request.searchMode,
            vector_fields=request.vectorFields,
            filters=request.filters,
            top=request.top,
            k=request.k,
            use_semantic_ranker=request.useSemanticRanker,
            include_debug_context=False,
            max_context_chars=settings.max_context_chars,
            max_chars_per_document=settings.max_chars_per_document,
            answer_max_completion_tokens=settings.answer_max_completion_tokens,
            critic_max_completion_tokens=settings.critic_max_completion_tokens,
            revision_max_completion_tokens=settings.revision_max_completion_tokens,
            max_llm_calls=settings.max_llm_calls_per_request,
            max_revision_count=settings.max_revision_count,
            max_recent_turns=settings.max_recent_turns,
            conversation_summary_max_chars=settings.conversation_summary_max_chars,
        )

    guardrail = graph_state.get("guardrail") or {}
    safety = graph_state.get("safety") or None

    if guardrail.get("allowed") is False:
        answer = (
            "I cannot help with that request. Please ask a safe, manual-related question about Sandvik rotary equipment."
        )
        answer_found = False
        confidence = 0.0
        citations = []
        used_citation_paths = []
        safety_summary = None
    else:
        answer = graph_state.get("final_answer") or (
            "I could not find enough information in the retrieved manual context to answer this question."
        )
        answer_found = bool(graph_state.get("answer_found", False))
        confidence = float(graph_state.get("final_confidence", 0.0))
        citations = graph_state.get("citations", [])
        used_citation_paths = graph_state.get("final_used_citation_paths", [])
        safety_summary = (
            {
                "safe": bool(safety.get("safe", False)),
                "requiresRevision": bool(safety.get("requiresRevision", False)),
            }
            if safety
            else None
        )

    response = {
        "requestId": request_id,
        "threadId": graph_state.get("thread_id", request.threadId or request_id),
        "answer": answer,
        "answerFound": answer_found,
        "confidence": confidence,
        "citations": citations,
        "usedCitationPaths": used_citation_paths,
        "safety": safety_summary,
        "latencyMs": timer["elapsedMs"],
    }

    log_event(
        event="ask_request_completed",
        request_id=request_id,
        threadId=response["threadId"],
        answerFound=response["answerFound"],
        confidence=response["confidence"],
        citationCount=len(response["citations"]),
        usedCitationPathCount=len(response["usedCitationPaths"]),
        safetySafe=response["safety"]["safe"] if response["safety"] else None,
        safetyRequiresRevision=response["safety"]["requiresRevision"] if response["safety"] else None,
        llmCallsUsed=graph_state.get("budgets", {}).get("llmCallsUsed"),
        revisionCount=graph_state.get("budgets", {}).get("revisionCount"),
        errorCount=len(graph_state.get("errors", [])),
        endpointLatencyMs=timer["elapsedMs"],
    )

    return response



@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    request_id = new_request_id()

    log_event(
        event="chat_request_received",
        request_id=request_id,
        threadId=request.threadId,
        userId=request.userId,
        message=request.message,
        searchMode=request.searchMode,
        filters=request.filters,
        top=request.top,
        k=request.k,
        vectorFields=request.vectorFields,
        useSemanticRanker=request.useSemanticRanker,
    )

    with elapsed_timer() as timer:
        graph_state = run_rag_graph(
            request_id=request_id,
            question=request.message,
            thread_id=request.threadId,
            user_id=request.userId,
            search_mode=request.searchMode,
            vector_fields=request.vectorFields,
            filters=request.filters,
            top=request.top,
            k=request.k,
            use_semantic_ranker=request.useSemanticRanker,
            include_debug_context=False,
            max_context_chars=settings.max_context_chars,
            max_chars_per_document=settings.max_chars_per_document,
            answer_max_completion_tokens=settings.answer_max_completion_tokens,
            critic_max_completion_tokens=settings.critic_max_completion_tokens,
            revision_max_completion_tokens=settings.revision_max_completion_tokens,
            max_llm_calls=settings.max_llm_calls_per_request,
            max_revision_count=settings.max_revision_count,
            max_recent_turns=settings.max_recent_turns,
            conversation_summary_max_chars=settings.conversation_summary_max_chars,
        )

    guardrail = graph_state.get("guardrail") or {}
    query_understanding = graph_state.get("query_understanding") or {}
    detected_entities = query_understanding.get("detectedEntities") or {}
    safety = graph_state.get("safety") or None

    if guardrail.get("allowed") is False:
        answer = (
            "I cannot help with that request. Please ask a safe, manual-related question about Sandvik rotary equipment."
        )
        answer_found = False
        confidence = 0.0
        citations = []
        used_citation_paths = []
        safety_summary = None
    else:
        answer = graph_state.get("final_answer") or (
            "I could not find enough information in the retrieved manual context to answer this question."
        )
        answer_found = bool(graph_state.get("answer_found", False))
        confidence = float(graph_state.get("final_confidence", 0.0))
        citations = graph_state.get("citations", [])
        used_citation_paths = graph_state.get("final_used_citation_paths", [])
        safety_summary = (
            {
                "safe": bool(safety.get("safe", False)),
                "requiresRevision": bool(safety.get("requiresRevision", False)),
            }
            if safety
            else None
        )

    detected_context = {
        "intent": query_understanding.get("intent"),
        "machine": detected_entities.get("machine"),
        "baseMachine": detected_entities.get("baseMachine"),
        "serialNumber": detected_entities.get("serialNumber"),
        "manualType": detected_entities.get("manualType"),
        "component": detected_entities.get("component"),
        "procedureType": detected_entities.get("procedureType"),
        "filters": query_understanding.get("filters") or {},
        "rewrittenQuery": query_understanding.get("rewrittenQuery"),
    }

    memory_summary = {
        "recentTurnCount": len(graph_state.get("recent_turns", [])),
        "hasConversationSummary": bool(graph_state.get("conversation_summary")),
    }

    response = {
        "requestId": request_id,
        "threadId": graph_state.get("thread_id", request.threadId or request_id),
        "answer": answer,
        "answerFound": answer_found,
        "confidence": confidence,
        "detectedContext": detected_context,
        "citations": citations,
        "usedCitationPaths": used_citation_paths,
        "safety": safety_summary,
        "memory": memory_summary,
        "latencyMs": timer["elapsedMs"],
    }

    log_event(
        event="chat_request_completed",
        request_id=request_id,
        threadId=response["threadId"],
        answerFound=response["answerFound"],
        confidence=response["confidence"],
        detectedIntent=detected_context.get("intent"),
        detectedBaseMachine=detected_context.get("baseMachine"),
        detectedComponent=detected_context.get("component"),
        filterCount=len(detected_context.get("filters", {})),
        citationCount=len(response["citations"]),
        usedCitationPathCount=len(response["usedCitationPaths"]),
        safetySafe=response["safety"]["safe"] if response["safety"] else None,
        safetyRequiresRevision=response["safety"]["requiresRevision"] if response["safety"] else None,
        recentTurnCount=response["memory"]["recentTurnCount"],
        llmCallsUsed=graph_state.get("budgets", {}).get("llmCallsUsed"),
        revisionCount=graph_state.get("budgets", {}).get("revisionCount"),
        errorCount=len(graph_state.get("errors", [])),
        endpointLatencyMs=timer["elapsedMs"],
    )

    return response


@app.post("/chat/memory/clear", response_model=ClearChatMemoryResponse)
async def clear_chat_memory(request: ClearChatMemoryRequest):
    request_id = new_request_id()

    log_event(
        event="chat_memory_clear_requested",
        request_id=request_id,
        threadId=request.threadId,
    )

    repo = get_memory_repository()
    repo.clear_memory(request.threadId)

    log_event(
        event="chat_memory_cleared",
        request_id=request_id,
        threadId=request.threadId,
    )

    return {
        "requestId": request_id,
        "threadId": request.threadId,
        "cleared": True,
    }
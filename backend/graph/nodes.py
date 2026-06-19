from __future__ import annotations

from typing import Any

from backend.agents.answer_generation_agent import generate_answer_from_context
from backend.agents.grounding_critic_agent import evaluate_grounding
from backend.agents.input_guardrail_agent import run_input_guardrail_agent
from backend.agents.query_understanding_agent import (
    extract_base_machine_from_text,
    extract_component_from_text,
    understand_query,
)
from backend.agents.revision_agent import revise_answer
from backend.agents.safety_critic_agent import evaluate_safety
from backend.context.context_builder import build_context_from_documents
from backend.context.image_reranker import rerank_image_references
from backend.context.image_reference_extractor import (
    extract_image_references_from_documents,
    extract_image_references_from_context_text,
    extract_image_references_from_single_context_for_citation,
    filter_image_references_for_used_citations,
    extract_image_references_from_documents,
    filter_image_references_for_used_citations,
)
from backend.context.image_reference_resolver import resolve_image_references
from backend.formatting.answer_formatter import format_answer_text
from backend.memory.factory import get_memory_repository
from backend.memory.models import ActiveConversationContext, ChatMessage
from backend.graph.state import (
    RagGraphState,
    add_graph_error,
    add_trace_step,
    increment_llm_call_count,
    increment_revision_count,
    llm_budget_remaining,
    revision_budget_remaining,
)
from backend.observability.timing import elapsed_timer
from backend.retrieval.search_executor import execute_search

from backend.agents.image_retrieval_agent import (
    extract_candidate_images_from_chunks,
    retrieve_relevant_images_for_final_answer,
)



def _request_id(state: RagGraphState) -> str | None:
    return state.get("request_id")


def _question_for_agents(state: RagGraphState) -> str:
    sanitized = state.get("sanitized_question") or ""
    current = state.get("current_question") or ""
    return sanitized or current


QUERY_UNDERSTANDING_FILTER_CONFIDENCE_THRESHOLD = 0.75


def _query_understanding_for_retrieval(state: RagGraphState) -> dict[str, Any]:
    query_understanding = state.get("query_understanding") or {}
    rewritten_query = (query_understanding.get("rewrittenQuery") or "").strip()

    return {
        "retrievalQuery": rewritten_query or _question_for_agents(state),
        "rawFilters": query_understanding.get("filters") or {},
        "filterConfidence": query_understanding.get("filterConfidence") or {},
    }


def _confident_query_understanding_filters(state: RagGraphState) -> dict[str, str]:
    query_understanding = state.get("query_understanding") or {}
    raw_filters = query_understanding.get("filters") or {}
    filter_confidence = query_understanding.get("filterConfidence") or {}

    confident_filters: dict[str, str] = {}

    for field_name, field_value in raw_filters.items():
        if field_value is None:
            continue

        confidence = filter_confidence.get(field_name, 0.0)

        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            confidence_value = 0.0

        if confidence_value >= QUERY_UNDERSTANDING_FILTER_CONFIDENCE_THRESHOLD:
            confident_filters[field_name] = str(field_value)

    return confident_filters


def _merged_retrieval_filters(state: RagGraphState) -> dict[str, str]:
    query_filters = _confident_query_understanding_filters(state)
    request_filters = state.get("filters") or {}

    # Explicit request filters win over model-extracted filters.
    return {
        **query_filters,
        **request_filters,
    }


def _recent_turns_text(state: RagGraphState, max_chars: int = 4000) -> str:
    recent_turns = state.get("recent_turns") or []
    parts: list[str] = []

    for turn in recent_turns[-6:]:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if content:
            parts.append(f"{role}: {content}")

    combined = "\n".join(parts)
    return combined[-max_chars:]


def _apply_memory_context_to_query_understanding(
    state: RagGraphState,
    result: Any,
) -> Any:
    """Use recent turns as deterministic context when current query lacks machine/component.

    This does not override entities/filters already detected from the current question.
    """

    memory_text = _recent_turns_text(state)
    active_context = state.get("active_context") or {}

    if not memory_text and not active_context:
        return result

    detected_memory_base_machine = (
        active_context.get("baseMachine")
        or extract_base_machine_from_text(memory_text)
    )
    detected_memory_component = (
        active_context.get("component")
        or extract_component_from_text(memory_text)
    )

    if detected_memory_base_machine and not result.detectedEntities.baseMachine:
        result.detectedEntities.baseMachine = detected_memory_base_machine

        if "baseMachine" not in result.filters:
            result.filters["baseMachine"] = detected_memory_base_machine
            result.filterConfidence["baseMachine"] = 0.9

    if detected_memory_component and not result.detectedEntities.component:
        result.detectedEntities.component = detected_memory_component

    if detected_memory_base_machine:
        rewritten_query = result.rewrittenQuery or _question_for_agents(state)

        if detected_memory_base_machine.lower() not in rewritten_query.lower():
            result.rewrittenQuery = f"{rewritten_query} {detected_memory_base_machine}".strip()

    return result



def load_memory_node(state: RagGraphState) -> RagGraphState:
    request_id = _request_id(state)
    thread_id = state.get("thread_id") or request_id or "default-thread"

    with elapsed_timer() as timer:
        try:
            repo = get_memory_repository()
            memory = repo.get_memory(thread_id)

            recent_turns = [
                {
                    "role": turn.role,
                    "content": turn.content,
                    "timestamp": turn.timestamp,
                    "requestId": turn.requestId,
                    "metadata": turn.metadata,
                }
                for turn in memory.recentTurns
            ]

            state["conversation_summary"] = memory.conversationSummary
            state["recent_turns"] = recent_turns
            state["messages"] = recent_turns
            state["active_context"] = memory.activeContext.model_dump()

            add_trace_step(
                state,
                node="load_memory",
                event="completed",
                latency_ms=timer["elapsedMs"],
                input_summary={"threadId": thread_id},
                output_summary={
                    "recentTurnCount": len(recent_turns),
                    "hasConversationSummary": bool(memory.conversationSummary),
                    "activeContext": memory.activeContext.model_dump(),
                },
            )

            return state

        except Exception as exc:
            add_graph_error(
                state,
                node="load_memory",
                error_type=type(exc).__name__,
                message=str(exc),
                recoverable=True,
            )

            state["conversation_summary"] = ""
            state["recent_turns"] = []
            state["messages"] = []

            add_trace_step(
                state,
                node="load_memory",
                event="failed",
                latency_ms=timer["elapsedMs"],
                error=str(exc),
            )

            return state


def input_guardrail_node(state: RagGraphState) -> RagGraphState:
    request_id = _request_id(state)

    with elapsed_timer() as timer:
        try:
            result = run_input_guardrail_agent(
                question=state.get("current_question", ""),
                request_id=request_id,
            )

            state["guardrail"] = result.model_dump()
            state["sanitized_question"] = result.sanitizedQuestion or state.get("current_question", "")

            add_trace_step(
                state,
                node="input_guardrail",
                event="completed",
                latency_ms=timer["elapsedMs"],
                input_summary={
                    "question": state.get("current_question", ""),
                },
                output_summary={
                    "allowed": result.allowed,
                    "riskLevel": result.riskLevel,
                    "mode": "deterministic_or_llm",
                },
            )

            if not result.allowed:
                state["answer_found"] = False
                state["final_answer"] = (
                    "I cannot help with that request. Please ask a safe, manual-related question about Sandvik rotary equipment."
                )
                state["final_confidence"] = 0.0
                state["final_used_citation_paths"] = []

            return state

        except Exception as exc:
            add_graph_error(
                state,
                node="input_guardrail",
                error_type=type(exc).__name__,
                message=str(exc),
                recoverable=False,
            )
            add_trace_step(
                state,
                node="input_guardrail",
                event="failed",
                latency_ms=timer["elapsedMs"],
                error=str(exc),
            )

            state["guardrail"] = {
                "allowed": False,
                "sanitizedQuestion": "",
                "riskLevel": "high",
                "reason": "Input guardrail failed closed.",
            }
            state["answer_found"] = False
            state["final_answer"] = (
                "I cannot help with that request. Please ask a safe, manual-related question about Sandvik rotary equipment."
            )
            state["final_confidence"] = 0.0
            state["final_used_citation_paths"] = []
            return state




def query_understanding_node(state: RagGraphState) -> RagGraphState:
    request_id = _request_id(state)
    guardrail = state.get("guardrail") or {}

    if guardrail and guardrail.get("allowed") is False:
        add_trace_step(
            state,
            node="query_understanding",
            event="skipped",
            input_summary={"reason": "Guardrail blocked request."},
        )
        return state

    if not llm_budget_remaining(state):
        state["query_understanding"] = {
            "intent": "unknown",
            "confidence": 0.0,
            "rewrittenQuery": _question_for_agents(state),
            "keywords": [],
            "detectedEntities": {
                "machine": None,
                "baseMachine": None,
                "serialNumber": None,
                "manualType": None,
                "component": None,
                "procedureType": None,
            },
            "filters": {},
            "filterConfidence": {},
            "needsClarification": False,
            "clarificationQuestion": None,
            "reason": "Query understanding skipped because LLM budget was exhausted.",
        }

        add_trace_step(
            state,
            node="query_understanding",
            event="skipped",
            input_summary={"reason": "LLM budget exhausted."},
            output_summary={
                "intent": "unknown",
                "confidence": 0.0,
                "filterCount": 0,
            },
        )
        return state

    increment_llm_call_count(state)

    with elapsed_timer() as timer:
        try:
            result = understand_query(
                question=_question_for_agents(state),
                conversation_summary=state.get("conversation_summary", ""),
                recent_turns=state.get("recent_turns", []),
                request_id=request_id,
            )

            result = _apply_memory_context_to_query_understanding(
                state=state,
                result=result,
            )

            state["query_understanding"] = result.model_dump()

            add_trace_step(
                state,
                node="query_understanding",
                event="completed",
                latency_ms=timer["elapsedMs"],
                input_summary={
                    "question": _question_for_agents(state),
                    "recentTurnCount": len(state.get("recent_turns", [])),
                    "hasConversationSummary": bool(state.get("conversation_summary")),
                },
                output_summary={
                    "intent": result.intent,
                    "confidence": result.confidence,
                    "rewrittenQuery": result.rewrittenQuery,
                    "filters": result.filters,
                    "needsClarification": result.needsClarification,
                },
            )

            return state

        except Exception as exc:
            add_graph_error(
                state,
                node="query_understanding",
                error_type=type(exc).__name__,
                message=str(exc),
                recoverable=True,
            )

            state["query_understanding"] = {
                "intent": "unknown",
                "confidence": 0.0,
                "rewrittenQuery": _question_for_agents(state),
                "keywords": [],
                "detectedEntities": {
                    "machine": None,
                    "baseMachine": None,
                    "serialNumber": None,
                    "manualType": None,
                    "component": None,
                    "procedureType": None,
                },
                "filters": {},
                "filterConfidence": {},
                "needsClarification": False,
                "clarificationQuestion": None,
                "reason": "Query understanding node failed; using original question without filters.",
            }

            add_trace_step(
                state,
                node="query_understanding",
                event="failed",
                latency_ms=timer["elapsedMs"],
                error=str(exc),
            )

            return state


def retrieval_node(state: RagGraphState) -> RagGraphState:
    request_id = _request_id(state)

    guardrail = state.get("guardrail") or {}
    if guardrail and guardrail.get("allowed") is False:
        add_trace_step(
            state,
            node="retrieval",
            event="skipped",
            input_summary={"reason": "Guardrail blocked request."},
        )
        return state

    with elapsed_timer() as timer:
        try:
            retrieval_plan = _query_understanding_for_retrieval(state)
            retrieval_query = retrieval_plan["retrievalQuery"]
            applied_filters = _merged_retrieval_filters(state)

            result = execute_search(
                query=retrieval_query,
                search_mode=state.get("search_mode", "hybrid"),
                vector_fields=state.get("vector_fields", ["contentVector"]),
                filters=applied_filters,
                top=int(state.get("top", 3)),
                k=int(state.get("k", 50)),
                use_semantic_ranker=bool(state.get("use_semantic_ranker", False)),
                request_id=request_id,
            )

            state["retrieval"] = result

            add_trace_step(
                state,
                node="retrieval",
                event="completed",
                latency_ms=timer["elapsedMs"],
                input_summary={
                    "originalQuestion": _question_for_agents(state),
                    "retrievalQuery": retrieval_query,
                    "searchMode": state.get("search_mode", "hybrid"),
                    "requestFilters": state.get("filters", {}),
                    "queryUnderstandingFilters": retrieval_plan["rawFilters"],
                    "queryUnderstandingFilterConfidence": retrieval_plan["filterConfidence"],
                    "appliedFilters": applied_filters,
                    "top": state.get("top", 3),
                    "k": state.get("k", 50),
                },
                output_summary={
                    "resultCount": result.get("resultCount"),
                    "count": result.get("count"),
                },
            )

            return state

        except Exception as exc:
            add_graph_error(
                state,
                node="retrieval",
                error_type=type(exc).__name__,
                message=str(exc),
                recoverable=True,
            )
            add_trace_step(
                state,
                node="retrieval",
                event="failed",
                latency_ms=timer["elapsedMs"],
                error=str(exc),
            )

            state["retrieval"] = {
                "documents": [],
                "resultCount": 0,
                "count": 0,
                "latencyMs": timer["elapsedMs"],
            }
            state["answer_found"] = False
            state["final_answer"] = "I could not retrieve relevant manual context for this question."
            state["final_confidence"] = 0.0
            state["final_used_citation_paths"] = []

    try:
        retrieval_payload = state.get("retrieval", {}) or {}
        retrieval_documents = (
            retrieval_payload.get("documents")
            or retrieval_payload.get("results")
            or retrieval_payload.get("value")
            or retrieval_payload.get("items")
            or []
        )

        if not retrieval_documents:
            retrieval_documents = state.get("used_documents", []) or []

        first_document = retrieval_documents[0] if retrieval_documents else {}
        first_content = str(first_document.get("content") or "") if isinstance(first_document, dict) else ""

        retrieval_candidate_images = extract_candidate_images_from_chunks(
            documents=retrieval_documents,
            citations=state.get("citations", []),
        )

        state["candidate_image_references"] = retrieval_candidate_images

        image_debug = dict(state.get("image_reference_debug", {}) or {})
        image_debug.update(
            {
                "imagePipelineBuild": "phase51-image-agent-trace",
                "imageAgentRan": True,
                "retrievalExtractionRan": True,
                "retrievalDocumentCountForImageExtraction": len(retrieval_documents or []),
                "retrievalCandidateImageCount": len(retrieval_candidate_images or []),
                "retrievalFirstDocumentKeys": sorted(list(first_document.keys())) if isinstance(first_document, dict) else [],
                "retrievalFirstContentLength": len(first_content),
                "retrievalFirstContainsPng": ".png" in first_content.lower(),
                "retrievalFirstContainsGuid": "GUID-" in first_content,
                "retrievalFirstTitle": first_document.get("title") if isinstance(first_document, dict) else None,
                "retrievalFirstCitationPath": first_document.get("citationPath") if isinstance(first_document, dict) else None,
            }
        )
        state["image_reference_debug"] = image_debug

        state = add_trace_step(
            state,
            node="image_retrieval_agent",
            event="retrieval_candidate_extraction_completed",
            output_summary={
                "imagePipelineBuild": "phase51-image-agent-trace",
                "retrievalDocumentCount": len(retrieval_documents or []),
                "firstDocumentContentLength": len(first_content),
                "firstDocumentContainsPng": ".png" in first_content.lower(),
                "firstDocumentContainsGuid": "GUID-" in first_content,
                "candidateImageCount": len(retrieval_candidate_images or []),
            },
        )

    except Exception as exc:
        errors = list(state.get("image_reference_errors", []))
        errors.append(
            {
                "stage": "retrieval_node_image_candidate_extraction",
                "message": str(exc),
                "recoverable": True,
            }
        )
        state["image_reference_errors"] = errors

        image_debug = dict(state.get("image_reference_debug", {}) or {})
        image_debug.update(
            {
                "imagePipelineBuild": "phase51-image-agent-trace",
                "imageAgentRan": False,
                "retrievalExtractionRan": False,
                "retrievalExtractionError": str(exc),
            }
        )
        state["image_reference_debug"] = image_debug

        state = add_trace_step(
            state,
            node="image_retrieval_agent",
            event="retrieval_candidate_extraction_failed",
            error=str(exc),
            output_summary={
                "imagePipelineBuild": "phase51-image-agent-trace",
            },
        )
    return state


def context_builder_node(state: RagGraphState) -> RagGraphState:
    request_id = _request_id(state)

    retrieval = state.get("retrieval") or {}
    documents = retrieval.get("documents", [])

    if not documents:
        add_trace_step(
            state,
            node="context_builder",
            event="skipped",
            input_summary={"reason": "No retrieved documents."},
        )
        state["context"] = ""
        state["context_char_count"] = 0
        state["citations"] = []
        state["used_documents"] = []
        state["skipped_documents"] = []
        return state

    budgets = state.get("budgets", {})

    with elapsed_timer() as timer:
        try:
            result = build_context_from_documents(
                documents=documents,
                max_context_chars=budgets.get("maxContextChars"),
                max_chars_per_document=budgets.get("maxCharsPerDocument"),
                request_id=request_id,
            )




            try:
                raw_image_documents = documents

                candidate_image_references = extract_candidate_images_from_chunks(
                    documents=raw_image_documents,
                    citations=state.get("citations", []),
                )

                state["candidate_image_references"] = candidate_image_references

            except Exception as exc:
                errors = list(state.get("image_reference_errors", []))
                errors.append(
                    {
                        "stage": "context_builder_image_candidate_extraction",
                        "message": str(exc),
                        "recoverable": True,
                    }
                )
                state["candidate_image_references"] = []
                state["image_reference_errors"] = errors




            state["context"] = result["context"]
            state["context_char_count"] = result["contextCharCount"]
            state["citations"] = result["citations"]
            state["used_documents"] = result["usedDocuments"]
            state["skipped_documents"] = result["skippedDocuments"]
            

            add_trace_step(
                state,
                node="context_builder",
                event="completed",
                latency_ms=timer["elapsedMs"],
                input_summary={
                    "documentCount": len(documents),
                },
                output_summary={
                    "contextCharCount": result["contextCharCount"],
                    "usedDocumentCount": result["usedDocumentCount"],
                    "citationCount": len(result["citations"]),
                },
            )

            return state

        except Exception as exc:
            add_graph_error(
                state,
                node="context_builder",
                error_type=type(exc).__name__,
                message=str(exc),
                recoverable=True,
            )
            add_trace_step(
                state,
                node="context_builder",
                event="failed",
                latency_ms=timer["elapsedMs"],
                error=str(exc),
            )

            state["context"] = ""
            state["context_char_count"] = 0
            state["citations"] = []
            state["used_documents"] = []
            state["skipped_documents"] = []
            return state


def answer_generation_node(state: RagGraphState) -> RagGraphState:
    request_id = _request_id(state)

    if not state.get("context"):
        add_trace_step(
            state,
            node="answer_generation",
            event="skipped",
            input_summary={"reason": "No context available."},
        )
        state["answer"] = {
            "answer": "I could not find enough information in the retrieved manual context to answer this question.",
            "usedCitationPaths": [],
            "confidence": 0.0,
            "answerFound": False,
        }
        state["answer_found"] = False
        state["final_answer"] = state["answer"]["answer"]
        state["final_confidence"] = 0.0
        state["final_used_citation_paths"] = []
        return state

    if not llm_budget_remaining(state):
        add_trace_step(
            state,
            node="answer_generation",
            event="skipped",
            input_summary={"reason": "LLM budget exhausted."},
        )
        state["answer_found"] = False
        state["final_answer"] = "The answer could not be generated because the LLM call budget was exhausted."
        state["final_confidence"] = 0.0
        state["final_used_citation_paths"] = []
        return state

    increment_llm_call_count(state)

    with elapsed_timer() as timer:
        try:
            result = generate_answer_from_context(
                question=_question_for_agents(state),
                context=state.get("context", ""),
                citations=state.get("citations", []),
                request_id=request_id,
            )

            state["answer"] = result.model_dump()
            state["answer_found"] = result.answerFound
            state["final_answer"] = result.answer
            state["final_confidence"] = result.confidence
            state["final_used_citation_paths"] = result.usedCitationPaths

            add_trace_step(
                state,
                node="answer_generation",
                event="completed",
                latency_ms=timer["elapsedMs"],
                output_summary={
                    "answerFound": result.answerFound,
                    "confidence": result.confidence,
                    "usedCitationCount": len(result.usedCitationPaths),
                },
            )

            return state

        except Exception as exc:
            add_graph_error(
                state,
                node="answer_generation",
                error_type=type(exc).__name__,
                message=str(exc),
                recoverable=True,
            )
            add_trace_step(
                state,
                node="answer_generation",
                event="failed",
                latency_ms=timer["elapsedMs"],
                error=str(exc),
            )

            state["answer"] = {
                "answer": "The answer generation step failed.",
                "usedCitationPaths": [],
                "confidence": 0.0,
                "answerFound": False,
            }
            state["answer_found"] = False
            state["final_answer"] = state["answer"]["answer"]
            state["final_confidence"] = 0.0
            state["final_used_citation_paths"] = []
            return state


def grounding_critic_node(state: RagGraphState) -> RagGraphState:
    request_id = _request_id(state)

    answer = state.get("answer") or {}
    if not answer.get("answerFound"):
        add_trace_step(
            state,
            node="grounding_critic",
            event="skipped",
            input_summary={"reason": "No answer found."},
        )
        return state

    if not llm_budget_remaining(state):
        add_trace_step(
            state,
            node="grounding_critic",
            event="skipped",
            input_summary={"reason": "LLM budget exhausted."},
        )
        state["grounding"] = {
            "grounded": False,
            "requiresRevision": True,
            "unsupportedClaims": [],
            "missingCitations": [],
            "reason": "Grounding critic skipped because LLM budget was exhausted.",
            "confidence": 0.0,
        }
        return state

    increment_llm_call_count(state)

    with elapsed_timer() as timer:
        try:
            result = evaluate_grounding(
                question=_question_for_agents(state),
                answer=answer.get("answer", ""),
                context=state.get("context", ""),
                citations=state.get("citations", []),
                request_id=request_id,
            )

            state["grounding"] = result.model_dump()

            add_trace_step(
                state,
                node="grounding_critic",
                event="completed",
                latency_ms=timer["elapsedMs"],
                output_summary={
                    "grounded": result.grounded,
                    "requiresRevision": result.requiresRevision,
                    "confidence": result.confidence,
                },
            )

            return state

        except Exception as exc:
            add_graph_error(
                state,
                node="grounding_critic",
                error_type=type(exc).__name__,
                message=str(exc),
                recoverable=True,
            )
            add_trace_step(
                state,
                node="grounding_critic",
                event="failed",
                latency_ms=timer["elapsedMs"],
                error=str(exc),
            )

            state["grounding"] = {
                "grounded": False,
                "requiresRevision": True,
                "unsupportedClaims": [],
                "missingCitations": [],
                "reason": "Grounding critic failed and answer requires review.",
                "confidence": 0.0,
            }
            return state


def revision_node(state: RagGraphState) -> RagGraphState:
    request_id = _request_id(state)

    answer = state.get("answer") or {}
    grounding = state.get("grounding") or {}

    should_revise = (
        bool(answer.get("answerFound"))
        and bool(grounding.get("requiresRevision"))
        and revision_budget_remaining(state)
        and llm_budget_remaining(state)
    )

    if not should_revise:
        add_trace_step(
            state,
            node="revision",
            event="skipped",
            input_summary={
                "answerFound": answer.get("answerFound"),
                "requiresRevision": grounding.get("requiresRevision"),
                "revisionBudgetRemaining": revision_budget_remaining(state),
                "llmBudgetRemaining": llm_budget_remaining(state),
            },
        )
        return state

    increment_llm_call_count(state)
    increment_revision_count(state)

    with elapsed_timer() as timer:
        try:
            from backend.models import GroundingCriticResult

            grounding_result = GroundingCriticResult(**grounding)

            result = revise_answer(
                question=_question_for_agents(state),
                answer=answer.get("answer", ""),
                grounding_result=grounding_result,
                context=state.get("context", ""),
                citations=state.get("citations", []),
                request_id=request_id,
            )

            state["revision"] = result.model_dump()

            if result.revisionApplied:
                state["final_answer"] = result.revisedAnswer
                state["final_confidence"] = result.confidence
                state["final_used_citation_paths"] = result.usedCitationPaths

            add_trace_step(
                state,
                node="revision",
                event="completed",
                latency_ms=timer["elapsedMs"],
                output_summary={
                    "revisionApplied": result.revisionApplied,
                    "confidence": result.confidence,
                },
            )

            return state

        except Exception as exc:
            add_graph_error(
                state,
                node="revision",
                error_type=type(exc).__name__,
                message=str(exc),
                recoverable=True,
            )
            add_trace_step(
                state,
                node="revision",
                event="failed",
                latency_ms=timer["elapsedMs"],
                error=str(exc),
            )
            return state


def safety_critic_node(state: RagGraphState) -> RagGraphState:
    request_id = _request_id(state)

    if not state.get("answer_found"):
        add_trace_step(
            state,
            node="safety_critic",
            event="skipped",
            input_summary={"reason": "No answer found."},
        )
        return state

    if not llm_budget_remaining(state):
        add_trace_step(
            state,
            node="safety_critic",
            event="skipped",
            input_summary={"reason": "LLM budget exhausted."},
        )
        state["safety"] = {
            "safe": False,
            "requiresRevision": True,
            "safetyIssues": [],
            "missingWarnings": [],
            "unsafeOrUnsupportedInstructions": [],
            "inventedSafetyCriticalDetails": [],
            "reason": "Safety critic skipped because LLM budget was exhausted.",
            "confidence": 0.0,
        }
        return state

    increment_llm_call_count(state)

    with elapsed_timer() as timer:
        try:
            result = evaluate_safety(
                question=_question_for_agents(state),
                answer=state.get("final_answer", ""),
                context=state.get("context", ""),
                citations=state.get("citations", []),
                request_id=request_id,
            )

            state["safety"] = result.model_dump()

            add_trace_step(
                state,
                node="safety_critic",
                event="completed",
                latency_ms=timer["elapsedMs"],
                output_summary={
                    "safe": result.safe,
                    "requiresRevision": result.requiresRevision,
                    "confidence": result.confidence,
                },
            )

            return state

        except Exception as exc:
            add_graph_error(
                state,
                node="safety_critic",
                error_type=type(exc).__name__,
                message=str(exc),
                recoverable=True,
            )
            add_trace_step(
                state,
                node="safety_critic",
                event="failed",
                latency_ms=timer["elapsedMs"],
                error=str(exc),
            )

            state["safety"] = {
                "safe": False,
                "requiresRevision": True,
                "safetyIssues": [],
                "missingWarnings": [],
                "unsafeOrUnsupportedInstructions": [],
                "inventedSafetyCriticalDetails": [],
                "reason": "Safety critic failed and answer requires review.",
                "confidence": 0.0,
            }
            return state


def final_response_node(state: RagGraphState) -> RagGraphState:
    answer = state.get("answer") or {}

    if not answer.get("answerFound"):
        state["answer_found"] = False
        state["final_answer"] = state.get("final_answer") or answer.get(
            "answer",
            "I could not find enough information in the retrieved manual context to answer this question.",
        )
        state["final_confidence"] = 0.0
        state["final_used_citation_paths"] = []
    else:
        state["answer_found"] = True
        state["final_answer"] = state.get("final_answer") or answer.get("answer", "")
        state["final_confidence"] = float(state.get("final_confidence", answer.get("confidence", 0.0)))
        state["final_used_citation_paths"] = state.get(
            "final_used_citation_paths",
            answer.get("usedCitationPaths", []),
        )

    try:
        image_result = retrieve_relevant_images_for_final_answer(
            question=state.get("current_question", "") or state.get("sanitized_question", ""),
            final_answer=state.get("final_answer", ""),
            candidate_image_references=state.get("candidate_image_references", []),
            final_used_citation_paths=state.get("final_used_citation_paths", []),
            max_images=3,
        )

        state["candidate_image_references"] = image_result.get("candidateImageReferences", [])
        state["image_references"] = image_result.get("imageReferences", [])
        state["image_reference_debug"] = image_result.get("imageReferenceDebug", {})
        state["image_reference_errors"] = image_result.get("imageReferenceErrors", [])

    except Exception as exc:
        errors = list(state.get("image_reference_errors", []))
        errors.append(
            {
                "stage": "final_response_image_retrieval_agent",
                "message": str(exc),
                "recoverable": True,
            }
        )
        state["image_references"] = []
        state["image_reference_debug"] = {
            "selectionMode": "failed",
        }
        state["image_reference_errors"] = errors







    state["final_answer"] = format_answer_text(state.get("final_answer", ""))

    add_trace_step(
        state,
        node="final_response",
        event="completed",
        output_summary={
            "answerFound": state.get("answer_found"),
            "finalConfidence": state.get("final_confidence"),
            "hasGrounding": state.get("grounding") is not None,
            "hasRevision": state.get("revision") is not None,
            "hasSafety": state.get("safety") is not None,
        },
    )

    state = _attach_debug_image_references(state)
    return state







def _attach_debug_image_references(state: RagGraphState) -> RagGraphState:
    if not state.get("enable_image_references", False):
        debug = dict(state.get("image_reference_debug", {}) or {})
        debug.update(
            {
                "imagePipelineBuild": "phase54-debug-only-image-gate",
                "imageAgentSkipped": True,
                "skipReason": "enable_image_references is false",
            }
        )
        state["image_reference_debug"] = debug
        state["candidate_image_references"] = []
        state["image_references"] = []
        return state

    """Finalize image refs from retrieval-time candidates and emit trace diagnostics."""
    try:
        image_result = retrieve_relevant_images_for_final_answer(
            question=state.get("current_question", "") or state.get("sanitized_question", ""),
            final_answer=state.get("final_answer", ""),
            candidate_image_references=state.get("candidate_image_references", []),
            final_used_citation_paths=state.get("final_used_citation_paths", []),
            max_images=3,
        )

        incoming_debug = dict(state.get("image_reference_debug", {}) or {})
        selection_debug = dict(image_result.get("imageReferenceDebug", {}) or {})
        merged_debug = {
            **incoming_debug,
            **selection_debug,
            "imagePipelineBuild": "phase51-image-agent-trace",
            "finalSelectionRan": True,
            "finalSelectionStage": "final_response_node",
        }

        state["candidate_image_references"] = image_result.get("candidateImageReferences", [])
        state["image_references"] = image_result.get("imageReferences", [])
        state["image_reference_debug"] = merged_debug

        existing_errors = list(state.get("image_reference_errors", []) or [])
        state["image_reference_errors"] = existing_errors + list(
            image_result.get("imageReferenceErrors", []) or []
        )

        state = add_trace_step(
            state,
            node="image_retrieval_agent",
            event="final_image_selection_completed",
            output_summary={
                "imagePipelineBuild": "phase51-image-agent-trace",
                "candidateImageCountBeforeFilter": merged_debug.get("candidateImageCountBeforeFilter", 0),
                "finalUsedCitationPathCount": merged_debug.get("finalUsedCitationPathCount", 0),
                "usedCitationImageCount": merged_debug.get("usedCitationImageCount", 0),
                "resolvedImageCount": merged_debug.get("resolvedImageCount", 0),
                "displayEligibleImageCount": merged_debug.get("displayEligibleImageCount", 0),
                "selectedImageCount": merged_debug.get("selectedImageCount", 0),
            },
        )

    except Exception as exc:
        errors = list(state.get("image_reference_errors", []) or [])
        errors.append(
            {
                "stage": "final_response_image_retrieval_agent",
                "message": str(exc),
                "recoverable": True,
            }
        )
        state["image_reference_errors"] = errors

        debug = dict(state.get("image_reference_debug", {}) or {})
        debug.update(
            {
                "imagePipelineBuild": "phase51-image-agent-trace",
                "finalSelectionRan": False,
                "finalSelectionStage": "final_response_node_failed",
                "finalSelectionError": str(exc),
            }
        )
        state["image_reference_debug"] = debug
        state["image_references"] = []

        state = add_trace_step(
            state,
            node="image_retrieval_agent",
            event="final_image_selection_failed",
            error=str(exc),
            output_summary={
                "imagePipelineBuild": "phase51-image-agent-trace",
            },
        )

    return state
def _updated_active_context_from_state(
    current_context: ActiveConversationContext,
    state: RagGraphState,
) -> ActiveConversationContext:
    query_understanding = state.get("query_understanding") or {}
    detected_entities = query_understanding.get("detectedEntities") or {}

    context = current_context.model_copy(deep=True)

    for field_name in ["machine", "baseMachine", "serialNumber", "manualType", "component"]:
        value = detected_entities.get(field_name)
        if value:
            setattr(context, field_name, value)

    intent = query_understanding.get("intent")
    if intent and intent != "unknown":
        context.intent = intent

    return context


def _active_context_from_query_understanding_metadata(
    current_context: ActiveConversationContext,
    query_understanding: dict[str, Any] | None,
) -> ActiveConversationContext:
    """Fallback active-context updater using queryUnderstanding metadata."""

    if not query_understanding:
        return current_context

    detected_entities = query_understanding.get("detectedEntities") or {}

    context = current_context.model_copy(deep=True)

    for field_name in ["machine", "baseMachine", "serialNumber", "manualType", "component"]:
        value = detected_entities.get(field_name)
        if value:
            setattr(context, field_name, value)

    intent = query_understanding.get("intent")
    if intent and intent != "unknown":
        context.intent = intent

    return context


def _truncate_summary_text(value: str, max_chars: int) -> str:
    cleaned = " ".join((value or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _updated_conversation_summary_from_state(
    current_summary: str,
    active_context: "ActiveConversationContext",
    state: "RagGraphState",
) -> str:
    """Create a compact deterministic summary for memory persistence.

    This avoids an extra LLM call while still making future query understanding
    aware of the active machine/component context and latest exchange.
    """
    budgets = state.get("budgets") or {}
    max_chars = int(budgets.get("conversationSummaryMaxChars", 2000))

    context_parts = []
    if active_context.machine:
        context_parts.append(f"machine={active_context.machine}")
    if active_context.baseMachine:
        context_parts.append(f"baseMachine={active_context.baseMachine}")
    if active_context.serialNumber:
        context_parts.append(f"serialNumber={active_context.serialNumber}")
    if active_context.manualType:
        context_parts.append(f"manualType={active_context.manualType}")
    if active_context.component:
        context_parts.append(f"component={active_context.component}")
    if active_context.intent:
        context_parts.append(f"intent={active_context.intent}")
    context_text = "; ".join(context_parts) if context_parts else "none"

    user_question = state.get("current_question", "")
    final_answer = state.get("final_answer", "")
    answer_found = state.get("answer_found")
    final_confidence = state.get("final_confidence")
    used_paths = state.get("final_used_citation_paths", []) or []

    answer_preview = _truncate_summary_text(final_answer, 500)
    new_summary = (
        f"Active context: {context_text}. "
        f"Latest user request: {user_question}. "
        f"Latest answerFound={answer_found}, confidence={final_confidence}. "
        f"Latest answer summary: {answer_preview} "
        f"Used citation path count: {len(used_paths)}."
    )

    if current_summary:
        combined = (
            f"Previous summary: {_truncate_summary_text(current_summary, 700)} "
            f"Updated summary: {new_summary}"
        )
    else:
        combined = new_summary

    return _truncate_summary_text(combined, max_chars)


def save_memory_node(state: RagGraphState) -> RagGraphState:
    request_id = _request_id(state)
    thread_id = state.get("thread_id") or request_id or "default-thread"

    with elapsed_timer() as timer:
        try:
            repo = get_memory_repository()

            user_message = ChatMessage(
                role="user",
                content=state.get("current_question", ""),
                requestId=request_id,
                metadata={
                    "sanitizedQuestion": state.get("sanitized_question"),
                    "queryUnderstanding": state.get("query_understanding"),
                },
            )

            assistant_message = ChatMessage(
                role="assistant",
                content=state.get("final_answer", ""),
                requestId=request_id,
                metadata={
                    "answerFound": state.get("answer_found"),
                    "finalConfidence": state.get("final_confidence"),
                    "finalUsedCitationPaths": state.get("final_used_citation_paths", []),
                    "safety": state.get("safety"),
                    "citationCount": len(state.get("citations", [])),
                },
            )

            memory = repo.append_turns(
                thread_id=thread_id,
                user_message=user_message,
                assistant_message=assistant_message,
                max_recent_turns=int((state.get("budgets") or {}).get("maxRecentTurns", 4)) * 2,
            )

            memory.activeContext = _updated_active_context_from_state(
                current_context=memory.activeContext,
                state=state,
            )
            memory.activeContext = _active_context_from_query_understanding_metadata(
                current_context=memory.activeContext,
                query_understanding=user_message.metadata.get("queryUnderstanding"),
            )
            memory.conversationSummary = _updated_conversation_summary_from_state(
                current_summary=memory.conversationSummary,
                active_context=memory.activeContext,
                state=state,
            )

            repo.save_memory(memory)

            recent_turns = [
                {
                    "role": turn.role,
                    "content": turn.content,
                    "timestamp": turn.timestamp,
                    "requestId": turn.requestId,
                    "metadata": turn.metadata,
                }
                for turn in memory.recentTurns
            ]

            state["recent_turns"] = recent_turns
            state["messages"] = recent_turns
            state["active_context"] = memory.activeContext.model_dump()
            state["conversation_summary"] = memory.conversationSummary

            add_trace_step(
                state,
                node="save_memory",
                event="completed",
                latency_ms=timer["elapsedMs"],
                input_summary={
                    "threadId": thread_id,
                    "answerFound": state.get("answer_found"),
                },
                output_summary={
                    "recentTurnCount": len(recent_turns),
                    "activeContext": memory.activeContext.model_dump(),
                    "hasConversationSummary": bool(memory.conversationSummary),
                    "conversationSummaryCharCount": len(memory.conversationSummary or ""),
                },
            )

            return state

        except Exception as exc:
            add_graph_error(
                state,
                node="save_memory",
                error_type=type(exc).__name__,
                message=str(exc),
                recoverable=True,
            )

            add_trace_step(
                state,
                node="save_memory",
                event="failed",
                latency_ms=timer["elapsedMs"],
                error=str(exc),
            )

            return state
        



GRAPH_NODE_FUNCTIONS: dict[str, Any] = {
    "load_memory": load_memory_node,
    "input_guardrail": input_guardrail_node,
    "query_understanding": query_understanding_node,
    "retrieval": retrieval_node,
    "context_builder": context_builder_node,
    "answer_generation": answer_generation_node,
    "grounding_critic": grounding_critic_node,
    "revision": revision_node,
    "safety_critic": safety_critic_node,
    "final_response": final_response_node,
    "save_memory": save_memory_node,
}

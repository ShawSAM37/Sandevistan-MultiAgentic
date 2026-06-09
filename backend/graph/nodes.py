from __future__ import annotations

from typing import Any

from backend.agents.answer_generation_agent import generate_answer_from_context
from backend.agents.grounding_critic_agent import evaluate_grounding
from backend.agents.input_guardrail_agent import run_input_guardrail_agent
from backend.agents.query_understanding_agent import understand_query
from backend.agents.revision_agent import revise_answer
from backend.agents.safety_critic_agent import evaluate_safety
from backend.context.context_builder import build_context_from_documents
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

    return state


GRAPH_NODE_FUNCTIONS: dict[str, Any] = {
    "input_guardrail": input_guardrail_node,
    "query_understanding": query_understanding_node,
    "retrieval": retrieval_node,
    "context_builder": context_builder_node,
    "answer_generation": answer_generation_node,
    "grounding_critic": grounding_critic_node,
    "revision": revision_node,
    "safety_critic": safety_critic_node,
    "final_response": final_response_node,
}

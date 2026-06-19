from __future__ import annotations

from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from backend.graph.nodes import (
    answer_generation_node,
    context_builder_node,
    final_response_node,
    grounding_critic_node,
    load_memory_node,
    input_guardrail_node,
    query_understanding_node,
    retrieval_node,
    revision_node,
    safety_critic_node,
    save_memory_node,
)
from backend.graph.state import RagGraphState, create_initial_graph_state
from backend.observability.logger import log_event


@lru_cache(maxsize=1)
def build_rag_graph():
    graph = StateGraph(RagGraphState)

    graph.add_node("load_memory", load_memory_node)
    graph.add_node("input_guardrail", input_guardrail_node)
    graph.add_node("query_understanding", query_understanding_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("context_builder", context_builder_node)
    graph.add_node("answer_generation", answer_generation_node)
    graph.add_node("grounding_critic", grounding_critic_node)
    graph.add_node("revision", revision_node)
    graph.add_node("safety_critic", safety_critic_node)
    graph.add_node("final_response", final_response_node)
    graph.add_node("save_memory", save_memory_node)

    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "input_guardrail")
    graph.add_edge("input_guardrail", "query_understanding")
    graph.add_edge("query_understanding", "retrieval")
    graph.add_edge("retrieval", "context_builder")
    graph.add_edge("context_builder", "answer_generation")
    graph.add_edge("answer_generation", "grounding_critic")
    graph.add_edge("grounding_critic", "revision")
    graph.add_edge("revision", "safety_critic")
    graph.add_edge("safety_critic", "final_response")
    graph.add_edge("final_response", "save_memory")
    graph.add_edge("save_memory", END)

    return graph.compile()

def run_rag_graph(
    *,
    request_id: str,
    question: str,
    thread_id: str | None = None,
    user_id: str | None = None,
    search_mode: str = "hybrid",
    vector_fields: list[str] | None = None,
    filters: dict[str, str] | None = None,
    top: int = 3,
    k: int = 50,
    use_semantic_ranker: bool = False,
    include_debug_context: bool = False,
    max_llm_calls: int = 6,
    enable_image_references: bool = False,
    max_revision_count: int = 1,
    max_context_chars: int = 12000,
    max_chars_per_document: int = 2500,
    answer_max_completion_tokens: int = 800,
    critic_max_completion_tokens: int = 1000,
    revision_max_completion_tokens: int = 1000,
    max_recent_turns: int = 4,
    conversation_summary_max_chars: int = 2000,
) -> RagGraphState:
    initial_state = create_initial_graph_state(
        request_id=request_id,
        question=question,
        thread_id=thread_id,
        user_id=user_id,
        search_mode=search_mode,
        vector_fields=vector_fields,
        filters=filters,
        top=top,
        k=k,
        use_semantic_ranker=use_semantic_ranker,
        include_debug_context=include_debug_context,
        max_llm_calls=max_llm_calls,
        max_revision_count=max_revision_count,
        max_context_chars=max_context_chars,
        max_chars_per_document=max_chars_per_document,
        answer_max_completion_tokens=answer_max_completion_tokens,
        critic_max_completion_tokens=critic_max_completion_tokens,
        revision_max_completion_tokens=revision_max_completion_tokens,
        max_recent_turns=max_recent_turns,
        conversation_summary_max_chars=conversation_summary_max_chars,
    )
    initial_state["enable_image_references"] = enable_image_references
    

    # Debug-only image retrieval gate.
    # Public /ask and /chat should keep this False by default.
    # /debug/graph-answer can pass True explicitly.
    initial_state["enable_image_references"] = enable_image_references

    log_event(
        event="rag_graph_started",
        request_id=request_id,
        threadId=initial_state["thread_id"],
        searchMode=search_mode,
        top=top,
        k=k,
        useSemanticRanker=use_semantic_ranker,
        maxLlmCalls=max_llm_calls,
        maxRevisionCount=max_revision_count,
        enableImageReferences=enable_image_references,
    )

    graph = build_rag_graph()

    result: RagGraphState = graph.invoke(
        initial_state,
        config={
            "configurable": {
                "thread_id": initial_state["thread_id"],
            }
        },
    )

    log_event(
        event="rag_graph_completed",
        request_id=request_id,
        threadId=result.get("thread_id"),
        answerFound=result.get("answer_found"),
        finalConfidence=result.get("final_confidence"),
        llmCallsUsed=result.get("budgets", {}).get("llmCallsUsed"),
        revisionCount=result.get("budgets", {}).get("revisionCount"),
        traceStepCount=len(result.get("trace_steps", [])),
        errorCount=len(result.get("errors", [])),
        enableImageReferences=result.get("enable_image_references"),
        imageReferenceCount=len(result.get("image_references", []) or []),
    )

    return result


def graph_state_to_debug_response(state: RagGraphState) -> dict[str, Any]:
    response = {
        "requestId": state.get("request_id"),
        "threadId": state.get("thread_id"),
        "query": state.get("current_question"),
        "sanitizedQuestion": state.get("sanitized_question"),
        "guardrail": state.get("guardrail"),
        "queryUnderstanding": state.get("query_understanding"),
        "answer": (state.get("answer") or {}).get("answer") if state.get("answer") else None,
        "answerFound": state.get("answer_found", False),
        "confidence": (state.get("answer") or {}).get("confidence", 0.0) if state.get("answer") else 0.0,
        "usedCitationPaths": (state.get("answer") or {}).get("usedCitationPaths", []) if state.get("answer") else [],
        "citations": state.get("citations", []),
        "usedDocuments": state.get("used_documents", []),
        "grounding": state.get("grounding"),
        "revision": state.get("revision"),
        "safety": state.get("safety"),
        "revisionAttempted": state.get("revision") is not None,
        "revisionCount": state.get("budgets", {}).get("revisionCount", 0),
        "finalAnswer": state.get("final_answer", ""),
        "finalUsedCitationPaths": state.get("final_used_citation_paths", []),
        "finalConfidence": state.get("final_confidence", 0.0),
        "candidateImageReferences": state.get("candidate_image_references", []),
        "imageReferences": state.get("image_references", []),
        "imageReferenceErrors": state.get("image_reference_errors", []),
        "imageReferenceDebug": state.get("image_reference_debug", {}),
        "retrieval": None,
        "contextCharCount": state.get("context_char_count", 0),
        "usedDocumentCount": len(state.get("used_documents", [])),
        "skippedDocumentCount": len(state.get("skipped_documents", [])),
        "budgets": state.get("budgets", {}),
        "traceSteps": state.get("trace_steps", []),
        "errors": state.get("errors", []),
        "memory": {
            "recentTurnCount": len(state.get("recent_turns", [])),
            "hasConversationSummary": bool(state.get("conversation_summary")),
            "activeContext": state.get("active_context", {}),
        },
    }

    retrieval = state.get("retrieval")
    if retrieval:
        response["retrieval"] = {
            "resultCount": retrieval.get("resultCount"),
            "count": retrieval.get("count"),
            "latencyMs": retrieval.get("latencyMs"),
            "searchMode": state.get("search_mode"),
            "vectorFields": state.get("vector_fields"),
            "requestFilters": state.get("filters"),
            "retrievalQuery": retrieval.get("retrievalQuery"),
            "appliedFilters": retrieval.get("appliedFilters"),
            "queryUnderstandingFilters": retrieval.get("queryUnderstandingFilters"),
            "queryUnderstandingFilterConfidence": retrieval.get("queryUnderstandingFilterConfidence"),
        }

    if state.get("include_debug_context"):
        response["context"] = state.get("context", "")

    return response

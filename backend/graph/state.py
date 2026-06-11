from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, TypedDict


GraphNodeName = Literal[
    "input_guardrail",
    "retrieval",
    "context_builder",
    "answer_generation",
    "grounding_critic",
    "revision",
    "safety_critic",
    "final_response",
]


class GraphTraceStep(TypedDict, total=False):
    node: str
    event: str
    timestamp: str
    latencyMs: int | None
    inputSummary: dict[str, Any]
    outputSummary: dict[str, Any]
    error: str | None


class GraphError(TypedDict, total=False):
    node: str
    errorType: str
    message: str
    recoverable: bool
    timestamp: str


class GraphBudgets(TypedDict, total=False):
    maxLlmCalls: int
    llmCallsUsed: int
    maxRevisionCount: int
    revisionCount: int
    maxContextChars: int
    maxCharsPerDocument: int
    answerMaxCompletionTokens: int
    criticMaxCompletionTokens: int
    revisionMaxCompletionTokens: int
    maxRecentTurns: int
    conversationSummaryMaxChars: int


class RagGraphState(TypedDict, total=False):
    # Identity / tracing
    request_id: str
    thread_id: str
    user_id: str | None

    # Conversation memory
    messages: list[dict[str, Any]]
    conversation_summary: str
    recent_turns: list[dict[str, Any]]
    active_context: dict[str, Any]

    # Current request
    current_question: str
    sanitized_question: str

    # Runtime request config
    search_mode: str
    vector_fields: list[str]
    filters: dict[str, str]
    top: int
    k: int
    use_semantic_ranker: bool
    include_debug_context: bool

    # Agent outputs
    guardrail: dict[str, Any] | None
    query_understanding: dict[str, Any] | None
    retrieval: dict[str, Any] | None
    context: str
    context_char_count: int
    citations: list[dict[str, Any]]
    used_documents: list[dict[str, Any]]
    skipped_documents: list[dict[str, Any]]

    answer: dict[str, Any] | None
    grounding: dict[str, Any] | None
    revision: dict[str, Any] | None
    safety: dict[str, Any] | None

    # Final response
    final_answer: str
    final_confidence: float
    final_used_citation_paths: list[str]
    answer_found: bool

    # Control / budgets
    budgets: GraphBudgets

    # Observability
    trace_steps: list[GraphTraceStep]
    errors: list[GraphError]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_initial_graph_state(
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
    max_revision_count: int = 1,
    max_context_chars: int = 12000,
    max_chars_per_document: int = 2500,
    answer_max_completion_tokens: int = 800,
    critic_max_completion_tokens: int = 1000,
    revision_max_completion_tokens: int = 1000,
    max_recent_turns: int = 4,
    conversation_summary_max_chars: int = 2000,
) -> RagGraphState:
    return {
        "request_id": request_id,
        "thread_id": thread_id or request_id,
        "user_id": user_id,
        "messages": [],
        "conversation_summary": "",
        "recent_turns": [],
        "active_context": {},
        "current_question": question,
        "sanitized_question": "",
        "search_mode": search_mode,
        "vector_fields": vector_fields or ["contentVector"],
        "filters": filters or {},
        "top": top,
        "k": k,
        "use_semantic_ranker": use_semantic_ranker,
        "include_debug_context": include_debug_context,
        "guardrail": None,
        "query_understanding": None,
        "retrieval": None,
        "context": "",
        "context_char_count": 0,
        "citations": [],
        "used_documents": [],
        "skipped_documents": [],
        "answer": None,
        "grounding": None,
        "revision": None,
        "safety": None,
        "final_answer": "",
        "final_confidence": 0.0,
        "final_used_citation_paths": [],
        "answer_found": False,
        "budgets": {
            "maxLlmCalls": max_llm_calls,
            "llmCallsUsed": 0,
            "maxRevisionCount": max_revision_count,
            "revisionCount": 0,
            "maxContextChars": max_context_chars,
            "maxCharsPerDocument": max_chars_per_document,
            "answerMaxCompletionTokens": answer_max_completion_tokens,
            "criticMaxCompletionTokens": critic_max_completion_tokens,
            "revisionMaxCompletionTokens": revision_max_completion_tokens,
            "maxRecentTurns": max_recent_turns,
            "conversationSummaryMaxChars": conversation_summary_max_chars,
        },
        "trace_steps": [],
        "errors": [],
    }


def add_trace_step(
    state: RagGraphState,
    *,
    node: str,
    event: str,
    latency_ms: int | None = None,
    input_summary: dict[str, Any] | None = None,
    output_summary: dict[str, Any] | None = None,
    error: str | None = None,
) -> RagGraphState:
    trace_steps = list(state.get("trace_steps", []))

    trace_steps.append(
        {
            "node": node,
            "event": event,
            "timestamp": utc_now_iso(),
            "latencyMs": latency_ms,
            "inputSummary": input_summary or {},
            "outputSummary": output_summary or {},
            "error": error,
        }
    )

    state["trace_steps"] = trace_steps
    return state


def add_graph_error(
    state: RagGraphState,
    *,
    node: str,
    error_type: str,
    message: str,
    recoverable: bool = True,
) -> RagGraphState:
    errors = list(state.get("errors", []))

    errors.append(
        {
            "node": node,
            "errorType": error_type,
            "message": message,
            "recoverable": recoverable,
            "timestamp": utc_now_iso(),
        }
    )

    state["errors"] = errors
    return state


def increment_llm_call_count(state: RagGraphState) -> RagGraphState:
    budgets = dict(state.get("budgets", {}))
    budgets["llmCallsUsed"] = int(budgets.get("llmCallsUsed", 0)) + 1
    state["budgets"] = budgets
    return state


def llm_budget_remaining(state: RagGraphState) -> bool:
    budgets = state.get("budgets", {})
    return int(budgets.get("llmCallsUsed", 0)) < int(budgets.get("maxLlmCalls", 6))


def revision_budget_remaining(state: RagGraphState) -> bool:
    budgets = state.get("budgets", {})
    return int(budgets.get("revisionCount", 0)) < int(budgets.get("maxRevisionCount", 1))


def increment_revision_count(state: RagGraphState) -> RagGraphState:
    budgets = dict(state.get("budgets", {}))
    budgets["revisionCount"] = int(budgets.get("revisionCount", 0)) + 1
    state["budgets"] = budgets
    return state

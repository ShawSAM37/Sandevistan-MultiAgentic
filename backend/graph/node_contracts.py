from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GraphNodeContract:
    name: str
    model_role: str | None
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    max_output_tokens: int | None
    fallback_behavior: str
    started_event: str
    completed_event: str
    failed_event: str


INPUT_GUARDRAIL_NODE = GraphNodeContract(
    name="input_guardrail",
    model_role="guardrail",
    reads=("current_question",),
    writes=("guardrail", "sanitized_question"),
    max_output_tokens=200,
    fallback_behavior="Block request if deterministic or model guardrail fails closed.",
    started_event="graph_input_guardrail_started",
    completed_event="graph_input_guardrail_completed",
    failed_event="graph_input_guardrail_failed",
)


QUERY_UNDERSTANDING_NODE = GraphNodeContract(
    name="query_understanding",
    model_role="planner",
    reads=("current_question", "conversation_summary", "recent_turns", "guardrail", "budgets"),
    writes=("query_understanding",),
    max_output_tokens=300,
    fallback_behavior="Use original sanitized question with no filters if query understanding fails.",
    started_event="graph_query_understanding_started",
    completed_event="graph_query_understanding_completed",
    failed_event="graph_query_understanding_failed",
)


RETRIEVAL_NODE = GraphNodeContract(
    name="retrieval",
    model_role=None,
    reads=("sanitized_question", "search_mode", "vector_fields", "filters", "top", "k"),
    writes=("retrieval",),
    max_output_tokens=None,
    fallback_behavior="Return no-answer if retrieval fails or no documents are found.",
    started_event="graph_retrieval_started",
    completed_event="graph_retrieval_completed",
    failed_event="graph_retrieval_failed",
)


CONTEXT_BUILDER_NODE = GraphNodeContract(
    name="context_builder",
    model_role=None,
    reads=("retrieval", "budgets"),
    writes=("context", "context_char_count", "citations", "used_documents", "skipped_documents"),
    max_output_tokens=None,
    fallback_behavior="Return no-answer if context cannot be built.",
    started_event="graph_context_builder_started",
    completed_event="graph_context_builder_completed",
    failed_event="graph_context_builder_failed",
)


ANSWER_GENERATION_NODE = GraphNodeContract(
    name="answer_generation",
    model_role="answer",
    reads=("sanitized_question", "context", "citations"),
    writes=("answer", "answer_found"),
    max_output_tokens=800,
    fallback_behavior="Return structured answerFound=false on model failure or rate limit.",
    started_event="graph_answer_generation_started",
    completed_event="graph_answer_generation_completed",
    failed_event="graph_answer_generation_failed",
)


GROUNDING_CRITIC_NODE = GraphNodeContract(
    name="grounding_critic",
    model_role="critic",
    reads=("sanitized_question", "answer", "context", "citations"),
    writes=("grounding",),
    max_output_tokens=1000,
    fallback_behavior="Mark requiresRevision=true if critic fails.",
    started_event="graph_grounding_critic_started",
    completed_event="graph_grounding_critic_completed",
    failed_event="graph_grounding_critic_failed",
)


REVISION_NODE = GraphNodeContract(
    name="revision",
    model_role="answer",
    reads=("sanitized_question", "answer", "grounding", "context", "citations", "budgets"),
    writes=("revision", "final_answer", "final_used_citation_paths", "final_confidence"),
    max_output_tokens=1000,
    fallback_behavior="Keep original answer if revision fails.",
    started_event="graph_revision_started",
    completed_event="graph_revision_completed",
    failed_event="graph_revision_failed",
)


SAFETY_CRITIC_NODE = GraphNodeContract(
    name="safety_critic",
    model_role="critic",
    reads=("sanitized_question", "final_answer", "context", "citations"),
    writes=("safety",),
    max_output_tokens=1000,
    fallback_behavior="Mark requiresRevision=true/manual review if safety critic fails.",
    started_event="graph_safety_critic_started",
    completed_event="graph_safety_critic_completed",
    failed_event="graph_safety_critic_failed",
)


FINAL_RESPONSE_NODE = GraphNodeContract(
    name="final_response",
    model_role=None,
    reads=("answer", "revision", "safety", "citations", "used_documents"),
    writes=("final_answer", "final_confidence", "final_used_citation_paths"),
    max_output_tokens=None,
    fallback_behavior="Return safest available answer or no-answer fallback.",
    started_event="graph_final_response_started",
    completed_event="graph_final_response_completed",
    failed_event="graph_final_response_failed",
)


GRAPH_NODE_CONTRACTS = {
    contract.name: contract
    for contract in (
        INPUT_GUARDRAIL_NODE,
        QUERY_UNDERSTANDING_NODE,
        RETRIEVAL_NODE,
        CONTEXT_BUILDER_NODE,
        ANSWER_GENERATION_NODE,
        GROUNDING_CRITIC_NODE,
        REVISION_NODE,
        SAFETY_CRITIC_NODE,
        FINAL_RESPONSE_NODE,
    )
}

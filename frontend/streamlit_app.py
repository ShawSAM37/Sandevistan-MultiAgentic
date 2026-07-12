from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any
from urllib.parse import urljoin

import requests
import streamlit as st

try:
    import urllib3
except Exception:  # pragma: no cover
    urllib3 = None


DEFAULT_BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:8000",
)

DEFAULT_IMAGE_TEST_QUERY = (
    "In the Emergency stop button section under Complementary protective measures, "
    "what are the emergency stop buttons used for, how do you reset one, and where "
    "are the 8 emergency stop buttons located? for DR416i"
)

SAFE_REFUSAL_TEXT = (
    "I cannot help with that request. Please ask a safe, manual-related question "
    "about Sandvik rotary equipment."
)


# =============================================================================
# Utilities
# =============================================================================


def normalize_backend_url(value: str) -> str:
    return (value or "").strip().rstrip("/")


def disable_insecure_warnings_if_needed(verify_ssl: bool) -> None:
    if not verify_ssl and urllib3 is not None:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def count_items(value: Any) -> int:
    return len(as_list(value))


def get_answer_text(result: dict[str, Any]) -> str:
    return (
        result.get("finalAnswer")
        or result.get("answer")
        or result.get("message")
        or ""
    )


def get_used_citation_paths(result: dict[str, Any]) -> set[str]:
    values = result.get("finalUsedCitationPaths") or result.get("usedCitationPaths") or []
    if isinstance(values, str):
        return {values}
    return {str(value) for value in values if value}


# =============================================================================
# Backend calls — final delivery UI always uses /debug/graph-answer
# =============================================================================


def warm_backend_health(
    *,
    backend_url: str,
    timeout_seconds: int,
    verify_ssl: bool,
) -> None:
    """Best-effort warm-up for Azure Container Apps before heavy debug calls."""
    try:
        requests.get(
            normalize_backend_url(backend_url) + "/health",
            timeout=min(20, max(5, timeout_seconds // 6)),
            verify=verify_ssl,
        )
    except Exception:
        return


def post_debug_graph_answer(
    *,
    backend_url: str,
    question: str,
    thread_id: str,
    timeout_seconds: int,
    verify_ssl: bool,
    max_attempts: int = 3,
) -> dict[str, Any]:
    """Call image-enabled debug graph endpoint with retry handling."""
    normalized_backend_url = normalize_backend_url(backend_url)
    warm_backend_health(
        backend_url=normalized_backend_url,
        timeout_seconds=timeout_seconds,
        verify_ssl=verify_ssl,
    )

    url = normalized_backend_url + "/debug/graph-answer"
    payload = {
        "query": question,
        "threadId": thread_id,
        "searchMode": "hybrid",
        "vectorFields": ["contentVector"],
        "useSemanticRanker": False,
        "includeDebugContext": False,
    }

    transient_statuses = {502, 503, 504}
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=timeout_seconds,
                verify=verify_ssl,
            )

            try:
                body = response.json()
            except Exception:
                body = {"rawText": response.text}

            if response.status_code in transient_statuses and attempt < max_attempts:
                time.sleep(4 * attempt)
                continue

            if response.status_code >= 400:
                raise RuntimeError(
                    f"POST {url} failed with HTTP {response.status_code}: "
                    f"{json.dumps(body, indent=2, ensure_ascii=False)}"
                )

            return body

        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
            if attempt < max_attempts:
                time.sleep(4 * attempt)
                continue
            raise RuntimeError(
                f"POST {url} failed after {max_attempts} attempts: {exc}"
            ) from exc

    if last_error:
        raise RuntimeError(f"POST {url} failed: {last_error}") from last_error
    raise RuntimeError(f"POST {url} failed for an unknown reason.")


# =============================================================================
# Session state
# =============================================================================


def ensure_state() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"streamlit-thread-{uuid.uuid4()}"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_error" not in st.session_state:
        st.session_state.last_error = None
    if "backend_url" not in st.session_state:
        st.session_state.backend_url = DEFAULT_BACKEND_URL


def new_chat() -> None:
    st.session_state.thread_id = f"streamlit-thread-{uuid.uuid4()}"
    st.session_state.messages = []
    st.session_state.last_error = None


# =============================================================================
# Rendering helpers
# =============================================================================


def stream_markdown_text(
    text: str,
    *,
    chunk_size: int = 12,
    delay_seconds: float = 0.012,
) -> None:
    """Frontend typewriter effect. Backend response is still returned as one payload."""
    if not text:
        return

    placeholder = st.empty()
    words = text.split(" ")

    for index in range(0, len(words), chunk_size):
        rendered = " ".join(words[: index + chunk_size])
        placeholder.markdown(rendered + " ▌")
        time.sleep(delay_seconds)

    placeholder.markdown(text)


def render_image_references(
    image_references: list[dict[str, Any]] | None,
    *,
    backend_url: str,
) -> None:
    if not image_references:
        return

    displayable_images = [
        image_ref
        for image_ref in image_references
        if isinstance(image_ref, dict)
        and image_ref.get("displayEligible", False)
        and image_ref.get("renderUrl")
    ]

    if not displayable_images:
        return

    st.markdown("### Related manual images")

    for index, image_ref in enumerate(displayable_images, start=1):
        render_url = str(image_ref.get("renderUrl") or "")
        full_url = urljoin(normalize_backend_url(backend_url) + "/", render_url.lstrip("/"))

        file_name = image_ref.get("fileName") or "manual image"
        title = image_ref.get("title")
        relevance = image_ref.get("relevance")
        relevance_score = image_ref.get("relevanceScore")

        caption_parts = [f"Image {index}: {file_name}"]
        if title:
            caption_parts.append(str(title))
        if relevance:
            caption_parts.append(f"relevance: {relevance}")
        if relevance_score is not None:
            caption_parts.append(f"score: {relevance_score}")

        with st.container(border=True):
            st.image(
                full_url,
                caption=" | ".join(caption_parts),
                use_container_width=True,
            )
            with st.expander("Image source details", expanded=False):
                st.json(
                    {
                        "fileName": image_ref.get("fileName"),
                        "title": image_ref.get("title"),
                        "citationPath": image_ref.get("citationPath"),
                        "blobName": image_ref.get("blobName"),
                        "renderUrl": image_ref.get("renderUrl"),
                        "relevance": image_ref.get("relevance"),
                        "relevanceScore": image_ref.get("relevanceScore"),
                        "rerankerReason": image_ref.get("rerankerReason"),
                        "resolutionReason": image_ref.get("resolutionReason"),
                    }
                )


def render_citation_card(citation: dict[str, Any], used_paths: set[str]) -> None:
    citation_id = citation.get("citationId")
    title = citation.get("title") or "Untitled"
    citation_path = citation.get("citationPath") or ""
    machine = citation.get("machine") or "Unknown machine"
    base_machine = citation.get("baseMachine") or "Unknown base machine"
    serial_number = citation.get("serialNumber") or "Unknown serial"
    manual_type = citation.get("manualType") or "Unknown manual type"
    used_label = "Used in answer" if citation_path in used_paths else "Retrieved"

    with st.expander(f"[{citation_id}] {title} — {used_label}"):
        st.markdown(f"**Machine:** {machine}")
        st.markdown(f"**Base machine:** {base_machine}")
        st.markdown(f"**Serial number:** {serial_number}")
        st.markdown(f"**Manual type:** {manual_type}")
        st.code(citation_path, language="text")


def render_assistant_response(
    message: dict[str, Any],
    *,
    backend_url: str,
) -> None:
    result = message.get("result") or {}
    answer = get_answer_text(result) or message.get("content") or ""
    answer_found = bool(result.get("answerFound", False))
    confidence = result.get("confidence") or result.get("finalConfidence")
    safety = result.get("safety")
    citations = as_list(result.get("citations"))
    used_paths = get_used_citation_paths(result)
    latency_ms = result.get("latencyMs") or result.get("endpointLatencyMs")
    request_id = result.get("requestId")

    with st.chat_message("assistant"):
        if answer_found:
            if message.get("stream", False):
                stream_markdown_text(answer)
                message["stream"] = False
            else:
                st.markdown(answer)
        else:
            st.warning(answer or SAFE_REFUSAL_TEXT)

        render_image_references(
            as_list(result.get("imageReferences")),
            backend_url=backend_url,
        )

        metadata_cols = st.columns([1, 1, 1])
        with metadata_cols[0]:
            if confidence is not None:
                try:
                    st.caption(f"Confidence: {float(confidence):.2f}")
                except Exception:
                    st.caption(f"Confidence: {confidence}")
        with metadata_cols[1]:
            if safety is None:
                st.caption("Safety: N/A")
            elif safety.get("safe") is True and safety.get("requiresRevision") is False:
                st.caption("Safety: Safe")
            else:
                st.caption("Safety: Review")
        with metadata_cols[2]:
            if latency_ms is not None:
                st.caption(f"Latency: {latency_ms} ms")

        if citations:
            st.markdown("**Citations**")
            for citation in citations:
                if isinstance(citation, dict):
                    render_citation_card(citation, used_paths)

        with st.expander("Response details", expanded=False):
            st.markdown(f"**Request ID:** `{request_id}`")
            st.markdown(f"**Thread ID:** `{result.get('threadId')}`")
            st.json(
                {
                    "answerFound": answer_found,
                    "confidence": confidence,
                    "safety": safety,
                    "usedCitationPaths": sorted(used_paths),
                    "latencyMs": latency_ms,
                    "imageReferenceDebug": result.get("imageReferenceDebug"),
                    "imageReferenceErrors": result.get("imageReferenceErrors"),
                }
            )


def render_chat_history(*, backend_url: str) -> None:
    for message in st.session_state.messages:
        role = message.get("role")
        if role == "user":
            with st.chat_message("user"):
                st.markdown(message.get("content", ""))
        elif role == "assistant":
            render_assistant_response(message, backend_url=backend_url)


# =============================================================================
# Main app
# =============================================================================


def main() -> None:
    st.set_page_config(
        page_title="Sandevistan",
        page_icon="⚙️",
        layout="wide",
    )

    ensure_state()

    with st.sidebar:
        st.title("⚙️ Sandevistan")
        st.caption("Final delivery UI: chat, memory thread, citations, and relevant manual images.")

        if st.button("New chat", type="primary"):
            new_chat()
            st.rerun()

        st.divider()
        st.subheader("Connection")
        backend_url = st.text_input(
            "Backend URL",
            value=st.session_state.backend_url,
        )
        st.session_state.backend_url = normalize_backend_url(backend_url)

        verify_ssl = st.checkbox(
            "Verify TLS certificate",
            value=os.getenv("STREAMLIT_ASK_INSECURE", "true").lower()
            not in {"1", "true", "yes"},
            help="Disable if local corporate certificates cause SSL errors.",
        )

        timeout_seconds = st.number_input(
            "Timeout seconds",
            min_value=10,
            max_value=300,
            value=300,
            step=5,
        )

        frontend_streaming = st.checkbox(
            "Frontend text streaming",
            value=True,
            help="Displays the completed backend answer with a typewriter effect.",
        )

        st.divider()
        st.subheader("Thread")
        st.code(st.session_state.thread_id, language="text")

        if st.button("Show DR416i image test query"):
            st.session_state.prefill_question = DEFAULT_IMAGE_TEST_QUERY
            st.rerun()

        st.caption(
            "This UI always calls /debug/graph-answer so imageReferences are returned. "
            "The same threadId is sent on every turn for backend memory."
        )

    disable_insecure_warnings_if_needed(verify_ssl)

    st.title("Sandevistan Manual Assistant")
    st.caption("Ask one clear question about Sandvik rotary equipment manuals.")

    render_chat_history(backend_url=st.session_state.backend_url)

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    if "prefill_question" in st.session_state:
        st.info("Copy this test query into the chat box:")
        st.code(st.session_state.pop("prefill_question"), language="text")

    question = st.chat_input("Ask about a machine, procedure, warning, or troubleshooting topic...")

    if question:
        st.session_state.last_error = None
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching manuals, checking safety, and selecting relevant images..."):
                try:
                    result = post_debug_graph_answer(
                        backend_url=st.session_state.backend_url,
                        question=question,
                        thread_id=st.session_state.thread_id,
                        timeout_seconds=int(timeout_seconds),
                        verify_ssl=verify_ssl,
                    )

                    assistant_message = {
                        "role": "assistant",
                        "content": get_answer_text(result),
                        "result": result,
                        "stream": bool(frontend_streaming and result.get("answerFound", False)),
                    }
                    st.session_state.messages.append(assistant_message)
                    st.rerun()

                except Exception as exc:
                    st.session_state.last_error = str(exc)
                    st.error(st.session_state.last_error)


if __name__ == "__main__":
    main()

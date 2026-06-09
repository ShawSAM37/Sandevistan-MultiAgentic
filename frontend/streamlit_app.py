from __future__ import annotations

import json
import os
import uuid
from typing import Any

import requests
import streamlit as st

try:
    import urllib3
except Exception:  # pragma: no cover
    urllib3 = None


DEFAULT_BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "https://ca-sandevistan-backend.graymushroom-28ea90b0.swedencentral.azurecontainerapps.io",
)

SAFE_REFUSAL_TEXT = (
    "I cannot help with that request. Please ask a safe, manual-related question "
    "about Sandvik rotary equipment."
)


def normalize_backend_url(value: str) -> str:
    return value.strip().rstrip("/")


def disable_insecure_warnings_if_needed(verify_ssl: bool) -> None:
    if not verify_ssl and urllib3 is not None:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def post_ask(
    *,
    backend_url: str,
    question: str,
    thread_id: str,
    timeout_seconds: int,
    verify_ssl: bool,
) -> dict[str, Any]:
    url = normalize_backend_url(backend_url) + "/ask"

    payload = {
        "question": question,
        "threadId": thread_id,
        "searchMode": "hybrid",
        "vectorFields": ["contentVector"],
        "useSemanticRanker": False,
    }

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

    if response.status_code >= 400:
        raise RuntimeError(
            f"POST {url} failed with HTTP {response.status_code}: "
            f"{json.dumps(body, indent=2, ensure_ascii=False)}"
        )

    return body


def ensure_state() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"streamlit-thread-{uuid.uuid4()}"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "last_error" not in st.session_state:
        st.session_state.last_error = None


def new_chat() -> None:
    st.session_state.thread_id = f"streamlit-thread-{uuid.uuid4()}"
    st.session_state.messages = []
    st.session_state.last_error = None


def render_citation_card(citation: dict[str, Any], used_paths: set[str]) -> None:
    citation_id = citation.get("citationId")
    title = citation.get("title") or "Untitled"
    path = citation.get("citationPath") or ""
    machine = citation.get("machine") or "Unknown machine"
    base_machine = citation.get("baseMachine") or "Unknown base machine"
    serial_number = citation.get("serialNumber") or "Unknown serial"
    manual_type = citation.get("manualType") or "Unknown manual type"

    used_label = "Used in answer" if path in used_paths else "Retrieved"

    with st.expander(f"[{citation_id}] {title} — {used_label}"):
        st.markdown(f"**Machine:** {machine}")
        st.markdown(f"**Base machine:** {base_machine}")
        st.markdown(f"**Serial number:** {serial_number}")
        st.markdown(f"**Manual type:** {manual_type}")
        st.code(path, language="text")


def render_assistant_response(message: dict[str, Any]) -> None:
    result = message.get("result") or {}
    answer = result.get("answer") or message.get("content") or ""
    answer_found = bool(result.get("answerFound", False))
    confidence = result.get("confidence")
    safety = result.get("safety")
    citations = result.get("citations") or []
    used_paths = set(result.get("usedCitationPaths") or [])
    latency_ms = result.get("latencyMs")
    request_id = result.get("requestId")

    with st.chat_message("assistant"):
        if answer_found:
            st.markdown(answer)
        else:
            st.warning(answer or SAFE_REFUSAL_TEXT)

        metadata_cols = st.columns([1, 1, 1])

        with metadata_cols[0]:
            if confidence is not None:
                st.caption(f"Confidence: {float(confidence):.2f}")

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
                render_citation_card(citation, used_paths)

        with st.expander("Response details", expanded=False):
            st.markdown(f"**Request ID:** `{request_id}`")
            st.markdown(f"**Thread ID:** `{result.get('threadId')}`")
            st.json(
                {
                    "answerFound": answer_found,
                    "confidence": confidence,
                    "safety": safety,
                    "usedCitationPaths": list(used_paths),
                    "latencyMs": latency_ms,
                }
            )


def render_chat_history() -> None:
    for message in st.session_state.messages:
        role = message.get("role")

        if role == "user":
            with st.chat_message("user"):
                st.markdown(message.get("content", ""))

        elif role == "assistant":
            render_assistant_response(message)


def main() -> None:
    st.set_page_config(
        page_title="Sandevistan",
        page_icon="⚙️",
        layout="wide",
    )

    ensure_state()

    with st.sidebar:
        st.title("⚙️ Sandevistan")

        st.caption("Chat interface for Sandvik rotary instruction manuals.")

        if st.button("New chat", type="primary"):
            new_chat()
            st.rerun()

        st.divider()

        with st.expander("Connection settings", expanded=False):
            backend_url = st.text_input(
                "Backend URL",
                value=DEFAULT_BACKEND_URL,
            )

            verify_ssl = st.checkbox(
                "Verify TLS certificate",
                value=os.getenv("STREAMLIT_ASK_INSECURE", "").lower()
                not in {"1", "true", "yes"},
                help="Disable only if your local Python environment has corporate certificate issues.",
            )

            timeout_seconds = st.number_input(
                "Timeout seconds",
                min_value=10,
                max_value=180,
                value=90,
                step=5,
            )

        st.divider()

        st.caption("Thread")
        st.code(st.session_state.thread_id, language="text")

        st.caption(
            "Note: full multi-turn backend memory will arrive in the upcoming /chat endpoint. "
            "This UI already preserves frontend chat history and passes a stable threadId."
        )

    disable_insecure_warnings_if_needed(verify_ssl)

    st.title("Sandevistan Manual Assistant")
    st.caption("Ask one clear question about Sandvik rotary equipment manuals.")

    render_chat_history()

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    question = st.chat_input("Ask about a machine, procedure, warning, or troubleshooting topic...")

    if question:
        st.session_state.last_error = None

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching manuals and checking safety..."):
                try:
                    result = post_ask(
                        backend_url=backend_url,
                        question=question,
                        thread_id=st.session_state.thread_id,
                        timeout_seconds=int(timeout_seconds),
                        verify_ssl=verify_ssl,
                    )

                    assistant_message = {
                        "role": "assistant",
                        "content": result.get("answer", ""),
                        "result": result,
                    }

                    st.session_state.messages.append(assistant_message)
                    st.rerun()

                except Exception as exc:
                    st.session_state.last_error = str(exc)
                    st.error(st.session_state.last_error)


if __name__ == "__main__":
    main()

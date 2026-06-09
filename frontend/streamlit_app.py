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


def normalize_backend_url(value: str) -> str:
    return value.strip().rstrip("/")


def post_ask(
    *,
    backend_url: str,
    payload: dict[str, Any],
    timeout_seconds: int,
    verify_ssl: bool,
) -> dict[str, Any]:
    url = normalize_backend_url(backend_url) + "/ask"

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


def render_citations(citations: list[dict[str, Any]], used_paths: list[str]) -> None:
    if not citations:
        st.info("No citations returned.")
        return

    used_path_set = set(used_paths or [])

    for citation in citations:
        citation_id = citation.get("citationId")
        title = citation.get("title") or "Untitled"
        path = citation.get("citationPath") or ""
        machine = citation.get("machine") or "Unknown machine"
        manual_type = citation.get("manualType") or "Unknown manual type"
        base_machine = citation.get("baseMachine") or "Unknown base machine"
        serial_number = citation.get("serialNumber") or "Unknown serial"

        used_badge = "✅ Used" if path in used_path_set else "Referenced"

        with st.expander(f"[{citation_id}] {title} — {used_badge}"):
            st.write(f"**Machine:** {machine}")
            st.write(f"**Base machine:** {base_machine}")
            st.write(f"**Serial number:** {serial_number}")
            st.write(f"**Manual type:** {manual_type}")
            st.code(path, language="text")


def render_response(result: dict[str, Any]) -> None:
    answer = result.get("answer") or ""
    answer_found = bool(result.get("answerFound", False))
    confidence = float(result.get("confidence", 0.0) or 0.0)
    safety = result.get("safety")
    latency_ms = result.get("latencyMs")
    request_id = result.get("requestId")
    thread_id = result.get("threadId")

    st.subheader("Answer")

    if answer_found:
        st.success("Answer found")
    else:
        st.warning("No grounded answer found / request blocked")

    st.markdown(answer)

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Confidence", f"{confidence:.2f}")

    with col2:
        st.metric("Latency", f"{latency_ms} ms" if latency_ms is not None else "N/A")

    with col3:
        if safety is None:
            st.metric("Safety", "N/A")
        elif safety.get("safe") is True and safety.get("requiresRevision") is False:
            st.metric("Safety", "Safe")
        else:
            st.metric("Safety", "Review")

    with st.expander("Request metadata"):
        st.write(f"**Request ID:** `{request_id}`")
        st.write(f"**Thread ID:** `{thread_id}`")
        st.json(
            {
                "answerFound": answer_found,
                "confidence": confidence,
                "safety": safety,
                "latencyMs": latency_ms,
                "usedCitationPaths": result.get("usedCitationPaths", []),
            }
        )

    st.subheader("Citations")
    render_citations(
        citations=result.get("citations", []) or [],
        used_paths=result.get("usedCitationPaths", []) or [],
    )


def init_session_state() -> None:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = f"streamlit-thread-{uuid.uuid4()}"

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    if "last_payload" not in st.session_state:
        st.session_state.last_payload = None


def main() -> None:
    st.set_page_config(
        page_title="Sandevistan RAG",
        page_icon="⚙️",
        layout="wide",
    )

    init_session_state()

    st.title("⚙️ Sandevistan Multi-Agentic RAG")
    st.caption("Production `/ask` frontend for Sandvik rotary instruction manual questions.")

    with st.sidebar:
        st.header("Backend")

        backend_url = st.text_input(
            "Backend URL",
            value=DEFAULT_BACKEND_URL,
            help="Base URL of the FastAPI backend. The app calls POST /ask.",
        )

        verify_ssl = st.checkbox(
            "Verify TLS certificate",
            value=os.getenv("STREAMLIT_ASK_INSECURE", "").lower() not in {"1", "true", "yes"},
            help="Disable only if local/corporate Python certificate trust causes SSL errors.",
        )

        if not verify_ssl and urllib3 is not None:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        timeout_seconds = st.number_input(
            "Timeout seconds",
            min_value=10,
            max_value=180,
            value=90,
            step=5,
        )

        st.divider()

        st.header("Thread")

        st.text_input(
            "Thread ID",
            key="thread_id",
            help="Used for future conversation continuity. Currently passed through to backend graph state.",
        )

        if st.button("New thread"):
            st.session_state.thread_id = f"streamlit-thread-{uuid.uuid4()}"
            st.session_state.last_result = None
            st.session_state.last_payload = None
            st.rerun()

        st.divider()

        st.header("Retrieval")

        search_mode = st.selectbox(
            "Search mode",
            options=["hybrid", "keyword", "vector"],
            index=0,
        )

        vector_field_options = ["contentVector", "titleVector"]
        vector_fields = st.multiselect(
            "Vector fields",
            options=vector_field_options,
            default=["contentVector"],
        )

        use_semantic_ranker = st.checkbox(
            "Use semantic ranker",
            value=False,
        )

        top = st.slider(
            "Top documents",
            min_value=1,
            max_value=20,
            value=3,
        )

        k = st.slider(
            "Vector K",
            min_value=1,
            max_value=100,
            value=50,
        )

        st.divider()

        st.header("Optional filters")

        machine = st.text_input("machine")
        base_machine = st.text_input("baseMachine")
        serial_number = st.text_input("serialNumber")
        manual_type = st.text_input("manualType")

    question = st.text_area(
        "Ask a manual-related question",
        value="How do I replace the hydraulic filter?",
        height=120,
    )

    col_ask, col_clear = st.columns([1, 1])

    with col_ask:
        ask_clicked = st.button("Ask Sandevistan", type="primary")

    with col_clear:
        clear_clicked = st.button("Clear result")

    if clear_clicked:
        st.session_state.last_result = None
        st.session_state.last_payload = None
        st.rerun()

    if ask_clicked:
        filters: dict[str, str] = {}

        if machine.strip():
            filters["machine"] = machine.strip()
        if base_machine.strip():
            filters["baseMachine"] = base_machine.strip()
        if serial_number.strip():
            filters["serialNumber"] = serial_number.strip()
        if manual_type.strip():
            filters["manualType"] = manual_type.strip()

        payload = {
            "question": question,
            "threadId": st.session_state.thread_id,
            "searchMode": search_mode,
            "vectorFields": vector_fields or ["contentVector"],
            "filters": filters,
            "top": top,
            "k": k,
            "useSemanticRanker": use_semantic_ranker,
        }

        st.session_state.last_payload = payload

        with st.spinner("Asking Sandevistan..."):
            try:
                result = post_ask(
                    backend_url=backend_url,
                    payload=payload,
                    timeout_seconds=int(timeout_seconds),
                    verify_ssl=verify_ssl,
                )
                st.session_state.last_result = result
            except Exception as exc:
                st.session_state.last_result = None
                st.error(str(exc))

    if st.session_state.last_payload is not None:
        with st.expander("Last request payload"):
            st.json(st.session_state.last_payload)

    if st.session_state.last_result is not None:
        render_response(st.session_state.last_result)


if __name__ == "__main__":
    main()

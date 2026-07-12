from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

try:
    import urllib3
except Exception:
    urllib3 = None


DEFAULT_BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://localhost:8000",
)

TRANSIENT_STATUSES = {502, 503, 504}


@dataclass
class TestCase:
    name: str
    query: str
    expected_keywords: list[str] = field(default_factory=list)
    expected_any_keywords: list[str] = field(default_factory=list)
    expected_image_files: list[str] = field(default_factory=list)
    min_selected_images: int = 0
    require_answer_found: bool = True
    notes: str = ""


@dataclass
class ChatTurn:
    query: str
    expected_keywords: list[str] = field(default_factory=list)
    expected_any_keywords: list[str] = field(default_factory=list)
    min_selected_images: int = 0
    expected_image_files: list[str] = field(default_factory=list)


@dataclass
class ChatScenario:
    name: str
    turns: list[ChatTurn]
    notes: str = ""


SINGLE_TURN_CASES: list[TestCase] = [
    TestCase(
        name="Feed chain tension measurement and sag specification",
        query=(
            "For DR416i, how do I check feed chain tension, what chain sag range is acceptable, "
            "and what safety preparations are required before measuring it?"
        ),
        expected_keywords=["feed chain", "180", "205", "lock", "tag"],
        expected_any_keywords=["7 to 8", "7-8", "180 to 205 mm"],
        min_selected_images=1,
        notes="Feed chain test based on the chunk with daily tension check and 180-205 mm sag range.",
    ),
    TestCase(
        name="Lower roller replacement procedure",
        query="For the undercarriage lower rollers, what are lower rollers used for and what is the replacement procedure?",
        expected_keywords=["lower roller", "track", "slacken", "locking", "bolts"],
        expected_any_keywords=["raise the undercarriage", "new roller", "replace"],
        min_selected_images=1,
        notes="Lower roller function/replacement image-heavy retrieval test.",
    ),
    TestCase(
        name="Electrical isolation station location and switches",
        query="Where is the electrical isolation station located and what are the numbered switches/components in it?",
        expected_keywords=["electrical isolation", "front right", "starter", "battery"],
        expected_any_keywords=["fuel fill", "main circuit breaker", "maintenance light"],
        expected_image_files=["GUID-9E77F3DA-4C56-4EE7-B674-51DB5643BE32-low.png"],
        min_selected_images=1,
        notes="Electrical isolation station table/image test.",
    ),
    TestCase(
        name="Servo filter replacement procedure",
        query="How do I replace servo replenishing and cooling filter elements, including safety precautions and torque?",
        expected_keywords=["servo", "filter", "lock", "tag", "hydraulic pressure"],
        expected_any_keywords=["50 to 55", "68 to 75", "check for leaks"],
        min_selected_images=1,
        notes="Servo filter procedure with torque and safety.",
    ),
    TestCase(
        name="Compressor components and receiver tank diagrams",
        query="For compressor components, what hazards are listed and what receiver tank components are shown in the diagrams?",
        expected_keywords=["compressor", "receiver tank", "stored energy", "drain"],
        expected_any_keywords=["safety relief valve", "pressure gauge", "separator element"],
        min_selected_images=1,
        notes="Compressor warning/icon vs diagram ranking test.",
    ),
    TestCase(
        name="Cab pressurization troubleshooting table",
        query="In cab pressurization and air quality troubleshooting, what causes low pressure, poor cabin sealing, error messages, and terminal connection errors?",
        expected_keywords=["low pressure", "dirty filter", "door", "windows"],
        expected_any_keywords=["terminal", "faulty components", "properly connected"],
        min_selected_images=0,
        notes="Table-only text grounding test.",
    ),
    TestCase(
        name="Access points and emergency egress",
        query="What access points and emergency egress ladders are described, and what caution is given about the emergency egress ladder?",
        expected_keywords=["access", "egress", "ladder", "falling object"],
        expected_any_keywords=["3-point", "main access ladder", "emergency exit"],
        min_selected_images=1,
        notes="Access/egress location image test.",
    ),
    TestCase(
        name="Negative grounding: lower roller torque not specified",
        query="For lower roller replacement, what exact torque value should be used for the new lower roller bolts?",
        expected_any_keywords=["does not", "not provide", "not specify", "no torque", "limitations"],
        min_selected_images=0,
        require_answer_found=False,
        notes="Should not hallucinate a torque value when the retrieved chunk does not specify one.",
    ),
]


CHAT_SCENARIOS: list[ChatScenario] = [
    ChatScenario(
        name="Memory: feed chain tension follow-up",
        turns=[
            ChatTurn(
                query="For DR416i, explain how to check feed chain tension.",
                expected_keywords=["feed chain", "tension"],
                min_selected_images=1,
            ),
            ChatTurn(
                query="What sag range should that measurement be in?",
                expected_keywords=["180", "205"],
                expected_any_keywords=["7 to 8", "7-8", "inches"],
            ),
            ChatTurn(
                query="Before measuring it, what safety steps from that same procedure should I follow?",
                expected_keywords=["stop", "lock", "tag"],
                expected_any_keywords=["mast rest", "electrical isolation", "drill rod"],
            ),
        ],
    ),
    ChatScenario(
        name="Memory: electrical isolation station follow-up",
        turns=[
            ChatTurn(
                query="Explain the electrical isolation station and where it is located.",
                expected_keywords=["electrical isolation", "front right"],
                min_selected_images=1,
            ),
            ChatTurn(
                query="In that same station, which switch disconnects power from the battery?",
                expected_keywords=["battery", "isolator"],
                expected_any_keywords=["switch (2)", "switch 2", "battery isolator switch"],
            ),
            ChatTurn(
                query="And which switch stops all hydraulic functions on the machine?",
                expected_keywords=["fuel", "fill", "hydraulic"],
                expected_any_keywords=["fuel fill isolator", "switch (9)", "switch 9"],
            ),
        ],
    ),
    ChatScenario(
        name="Memory: emergency stop image and safety follow-up",
        turns=[
            ChatTurn(
                query=(
                    "In the Emergency stop button section under Complementary protective measures, "
                    "what are the emergency stop buttons used for, how do you reset one, and where are the 8 emergency stop buttons located? for DR416i"
                ),
                expected_keywords=["emergency stop", "reset", "8"],
                expected_any_keywords=["drill guardrail", "joystick", "drill mast"],
                min_selected_images=1,
                expected_image_files=["GUID-AU178D6052-24DE-43C2-B8B6-7C6867A2E0-low.png"],
            ),
            ChatTurn(
                query="From that same section, summarize only the safety warning.",
                expected_keywords=["malfunctioning", "death", "severe injury"],
                expected_any_keywords=["test", "start of each shift"],
            ),
        ],
    ),
]


def disable_warnings_if_needed(verify_tls: bool) -> None:
    if not verify_tls and urllib3 is not None:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def normalize_backend_url(url: str) -> str:
    return (url or "").strip().rstrip("/")


def answer_text(result: dict[str, Any]) -> str:
    return result.get("finalAnswer") or result.get("answer") or ""


def post_debug_graph_answer(
    *,
    backend_url: str,
    query: str,
    thread_id: str,
    timeout: int,
    verify_tls: bool,
    max_attempts: int = 3,
) -> dict[str, Any]:
    url = normalize_backend_url(backend_url) + "/debug/graph-answer"
    payload = {
        "query": query,
        "threadId": thread_id,
        "searchMode": "hybrid",
        "vectorFields": ["contentVector"],
        "useSemanticRanker": False,
        "includeDebugContext": False,
    }

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout, verify=verify_tls)
            try:
                body = response.json()
            except Exception:
                body = {"rawText": response.text}

            if response.status_code in TRANSIENT_STATUSES and attempt < max_attempts:
                time.sleep(4 * attempt)
                continue

            if response.status_code >= 400:
                raise RuntimeError(
                    f"POST {url} failed with HTTP {response.status_code}: "
                    f"{json.dumps(body, indent=2, ensure_ascii=False)}"
                )
            return body
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            if attempt < max_attempts:
                time.sleep(4 * attempt)
                continue
            raise RuntimeError(f"POST {url} failed after {max_attempts} attempts: {exc}") from exc

    if last_exc:
        raise RuntimeError(f"POST {url} failed: {last_exc}") from last_exc
    raise RuntimeError(f"POST {url} failed unexpectedly")


def image_files(result: dict[str, Any]) -> list[str]:
    images = result.get("imageReferences") or []
    if not isinstance(images, list):
        return []
    return [str(image.get("fileName")) for image in images if isinstance(image, dict)]


def selected_image_count(result: dict[str, Any]) -> int:
    images = result.get("imageReferences") or []
    return len(images) if isinstance(images, list) else 0


def evaluate_result(
    *,
    result: dict[str, Any],
    expected_keywords: list[str],
    expected_any_keywords: list[str],
    expected_image_files: list[str],
    min_selected_images: int,
    require_answer_found: bool,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    text = answer_text(result).lower()

    if require_answer_found and result.get("answerFound") is not True:
        failures.append(f"Expected answerFound=True, got {result.get('answerFound')}")

    for keyword in expected_keywords:
        if keyword.lower() not in text:
            failures.append(f"Missing required keyword in answer: {keyword!r}")

    if expected_any_keywords and not any(keyword.lower() in text for keyword in expected_any_keywords):
        failures.append(
            "Expected at least one of these keywords in answer: "
            + ", ".join(repr(k) for k in expected_any_keywords)
        )

    if selected_image_count(result) < min_selected_images:
        failures.append(
            f"Expected at least {min_selected_images} selected image(s), got {selected_image_count(result)}"
        )

    files = set(image_files(result))
    for expected in expected_image_files:
        if expected not in files:
            failures.append(f"Expected selected image file not found: {expected}")

    image_errors = result.get("imageReferenceErrors") or []
    if image_errors:
        failures.append(f"Expected no imageReferenceErrors, got: {image_errors}")

    return not failures, failures


def print_result_summary(result: dict[str, Any]) -> None:
    debug = result.get("imageReferenceDebug") or {}
    print("answerFound              :", result.get("answerFound"))
    print("confidence               :", result.get("confidence") or result.get("finalConfidence"))
    print("citationCount            :", len(result.get("citations") or []))
    print("selectedImageCount       :", selected_image_count(result))
    print("selectedImageFiles       :", image_files(result))
    print("imageReferenceDebug      :", debug)
    print("imageReferenceErrors     :", result.get("imageReferenceErrors") or [])
    preview = answer_text(result).replace("\n", " ")[:500]
    print("answerPreview            :", preview)


def run_single_cases(args: argparse.Namespace) -> tuple[int, int, list[dict[str, Any]]]:
    passed = failed = 0
    raw_results: list[dict[str, Any]] = []
    print("\n" + "=" * 100)
    print("SINGLE-TURN CASES")
    print("=" * 100)

    for case in SINGLE_TURN_CASES:
        print("\n" + "-" * 100)
        print("CASE:", case.name)
        print("QUERY:", case.query)
        if case.notes:
            print("NOTES:", case.notes)
        thread_id = f"rag-eval-single-{uuid.uuid4()}"
        started = time.perf_counter()
        try:
            result = post_debug_graph_answer(
                backend_url=args.backend_url,
                query=case.query,
                thread_id=thread_id,
                timeout=args.timeout,
                verify_tls=not args.insecure,
            )
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            print("elapsedMs                :", elapsed_ms)
            print_result_summary(result)
            ok, failures = evaluate_result(
                result=result,
                expected_keywords=case.expected_keywords,
                expected_any_keywords=case.expected_any_keywords,
                expected_image_files=case.expected_image_files,
                min_selected_images=case.min_selected_images,
                require_answer_found=case.require_answer_found,
            )
            raw_results.append({"type": "single", "name": case.name, "query": case.query, "result": result, "ok": ok, "failures": failures})
            if ok:
                print("PASS")
                passed += 1
            else:
                print("FAIL")
                for failure in failures:
                    print(" -", failure)
                failed += 1
        except Exception as exc:
            print("FAIL: exception:", exc)
            raw_results.append({"type": "single", "name": case.name, "query": case.query, "exception": str(exc), "ok": False})
            failed += 1
    return passed, failed, raw_results


def run_chat_scenarios(args: argparse.Namespace) -> tuple[int, int, list[dict[str, Any]]]:
    passed = failed = 0
    raw_results: list[dict[str, Any]] = []
    print("\n" + "=" * 100)
    print("MULTI-TURN MEMORY SCENARIOS")
    print("=" * 100)

    for scenario in CHAT_SCENARIOS:
        print("\n" + "-" * 100)
        print("SCENARIO:", scenario.name)
        thread_id = f"rag-eval-chat-{uuid.uuid4()}"
        scenario_ok = True
        scenario_failures: list[str] = []
        turn_results: list[dict[str, Any]] = []

        for index, turn in enumerate(scenario.turns, start=1):
            print("\nTURN", index)
            print("QUERY:", turn.query)
            started = time.perf_counter()
            try:
                result = post_debug_graph_answer(
                    backend_url=args.backend_url,
                    query=turn.query,
                    thread_id=thread_id,
                    timeout=args.timeout,
                    verify_tls=not args.insecure,
                )
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                print("elapsedMs                :", elapsed_ms)
                print_result_summary(result)
                ok, failures = evaluate_result(
                    result=result,
                    expected_keywords=turn.expected_keywords,
                    expected_any_keywords=turn.expected_any_keywords,
                    expected_image_files=turn.expected_image_files,
                    min_selected_images=turn.min_selected_images,
                    require_answer_found=True,
                )
                turn_results.append({"turn": index, "query": turn.query, "result": result, "ok": ok, "failures": failures})
                if ok:
                    print("TURN PASS")
                else:
                    scenario_ok = False
                    scenario_failures.extend([f"Turn {index}: {failure}" for failure in failures])
                    print("TURN FAIL")
                    for failure in failures:
                        print(" -", failure)
            except Exception as exc:
                scenario_ok = False
                scenario_failures.append(f"Turn {index}: exception: {exc}")
                turn_results.append({"turn": index, "query": turn.query, "exception": str(exc), "ok": False})
                print("TURN FAIL: exception:", exc)

        raw_results.append({"type": "chat_scenario", "name": scenario.name, "threadId": thread_id, "turns": turn_results, "ok": scenario_ok, "failures": scenario_failures})
        if scenario_ok:
            print("SCENARIO PASS")
            passed += 1
        else:
            print("SCENARIO FAIL")
            for failure in scenario_failures:
                print(" -", failure)
            failed += 1
    return passed, failed, raw_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Final delivery RAG + image + memory regression suite.")
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--insecure", action="store_true", help="Disable TLS certificate verification.")
    parser.add_argument("--skip-single", action="store_true")
    parser.add_argument("--skip-chat", action="store_true")
    parser.add_argument("--out", default="rag_image_memory_test_results.json")
    args = parser.parse_args()

    args.backend_url = normalize_backend_url(args.backend_url)
    disable_warnings_if_needed(not args.insecure)
    print("Backend URL:", args.backend_url)
    print("Timeout    :", args.timeout)
    print("TLS verify :", not args.insecure)

    all_raw_results: list[dict[str, Any]] = []
    total_passed = total_failed = 0

    if not args.skip_single:
        passed, failed, raw = run_single_cases(args)
        total_passed += passed
        total_failed += failed
        all_raw_results.extend(raw)
    if not args.skip_chat:
        passed, failed, raw = run_chat_scenarios(args)
        total_passed += passed
        total_failed += failed
        all_raw_results.extend(raw)

    output = {"backendUrl": args.backend_url, "totalPassed": total_passed, "totalFailed": total_failed, "results": all_raw_results}
    Path(args.out).write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print("Passed:", total_passed)
    print("Failed:", total_failed)
    print("Results written to:", args.out)
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

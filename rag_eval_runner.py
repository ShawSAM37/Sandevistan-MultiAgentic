#!/usr/bin/env python3
"""
Generic RAG evaluation runner for rag_manual_eval_dataset.json.

What it tests:
- Single-turn question answering
- Multi-turn chat memory / follow-up handling
- Topic switching without stale context bleed
- Negative/missing-info questions
- Simple lexical answer checks via expected_answer_terms
- Optional citation/chunk-id checks if your API returns citations/retrieved chunks

Environment variables:
  RAG_BASE_URL       Base URL, e.g. http://localhost:8000
  RAG_ASK_PATH       Ask endpoint path, default: /ask
  RAG_API_KEY        Optional bearer token
  RAG_TIMEOUT        Request timeout seconds, default: 60

Default request body:
  {
    "question": "...",
    "sessionId": "...",
    "chatId": "...",
    "metadata": {"evalTestId": "..."}
  }

If your API uses different field names, edit build_payload().
"""

import argparse
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    print("Missing dependency: requests. Install with: pip install requests", file=sys.stderr)
    raise


def normalize(text: Any) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def term_present(answer: str, term: str) -> bool:
    return normalize(term) in normalize(answer)


def extract_answer(resp_json: Any) -> str:
    """Try common response shapes. Edit this if your API has a custom schema."""
    if isinstance(resp_json, str):
        return resp_json
    if not isinstance(resp_json, dict):
        return json.dumps(resp_json, ensure_ascii=False)

    common_keys = [
        "answer", "final_answer", "response", "content", "message", "text", "output"
    ]
    for key in common_keys:
        val = resp_json.get(key)
        if isinstance(val, str):
            return val
        if isinstance(val, dict):
            nested = extract_answer(val)
            if nested:
                return nested

    # OpenAI-like shapes
    choices = resp_json.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            return msg["content"]

    return json.dumps(resp_json, ensure_ascii=False)


def extract_citation_text(resp_json: Any) -> str:
    """Flatten likely citation/retrieval fields for optional chunk checks."""
    if not isinstance(resp_json, dict):
        return ""
    fields = []
    for key in ["citations", "sources", "sourceDocuments", "documents", "retrieved", "contexts", "references", "debug"]:
        if key in resp_json:
            fields.append(json.dumps(resp_json[key], ensure_ascii=False))
    return " ".join(fields)


def citation_mentions_expected_chunks(resp_json: Any, expected_chunks: List[int]) -> Optional[bool]:
    """
    Optional heuristic. Returns:
      True/False if citations are present,
      None if no citation-like fields are present.
    """
    citation_text = extract_citation_text(resp_json)
    if not citation_text.strip():
        return None
    lower = citation_text.lower()
    for ch in expected_chunks:
        patterns = [f"chunk {ch}", f"chunk_{ch}", f"chunk-{ch}", f"\"chunk\": {ch}", f"\"chunk_id\": {ch}", f"\"chunkId\": {ch}"]
        if any(p.lower() in lower for p in patterns):
            return True
    return False if expected_chunks else None


def build_payload(question: str, session_id: str, chat_id: str, test_id: str, turn: Optional[int] = None) -> Dict[str, Any]:
    """Edit this function to match your /ask request schema."""
    return {
        "question": question,
        "sessionId": session_id,
        "chatId": chat_id,
        "metadata": {
            "evalTestId": test_id,
            "evalTurn": turn,
        },
    }


def call_rag(base_url: str, ask_path: str, payload: Dict[str, Any], api_key: Optional[str], timeout: int) -> Tuple[int, Any, float]:
    url = base_url.rstrip("/") + "/" + ask_path.lstrip("/")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    start = time.time()
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    elapsed = time.time() - start
    try:
        body = r.json()
    except Exception:
        body = r.text
    return r.status_code, body, elapsed


@dataclass
class EvalResult:
    id: str
    type: str
    category: str
    turn: Optional[int]
    question: str
    status_code: Optional[int]
    passed: bool
    score: float
    latency_sec: Optional[float]
    missing_terms: List[str]
    forbidden_terms_found: List[str]
    citation_check: Optional[bool]
    answer_preview: str
    error: Optional[str] = None


def evaluate_answer(test: Dict[str, Any], answer: str, resp_json: Any) -> Tuple[bool, float, List[str], List[str], Optional[bool]]:
    expected_terms = test.get("expected_answer_terms", []) or []
    must_not = test.get("must_not_contain", []) or []

    missing = [t for t in expected_terms if not term_present(answer, t)]
    forbidden = [t for t in must_not if term_present(answer, t)]

    if expected_terms:
        term_score = (len(expected_terms) - len(missing)) / len(expected_terms)
    else:
        term_score = 1.0

    citation_check = citation_mentions_expected_chunks(resp_json, test.get("expected_chunks", []) or [])

    # Pass rule: >= 60% expected terms, no forbidden terms, HTTP handled elsewhere.
    passed = term_score >= 0.60 and not forbidden
    return passed, round(term_score, 3), missing, forbidden, citation_check


def run_single_test(test: Dict[str, Any], config: Dict[str, Any]) -> EvalResult:
    session_id = f"eval-single-{test['id']}-{uuid.uuid4().hex[:8]}"
    chat_id = session_id
    payload = build_payload(test["question"], session_id, chat_id, test["id"])
    try:
        status, body, latency = call_rag(config["base_url"], config["ask_path"], payload, config.get("api_key"), config["timeout"])
        answer = extract_answer(body)
        passed, score, missing, forbidden, citation_check = evaluate_answer(test, answer, body)
        if status >= 400:
            passed = False
        return EvalResult(
            id=test["id"], type=test.get("type", "single_turn"), category=test.get("category", ""), turn=None,
            question=test["question"], status_code=status, passed=passed, score=score, latency_sec=round(latency, 3),
            missing_terms=missing, forbidden_terms_found=forbidden, citation_check=citation_check,
            answer_preview=answer[:500], error=None
        )
    except Exception as e:
        return EvalResult(
            id=test["id"], type=test.get("type", "single_turn"), category=test.get("category", ""), turn=None,
            question=test["question"], status_code=None, passed=False, score=0.0, latency_sec=None,
            missing_terms=test.get("expected_answer_terms", []), forbidden_terms_found=[], citation_check=None,
            answer_preview="", error=str(e)
        )


def run_chat_session(session: Dict[str, Any], config: Dict[str, Any]) -> List[EvalResult]:
    results = []
    session_id = f"eval-chat-{session['id']}-{uuid.uuid4().hex[:8]}"
    chat_id = session_id
    for turn_obj in session.get("turns", []):
        turn = turn_obj.get("turn")
        test_id = f"{session['id']}-T{turn}"
        question = turn_obj["user"]
        pseudo_test = {
            "id": test_id,
            "type": session.get("type", "multi_turn"),
            "category": session.get("category", ""),
            "question": question,
            "expected_chunks": turn_obj.get("expected_chunks", []),
            "expected_answer_terms": turn_obj.get("expected_answer_terms", []),
            "must_not_contain": turn_obj.get("must_not_contain", []),
        }
        payload = build_payload(question, session_id, chat_id, test_id, turn)
        try:
            status, body, latency = call_rag(config["base_url"], config["ask_path"], payload, config.get("api_key"), config["timeout"])
            answer = extract_answer(body)
            passed, score, missing, forbidden, citation_check = evaluate_answer(pseudo_test, answer, body)
            if status >= 400:
                passed = False
            results.append(EvalResult(
                id=test_id, type=session.get("type", "multi_turn"), category=session.get("category", ""), turn=turn,
                question=question, status_code=status, passed=passed, score=score, latency_sec=round(latency, 3),
                missing_terms=missing, forbidden_terms_found=forbidden, citation_check=citation_check,
                answer_preview=answer[:500], error=None
            ))
        except Exception as e:
            results.append(EvalResult(
                id=test_id, type=session.get("type", "multi_turn"), category=session.get("category", ""), turn=turn,
                question=question, status_code=None, passed=False, score=0.0, latency_sec=None,
                missing_terms=pseudo_test.get("expected_answer_terms", []), forbidden_terms_found=[], citation_check=None,
                answer_preview="", error=str(e)
            ))
    return results


def flatten_tests(dataset: Dict[str, Any], mode: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    singles = []
    chats = []
    if mode in {"all", "single"}:
        singles.extend(dataset.get("single_turn_tests", []))
    if mode in {"all", "negative"}:
        singles.extend(dataset.get("negative_tests", []))
    if mode in {"all", "chat"}:
        chats.extend(dataset.get("chat_sessions", []))
    if mode == "quick":
        ids = set(dataset.get("quick_regression_ids", []))
        singles.extend([t for t in dataset.get("single_turn_tests", []) + dataset.get("negative_tests", []) if t.get("id") in ids])
        chats.extend([s for s in dataset.get("chat_sessions", []) if s.get("id") in ids])
    return singles, chats


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAG evaluation tests.")
    parser.add_argument("--dataset", default="rag_manual_eval_dataset.json")
    parser.add_argument("--mode", choices=["all", "quick", "single", "chat", "negative"], default="quick")
    parser.add_argument("--out", default="rag_eval_results.json")
    parser.add_argument("--base-url", default=os.getenv("RAG_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--ask-path", default=os.getenv("RAG_ASK_PATH", "/ask"))
    parser.add_argument("--api-key", default=os.getenv("RAG_API_KEY"))
    parser.add_argument("--timeout", type=int, default=int(os.getenv("RAG_TIMEOUT", "60")))
    args = parser.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    config = {"base_url": args.base_url, "ask_path": args.ask_path, "api_key": args.api_key, "timeout": args.timeout}
    singles, chats = flatten_tests(dataset, args.mode)

    results: List[EvalResult] = []
    print(f"Running mode={args.mode}; single/negative tests={len(singles)}, chat sessions={len(chats)}")
    print(f"Endpoint: {args.base_url.rstrip('/')}/{args.ask_path.lstrip('/')}")

    for test in singles:
        res = run_single_test(test, config)
        results.append(res)
        print(("PASS" if res.passed else "FAIL"), res.id, f"score={res.score}", f"latency={res.latency_sec}")

    for session in chats:
        print(f"Running chat session {session['id']}...")
        session_results = run_chat_session(session, config)
        results.extend(session_results)
        for res in session_results:
            print(("PASS" if res.passed else "FAIL"), res.id, f"score={res.score}", f"latency={res.latency_sec}")

    total = len(results)
    passed = sum(1 for r in results if r.passed)
    avg_score = round(sum(r.score for r in results) / total, 3) if total else 0.0
    avg_latency = round(sum(r.latency_sec or 0 for r in results) / total, 3) if total else 0.0

    report = {
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": round(passed / total, 3) if total else 0.0,
            "avg_score": avg_score,
            "avg_latency_sec": avg_latency,
            "mode": args.mode,
            "base_url": args.base_url,
            "ask_path": args.ask_path,
        },
        "results": [asdict(r) for r in results]
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nSummary")
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote results to {args.out}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())

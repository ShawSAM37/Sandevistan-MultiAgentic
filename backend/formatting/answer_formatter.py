from __future__ import annotations

import json
import re
from typing import Any


_JSON_OBJECT_START = re.compile(r"^\s*\{", flags=re.DOTALL)
_JSON_ARRAY_START = re.compile(r"^\s*\[", flags=re.DOTALL)


def _strip_outer_code_fence(value: str) -> str:
    text = value.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json|markdown|md|text)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    return text


def _looks_like_json(value: str) -> bool:
    text = _strip_outer_code_fence(value)

    return bool(_JSON_OBJECT_START.match(text) or _JSON_ARRAY_START.match(text))


def _try_parse_json(value: str) -> Any | None:
    text = _strip_outer_code_fence(value)

    if not _looks_like_json(text):
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _humanize_heading(value: str) -> str:
    heading = str(value).strip()

    heading = heading.replace("_", " ")
    heading = re.sub(r"\s+", " ", heading)

    return heading[:1].upper() + heading[1:]


def _stringify_scalar(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, bool):
        return "true" if value else "false"

    return str(value).strip()


def _list_to_markdown(items: list[Any]) -> str:
    lines: list[str] = []

    for index, item in enumerate(items, start=1):
        if isinstance(item, dict):
            nested = _dict_to_markdown(item, heading_level=3).strip()
            if nested:
                lines.append(f"{index}. {nested}")
        elif isinstance(item, list):
            nested = _list_to_markdown(item).strip()
            if nested:
                lines.append(f"{index}. {nested}")
        else:
            scalar = _stringify_scalar(item)
            if scalar:
                # Preserve already-numbered procedure steps.
                if re.match(r"^\d+\.\s+", scalar):
                    lines.append(scalar)
                else:
                    lines.append(f"- {scalar}")

    return "\n".join(lines)


def _dict_to_markdown(data: dict[str, Any], heading_level: int = 2) -> str:
    sections: list[str] = []
    heading_prefix = "#" * max(1, min(heading_level, 6))

    for key, value in data.items():
        if value is None:
            continue

        heading = _humanize_heading(key)

        if isinstance(value, dict):
            body = _dict_to_markdown(value, heading_level=heading_level + 1).strip()
        elif isinstance(value, list):
            body = _list_to_markdown(value).strip()
        else:
            body = _stringify_scalar(value)

        if not body:
            continue

        sections.append(f"{heading_prefix} {heading}\n\n{body}")

    return "\n\n".join(sections).strip()


def _normalize_markdown_text(value: str) -> str:
    text = value.strip()

    # Convert escaped newlines that sometimes appear in model output.
    text = text.replace("\\r\\n", "\n")
    text = text.replace("\\n", "\n")

    # Remove accidental wrapping quotes around full answer.
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        try:
            decoded = json.loads(text)
            if isinstance(decoded, str):
                text = decoded
        except json.JSONDecodeError:
            text = text[1:-1]

    # Normalize excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def format_answer_text(answer: Any) -> str:
    """Normalize final answer text for user-facing endpoints.

    Behavior:
    - If answer is already Markdown/plain text, normalize whitespace only.
    - If answer is a dict/list object, convert to Markdown.
    - If answer is a JSON-looking string, parse and convert to Markdown.
    - If parsing fails, return normalized text unchanged.
    """

    if answer is None:
        return ""

    if isinstance(answer, dict):
        return _dict_to_markdown(answer)

    if isinstance(answer, list):
        return _list_to_markdown(answer)

    text = _normalize_markdown_text(str(answer))

    parsed = _try_parse_json(text)

    if isinstance(parsed, dict):
        return _dict_to_markdown(parsed)

    if isinstance(parsed, list):
        return _list_to_markdown(parsed)

    return text

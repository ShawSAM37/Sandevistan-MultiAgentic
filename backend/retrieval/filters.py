from backend.constants import USER_FILTER_FIELDS


def escape_odata_string(value: str) -> str:
    return value.replace("'", "''")


def build_filter_expression(filters: dict[str, str] | None) -> str | None:
    if not filters:
        return None

    expressions: list[str] = []

    for field, value in filters.items():
        if field not in USER_FILTER_FIELDS:
            raise ValueError(f"Unsupported filter field for V1: {field}")

        if value is None:
            continue

        text = str(value).strip()

        if not text:
            continue

        escaped_value = escape_odata_string(text)
        expressions.append(f"{field} eq '{escaped_value}'")

    if not expressions:
        return None

    return " and ".join(expressions)

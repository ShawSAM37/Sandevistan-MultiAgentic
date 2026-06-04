import argparse
import ast
import csv
import json
import math
import os
import sys
import time
from typing import Any

import requests


APPROVED_FIELDS = [
    "id",
    "content",
    "contentVector",
    "title",
    "titleVector",
    "manualType",
    "baseMachine",
    "serialNumber",
    "machine",
    "citationPath",
]

TEXT_FIELDS = [
    "id",
    "content",
    "title",
    "manualType",
    "baseMachine",
    "serialNumber",
    "machine",
    "citationPath",
]

VECTOR_FIELDS = [
    "contentVector",
    "titleVector",
]

VECTOR_DIMENSIONS = 1024


def clean_text(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    if text == "":
        return None

    return text


def parse_vector(value: Any, field_name: str) -> list[float]:
    if value is None:
        raise ValueError(f"{field_name} is missing.")

    text = str(value).strip()

    if not text:
        raise ValueError(f"{field_name} is empty.")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except Exception as exc:
            raise ValueError(f"{field_name} could not be parsed as a list.") from exc

    if not isinstance(parsed, list):
        raise ValueError(f"{field_name} is not a list.")

    if len(parsed) != VECTOR_DIMENSIONS:
        raise ValueError(
            f"{field_name} has invalid dimension count. "
            f"Expected {VECTOR_DIMENSIONS}, got {len(parsed)}."
        )

    vector: list[float] = []

    for item in parsed:
        try:
            number = float(item)
        except Exception as exc:
            raise ValueError(f"{field_name} contains a non-numeric value.") from exc

        if math.isnan(number) or math.isinf(number):
            raise ValueError(f"{field_name} contains NaN or Inf.")

        vector.append(number)

    return vector


def map_csv_row_to_document(row: dict[str, Any]) -> dict[str, Any]:
    document: dict[str, Any] = {
        "@search.action": "mergeOrUpload",
    }

    for field in TEXT_FIELDS:
        document[field] = clean_text(row.get(field))

    document_id = document.get("id")
    if not document_id:
        raise ValueError("Document id is required.")

    if not document.get("content"):
        raise ValueError(f"Document {document_id} has empty content.")

    for vector_field in VECTOR_FIELDS:
        document[vector_field] = parse_vector(row.get(vector_field), vector_field)

    return document


def estimate_payload_size_bytes(documents: list[dict[str, Any]]) -> int:
    payload = {"value": documents}
    return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def upload_batch(
    endpoint: str,
    index_name: str,
    api_key: str,
    api_version: str,
    documents: list[dict[str, Any]],
) -> dict[str, Any]:
    if not endpoint:
        raise ValueError("endpoint is required for upload.")

    if not api_key:
        raise ValueError("api_key is required for upload.")

    url = f"{endpoint.rstrip('/')}/indexes/{index_name}/docs/index?api-version={api_version}"

    headers = {
        "Content-Type": "application/json",
        "api-key": api_key,
    }

    response = requests.post(
        url,
        headers=headers,
        json={"value": documents},
        timeout=120,
    )

    if response.status_code >= 400:
        raise RuntimeError(
            "Azure AI Search upload failed. "
            f"Status={response.status_code}. "
            f"Body={response.text[:2000]}"
        )

    return response.json()


def inspect_csv(csv_path: str, encoding: str) -> list[str]:
    with open(csv_path, "r", encoding=encoding, newline="") as file:
        reader = csv.DictReader(file)
        return reader.fieldnames or []


def validate_csv_columns(fieldnames: list[str] | None) -> None:
    if not fieldnames:
        raise ValueError("CSV has no header row.")

    missing = [field for field in APPROVED_FIELDS if field not in fieldnames]

    if missing:
        raise ValueError(f"CSV is missing required approved fields: {missing}")


def flush_batch(
    current_batch: list[dict[str, Any]],
    endpoint: str,
    index_name: str,
    api_key: str,
    api_version: str,
    dry_run: bool,
    total_uploaded: int,
) -> int:
    if not current_batch:
        return total_uploaded

    payload_size = estimate_payload_size_bytes(current_batch)

    if dry_run:
        print(
            "[DRY RUN] Would upload batch: "
            f"docs={len(current_batch)}, "
            f"payloadBytes={payload_size}"
        )
        return total_uploaded

    result = upload_batch(
        endpoint=endpoint,
        index_name=index_name,
        api_key=api_key,
        api_version=api_version,
        documents=current_batch,
    )

    result_items = result.get("value", [])
    failed_items = [
        item for item in result_items
        if not item.get("succeeded", False)
    ]

    failed_items = [
        item for item in result_items
        if item.get("status") is not True
    ]

    total_uploaded += len(current_batch)

    print(
        "[UPLOAD] "
        f"uploaded={total_uploaded}, "
        f"batchDocs={len(current_batch)}, "
        f"payloadBytes={payload_size}, "
        f"resultCount={len(result_items)}"
    )

    return total_uploaded


def ingest_csv(
    csv_path: str,
    endpoint: str,
    index_name: str,
    api_key: str,
    api_version: str,
    batch_size: int,
    max_payload_mb: float,
    limit: int | None,
    dry_run: bool,
    encoding: str,
) -> None:
    start_time = time.time()

    total_seen = 0
    total_valid = 0
    total_uploaded = 0
    total_failed = 0

    current_batch: list[dict[str, Any]] = []
    max_payload_bytes = int(max_payload_mb * 1024 * 1024)

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")

    if max_payload_bytes <= 0:
        raise ValueError("max_payload_mb must be greater than 0.")

    with open(csv_path, "r", encoding=encoding, newline="") as file:
        reader = csv.DictReader(file)

        validate_csv_columns(reader.fieldnames)

        print(f"CSV columns: {reader.fieldnames}")
        print(f"Using approved fields only: {APPROVED_FIELDS}")
        print(f"Dry run: {dry_run}")
        print(f"Batch size: {batch_size}")
        print(f"Max payload MB: {max_payload_mb}")
        print(f"Limit: {limit}")
        print("")

        for row in reader:
            if limit is not None and total_seen >= limit:
                break

            total_seen += 1

            try:
                document = map_csv_row_to_document(row)
                total_valid += 1
            except Exception as exc:
                total_failed += 1
                print(f"[WARN] Skipping row {total_seen}: {exc}")
                continue

            if len(current_batch) >= batch_size:
                total_uploaded = flush_batch(
                    current_batch=current_batch,
                    endpoint=endpoint,
                    index_name=index_name,
                    api_key=api_key,
                    api_version=api_version,
                    dry_run=dry_run,
                    total_uploaded=total_uploaded,
                )
                current_batch = []

            candidate_batch = current_batch + [document]
            candidate_payload_size = estimate_payload_size_bytes(candidate_batch)

            if candidate_payload_size > max_payload_bytes and current_batch:
                total_uploaded = flush_batch(
                    current_batch=current_batch,
                    endpoint=endpoint,
                    index_name=index_name,
                    api_key=api_key,
                    api_version=api_version,
                    dry_run=dry_run,
                    total_uploaded=total_uploaded,
                )
                current_batch = [document]
            else:
                current_batch.append(document)

        if current_batch:
            total_uploaded = flush_batch(
                current_batch=current_batch,
                endpoint=endpoint,
                index_name=index_name,
                api_key=api_key,
                api_version=api_version,
                dry_run=dry_run,
                total_uploaded=total_uploaded,
            )

    elapsed = time.time() - start_time

    print("")
    print("Ingestion summary")
    print("-----------------")
    print(f"totalSeen={total_seen}")
    print(f"totalValid={total_valid}")
    print(f"totalUploaded={total_uploaded}")
    print(f"totalFailed={total_failed}")
    print(f"elapsedSeconds={elapsed:.2f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream a large CSV file into Azure AI Search using approved V1 fields."
    )

    parser.add_argument(
        "--csv-path",
        required=True,
        help="Path to the source CSV file.",
    )

    parser.add_argument(
        "--endpoint",
        default=os.getenv("AZURE_SEARCH_ENDPOINT"),
        help="Azure AI Search endpoint. Not required for --dry-run or --inspect-only.",
    )

    parser.add_argument(
        "--index-name",
        default=os.getenv("AZURE_SEARCH_INDEX_NAME", "rotary-instruction-manuals"),
        help="Azure AI Search index name.",
    )

    parser.add_argument(
        "--api-key",
        default=os.getenv("AZURE_SEARCH_ADMIN_KEY"),
        help="Azure AI Search admin/query key. Not required for --dry-run or --inspect-only.",
    )

    parser.add_argument(
        "--api-version",
        default=os.getenv("AZURE_SEARCH_API_VERSION", "2024-07-01"),
        help="Azure AI Search API version.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Maximum number of documents per upload batch.",
    )

    parser.add_argument(
        "--max-payload-mb",
        type=float,
        default=8.0,
        help="Approximate maximum JSON payload size per upload batch in MB.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of CSV rows to read. Useful for tests.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate rows without uploading to Azure AI Search.",
    )

    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Only print CSV columns and exit.",
    )

    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="CSV file encoding.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.inspect_only:
        columns = inspect_csv(args.csv_path, args.encoding)
        print("CSV columns:")
        for column in columns:
            print(f"- {column}")
        return

    if not args.dry_run:
        if not args.endpoint:
            raise ValueError("Search endpoint is required. Use --endpoint or AZURE_SEARCH_ENDPOINT.")

        if not args.api_key:
            raise ValueError("Search API key is required. Use --api-key or AZURE_SEARCH_ADMIN_KEY.")

    endpoint = args.endpoint.rstrip("/") if args.endpoint else ""
    api_key = args.api_key or ""

    ingest_csv(
        csv_path=args.csv_path,
        endpoint=endpoint,
        index_name=args.index_name,
        api_key=api_key,
        api_version=args.api_version,
        batch_size=args.batch_size,
        max_payload_mb=args.max_payload_mb,
        limit=args.limit,
        dry_run=args.dry_run,
        encoding=args.encoding,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(130)

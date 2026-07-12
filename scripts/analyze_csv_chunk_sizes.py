import argparse
import csv
import math
from statistics import mean, median


def percentile(values: list[int], p: float) -> float:
    if not values:
        return 0.0

    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * p
    lower = math.floor(index)
    upper = math.ceil(index)

    if lower == upper:
        return float(sorted_values[int(index)])

    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    weight = index - lower

    return lower_value * (1 - weight) + upper_value * weight


def estimate_tokens_from_chars(char_count: int, chars_per_token: float = 4.0) -> int:
    return math.ceil(char_count / chars_per_token)


def estimate_context_block_chars(
    content_chars: int,
    title_chars: int = 0,
    citation_path_chars: int = 0,
    metadata_overhead_chars: int = 350,
) -> int:
    return content_chars + title_chars + citation_path_chars + metadata_overhead_chars


def analyze_csv(
    csv_path: str,
    encoding: str,
    limit: int | None,
    chars_per_token: float,
) -> None:
    content_lengths: list[int] = []
    title_lengths: list[int] = []
    citation_path_lengths: list[int] = []
    block_lengths: list[int] = []

    total_rows = 0
    rows_with_content = 0
    empty_content_rows = 0

    with open(csv_path, "r", encoding=encoding, newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("CSV has no header row.")

        required = ["id", "content", "title", "citationPath"]
        missing = [field for field in required if field not in reader.fieldnames]

        if missing:
            raise ValueError(f"CSV is missing required fields: {missing}")

        for row in reader:
            if limit is not None and total_rows >= limit:
                break

            total_rows += 1

            content = (row.get("content") or "").strip()
            title = (row.get("title") or "").strip()
            citation_path = (row.get("citationPath") or "").strip()

            content_len = len(content)
            title_len = len(title)
            citation_path_len = len(citation_path)

            if content_len == 0:
                empty_content_rows += 1
                continue

            rows_with_content += 1

            content_lengths.append(content_len)
            title_lengths.append(title_len)
            citation_path_lengths.append(citation_path_len)

            block_lengths.append(
                estimate_context_block_chars(
                    content_chars=content_len,
                    title_chars=title_len,
                    citation_path_chars=citation_path_len,
                )
            )

    if not content_lengths:
        raise ValueError("No non-empty content rows found.")

    print("")
    print("CSV Chunk Size Analysis")
    print("=======================")
    print(f"csvPath={csv_path}")
    print(f"totalRowsScanned={total_rows}")
    print(f"rowsWithContent={rows_with_content}")
    print(f"emptyContentRows={empty_content_rows}")
    print(f"charsPerTokenEstimate={chars_per_token}")
    print("")

    print("Content character lengths")
    print("-------------------------")
    print(f"avg={mean(content_lengths):.2f}")
    print(f"median={median(content_lengths):.2f}")
    print(f"p75={percentile(content_lengths, 0.75):.2f}")
    print(f"p90={percentile(content_lengths, 0.90):.2f}")
    print(f"p95={percentile(content_lengths, 0.95):.2f}")
    print(f"p99={percentile(content_lengths, 0.99):.2f}")
    print(f"max={max(content_lengths)}")
    print("")

    print("Estimated content tokens")
    print("------------------------")
    print(f"avgTokens={estimate_tokens_from_chars(int(mean(content_lengths)), chars_per_token)}")
    print(f"medianTokens={estimate_tokens_from_chars(int(median(content_lengths)), chars_per_token)}")
    print(f"p90Tokens={estimate_tokens_from_chars(int(percentile(content_lengths, 0.90)), chars_per_token)}")
    print(f"p95Tokens={estimate_tokens_from_chars(int(percentile(content_lengths, 0.95)), chars_per_token)}")
    print(f"p99Tokens={estimate_tokens_from_chars(int(percentile(content_lengths, 0.99)), chars_per_token)}")
    print(f"maxTokens={estimate_tokens_from_chars(max(content_lengths), chars_per_token)}")
    print("")

    print("Estimated context block character lengths")
    print("-----------------------------------------")
    print("Includes rough metadata overhead per document.")
    print(f"avgBlockChars={mean(block_lengths):.2f}")
    print(f"medianBlockChars={median(block_lengths):.2f}")
    print(f"p90BlockChars={percentile(block_lengths, 0.90):.2f}")
    print(f"p95BlockChars={percentile(block_lengths, 0.95):.2f}")
    print(f"p99BlockChars={percentile(block_lengths, 0.99):.2f}")
    print(f"maxBlockChars={max(block_lengths)}")
    print("")

    avg_block_chars = mean(block_lengths)
    p90_block_chars = percentile(block_lengths, 0.90)
    p95_block_chars = percentile(block_lengths, 0.95)

    print("Estimated context sizes by topK")
    print("-------------------------------")

    for top_k in [3, 5, 10]:
        avg_context_chars = int(avg_block_chars * top_k)
        p90_context_chars = int(p90_block_chars * top_k)
        p95_context_chars = int(p95_block_chars * top_k)

        print(f"topK={top_k}")
        print(f"  avgContextChars={avg_context_chars}")
        print(f"  avgContextTokens≈{estimate_tokens_from_chars(avg_context_chars, chars_per_token)}")
        print(f"  p90ContextChars={p90_context_chars}")
        print(f"  p90ContextTokens≈{estimate_tokens_from_chars(p90_context_chars, chars_per_token)}")
        print(f"  p95ContextChars={p95_context_chars}")
        print(f"  p95ContextTokens≈{estimate_tokens_from_chars(p95_context_chars, chars_per_token)}")

    print("")
    print("Suggested starting configuration")
    print("--------------------------------")

    suggested_max_chars_per_doc = int(min(max(percentile(content_lengths, 0.90), 1500), 5000))
    suggested_context_chars_top_5 = int(min(max(p90_block_chars * 5, 8000), 20000))

    print(f"MAX_CHARS_PER_DOCUMENT≈{suggested_max_chars_per_doc}")
    print(f"MAX_CONTEXT_CHARS for top=5≈{suggested_context_chars_top_5}")
    print("ANSWER_MAX_COMPLETION_TOKENS≈400-600")
    print("GUARDRAIL_MAX_COMPLETION_TOKENS≈150-250")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze local CSV chunk/content sizes for RAG token budgeting."
    )

    parser.add_argument("--csv-path", required=True)
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--chars-per-token",
        type=float,
        default=4.0,
        help="Rough token estimate. Default 4 chars/token.",
    )

    args = parser.parse_args()

    analyze_csv(
        csv_path=args.csv_path,
        encoding=args.encoding,
        limit=args.limit,
        chars_per_token=args.chars_per_token,
    )


if __name__ == "__main__":
    main()

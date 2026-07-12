from __future__ import annotations

import argparse
import json
import os
import re
import urllib3
from typing import Any

import requests


PNG_PATTERN = re.compile(
    r"GUID-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*-low\.png",
    flags=re.IGNORECASE,
)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def scan_text(value: Any) -> list[str]:
    if value is None:
        return []
    return sorted(set(PNG_PATTERN.findall(str(value))))


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe /debug/graph-answer retrieved chunks for PNG image filenames."
    )
    parser.add_argument(
        "--backend-url",
        default=os.getenv("BACKEND_URL"),
        required=False,
        help="Backend URL. Defaults to BACKEND_URL env var.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Query to send to /debug/graph-answer.",
    )
    parser.add_argument(
        "--out",
        default="debug-image-chunk-probe-result.json",
        help="Output JSON result path.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=1200,
        help="Number of content characters to preview per used document.",
    )

    args = parser.parse_args()

    if not args.backend_url:
        raise SystemExit("Missing --backend-url or BACKEND_URL env var.")

    # Suppress warnings caused by verify=False for internal/debug endpoints.
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    backend_url = args.backend_url.rstrip("/")
    url = f"{backend_url}/debug/graph-answer"

    response = requests.post(
        url,
        json={"query": args.query},
        timeout=180,
        verify=False,
    )
    response.raise_for_status()

    result = response.json()

    used_documents = as_list(result.get("usedDocuments"))
    final_used_paths = as_list(result.get("finalUsedCitationPaths"))
    candidate_refs = as_list(result.get("candidateImageReferences"))
    final_refs = as_list(result.get("imageReferences"))
    image_errors = as_list(result.get("imageReferenceErrors"))

    print("=" * 100)
    print("QUERY")
    print("=" * 100)
    print(args.query)
    print()

    print("=" * 100)
    print("GRAPH SUMMARY")
    print("=" * 100)
    print("answerFound              :", result.get("answerFound"))
    print("usedDocumentCount        :", len(used_documents))
    print("finalUsedCitationCount   :", len(final_used_paths))
    print("candidateImageRefCount   :", len(candidate_refs))
    print("finalImageRefCount       :", len(final_refs))
    print("imageErrorCount          :", len(image_errors))
    print()

    print("=" * 100)
    print("FINAL USED CITATION PATHS")
    print("=" * 100)
    print(json.dumps(final_used_paths, indent=2, ensure_ascii=False))
    print()

    total_png_matches = 0
    probe_docs: list[dict[str, Any]] = []

    for index, raw_doc in enumerate(used_documents, start=1):
        doc = raw_doc if isinstance(raw_doc, dict) else {}

        title = safe_text(doc.get("title"))
        citation_path = safe_text(doc.get("citationPath"))
        content = safe_text(doc.get("content"))

        title_matches = scan_text(title)
        path_matches = scan_text(citation_path)
        content_matches = scan_text(content)

        all_matches = sorted(set(title_matches + path_matches + content_matches))
        total_png_matches += len(all_matches)

        probe_doc = {
            "index": index,
            "title": title,
            "citationPath": citation_path,
            "contentLength": len(content),
            "containsGUID": "GUID-" in content,
            "containsPng": ".png" in content.lower(),
            "titleMatches": title_matches,
            "citationPathMatches": path_matches,
            "contentMatches": content_matches,
            "allMatches": all_matches,
            "contentPreview": content[: args.preview_chars],
        }
        probe_docs.append(probe_doc)

        print("=" * 100)
        print(f"USED DOCUMENT {index}")
        print("=" * 100)
        print("title        :", title)
        print("citationPath :", citation_path)
        print("contentLength:", len(content))
        print("contains GUID-:", "GUID-" in content)
        print("contains .png :", ".png" in content.lower())
        print("png matches   :", all_matches)
        print()
        print("CONTENT PREVIEW")
        print("-" * 100)
        print(content[: args.preview_chars])
        print()

    output = {
        "query": args.query,
        "answerFound": result.get("answerFound"),
        "finalUsedCitationPaths": final_used_paths,
        "candidateImageReferences": candidate_refs,
        "imageReferences": final_refs,
        "imageReferenceErrors": image_errors,
        "usedDocumentProbe": probe_docs,
        "totalPngMatchesInUsedDocuments": total_png_matches,
    }

    with open(args.out, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print("=" * 100)
    print("PROBE RESULT")
    print("=" * 100)
    print("totalPngMatchesInUsedDocuments:", total_png_matches)
    print("wrote:", args.out)

    if total_png_matches == 0:
        print()
        print("CONCLUSION: Real used chunks did not expose .png filenames.")
        print("The simple pipeline is correct, but ingestion/index content must include image filenames.")
    else:
        print()
        print("CONCLUSION: Real used chunks expose .png filenames.")
        print("The simple extractor + manifest + reranker pipeline can work directly.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

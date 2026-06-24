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


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def find_png_names(text: Any) -> list[str]:
    return sorted(set(PNG_PATTERN.findall(safe_str(text))))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe default Sandevistan /debug/graph-answer pipeline chunks for image names."
    )
    parser.add_argument(
        "--backend-url",
        default=os.getenv("BACKEND_URL"),
        help="Backend URL. Defaults to BACKEND_URL env var.",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Question to send through the default graph pipeline.",
    )
    parser.add_argument(
        "--out",
        default="default-pipeline-chunk-probe.json",
        help="Output JSON file.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=2000,
        help="Content preview chars per used document.",
    )

    args = parser.parse_args()

    if not args.backend_url:
        raise SystemExit("Missing --backend-url or BACKEND_URL env var.")

    # Suppress SSL warning because verify=False is used for the debug/internal call.
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
    citations = as_list(result.get("citations"))
    final_used_paths = as_list(result.get("finalUsedCitationPaths"))
    query_understanding = result.get("queryUnderstanding") or {}
    candidate_images = as_list(result.get("candidateImageReferences"))
    final_images = as_list(result.get("imageReferences"))
    image_errors = as_list(result.get("imageReferenceErrors"))

    print("=" * 100)
    print("DEFAULT GRAPH PIPELINE PROBE")
    print("=" * 100)
    print("query:", args.query)
    print("answerFound:", result.get("answerFound"))
    print("usedDocumentCount:", len(used_documents))
    print("citationCount:", len(citations))
    print("finalUsedCitationCount:", len(final_used_paths))
    print("candidateImageRefCount:", len(candidate_images))
    print("finalImageRefCount:", len(final_images))
    print("imageErrorCount:", len(image_errors))
    print()

    print("=" * 100)
    print("QUERY UNDERSTANDING")
    print("=" * 100)
    print(json.dumps(query_understanding, indent=2, ensure_ascii=False))
    print()

    print("=" * 100)
    print("FINAL USED CITATION PATHS")
    print("=" * 100)
    print(json.dumps(final_used_paths, indent=2, ensure_ascii=False))
    print()

    probe_docs: list[dict[str, Any]] = []
    total_png_matches = 0

    for idx, raw_doc in enumerate(used_documents, start=1):
        doc = raw_doc if isinstance(raw_doc, dict) else {}

        title = safe_str(doc.get("title"))
        citation_path = safe_str(doc.get("citationPath"))
        content = safe_str(doc.get("content"))

        doc_keys = sorted(list(doc.keys()))

        title_matches = find_png_names(title)
        path_matches = find_png_names(citation_path)
        content_matches = find_png_names(content)

        all_matches = sorted(set(title_matches + path_matches + content_matches))
        total_png_matches += len(all_matches)

        print("=" * 100)
        print(f"USED DOCUMENT {idx}")
        print("=" * 100)
        print("keys:", doc_keys)
        print("title:", title)
        print("citationPath:", citation_path)
        print("contentLength:", len(content))
        print("contains GUID-:", "GUID-" in content)
        print("contains .png:", ".png" in content.lower())
        print("pngMatches:", all_matches)
        print()
        print("CONTENT PREVIEW")
        print("-" * 100)
        print(content[: args.preview_chars])
        print()

        probe_docs.append(
            {
                "index": idx,
                "keys": doc_keys,
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
        )

    output = {
        "query": args.query,
        "answerFound": result.get("answerFound"),
        "queryUnderstanding": query_understanding,
        "finalUsedCitationPaths": final_used_paths,
        "candidateImageReferences": candidate_images,
        "imageReferences": final_images,
        "imageReferenceErrors": image_errors,
        "usedDocumentProbe": probe_docs,
        "totalPngMatchesInUsedDocuments": total_png_matches,
    }

    with open(args.out, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)

    print("=" * 100)
    print("FINAL PROBE RESULT")
    print("=" * 100)
    print("totalPngMatchesInUsedDocuments:", total_png_matches)
    print("wrote:", args.out)

    if total_png_matches > 0:
        print("CONCLUSION: Default graph chunks expose PNG image names. Simple pipeline can work directly.")
    else:
        print("CONCLUSION: Default graph chunks do not expose PNG image names for this query.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
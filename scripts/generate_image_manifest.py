from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


IMAGE_PATTERN = re.compile(
    r"^(?P<stem>GUID-[A-Za-z0-9-]+)-(?P<resolution>low|high|medium|thumb)\.(?P<extension>png|jpg|jpeg|svg|gif)$",
    flags=re.IGNORECASE,
)


def infer_manual_type(manual_title: str | None) -> str | None:
    if not manual_title:
        return None

    value = manual_title.lower()

    if "maintenance manual" in value:
        return "Maintenance Manual"
    if "operator" in value:
        return "Operator's Manual"
    if "service manual" in value:
        return "Service Manual"
    if "parts manual" in value:
        return "Parts Manual"

    return None


def parse_manual_folder(folder_name: str) -> dict[str, Any]:
    parts = folder_name.split("=")

    data: dict[str, Any] = {
        "manualFolder": folder_name,
        "manualSourceGuid": None,
        "manualTitle": None,
        "manualRevision": None,
        "exportType": None,
        "language": None,
        "manualType": None,
    }

    if len(parts) >= 1:
        data["manualSourceGuid"] = parts[0] or None
    if len(parts) >= 2:
        data["manualTitle"] = parts[1] or None
    if len(parts) >= 3:
        data["manualRevision"] = parts[2] or None
    if len(parts) >= 4:
        data["exportType"] = parts[3] or None
    if len(parts) >= 5:
        data["language"] = parts[4] or None

    data["manualType"] = infer_manual_type(data.get("manualTitle"))

    return data


def normalize_blob_path(path: Path) -> str:
    return "/".join(path.parts)


def build_manifest_row(
    *,
    root: Path,
    file_path: Path,
    blob_prefix: str,
    container_name: str,
) -> dict[str, Any]:
    relative_path = file_path.relative_to(root)
    parts = relative_path.parts

    product = parts[0] if len(parts) > 0 else None
    version = parts[1] if len(parts) > 1 else None
    manual_category = parts[2] if len(parts) > 2 else None
    manual_folder = parts[3] if len(parts) > 3 else None

    file_name = file_path.name

    match = IMAGE_PATTERN.match(file_name)
    if match:
        image_stem = match.group("stem")
        resolution = match.group("resolution").lower()
        extension = match.group("extension").lower()
    else:
        image_stem = file_path.stem
        resolution = None
        extension = file_path.suffix.lstrip(".").lower() or None

    manual_data = parse_manual_folder(manual_folder or "")

    blob_relative_path = normalize_blob_path(relative_path)
    blob_name = f"{blob_prefix.strip('/')}/{blob_relative_path}" if blob_prefix else blob_relative_path

    return {
        "imageId": file_name,
        "fileName": file_name,
        "imageStem": image_stem,
        "resolution": resolution,
        "extension": extension,
        "blobContainer": container_name,
        "blobName": blob_name,
        "localRelativePath": blob_relative_path,
        "product": product,
        "version": version,
        "manualCategory": manual_category,
        **manual_data,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            file.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Sandevistan image manifest JSONL from local manual image folders."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Root image folder, e.g. C:\\downloaded_blobs\\latest-manuals_png_only",
    )
    parser.add_argument(
        "--out",
        default="image-manifest.jsonl",
        help="Output JSONL manifest path.",
    )
    parser.add_argument(
        "--summary-out",
        default="image-manifest-summary.json",
        help="Output summary JSON path.",
    )
    parser.add_argument(
        "--duplicates-out",
        default="image-manifest-duplicates.json",
        help="Output duplicate filename report path.",
    )
    parser.add_argument(
        "--container",
        default="manual-images",
        help="Azure Blob container name to write into manifest.",
    )
    parser.add_argument(
        "--blob-prefix",
        default="latest-manuals_png_only",
        help="Blob prefix used when uploaded to Azure.",
    )

    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root}")

    image_files = sorted(
        [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".gif"}
        ]
    )

    rows = [
        build_manifest_row(
            root=root,
            file_path=file_path,
            blob_prefix=args.blob_prefix,
            container_name=args.container,
        )
        for file_path in image_files
    ]

    out_path = Path(args.out)
    summary_path = Path(args.summary_out)
    duplicates_path = Path(args.duplicates_out)

    write_jsonl(out_path, rows)

    by_file_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_product: dict[str, int] = defaultdict(int)
    by_manual_type: dict[str, int] = defaultdict(int)
    by_language: dict[str, int] = defaultdict(int)

    for row in rows:
        by_file_name[row["fileName"].lower()].append(row)
        by_product[row.get("product") or "UNKNOWN"] += 1
        by_manual_type[row.get("manualType") or "UNKNOWN"] += 1
        by_language[row.get("language") or "UNKNOWN"] += 1

    duplicates = {
        file_name: entries
        for file_name, entries in by_file_name.items()
        if len(entries) > 1
    }

    summary = {
        "root": str(root),
        "manifestPath": str(out_path.resolve()),
        "imageCount": len(rows),
        "uniqueFileNameCount": len(by_file_name),
        "duplicateFileNameCount": len(duplicates),
        "products": dict(sorted(by_product.items())),
        "manualTypes": dict(sorted(by_manual_type.items())),
        "languages": dict(sorted(by_language.items())),
        "container": args.container,
        "blobPrefix": args.blob_prefix,
    }

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    duplicates_path.write_text(
        json.dumps(duplicates, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

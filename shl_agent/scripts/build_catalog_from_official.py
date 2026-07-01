"""
Run this once you have the official catalog JSON (the link embedded in the
assignment PDF, page 4) to regenerate data/catalog.json from it instead of
the reconstructed fallback dataset.

Usage:
    python scripts/build_catalog_from_official.py path/to/shl_product_catalog.json

This script INTROSPECTS the file first (prints top-level shape + a sample
record) so field names can be confirmed/adjusted quickly - official schemas
are rarely identical to what you'd guess. Adjust FIELD_MAP below after
looking at the printed sample, then re-run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.json"

# Best-guess field mapping - update these keys after inspecting the real file.
FIELD_MAP = {
    "name": ["name", "title", "assessment_name"],
    "url": ["url", "link", "catalog_url"],
    "description": ["description", "desc", "summary"],
    "remote_testing": ["remote_testing", "remote"],
    "adaptive_irt": ["adaptive_irt", "adaptive", "irt"],
    "test_type_codes": ["test_type_codes", "test_type", "type_codes"],
    "duration_minutes": ["duration_minutes", "duration", "length"],
    "job_levels": ["job_levels", "job_level", "levels"],
    "languages": ["languages", "language"],
    "category": ["category", "type", "solution_type"],  # to filter individual vs job solutions
}


def first_present(d: dict, keys: list[str]):
    for k in keys:
        if k in d:
            return d[k]
    return None


def main(path: str):
    with open(path) as f:
        raw = json.load(f)

    records = raw if isinstance(raw, list) else raw.get("items") or raw.get("data") or raw.get("results")
    if records is None:
        print("Could not find a list of records - top-level keys were:", list(raw.keys()))
        print("Inspect the file manually and adjust this script's `records` extraction.")
        return

    print(f"Found {len(records)} records. Sample:")
    print(json.dumps(records[0], indent=2)[:2000])
    print("\nIf FIELD_MAP above doesn't match these keys, edit it and re-run.\n")

    out = []
    for i, r in enumerate(records):
        name = first_present(r, FIELD_MAP["name"])
        url = first_present(r, FIELD_MAP["url"])
        if not name or not url:
            continue
        out.append({
            "id": f"shl_{i+1:04d}",
            "name": name,
            "url": url,
            "description": first_present(r, FIELD_MAP["description"]) or "",
            "remote_testing": bool(first_present(r, FIELD_MAP["remote_testing"])),
            "adaptive_irt": bool(first_present(r, FIELD_MAP["adaptive_irt"])),
            "test_type_codes": list(first_present(r, FIELD_MAP["test_type_codes"]) or []),
            "test_type_labels": [],
            "test_type_text": None,
            "duration_minutes": first_present(r, FIELD_MAP["duration_minutes"]),
            "job_levels": first_present(r, FIELD_MAP["job_levels"]) or [],
            "languages": first_present(r, FIELD_MAP["languages"]) or [],
        })

    print(f"Converted {len(out)} / {len(records)} records.")
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/build_catalog_from_official.py <path_to_official_json>")
        sys.exit(1)
    main(sys.argv[1])

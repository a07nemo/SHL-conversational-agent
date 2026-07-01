"""
Loads the SHL Individual Test Solutions catalog and prepares it for retrieval.

Each catalog item looks like:
{
  "id": "shl_0001",
  "name": "Adobe Experience Manager (New)",
  "url": "https://www.shl.com/solutions/products/product-catalog/view/adobe-experience-manager-new/",
  "description": "...",
  "remote_testing": true,
  "adaptive_irt": false,
  "test_type_codes": ["K"],
  "test_type_labels": ["Knowledge & Skills"],
  "test_type_text": "Knowledge & Skills",
  "duration_minutes": 17,
  "job_levels": ["Mid-Professional", "Professional Individual Contributor"],
  "languages": ["English (USA)"]
}
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "catalog.json")

# Small alias table for common SHL shorthand/acronyms that won't share tokens
# with their catalog description text (pure keyword/BM25 retrieval would miss
# these otherwise). Keys are lowercase tokens a user might type.
ALIASES: dict[str, str] = {
    "opq": "occupational personality questionnaire",
    "opq32r": "occupational personality questionnaire",
    "gsa": "global skills assessment",
    "mq": "motivation questionnaire",
    "sjt": "situational judgement biodata",
    "ucf": "universal competency framework",
    "jfa": "job focused assessment",
    "verify": "verify ability aptitude cognitive",
}


@dataclass
class CatalogItem:
    id: str
    name: str
    url: str
    description: str
    remote_testing: bool
    adaptive_irt: bool
    test_type_codes: list[str]
    test_type_labels: list[str]
    test_type_text: Optional[str]
    duration_minutes: Optional[int]
    job_levels: list[str]
    languages: list[str]
    search_text: str = field(default="", repr=False)

    def to_recommendation(self) -> dict:
        """Shape required by the /chat response schema."""
        return {
            "name": self.name,
            "url": self.url,
            "test_type": "".join(self.test_type_codes) if self.test_type_codes else "Unknown",
        }

    def to_context_snippet(self) -> str:
        """Compact representation fed to the LLM as grounding context."""
        dur = f"{self.duration_minutes} min" if self.duration_minutes else "duration n/a"
        levels = ", ".join(self.job_levels) if self.job_levels else "n/a"
        desc = self.description[:280]
        return (
            f"[{self.id}] {self.name} | type: {self.test_type_text or '/'.join(self.test_type_codes)} "
            f"| {dur} | remote: {self.remote_testing} | adaptive/IRT: {self.adaptive_irt} "
            f"| job levels: {levels}\n    {desc}"
        )


def _build_search_text(item: dict) -> str:
    parts = [
        item["name"],
        item.get("description", ""),
        item.get("test_type_text") or "",
        " ".join(item.get("test_type_labels", [])),
        " ".join(item.get("job_levels", [])),
    ]
    return " ".join(p for p in parts if p)


class Catalog:
    def __init__(self, path: str = DATA_PATH):
        with open(path, "r") as f:
            raw = json.load(f)
        self.items: list[CatalogItem] = []
        self._by_id: dict[str, CatalogItem] = {}
        self._by_name_lower: dict[str, CatalogItem] = {}
        for entry in raw:
            item = CatalogItem(
                id=entry["id"],
                name=entry["name"],
                url=entry["url"],
                description=entry.get("description", ""),
                remote_testing=bool(entry.get("remote_testing")),
                adaptive_irt=bool(entry.get("adaptive_irt")),
                test_type_codes=entry.get("test_type_codes", []),
                test_type_labels=entry.get("test_type_labels", []),
                test_type_text=entry.get("test_type_text"),
                duration_minutes=entry.get("duration_minutes"),
                job_levels=entry.get("job_levels", []),
                languages=entry.get("languages", []),
            )
            item.search_text = _build_search_text(entry)
            self.items.append(item)
            self._by_id[item.id] = item
            self._by_name_lower[item.name.lower()] = item

    def __len__(self) -> int:
        return len(self.items)

    def get(self, item_id: str) -> Optional[CatalogItem]:
        return self._by_id.get(item_id)

    def get_by_name(self, name: str) -> Optional[CatalogItem]:
        return self._by_name_lower.get(name.lower())

    def find_names_mentioned_in(self, text: str) -> list[CatalogItem]:
        """Find catalog items whose exact name appears as a substring of `text`.

        Used to recover items that were recommended in earlier turns (the
        stateless API only carries plain-text history, so we recover prior
        structured recommendations by name-matching against the prose).
        Longer names are checked first to avoid partial-name false positives.
        """
        text_lower = text.lower()
        found = []
        for item in sorted(self.items, key=lambda i: -len(i.name)):
            if item.name.lower() in text_lower:
                found.append(item)
        return found


_catalog_singleton: Optional[Catalog] = None


def get_catalog() -> Catalog:
    global _catalog_singleton
    if _catalog_singleton is None:
        _catalog_singleton = Catalog()
    return _catalog_singleton

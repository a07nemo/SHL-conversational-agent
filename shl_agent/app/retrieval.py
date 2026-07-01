from __future__ import annotations

import re
from typing import Optional

from rank_bm25 import BM25Okapi

from .catalog import ALIASES, Catalog, CatalogItem

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def expand_aliases(text: str) -> str:
    """Append expansion terms for known SHL shorthand so BM25 has token overlap."""
    tokens = set(tokenize(text))
    extra = [ALIASES[t] for t in tokens if t in ALIASES]
    return text + " " + " ".join(extra) if extra else text


class Retriever:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        corpus = [tokenize(item.search_text) for item in catalog.items]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 20) -> list[CatalogItem]:
        if not query.strip():
            return []
        expanded = expand_aliases(query)
        tokens = tokenize(expanded)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            range(len(self.catalog.items)), key=lambda i: scores[i], reverse=True
        )
        results = []
        for i in ranked[: top_k * 2]:  # overshoot then filter zero scores
            if scores[i] > 0:
                results.append(self.catalog.items[i])
            if len(results) >= top_k:
                break
        return results


_retriever_singleton: Optional[Retriever] = None


def get_retriever(catalog: Catalog) -> Retriever:
    global _retriever_singleton
    if _retriever_singleton is None:
        _retriever_singleton = Retriever(catalog)
    return _retriever_singleton

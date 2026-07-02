from __future__ import annotations


def recall_at_k(recommended_names: list[str], relevant_names: list[str], k: int = 10) -> float:
    """Recall@K = |relevant ∩ top-K recommended| / |relevant|.

    Matching is case-insensitive on assessment name.
    """
    if not relevant_names:
        return 1.0  # nothing to find => trivially satisfied; harness shouldn't produce this
    top_k = {n.lower() for n in recommended_names[:k]}
    relevant = {n.lower() for n in relevant_names}
    hit = len(top_k & relevant)
    return hit / len(relevant)


def mean_recall_at_k(per_trace_recalls: list[float]) -> float:
    if not per_trace_recalls:
        return 0.0
    return sum(per_trace_recalls) / len(per_trace_recalls)

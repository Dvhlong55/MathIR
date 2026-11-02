"""Evaluation metrics for Mathematical Information Retrieval (mAP, Precision@K)."""

from typing import List


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Calculates Precision@K for a retrieved list against ground truth relevant items."""
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    rel_set = set(relevant)
    hits = sum(1 for item in top_k if item in rel_set)
    return hits / k


def average_precision(retrieved: List[str], relevant: List[str]) -> float:
    """Calculates Average Precision (AP) for a single query."""
    if not relevant:
        return 0.0
    rel_set = set(relevant)
    hits = 0
    score = 0.0
    for i, item in enumerate(retrieved):
        if item in rel_set:
            hits += 1
            score += hits / (i + 1)
    return score / len(relevant)


def mean_average_precision(all_retrieved: List[List[str]], all_relevant: List[List[str]]) -> float:
    """Calculates Mean Average Precision (mAP) across multiple queries."""
    if not all_retrieved or not all_relevant:
        return 0.0
    total = sum(average_precision(ret, rel) for ret, rel in zip(all_retrieved, all_relevant))
    return total / len(all_retrieved)

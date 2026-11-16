"""Evaluation metrics for Mathematical Information Retrieval (mAP, Precision@K, nDCG', MAP', P'@K, bpref)."""

import math
from typing import Dict, List, Optional


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


def _filter_judged(retrieved: List[str], judgments: Dict[str, int]) -> List[str]:
    """Filters out unjudged items (ARQMath prime metric convention)."""
    return [doc_id for doc_id in retrieved if doc_id in judgments]


def precision_at_k_prime(
    retrieved: List[str],
    judgments: Dict[str, int],
    k: int = 10,
    rel_threshold: int = 1,
) -> float:
    """Calculates P'@K over judged hits only (ARQMath convention)."""
    if k <= 0:
        return 0.0
    judged = _filter_judged(retrieved, judgments)[:k]
    hits = sum(1 for doc_id in judged if judgments[doc_id] >= rel_threshold)
    return hits / k


def average_precision_prime(
    retrieved: List[str],
    judgments: Dict[str, int],
    rel_threshold: int = 1,
) -> float:
    """Calculates AP' over judged hits only (ARQMath convention)."""
    total_rel = sum(1 for rel in judgments.values() if rel >= rel_threshold)
    if total_rel == 0:
        return 0.0
    judged = _filter_judged(retrieved, judgments)
    hits = 0
    score = 0.0
    for rank_idx, doc_id in enumerate(judged, start=1):
        if judgments[doc_id] >= rel_threshold:
            hits += 1
            score += hits / rank_idx
    return score / total_rel


def mean_average_precision_prime(
    all_retrieved: List[List[str]],
    all_judgments: List[Dict[str, int]],
    rel_threshold: int = 1,
) -> float:
    """Calculates MAP' across multiple queries using judged-only rankings."""
    if not all_retrieved or not all_judgments:
        return 0.0
    scores = [
        average_precision_prime(ret, jud, rel_threshold=rel_threshold)
        for ret, jud in zip(all_retrieved, all_judgments)
    ]
    return sum(scores) / len(scores)


def ndcg_prime(
    retrieved: List[str],
    judgments: Dict[str, int],
    k: Optional[int] = None,
    gain_mode: str = "linear",
) -> float:
    """Calculates nDCG' over judged hits only (ARQMath primary metric).

    Args:
        retrieved: Ranked list of document/formula IDs.
        judgments: Mapping from judged ID to non-negative graded relevance.
        k: Optional cutoff applied after filtering to judged items.
        gain_mode: `'linear'` (trec_eval style: `rel`) or `'exponential'` (`2**rel - 1`).
    """
    def _gain(rel: int) -> float:
        if rel <= 0:
            return 0.0
        if gain_mode == "linear":
            return float(rel)
        if gain_mode == "exponential":
            return float((2 ** rel) - 1)
        raise ValueError(f"Unknown gain_mode: {gain_mode}")

    judged = _filter_judged(retrieved, judgments)
    if k is not None:
        if k <= 0:
            return 0.0
        judged = judged[:k]

    ideal_rels = sorted((r for r in judgments.values() if r > 0), reverse=True)
    if k is not None:
        ideal_rels = ideal_rels[:k]
    idcg = sum(_gain(r) / math.log2(rank + 1) for rank, r in enumerate(ideal_rels, start=1))
    if idcg == 0.0:
        return 0.0

    dcg = sum(
        _gain(judgments[doc_id]) / math.log2(rank + 1)
        for rank, doc_id in enumerate(judged, start=1)
    )
    return dcg / idcg


def bpref(
    retrieved: List[str],
    judgments: Dict[str, int],
    rel_threshold: int = 1,
) -> float:
    """Calculates Buckley & Voorhees (2004) bpref for incomplete judgments (NTCIR-12)."""
    total_rel = sum(1 for r in judgments.values() if r >= rel_threshold)
    total_nonrel = sum(1 for r in judgments.values() if r < rel_threshold)
    if total_rel == 0:
        return 0.0

    denom = min(total_rel, total_nonrel)
    nonrel_seen = 0
    pref_sum = 0.0
    for doc_id in retrieved:
        if doc_id not in judgments:
            continue
        if judgments[doc_id] >= rel_threshold:
            if denom == 0:
                pref_sum += 1.0
            else:
                pref_sum += 1.0 - (min(nonrel_seen, total_rel) / denom)
        else:
            nonrel_seen += 1
    return pref_sum / total_rel


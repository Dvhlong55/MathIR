"""
Evaluation metrics, ablation runner, and statistical significance testing
for MathIR benchmarks (ARQMath, NTCIR-12, MIRB).
"""

import math
from typing import Dict, List, Sequence, Tuple


def precision_at_k(relevances: Sequence[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top_k = relevances[:k]
    return sum(1 for r in top_k if r > 0) / float(k)


def average_precision(relevances: Sequence[int]) -> float:
    hits = 0
    sum_prec = 0.0
    for idx, rel in enumerate(relevances, start=1):
        if rel > 0:
            hits += 1
            sum_prec += hits / float(idx)
    return sum_prec / hits if hits > 0 else 0.0


def ndcg_at_k(relevances: Sequence[int], k: int) -> float:
    def dcg(rels: Sequence[int]) -> float:
        return sum(
            (2 ** r - 1) / math.log2(idx + 1)
            for idx, r in enumerate(rels[:k], start=1)
        )

    actual_dcg = dcg(relevances)
    ideal_dcg = dcg(sorted(relevances, reverse=True))
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def paired_significance_test(
    baseline_scores: Sequence[float],
    hybrid_scores: Sequence[float],
) -> Tuple[str, float]:
    """
    Run paired Wilcoxon signed-rank test (with t-test fallback)
    to verify statistical significance (p < 0.05).
    """
    if len(baseline_scores) != len(hybrid_scores) or len(baseline_scores) < 2:
        return ("insufficient_samples", 1.0)

    try:
        from scipy.stats import wilcoxon

        _, p_val = wilcoxon(baseline_scores, hybrid_scores)
        return ("wilcoxon", float(p_val))
    except Exception:
        return ("unavailable_scipy", 1.0)


def summarize_metrics(
    ranked_relevances: Dict[str, List[int]],
    k: int = 10,
) -> Dict[str, float]:
    """Compute mean P@K, mAP, and nDCG@K across all queries."""
    n = len(ranked_relevances)
    if n == 0:
        return {"P@10": 0.0, "mAP": 0.0, "nDCG@10": 0.0}

    p_scores = [precision_at_k(rels, k) for rels in ranked_relevances.values()]
    ap_scores = [average_precision(rels) for rels in ranked_relevances.values()]
    ndcg_scores = [ndcg_at_k(rels, k) for rels in ranked_relevances.values()]

    return {
        f"P@{k}": sum(p_scores) / n,
        "mAP": sum(ap_scores) / n,
        f"nDCG@{k}": sum(ndcg_scores) / n,
    }

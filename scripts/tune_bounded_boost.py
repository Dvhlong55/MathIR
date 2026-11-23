"""Step 4: Tune Stage 3 bounded score boost on a validation retrieval split.

Expected input `.jsonl` rows (one candidate per line):
    {
      "query_id": "q1",
      "query_formula": "a^2 - b^2",
      "doc_id": "d12",
      "cand_formula": "(a-b)(a+b)",
      "stage12_score": 0.78,
      "relevance": 2
    }

Compares:
- Stage 1+2 only (`beta = 0`)
- Hard Rank-1 (`mode = hard_rank1`, reproducing `server.py`)
- Grid search over `residual` and `clipped_additive` bounded boost (`beta`, `min_context_score`)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.metrics import (
    average_precision_prime,
    bpref,
    ndcg_prime,
    precision_at_k_prime,
)
from src.retrieval.symbolic_verifier import TimeoutSymbolicVerifier, apply_bounded_boost


def load_validation_rows(path: str) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            grouped[str(row["query_id"])].append(row)
    return grouped


def evaluate_config(
    grouped: Dict[str, List[Dict[str, Any]]],
    equiv_cache: Dict[Tuple[str, str], bool],
    mode: str,
    beta: float,
    min_context_score: float,
) -> Dict[str, float]:
    ndcgs: List[float] = []
    maps: List[float] = []
    p10s: List[float] = []
    bprefs: List[float] = []

    for _qid, items in grouped.items():
        judgments = {str(it["doc_id"]): int(it["relevance"]) for it in items if "relevance" in it and it["relevance"] is not None}
        scored: List[Tuple[float, str]] = []
        for it in items:
            key = (str(it["query_formula"]), str(it["cand_formula"]))
            is_eq = equiv_cache.get(key, False)
            final_score = apply_bounded_boost(
                base_score=float(it["stage12_score"]),
                is_symbolic_equivalent=is_eq,
                mode=mode,
                beta=beta,
                min_context_score=min_context_score,
            )
            scored.append((final_score, str(it["doc_id"])))

        scored.sort(key=lambda x: x[0], reverse=True)
        ranked_ids = [doc_id for _, doc_id in scored]
        ndcgs.append(ndcg_prime(ranked_ids, judgments))
        maps.append(average_precision_prime(ranked_ids, judgments))
        p10s.append(precision_at_k_prime(ranked_ids, judgments, k=10))
        bprefs.append(bpref(ranked_ids, judgments))

    n = len(ndcgs) or 1
    return {
        "nDCG_prime": round(sum(ndcgs) / n, 4),
        "MAP_prime": round(sum(maps) / n, 4),
        "P10_prime": round(sum(p10s) / n, 4),
        "bpref": round(sum(bprefs) / n, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune bounded score boost on validation split.")
    parser.add_argument("--input", type=str, required=True, help="Validation .jsonl file.")
    parser.add_argument("--timeout", type=float, default=0.75)
    parser.add_argument("--backend", type=str, default="lark", choices=["lark", "antlr"])
    parser.add_argument("--output", type=str, help="Optional JSON output path.")
    args = parser.parse_args()

    grouped = load_validation_rows(args.input)
    unique_pairs = {
        (str(it["query_formula"]), str(it["cand_formula"]))
        for items in grouped.values()
        for it in items
    }

    equiv_cache: Dict[Tuple[str, str], bool] = {}
    with TimeoutSymbolicVerifier(timeout_seconds=args.timeout, latex_backend=args.backend) as verifier:
        for qf, cf in unique_pairs:
            res = verifier.verify_pair(qf, cf)
            equiv_cache[(qf, cf)] = res.is_symbolic_equivalent
        parse_summary = verifier.stats.summary()

    results: List[Dict[str, Any]] = []
    # Baseline 1: No Stage 3 boost
    base_metrics = evaluate_config(grouped, equiv_cache, mode="residual", beta=0.0, min_context_score=0.0)
    results.append({"mode": "stage12_only", "beta": 0.0, "min_context_score": 0.0, **base_metrics})

    # Baseline 2: Unconditional Rank-1 (server.py behavior)
    hard_metrics = evaluate_config(grouped, equiv_cache, mode="hard_rank1", beta=1.0, min_context_score=0.0)
    results.append({"mode": "hard_rank1", "beta": 1.0, "min_context_score": 0.0, **hard_metrics})

    # Grid search bounded boost
    for mode in ("residual", "clipped_additive"):
        for beta in (0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
            for tau in (0.0, 0.3, 0.5):
                metrics = evaluate_config(grouped, equiv_cache, mode=mode, beta=beta, min_context_score=tau)
                results.append({"mode": mode, "beta": beta, "min_context_score": tau, **metrics})

    results.sort(key=lambda r: (r["nDCG_prime"], r["MAP_prime"]), reverse=True)
    payload = {
        "best_config": results[0],
        "stage3_telemetry": parse_summary,
        "all_configs": results,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

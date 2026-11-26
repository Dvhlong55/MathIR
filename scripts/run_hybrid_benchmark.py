#!/usr/bin/env python3
"""
End-to-end Ablation & Benchmark Script for MathIR Hybrid System.
Evaluates Stage 1 (MathBERT + FAISS), Stage 1+2 (+ Structural Gate),
and Stage 1+2+3 (+ Timeout-guarded SymPy CAS) on data/processed/.
"""

import json
import os
import sys
import time
from typing import Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.benchmark import (
    average_precision,
    ndcg_at_k,
    paired_significance_test,
    precision_at_k,
    summarize_metrics,
)
from src.retrieval.faiss_index import MathFAISSIndex
from src.retrieval.hybrid_pipeline import CandidateItem, HybridMathIRPipeline
from src.retrieval.stage1_pipeline import MathBERTEmbedder
from src.retrieval.symbolic_verifier import TimeoutSymbolicVerifier, VerificationStatus


def load_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    data_dir = os.path.join(PROJECT_ROOT, "data", "processed")
    corpus_path = os.path.join(data_dir, "corpus_formulas.jsonl")
    queries_path = os.path.join(data_dir, "benchmark_queries.jsonl")
    qrels_path = os.path.join(data_dir, "benchmark_qrels.json")

    corpus = load_jsonl(corpus_path)
    queries = load_jsonl(queries_path)
    with open(qrels_path, "r", encoding="utf-8") as f:
        qrels: Dict[str, Dict[str, float]] = json.load(f)

    print(f"Loaded {len(corpus)} corpus formulas, {len(queries)} queries.")

    # Build doc_id -> latex map and FAISS index
    embedder = MathBERTEmbedder()
    texts = [item["latex"] for item in corpus]
    doc_ids = [item["doc_id"] for item in corpus]

    import numpy as np
    vecs_cache = os.path.join(data_dir, "corpus_vecs.npy")
    t0 = time.time()
    if os.path.exists(vecs_cache):
        print(f"Loading cached corpus vectors from {vecs_cache}...")
        corpus_vecs = np.load(vecs_cache)
    else:
        print("Encoding corpus with MathBERT in batches of 64...")
        corpus_vecs = np.vstack([embedder.encode(texts[i : i + 64]) for i in range(0, len(texts), 64)])
        np.save(vecs_cache, corpus_vecs)
    index = MathFAISSIndex(dim=corpus_vecs.shape[1])
    index.build(corpus_vecs, [{"idx": i, "doc_id": doc_ids[i], "formula": texts[i]} for i in range(len(corpus))])
    print(f"Indexed {len(corpus)} formulas in {time.time() - t0:.1f}s.")

    # Retrieve Top-50 per query once
    query_formulas = [q["formula"] for q in queries]
    query_vecs = np.vstack([embedder.encode(query_formulas[i : i + 32]) for i in range(0, len(query_formulas), 32)])

    configs = {
        "Stage1_MathBERT": HybridMathIRPipeline(use_stage2=False, use_stage3=False),
        "Stage1+2_StructuralGate": HybridMathIRPipeline(lambda_penalty=0.5, use_stage2=True, use_stage3=False),
        "Stage1+2+3_FullHybrid": HybridMathIRPipeline(lambda_penalty=0.5, cas_boost=0.25, use_stage2=True, use_stage3=True),
    }

    results_by_config: Dict[str, Dict[str, List[int]]] = {k: {} for k in configs}
    ndcg_per_query: Dict[str, List[float]] = {k: [] for k in configs}

    with TimeoutSymbolicVerifier(timeout_seconds=0.5) as verifier:
        for qi, q in enumerate(queries):
            qid = q["query_id"]
            q_latex = q["formula"]
            rel_map = qrels.get(qid, {})

            hits = index.search(query_vecs[qi : qi + 1], k=50)
            base_candidates: List[CandidateItem] = []
            for rank_i, hit in enumerate(hits):
                c_doc_id = hit["doc_id"]
                c_latex = hit["formula"]
                n_score = float(hit["score"])

                # Run Stage 3 CAS only on top-10 to keep evaluation fast
                cas_eq = None
                if rank_i < 10:
                    vres = verifier.verify_pair(q_latex, c_latex)
                    cas_eq = vres.status == VerificationStatus.EQUIV_PROVEN

                base_candidates.append(
                    CandidateItem(
                        doc_id=c_doc_id,
                        text=c_latex,
                        neural_score=n_score,
                        cas_equivalent=cas_eq,
                    )
                )

            for cfg_name, pipe in configs.items():
                cands_copy = [
                    CandidateItem(
                        doc_id=c.doc_id,
                        text=c.text,
                        neural_score=c.neural_score,
                        cas_equivalent=c.cas_equivalent,
                    )
                    for c in base_candidates
                ]
                ranked = pipe.rerank(q_latex, cands_copy)
                rels = [1 if rel_map.get(c.doc_id, 0) > 0 else 0 for c in ranked]
                results_by_config[cfg_name][qid] = rels
                ndcg_per_query[cfg_name].append(ndcg_at_k(rels, 10))

    summary = {}
    print("\n" + "=" * 70)
    print("  MathIR Hybrid Ablation Results (115 Queries: mseformula + wikiformula)")
    print("=" * 70)
    for cfg_name in configs:
        m = summarize_metrics(results_by_config[cfg_name], k=10)
        summary[cfg_name] = m
        print(f"{cfg_name:28s} | P@10: {m['P@10']:.4f} | mAP: {m['mAP']:.4f} | nDCG@10: {m['nDCG@10']:.4f}")

    test_name, p_val = paired_significance_test(
        ndcg_per_query["Stage1_MathBERT"],
        ndcg_per_query["Stage1+2+3_FullHybrid"],
    )
    print("-" * 70)
    print(f"Paired Significance ({test_name}, Stage1 vs FullHybrid nDCG@10): p = {p_val:.6f}")
    print("=" * 70)

    out_path = os.path.join(PROJECT_ROOT, "reports", "hybrid_benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "p_value": p_val, "test": test_name}, f, indent=2)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()

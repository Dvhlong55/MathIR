#!/usr/bin/env python3
"""
CLI script to download, sample, and preprocess MathIR benchmark formulas.
Course: SC203 - Scientific Method (APCS - HCMUS)
Research Group 2: Mathematical Information Retrieval (MathIR)

Usage:
    python scripts/extract_benchmark_samples.py --sample_size 50000 --output_dir data/processed
"""

import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import json
import argparse
from typing import Dict, List, Any

# Add repository root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.dataset_loader import MathBenchmarkLoader
from src.preprocessing.formula_sampler import FormulaSampler


def parse_args():
    parser = argparse.ArgumentParser(description="Extract and sample standardized MathIR benchmark datasets.")
    parser.add_argument(
        "--datasets",
        type=str,
        default="mseformula,wikiformula",
        help="Comma-separated list of datasets to process (mseformula, wikiformula)"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=50000,
        help="Target number of sampled formulas to output (e.g. 50000 to 100000)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Directory to save processed JSONL datasets"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data/raw",
        help="Directory to cache raw downloaded JSONL files"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # Resolve relative paths relative to repository root
    output_dir = os.path.join(project_root, args.output_dir) if not os.path.isabs(args.output_dir) else args.output_dir
    cache_dir = os.path.join(project_root, args.cache_dir) if not os.path.isabs(args.cache_dir) else args.cache_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    print("=" * 70)
    print("  MathIR Benchmark Extractor & Sampler")
    print(f"  Datasets:    {args.datasets}")
    print(f"  Target Size: {args.sample_size}")
    print(f"  Output Dir:  {output_dir}")
    print(f"  Cache Dir:   {cache_dir}")
    print("=" * 70)

    loader = MathBenchmarkLoader(cache_dir=cache_dir)
    selected_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    all_corpus: List[Dict[str, Any]] = []
    all_queries: List[Dict[str, Any]] = []
    merged_qrels: Dict[str, Dict[str, float]] = {}

    for ds_name in selected_datasets:
        print(f"\n[+] Processing dataset: {ds_name}...")
        try:
            corpus = loader.load_corpus(ds_name)
            queries = loader.load_queries(ds_name)
            qrels = loader.load_qrels(ds_name)

            print(f"    - Corpus documents loaded: {len(corpus):,}")
            print(f"    - Evaluation queries:      {len(queries):,}")
            print(f"    - Qrels topics:            {len(qrels):,}")

            all_corpus.extend(corpus)
            all_queries.extend(queries)
            merged_qrels.update(qrels)
        except Exception as e:
            print(f"    [!] Error loading {ds_name}: {e}")

    if not all_corpus:
        print("[!] No documents loaded. Exiting.")
        sys.exit(1)

    print(f"\n[+] Total combined corpus items: {len(all_corpus):,}")
    print(f"[+] Total benchmark queries:     {len(all_queries):,}")

    # Run Sampler
    print(f"\n[+] Sampling distinct formulas up to {args.sample_size:,}...")
    sampler = FormulaSampler(corpus=all_corpus, qrels=merged_qrels)
    sampled_formulas = sampler.sample(target_size=args.sample_size, seed=args.seed)

    print(f"    - Successfully sampled: {len(sampled_formulas):,} distinct formulas.")

    # Compute category statistics
    category_counts: Dict[str, int] = {}
    for item in sampled_formulas:
        for tag in item.get("tags", []):
            category_counts[tag] = category_counts.get(tag, 0) + 1

    print("\n[+] Mathematical Feature Distribution:")
    for tag, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(sampled_formulas)) * 100
        print(f"    - {tag:<20}: {count:>6,} ({pct:.1f}%)")

    # Save to output files
    corpus_out_file = os.path.join(output_dir, "corpus_formulas.jsonl")
    queries_out_file = os.path.join(output_dir, "benchmark_queries.jsonl")
    qrels_out_file = os.path.join(output_dir, "benchmark_qrels.json")
    stats_out_file = os.path.join(output_dir, "summary_stats.json")

    print(f"\n[+] Saving processed data to {output_dir}...")
    with open(corpus_out_file, "w", encoding="utf-8") as f:
        for item in sampled_formulas:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"    - Saved: {corpus_out_file}")

    with open(queries_out_file, "w", encoding="utf-8") as f:
        for q in all_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"    - Saved: {queries_out_file}")

    with open(qrels_out_file, "w", encoding="utf-8") as f:
        json.dump(merged_qrels, f, indent=2, ensure_ascii=False)
    print(f"    - Saved: {qrels_out_file}")

    summary_stats = {
        "total_sampled_formulas": len(sampled_formulas),
        "total_queries": len(all_queries),
        "total_qrels_topics": len(merged_qrels),
        "datasets_used": selected_datasets,
        "feature_distribution": category_counts,
        "files": {
            "corpus": corpus_out_file,
            "queries": queries_out_file,
            "qrels": qrels_out_file
        }
    }
    with open(stats_out_file, "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)
    print(f"    - Saved: {stats_out_file}")

    print("\n" + "=" * 70)
    print("  Data preparation complete! Stage 1 & Stage 2 ready.")
    print("=" * 70)


if __name__ == "__main__":
    main()

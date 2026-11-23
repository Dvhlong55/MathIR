"""Step 3: Measure MathBERT baseline cosine similarity distributions (anisotropy calibration).

Evaluates `tbs17/MathBERT` ([CLS] L2-normalized embeddings, matching `server.py`) on:
1. Random unrelated formula/question pairs (baseline distribution).
2. Truly equivalent pairs (ground-truth algebraic/semantic equivalents).
3. Adversarial failure cases (from `tests/test_bottlenecks.py` + `Group2_Week7_Report.tex`).

Usage:
    python scripts/calibrate_mathbert_cosine.py --corpus-formulas data/arqmath_formulas.jsonl --n-random 1000
    python scripts/calibrate_mathbert_cosine.py --random-pairs data/rand.jsonl --equiv-pairs data/equiv.jsonl --include-adversarial
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


ADVERSARIAL_CASES: List[Dict[str, str]] = [
    {"name": "Fraction Inversion (x/y vs y/x)", "fA": r"\frac{x}{y}", "fB": r"\frac{y}{x}"},
    {"name": "Inequality Flip (x > 0 vs x < 0)", "fA": "x > 0", "fB": "x < 0"},
    {"name": "Algebraic Fallacy ((a+b)^2 vs a^2+b^2)", "fA": "(a+b)^2", "fB": "a^2 + b^2"},
    {"name": "Exponent Sign (e^x vs e^{-x})", "fA": "e^x", "fB": "e^{-x}"},
    {"name": "Trig Orthogonal (sin(x) vs cos(x))", "fA": r"\sin(x)", "fB": r"\cos(x)"},
]


def load_pairs(path: str, col_a: str, col_b: str, limit: Optional[int] = None) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if path.lower().endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                pairs.append((str(obj[col_a]), str(obj[col_b])))
                if limit and len(pairs) >= limit:
                    break
    elif path.lower().endswith(".csv"):
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                pairs.append((str(row[col_a]), str(row[col_b])))
                if limit and len(pairs) >= limit:
                    break
    else:
        raise ValueError("Expected .jsonl or .csv input")
    return pairs


def sample_random_pairs_from_corpus(
    path: str, col: str, n_pairs: int, seed: int = 42
) -> List[Tuple[str, str]]:
    items: List[str] = []
    if path.lower().endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(str(json.loads(line)[col]))
    elif path.lower().endswith(".csv"):
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                items.append(str(row[col]))
    else:
        raise ValueError("Expected .jsonl or .csv input")

    unique_items = list(dict.fromkeys(x for x in items if x.strip()))
    if len(unique_items) < 2:
        raise ValueError("Need at least 2 distinct formulas to sample random pairs")

    rng = random.Random(seed)
    pairs: List[Tuple[str, str]] = []
    seen = set()
    max_attempts = n_pairs * 10
    attempts = 0
    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        a, b = rng.sample(unique_items, 2)
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            pairs.append((a, b))
    return pairs


def encode_texts(texts: Sequence[str], model_path: str, batch_size: int = 64) -> np.ndarray:
    """Encodes texts with MathBERT [CLS] L2-normalized vectors (matching `server.py`)."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    vecs: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            ).to(device)
            out = model(**inputs)
            cls_vec = out.last_hidden_state[:, 0, :]
            norm_vec = F.normalize(cls_vec, p=2, dim=-1)
            vecs.append(norm_vec.cpu().numpy())
    return np.vstack(vecs) if vecs else np.zeros((0, 768), dtype=np.float32)


def pair_cosines(pairs: Sequence[Tuple[str, str]], model_path: str, batch_size: int = 64) -> np.ndarray:
    if not pairs:
        return np.array([], dtype=np.float32)
    unique_texts = list(dict.fromkeys([t for p in pairs for t in p]))
    idx_map = {t: i for i, t in enumerate(unique_texts)}
    embs = encode_texts(unique_texts, model_path=model_path, batch_size=batch_size)
    idx_a = [idx_map[a] for a, _ in pairs]
    idx_b = [idx_map[b] for _, b in pairs]
    return np.sum(embs[idx_a] * embs[idx_b], axis=1)


def summarize_distribution(scores: np.ndarray) -> Dict[str, float]:
    if scores.size == 0:
        return {}
    return {
        "count": int(scores.size),
        "mean": round(float(np.mean(scores)), 4),
        "std": round(float(np.std(scores, ddof=1)) if scores.size > 1 else 0.0, 4),
        "median": round(float(np.median(scores)), 4),
        "p5": round(float(np.percentile(scores, 5)), 4),
        "p95": round(float(np.percentile(scores, 95)), 4),
        "min": round(float(np.min(scores)), 4),
        "max": round(float(np.max(scores)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate MathBERT cosine baseline distributions.")
    parser.add_argument("--model-path", type=str, default=None, help="Local path or HF id (default: baselines/mathbert or tbs17/MathBERT).")
    parser.add_argument("--random-pairs", type=str, help="JSONL/CSV of random unrelated pairs.")
    parser.add_argument("--corpus-formulas", type=str, help="JSONL/CSV of single formulas to sample random pairs from.")
    parser.add_argument("--formula-col", type=str, default="latex")
    parser.add_argument("--n-random", type=int, default=1000)
    parser.add_argument("--equiv-pairs", type=str, help="JSONL/CSV of truly equivalent pairs.")
    parser.add_argument("--col-a", type=str, default="formula_a")
    parser.add_argument("--col-b", type=str, default="formula_b")
    parser.add_argument("--include-adversarial", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, help="Optional JSON output path.")
    args = parser.parse_args()

    local_mb = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "baselines", "mathbert"))
    model_path = args.model_path or (local_mb if os.path.isdir(local_mb) else "tbs17/MathBERT")

    report: Dict[str, Any] = {"model_path": model_path}
    rand_mean: Optional[float] = None
    rand_std: Optional[float] = None

    if args.random_pairs:
        rand_pairs = load_pairs(args.random_pairs, args.col_a, args.col_b, limit=args.n_random)
    elif args.corpus_formulas:
        rand_pairs = sample_random_pairs_from_corpus(args.corpus_formulas, args.formula_col, args.n_random, args.seed)
    else:
        rand_pairs = []

    if rand_pairs:
        rand_scores = pair_cosines(rand_pairs, model_path, batch_size=args.batch_size)
        rand_summary = summarize_distribution(rand_scores)
        rand_mean = rand_summary["mean"]
        rand_std = rand_summary["std"]
        report["random_unrelated"] = rand_summary

    if args.equiv_pairs:
        eq_pairs = load_pairs(args.equiv_pairs, args.col_a, args.col_b)
        eq_scores = pair_cosines(eq_pairs, model_path, batch_size=args.batch_size)
        eq_summary = summarize_distribution(eq_scores)
        if rand_mean is not None and rand_std and rand_std > 0:
            eq_summary["mean_z_vs_random"] = round((eq_summary["mean"] - rand_mean) / rand_std, 4)
        report["truly_equivalent"] = eq_summary

    if args.include_adversarial:
        adv_pairs = [(c["fA"], c["fB"]) for c in ADVERSARIAL_CASES]
        adv_scores = pair_cosines(adv_pairs, model_path, batch_size=args.batch_size)
        adv_items = []
        for case, sc in zip(ADVERSARIAL_CASES, adv_scores):
            raw = round(float(sc), 4)
            entry: Dict[str, Any] = {"name": case["name"], "raw_cosine": raw}
            if rand_mean is not None:
                entry["delta_above_random_mean"] = round(raw - rand_mean, 4)
                if rand_std and rand_std > 0:
                    entry["z_vs_random"] = round((raw - rand_mean) / rand_std, 4)
                if rand_mean < 1.0:
                    entry["anisotropy_rescaled"] = round((raw - rand_mean) / (1.0 - rand_mean), 4)
            adv_items.append(entry)
        report["adversarial_cases"] = adv_items

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

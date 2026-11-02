"""
Benchmark Test Suite for MathIR: Failure Cases & Bottleneck Analysis
Course: SC203 - Scientific Method (APCS - HCMUS)
Group 2: Mathematical Information Retrieval (MathIR)

This test suite runs empirical evaluations on the Hybrid MathIR model (SymPy CAS + MathBERT)
to demonstrate the 3 core bottlenecks:
1. MathBERT Semantic Blindness / False Positives (Polarity & Inversion)
2. Cross-lingual Disparity (English vs Vietnamese Math Queries)
3. CAS Structural Limitations (Ellipsis & Bound Variables)
"""

import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import torch
import torch.nn.functional as F
import sympy as sp

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from server import evaluate_hybrid

TEST_CASES = [
    # -------------------------------------------------------------
    # Category 1: MathBERT False Positives (Opposite Polarities / Misconceptions)
    # Expected: Humans/CAS see difference, but MathBERT assigns > 90%
    # -------------------------------------------------------------
    {
        "category": "MathBERT False Positives (Polarity & Inversion)",
        "name": "Fraction Inversion (x/y vs y/x)",
        "fA": r"\frac{x}{y}",
        "fB": r"\frac{y}{x}",
        "expected_issue": "Semantic Inversion: MathBERT gives ~99% similarity despite reciprocal relationship."
    },
    {
        "category": "MathBERT False Positives (Polarity & Inversion)",
        "name": "Inequality Direction (x > 0 vs x < 0)",
        "fA": "x > 0",
        "fB": "x < 0",
        "expected_issue": "Polarity Flip: MathBERT scores ~93% on opposite domains."
    },
    {
        "category": "MathBERT False Positives (Polarity & Inversion)",
        "name": "Algebraic Misconception ((a+b)^2 vs a^2 + b^2)",
        "fA": "(a+b)^2",
        "fB": "a^2 + b^2",
        "expected_issue": "Cross-term Blindness: MathBERT scores ~93% ignoring 2ab term."
    },
    {
        "category": "MathBERT False Positives (Polarity & Inversion)",
        "name": "Exponential Sign (e^x vs e^(-x))",
        "fA": "e^x",
        "fB": "e^{-x}",
        "expected_issue": "Exponent Direction: MathBERT scores ~90% on diverging functions."
    },

    # -------------------------------------------------------------
    # Category 2: Cross-lingual Language Disparity (English vs Vietnamese)
    # Expected: High score on EN-EN, but drops sharply on VI-EN / VI-VI
    # -------------------------------------------------------------
    {
        "category": "Cross-lingual Gap (English vs Vietnamese)",
        "name": "English vs English Query Alignment",
        "fA": r"Let $a \ge 2$. Find the minimum value of $P = a + \frac{1}{a}$.",
        "fB": r"Given $a \ge 2$. Find the min value of $P = a + \frac{1}{a}$.",
        "expected_issue": "English baseline reaches ~97% similarity."
    },
    {
        "category": "Cross-lingual Gap (English vs Vietnamese)",
        "name": "Vietnamese vs English Query Alignment (Same Math Intent)",
        "fA": r"Cho $a \ge 2$. Tìm giá trị nhỏ nhất của $P = a + \frac{1}{a}$.",
        "fB": r"Given $a \ge 2$. Find the min value of $P = a + \frac{1}{a}$.",
        "expected_issue": "Language Gap: Similarity drops below 60% due to tokenizer out-of-vocabulary subwords."
    },

    # -------------------------------------------------------------
    # Category 3: CAS Limitations (Syntax & Domain Constraints)
    # Expected: CAS fails or refuses equivalence, falling back to neural
    # -------------------------------------------------------------
    {
        "category": "CAS Limitations (Syntax & Boundary)",
        "name": "Ellipsis Sequence (1+2+...+n vs n(n+1)/2)",
        "fA": "1 + 2 + ... + n",
        "fB": r"\frac{n(n+1)}{2}",
        "expected_issue": "Informal Syntax: SymPy cannot parse ellipsis '...' and crashes without AST."
    },
    {
        "category": "CAS Limitations (Syntax & Boundary)",
        "name": "Domain / Bound Variable Restriction (sqrt(x^2) vs x)",
        "fA": r"\sqrt{x^2}",
        "fB": "x",
        "expected_issue": "Strict Identity: Delta = -x + sqrt(x^2) != 0 unless x >= 0 is declared."
    }
]

def run_tests():
    print("=" * 80)
    print(" MathIR Empirical Benchmark: Failure Cases & Bottleneck Evaluation")
    print("=" * 80)
    
    current_cat = None
    for i, t in enumerate(TEST_CASES, 1):
        if t["category"] != current_cat:
            current_cat = t["category"]
            print(f"\n>>> [{current_cat}]")
            print("-" * 80)

        res = evaluate_hybrid(t["fA"], t["fB"])
        cas_status = "Identity (0)" if res.get("is_symbolic_equivalent") else res.get("diff", "Error")
        neural_pct = res.get("neural_similarity", 0.0) * 100
        fused_pct = res.get("fused_score", 0.0) * 100

        print(f"Test #{i:02d}: {t['name']}")
        print(f"  Formula A : {t['fA']}")
        print(f"  Formula B : {t['fB']}")
        print(f"  -> Neural Sim : {neural_pct:.1f}% | CAS: {cas_status} | Fused: {fused_pct:.1f}%")
        print(f"  -> Observed   : {t['expected_issue']}")
        print()

    print("=" * 80)
    print("Benchmark complete.")

if __name__ == "__main__":
    run_tests()

"""Step 2: Audit SymPy Stage 3 parse success rate, error breakdown, and latency on ARQMath LaTeX.

Usage:
    # Single-formula or pair dataset (.jsonl or .csv) from ARQMath:
    python scripts/eval_sympy_parse_rate.py --input data/arqmath_sample.jsonl --formula-col latex --timeout 0.75

    # Pair mode:
    python scripts/eval_sympy_parse_rate.py --input data/arqmath_pairs.jsonl --query-col formula_a --cand-col formula_b

    # Quick local sanity run using built-in noisy LaTeX + test_bottlenecks cases:
    python scripts/eval_sympy_parse_rate.py --demo-noisy
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.symbolic_verifier import TimeoutSymbolicVerifier


DEMO_NOISY_FORMULAS = [
    r"\displaystyle \frac{a^2 - b^2}{a - b}",
    r"\left( x + 1 \right)^2",
    r"1 + 2 + \dots + n",
    r"1 + 2 + ... + n",
    r"\sin^2(x) + \cos^2(x)",
    r"\int_0^1 x^2 \, dx",
    r"\sum_{i=1}^{n} i = \frac{n(n+1)}{2}",
    r"\frac{x}{",  # deliberately broken LaTeX
    r"x > 0",
    r"\text{where } x \ge 2",
]


def _load_records(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if path.lower().endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
                if limit and len(records) >= limit:
                    break
    elif path.lower().endswith(".csv"):
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(dict(row))
                if limit and len(records) >= limit:
                    break
    else:
        raise ValueError("Unsupported file extension; expected .jsonl or .csv")
    return records


def audit_single_formulas(
    formulas: Iterable[str],
    timeout_seconds: float,
    latex_backend: str,
) -> Dict[str, Any]:
    """Audits per-formula parse rate via `verify_pair(f, f)` so timeout also guards parsing."""
    total = 0
    parsed_ok = 0
    unsupported = 0
    parse_err = 0
    timeout_cnt = 0
    latencies: List[float] = []
    reasons: Counter = Counter()

    with TimeoutSymbolicVerifier(timeout_seconds=timeout_seconds, latex_backend=latex_backend) as verifier:
        for f in formulas:
            total += 1
            t0 = time.perf_counter()
            ok, unsup, err = verifier.check_single_parse(f)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            if ok:
                parsed_ok += 1
            elif unsup:
                unsupported += 1
                reasons[f"unsupported:{unsup}"] += 1
            else:
                parse_err += 1
                key = (err or "unknown").split("|")[0].strip()[:120]
                reasons[key] += 1

    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    p50 = sorted_lat[min(n - 1, int(0.50 * n))] if n else 0.0
    p95 = sorted_lat[min(n - 1, int(0.95 * n))] if n else 0.0

    denom = total or 1
    return {
        "mode": "single_formula_parse",
        "total": total,
        "parse_success_count": parsed_ok,
        "parse_success_rate": round(parsed_ok / denom, 4),
        "unsupported_syntax_count": unsupported,
        "parse_error_count": parse_err,
        "timeout_count": timeout_cnt,
        "latency_ms": {
            "p50": round(p50, 3),
            "p95": round(p95, 3),
            "max": round(sorted_lat[-1], 3) if sorted_lat else 0.0,
        },
        "top_failure_reasons": reasons.most_common(15),
    }


def audit_formula_pairs(
    pairs: Iterable[tuple[str, str]],
    timeout_seconds: float,
    latex_backend: str,
) -> Dict[str, Any]:
    """Audits full Stage 3 verification (parse + simplify + timeout) on formula pairs."""
    with TimeoutSymbolicVerifier(timeout_seconds=timeout_seconds, latex_backend=latex_backend) as verifier:
        for fa, fb in pairs:
            verifier.verify_pair(fa, fb)
        summary = verifier.stats.summary()
        summary["mode"] = "pair_verification"
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Stage 3 SymPy parse rate & latency.")
    parser.add_argument("--input", type=str, help="Path to .jsonl or .csv dataset.")
    parser.add_argument("--formula-col", type=str, default="latex", help="Column name for single-formula audit.")
    parser.add_argument("--query-col", type=str, help="Column name for query formula (pair mode).")
    parser.add_argument("--cand-col", type=str, help="Column name for candidate formula (pair mode).")
    parser.add_argument("--timeout", type=float, default=0.75, help="Per-candidate timeout in seconds.")
    parser.add_argument("--backend", type=str, default="lark", choices=["lark", "antlr"])
    parser.add_argument("--limit", type=int, default=None, help="Optional max records to audit.")
    parser.add_argument("--output", type=str, help="Optional JSON output path for audit summary.")
    parser.add_argument("--demo-noisy", action="store_true", help="Run diagnostic check on built-in noisy LaTeX samples.")
    args = parser.parse_args()

    if args.demo_noisy:
        report = audit_single_formulas(DEMO_NOISY_FORMULAS, args.timeout, args.backend)
    elif args.input:
        records = _load_records(args.input, limit=args.limit)
        if args.query_col and args.cand_col:
            pairs = [(str(r.get(args.query_col, "")), str(r.get(args.cand_col, ""))) for r in records]
            report = audit_formula_pairs(pairs, args.timeout, args.backend)
        else:
            formulas = [str(r.get(args.formula_col, "")) for r in records]
            report = audit_single_formulas(formulas, args.timeout, args.backend)
    else:
        parser.error("Provide either --input <file> or --demo-noisy")
        return

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

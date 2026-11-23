"""
Stage 2: Graded Structural Penalty Gate for Neuro-Symbolic MathIR.
Detects neural false positives (polarity flips, fraction inversions,
and root/trig operator mismatches) and applies a tunable soft penalty.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


OPPOSITE_RELATIONS = {
    (">", "<"),
    ("<", ">"),
    (r"\ge", r"\le"),
    (r"\le", r"\ge"),
    (">=", "<="),
    ("<=", ">="),
    (r"\geq", r"\leq"),
    (r"\leq", r"\geq"),
}

TRIG_FUNCTIONS = (r"\sin", r"\cos", r"\tan", r"\cot", r"\sec", r"\csc")


@dataclass
class GateResult:
    """Result of the structural penalty gate for a (query, candidate) pair."""
    penalty_factor: float
    reasons: List[str]


def _extract_relations(expr: str) -> List[str]:
    pattern = r"(\\geq|\\leq|\\ge|\\le|>=|<=|>|<)"
    return re.findall(pattern, expr)


def _extract_fractions(expr: str) -> List[Tuple[str, str]]:
    """Extract simple \\frac{num}{den} pairs (non-nested)."""
    return re.findall(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", expr)


def _extract_trig_ops(expr: str) -> List[str]:
    ops = []
    for fn in TRIG_FUNCTIONS:
        if fn in expr:
            ops.append(fn)
    return ops


def evaluate_structural_gate(
    query: str,
    candidate: str,
    lambda_penalty: float = 0.5,
    ast_overlap: Optional[float] = None,
) -> GateResult:
    """
    Compute a soft structural penalty factor in (0, 1] for (query, candidate).
    A factor of 1.0 means no structural mismatch detected.
    """
    if not (0.0 < lambda_penalty <= 1.0):
        raise ValueError("lambda_penalty must be in (0, 1].")

    penalty = 1.0
    reasons: List[str] = []

    # 1. Relational polarity flip check (e.g., x > 0 vs x < 0)
    q_rels = set(_extract_relations(query))
    c_rels = set(_extract_relations(candidate))
    if q_rels and c_rels:
        for qr in q_rels:
            for cr in c_rels:
                if (qr, cr) in OPPOSITE_RELATIONS:
                    penalty *= lambda_penalty
                    reasons.append(f"polarity_flip:{qr}_vs_{cr}")
                    break

    # 2. Fraction numerator/denominator inversion (e.g., \frac{x}{y} vs \frac{y}{x})
    q_fracs = _extract_fractions(query)
    c_fracs = _extract_fractions(candidate)
    for q_num, q_den in q_fracs:
        qn, qd = q_num.strip(), q_den.strip()
        for c_num, c_den in c_fracs:
            if qn == c_den.strip() and qd == c_num.strip() and qn != qd:
                penalty *= lambda_penalty
                reasons.append(f"fraction_inversion:{qn}/{qd}_vs_{qd}/{qn}")
                break

    # 3. Trigonometric / root operator mismatch (e.g., \sin(x) vs \cos(x))
    q_trigs = set(_extract_trig_ops(query))
    c_trigs = set(_extract_trig_ops(candidate))
    if q_trigs and c_trigs and q_trigs.isdisjoint(c_trigs):
        penalty *= lambda_penalty
        reasons.append(f"trig_mismatch:{sorted(q_trigs)}_vs_{sorted(c_trigs)}")

    # 4. Optional soft AST/OPT overlap modulation (from Tangent-S / Tangent-CFT)
    if ast_overlap is not None and 0.0 <= ast_overlap < 1.0 and reasons:
        # Slightly soften or strengthen based on structural overlap
        penalty = max(0.1, min(1.0, penalty * (0.5 + 0.5 * ast_overlap)))

    return GateResult(penalty_factor=penalty, reasons=reasons)

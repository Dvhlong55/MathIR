"""Unit tests for Stage 3 TimeoutSymbolicVerifier, bounded boost, and IR prime metrics."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.calibrate_mathbert_cosine import summarize_distribution
from scripts.tune_bounded_boost import evaluate_config
from src.evaluation.metrics import (
    average_precision_prime,
    bpref,
    ndcg_prime,
    precision_at_k_prime,
)
from src.retrieval.symbolic_verifier import (
    TimeoutSymbolicVerifier,
    VerificationStatus,
    apply_bounded_boost,
    clean_latex_noise,
    evaluate_pair_core,
)


def test_algebraic_equivalence_and_trig():
    res1 = evaluate_pair_core(r"a^2 - b^2", r"(a - b)(a + b)")
    assert res1.status == VerificationStatus.EQUIV_PROVEN
    assert res1.is_symbolic_equivalent is True

    res2 = evaluate_pair_core(r"\sin^2(x) + \cos^2(x)", r"1")
    assert res2.status == VerificationStatus.EQUIV_PROVEN
    assert res2.is_symbolic_equivalent is True


def test_polarity_and_inversion_rejected():
    res_ineq = evaluate_pair_core("x > 0", "x < 0")
    assert res_ineq.status == VerificationStatus.NOT_EQUIV
    assert res_ineq.is_symbolic_equivalent is False

    res_frac = evaluate_pair_core(r"\frac{x}{y}", r"\frac{y}{x}")
    assert res_frac.status == VerificationStatus.NOT_EQUIV
    assert res_frac.is_symbolic_equivalent is False


def test_equivalent_flipped_inequality():
    res = evaluate_pair_core("x > 0", "0 < x")
    assert res.status == VerificationStatus.EQUIV_PROVEN
    assert res.is_symbolic_equivalent is True


def test_ellipsis_safe_fallback():
    with TimeoutSymbolicVerifier(timeout_seconds=2.0) as verifier:
        res = verifier.verify_pair("1 + 2 + ... + n", r"\frac{n(n+1)}{2}")
        assert res.status == VerificationStatus.UNSUPPORTED_SYNTAX
        assert res.is_symbolic_equivalent is False

        res_dots = verifier.verify_pair(r"1 + 2 + \dots + n", r"\frac{n(n+1)}{2}")
        assert res_dots.status == VerificationStatus.UNSUPPORTED_SYNTAX
        assert res_dots.is_symbolic_equivalent is False


def test_noisy_latex_cleaning_and_parse_error_fallback():
    cleaned = clean_latex_noise(r"\displaystyle \left( \frac{x}{y} \right) \,")
    assert r"\displaystyle" not in cleaned
    assert r"\left" not in cleaned

    with TimeoutSymbolicVerifier(timeout_seconds=2.0) as verifier:
        res_bad = verifier.verify_pair(r"\frac{x}{", r"x")
        assert res_bad.status == VerificationStatus.PARSE_ERROR
        assert res_bad.is_symbolic_equivalent is False
        assert res_bad.parse_success_query is False
        assert res_bad.parse_success_cand is True


def test_timeout_enforced_and_worker_recovers():
    # High-degree multivariate polynomial expansion/simplification reliably exceeds 0.10s
    heavy_a = "(x + y + z + w + t + 1)**14"
    heavy_b = "(x + y + z + w + t)**14"
    with TimeoutSymbolicVerifier(timeout_seconds=0.10) as verifier:
        res_timeout = verifier.verify_pair(heavy_a, heavy_b)
        assert res_timeout.status == VerificationStatus.TIMEOUT
        assert res_timeout.is_symbolic_equivalent is False

        # Subsequent normal verification must still succeed after worker recovery
        res_ok = verifier.verify_pair("(x + 1)^2", "x^2 + 2x + 1")
        assert res_ok.status == VerificationStatus.EQUIV_PROVEN
        assert res_ok.is_symbolic_equivalent is True

        summary = verifier.stats.summary()
        assert summary["total_pairs"] == 2
        assert summary["counts"]["timeout"] == 1
        assert summary["counts"]["equiv_proven"] == 1


def test_bounded_score_boost_and_reranking():
    # Residual mode preserves relative ordering of context scores
    s_high = apply_bounded_boost(0.80, True, mode="residual", beta=0.25)
    s_low = apply_bounded_boost(0.40, True, mode="residual", beta=0.25)
    assert s_low < s_high < 1.0
    assert s_high == pytest.approx(0.85, abs=1e-6)

    # Low-context gate suppresses boost when s12 < min_context_score
    s_gated = apply_bounded_boost(0.20, True, mode="residual", beta=0.5, min_context_score=0.3)
    assert s_gated == pytest.approx(0.20, abs=1e-6)

    # Non-equivalent candidates remain untouched
    assert apply_bounded_boost(0.72, False, mode="residual", beta=0.5) == pytest.approx(0.72, abs=1e-6)

    with TimeoutSymbolicVerifier(timeout_seconds=2.0) as verifier:
        ranked = verifier.rerank_candidates(
            query_formula=r"a^2 - b^2",
            candidates=[
                {"id": "d1", "formula": r"a^2 + b^2", "stage12_score": 0.85},
                {"id": "d2", "formula": r"(a-b)(a+b)", "stage12_score": 0.75},
                {"id": "d3", "formula": r"1 + 2 + ... + n", "stage12_score": 0.60},
            ],
            boost_mode="residual",
            beta=0.5,
        )
        # d2 boosts from 0.75 -> 0.875 and overtakes d1 (0.85) without forcing 1.0
        assert [c["id"] for c in ranked] == ["d2", "d1", "d3"]
        assert ranked[0]["final_score"] == pytest.approx(0.875, abs=1e-6)
        assert ranked[2]["stage3_status"] == VerificationStatus.UNSUPPORTED_SYNTAX.value


def test_arqmath_prime_metrics_and_bpref():
    judgments = {"d1": 3, "d2": 1, "d3": 0, "d4": 0}
    # "u1" and "u2" are unjudged and should be ignored by prime metrics
    retrieved = ["u1", "d1", "u2", "d3", "d2", "d4"]

    assert precision_at_k_prime(retrieved, judgments, k=2) == pytest.approx(0.5)
    assert average_precision_prime(retrieved, judgments) == pytest.approx((1.0 / 1 + 2.0 / 3) / 2.0)
    assert ndcg_prime(["d1", "d2", "d3"], judgments, gain_mode="linear") == pytest.approx(1.0)
    assert bpref(retrieved, judgments) == pytest.approx((1.0 + 0.5) / 2.0)


def test_tune_evaluate_config_and_calibration_summary():
    import numpy as np

    grouped = {
        "q1": [
            {"doc_id": "d_wrong_ctx", "query_formula": "x^2", "cand_formula": "x*x", "stage12_score": 0.20, "relevance": 0},
            {"doc_id": "d_rel", "query_formula": "x^2", "cand_formula": "x^2 + 1", "stage12_score": 0.85, "relevance": 3},
        ]
    }
    cache = {("x^2", "x*x"): True, ("x^2", "x^2 + 1"): False}

    # Unconditional hard_rank1 pushes irrelevant context d_wrong_ctx (1.0) above d_rel (0.85)
    hard = evaluate_config(grouped, cache, mode="hard_rank1", beta=1.0, min_context_score=0.0)
    # Gated residual boost avoids boosting low-context false match (0.20 < 0.30)
    bounded = evaluate_config(grouped, cache, mode="residual", beta=0.25, min_context_score=0.30)
    assert bounded["nDCG_prime"] > hard["nDCG_prime"]
    assert bounded["nDCG_prime"] == pytest.approx(1.0)

    dist = summarize_distribution(np.array([0.70, 0.80, 0.90]))
    assert dist["count"] == 3
    assert dist["mean"] == pytest.approx(0.8)
    assert dist["median"] == pytest.approx(0.8)

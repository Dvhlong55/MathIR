"""
Unit tests for Tangent-CFT / Tangent-CFTED Baseline Retriever.
Course: SC203 - Scientific Method (APCS - HCMUS)
Research Group 2: Mathematical Information Retrieval (MathIR)
"""

import os
import sys
import unittest

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.retrieval.tangent_retriever import TangentBaseline


class TestTangentBaseline(unittest.TestCase):

    def setUp(self):
        self.retriever = TangentBaseline(use_ted=True)

    def test_score_pair_identical(self):
        """Identical formulas must receive perfect score 1.0."""
        score_info = self.retriever.score_pair(r"\frac{a+b}{c}", r"\frac{a+b}{c}")
        self.assertEqual(score_info["final_score"], 1.0)
        self.assertEqual(score_info["penalty_factor"], 1.0)
        self.assertFalse(score_info["conflict_diagnostics"]["has_critical_conflict"])

    def test_score_pair_ratio_inversion(self):
        """Reciprocal fraction \\frac{x}{y} vs \\frac{y}{x} must receive penalty 0.4."""
        score_info = self.retriever.score_pair(r"\frac{x}{y}", r"\frac{y}{x}")
        self.assertEqual(score_info["penalty_factor"], 0.4)
        self.assertTrue(score_info["conflict_diagnostics"]["is_ratio_inverted"])
        self.assertLess(score_info["final_score"], 0.5)

    def test_score_pair_polarity_flip(self):
        """Opposite inequalities x > 0 vs x < 0 must receive penalty 0.5."""
        score_info = self.retriever.score_pair("x > 0", "x < 0")
        self.assertEqual(score_info["penalty_factor"], 0.5)
        self.assertTrue(score_info["conflict_diagnostics"]["is_polarity_flipped"])

    def test_corpus_indexing_and_ranking(self):
        """Retrieval must rank exact structural match above adversarial inverted fractions."""
        corpus = [
            {"doc_id": "c_identical", "latex": r"\frac{x}{y}"},
            {"doc_id": "c_inverted", "latex": r"\frac{y}{x}"},
            {"doc_id": "c_ineq", "latex": "x > 0"},
            {"doc_id": "c_poly", "latex": "a^2 + b^2"}
        ]
        self.retriever.index_corpus(corpus)

        results = self.retriever.retrieve(r"\frac{x}{y}", top_k=5)
        self.assertGreater(len(results), 0)

        # Rank 1 must be exact match
        top_hit = results[0]
        self.assertEqual(top_hit["doc_id"], "c_identical")
        self.assertEqual(top_hit["score"], 1.0)

        # Inverted fraction must be penalized and ranked strictly below
        if len(results) > 1 and results[1]["doc_id"] == "c_inverted":
            self.assertLess(results[1]["score"], top_hit["score"])
            self.assertEqual(results[1]["penalty_factor"], 0.4)


if __name__ == "__main__":
    unittest.main()

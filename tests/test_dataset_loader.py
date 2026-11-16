"""
Unit tests for MathIR dataset loader and formula sampler.
Course: SC203 - Scientific Method (APCS - HCMUS)
Research Group 2: Mathematical Information Retrieval (MathIR)
"""

import os
import sys
import unittest

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.dataset_loader import clean_latex, extract_formulas_from_text
from src.preprocessing.formula_sampler import tag_formula_features, FormulaSampler


class TestDatasetLoader(unittest.TestCase):

    def test_clean_latex(self):
        """Verifies stripping of markdown and LaTeX delimiters."""
        self.assertEqual(clean_latex("$x > 0$"), "x > 0")
        self.assertEqual(clean_latex("$$\\frac{a}{b}$$"), "\\frac{a}{b}")
        self.assertEqual(clean_latex("\\[e^x = 1\\]"), "e^x = 1")
        self.assertEqual(clean_latex("\\(a + b\\)"), "a + b")
        self.assertEqual(clean_latex("   x + y   "), "x + y")

    def test_extract_formulas_from_text(self):
        """Verifies formula extraction from mixed text."""
        text = "Let $a \\ge 2$. Find the minimum of $P = a + \\frac{1}{a}$ and $$\\sum_{i=1}^n i$$."
        formulas = extract_formulas_from_text(text)
        self.assertEqual(len(formulas), 3)
        self.assertIn("a \\ge 2", formulas)
        self.assertIn("P = a + \\frac{1}{a}", formulas)
        self.assertIn("\\sum_{i=1}^n i", formulas)

    def test_tag_formula_features(self):
        """Verifies classification of mathematical structures."""
        tags_ineq = tag_formula_features("x > 0")
        self.assertIn("inequality", tags_ineq)

        tags_frac = tag_formula_features("\\frac{x}{y}")
        self.assertIn("fraction", tags_frac)

        tags_trig = tag_formula_features("\\sin(x) + \\cos(x)")
        self.assertIn("trigonometric", tags_trig)

        tags_calc = tag_formula_features("\\lim_{n \\to \\infty} \\sum_{i=1}^n i")
        self.assertIn("calculus", tags_calc)

        tags_ellipsis = tag_formula_features("1 + 2 + ... + n")
        self.assertIn("ellipsis_sequence", tags_ellipsis)

    def test_formula_sampler(self):
        """Verifies distinct sampling and qrels priority preservation."""
        corpus = [
            {
                "doc_id": "doc_1",
                "latex": "x > 0",
                "all_formulas": ["x > 0", "y < 0"],
                "title": "Positive domain",
                "source_dataset": "test_ds"
            },
            {
                "doc_id": "doc_2",
                "latex": "\\frac{a}{b}",
                "all_formulas": ["\\frac{a}{b}"],
                "title": "Fraction test",
                "source_dataset": "test_ds"
            },
            {
                "doc_id": "doc_3",
                "latex": "1 + 2 + ... + n",
                "all_formulas": ["1 + 2 + ... + n"],
                "title": "Sum test",
                "source_dataset": "test_ds"
            }
        ]
        qrels = {"query_1": {"doc_1": 3.0}}

        sampler = FormulaSampler(corpus=corpus, qrels=qrels)
        formulas = sampler.extract_distinct_formulas()

        # Should deduplicate and tag
        self.assertGreaterEqual(len(formulas), 3)
        
        # doc_1 should be tagged as benchmark relevant
        doc_1_item = next(f for f in formulas if f["doc_id"] == "doc_1")
        self.assertTrue(doc_1_item["is_benchmark_relevant"])

        # Sampling with target_size = 2 should keep priority item
        sampled = sampler.sample(target_size=2)
        self.assertEqual(len(sampled), 2)
        has_priority = any(f["is_benchmark_relevant"] for f in sampled)
        self.assertTrue(has_priority)


if __name__ == "__main__":
    unittest.main()

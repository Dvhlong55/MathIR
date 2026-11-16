"""
Unit tests for Mathematical AST & Tangent-S Symbol-Pair Tuple Extractor.
Course: SC203 - Scientific Method (APCS - HCMUS)
Research Group 2: Mathematical Information Retrieval (MathIR)

Verifies adversarial conflict detection across the 4 core neural failure modes:
1. Ratio Inversion: \\frac{x}{y} vs \\frac{y}{x}
2. Polarity Flip: x > 0 vs x < 0
3. Trig Orthogonality: \\sin(x) vs \\cos(x)
4. Exponent Direction: e^x vs e^{-x}
"""

import os
import sys
import unittest

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.representation.tree_parser import parse_latex_to_tree, MathTreeNode
from src.representation.tuple_extractor import (
    extract_symbol_tuples,
    compute_tuple_similarity,
    detect_structural_conflicts
)


class TestTreeParserAndTuples(unittest.TestCase):

    def test_fraction_parsing_and_inversion(self):
        """Verifies fraction parsing and detection of reciprocal inversion."""
        tree_a = parse_latex_to_tree(r"\frac{x}{y}")
        tree_b = parse_latex_to_tree(r"\frac{y}{x}")

        # Check node structure
        self.assertEqual(tree_a.label, r"\frac")
        self.assertEqual(len(tree_a.children), 2)
        self.assertEqual(tree_a.children[0].role, "numerator")
        self.assertEqual(tree_a.children[0].label, "x")
        self.assertEqual(tree_a.children[1].role, "denominator")
        self.assertEqual(tree_a.children[1].label, "y")

        # Check symbol tuples
        tuples_a = extract_symbol_tuples(tree_a)
        self.assertIn(r"(\frac, x, NUMERATOR)", tuples_a)
        self.assertIn(r"(\frac, y, DENOMINATOR)", tuples_a)

        # Detect structural conflict
        conflict = detect_structural_conflicts(tree_a, tree_b)
        self.assertTrue(conflict["is_ratio_inverted"])
        self.assertTrue(conflict["has_critical_conflict"])
        self.assertEqual(conflict["penalty_factor"], 0.4)

    def test_inequality_polarity_flip(self):
        """Verifies relational parsing and opposite inequality detection."""
        tree_gt = parse_latex_to_tree("x > 0")
        tree_lt = parse_latex_to_tree("x < 0")

        self.assertEqual(tree_gt.node_type, "relation")
        self.assertEqual(tree_gt.label, ">")
        self.assertEqual(tree_lt.label, "<")

        conflict = detect_structural_conflicts(tree_gt, tree_lt)
        self.assertTrue(conflict["is_polarity_flipped"])
        self.assertTrue(conflict["has_critical_conflict"])
        self.assertEqual(conflict["penalty_factor"], 0.5)

        # Test with LaTeX macros \ge vs \le
        tree_ge = parse_latex_to_tree(r"a \ge 2")
        tree_le = parse_latex_to_tree(r"a \le 2")
        conflict_macro = detect_structural_conflicts(tree_ge, tree_le)
        self.assertTrue(conflict_macro["is_polarity_flipped"])

    def test_trigonometric_orthogonality(self):
        """Verifies trigonometric function conflict detection (sin vs cos)."""
        tree_sin = parse_latex_to_tree(r"\sin(x)")
        tree_cos = parse_latex_to_tree(r"\cos(x)")

        self.assertEqual(tree_sin.node_type, "trigonometric")
        self.assertEqual(tree_cos.node_type, "trigonometric")

        conflict = detect_structural_conflicts(tree_sin, tree_cos)
        self.assertTrue(conflict["is_trig_mismatch"])
        self.assertEqual(conflict["penalty_factor"], 0.4)
        self.assertEqual(conflict["ast_overlap"], 0.0)

    def test_exponent_direction(self):
        """Verifies exponential sign reversal detection (e^x vs e^{-x})."""
        tree_pos = parse_latex_to_tree("e^x")
        tree_neg = parse_latex_to_tree("e^{-x}")

        conflict = detect_structural_conflicts(tree_pos, tree_neg)
        self.assertTrue(conflict["is_exponent_flipped"])
        self.assertEqual(conflict["penalty_factor"], 0.5)

    def test_identical_formula_alignment(self):
        """Verifies 100% overlap and penalty = 1.0 on identical structures."""
        formula = r"\frac{a + b}{c} \ge 2"
        tree_1 = parse_latex_to_tree(formula)
        tree_2 = parse_latex_to_tree(formula)

        conflict = detect_structural_conflicts(tree_1, tree_2)
        self.assertFalse(conflict["has_critical_conflict"])
        self.assertEqual(conflict["penalty_factor"], 1.0)
        self.assertEqual(conflict["ast_jaccard"], 1.0)
        self.assertEqual(conflict["ast_overlap"], 1.0)


if __name__ == "__main__":
    unittest.main()

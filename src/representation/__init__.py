"""Mathematical representation package for MathIR (AST, OPT, SLT & Tangent-S)."""

from .tree_parser import MathTreeNode, LaTeXTreeBuilder, parse_latex_to_tree
from .tuple_extractor import (
    extract_symbol_tuples,
    compute_tuple_similarity,
    detect_structural_conflicts
)

__all__ = [
    "MathTreeNode",
    "LaTeXTreeBuilder",
    "parse_latex_to_tree",
    "extract_symbol_tuples",
    "compute_tuple_similarity",
    "detect_structural_conflicts"
]

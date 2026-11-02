"""Formula canonicalization and variable renaming invariance (Hypothesis H1)."""

import sympy
from typing import Optional


class FormulaCanonicalizer:
    """Canonicalizes raw LaTeX / mathematical formulas into normalized representations.
    
    Helps achieve variable renaming invariance and algebraic simplification
    as outlined in Hypothesis H1.
    """

    def __init__(self):
        pass

    def simplify_expression(self, expr_str: str) -> str:
        """Parses and simplifies an algebraic expression via SymPy.
        
        Example: (x + 1)**2 -> x**2 + 2*x + 1
        """
        try:
            expr = sympy.sympify(expr_str)
            simplified = sympy.expand(expr)
            return str(simplified)
        except Exception:
            return expr_str

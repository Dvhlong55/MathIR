"""
Mathematical AST & Operator Tree (OPT) / Symbol Layout Tree (SLT) parser for MathIR.
Course: SC203 - Scientific Method (APCS - HCMUS)
Research Group 2: Mathematical Information Retrieval (MathIR)

Parses raw LaTeX formulas into 2D structural trees, capturing hierarchical operator precedence,
fraction numerator/denominator relationships, power/root layouts, and relational bounding.
Compatible with pure Python environments without requiring external Perl LaTeXML binaries.
"""

import re
from typing import List, Optional, Dict, Any, Tuple


class MathTreeNode:
    """Node in a mathematical Operator Tree (OPT) or Symbol Layout Tree (SLT)."""

    def __init__(
        self,
        label: str,
        node_type: str = "operator",
        role: Optional[str] = None,
        children: Optional[List["MathTreeNode"]] = None
    ):
        """Initializes a node.
        
        Args:
            label: Textual representation (e.g. '+', '\\frac', 'x', '2', '>', '\\sin').
            node_type: One of ('operator', 'relation', 'function', 'variable', 'number', 'layout').
            role: Semantic role ('numerator', 'denominator', 'base', 'exponent', 'arg', 'lhs', 'rhs', 'child').
            children: Child subtrees.
        """
        self.label = str(label).strip()
        self.node_type = node_type
        self.role = role or "child"
        self.children = children if children is not None else []

    def add_child(self, child: "MathTreeNode", role: Optional[str] = None) -> None:
        if role:
            child.role = role
        self.children.append(child)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Serializes node to recursive dictionary representation."""
        return {
            "label": self.label,
            "type": self.node_type,
            "role": self.role,
            "children": [c.to_dict() for c in self.children]
        }

    def __repr__(self) -> str:
        if not self.children:
            return f"Node({self.label}:{self.node_type})"
        child_str = ", ".join(repr(c) for c in self.children)
        return f"Node({self.label}:{self.node_type}, role={self.role}, [{child_str}])"


# Token patterns for LaTeX mathematical expressions
TOKEN_REGEX = re.compile(
    r"""
    (\\[a-zA-Z]+)          |  # LaTeX macros (\frac, \sin, \ge, etc.)
    (\d+(?:\.\d+)?)        |  # Numbers (integers or decimals)
    ([a-zA-Z])             |  # Single variable characters
    (>=|<=|==|!=)          |  # Multi-char ascii relations
    ([+\-*/=><^_(),{}[\]])    # Single-char operators and punctuation
    """,
    re.VERBOSE
)


def tokenize_latex(latex_str: str) -> List[str]:
    """Tokenizes a LaTeX string into atomic symbols and delimiters."""
    # Strip enclosing delimiters
    s = latex_str.strip()
    s = re.sub(r'^\$\$+|\$\$+$', '', s)
    s = re.sub(r'^\$+|\$+$', '', s)
    s = re.sub(r'^\\\[|\\\]$', '', s)
    s = re.sub(r'^\\\(|\\\)$', '', s)

    tokens = []
    for match in TOKEN_REGEX.finditer(s):
        tok = match.group(0).strip()
        if tok:
            tokens.append(tok)
    return tokens


def find_matching_brace(tokens: List[str], start_idx: int) -> int:
    """Finds the index of the matching closing brace '}' or ')' for the token at start_idx."""
    open_tok = tokens[start_idx]
    close_tok = "}" if open_tok == "{" else (")" if open_tok == "(" else "]")
    
    depth = 0
    for i in range(start_idx, len(tokens)):
        if tokens[i] == open_tok:
            depth += 1
        elif tokens[i] == close_tok:
            depth -= 1
            if depth == 0:
                return i
    return len(tokens) - 1


class LaTeXTreeBuilder:
    """Recursive descent parser to construct Operator Trees from LaTeX tokens."""

    RELATIONAL_OPERATORS = {
        "=": "equality",
        ">": "greater_than",
        "<": "less_than",
        r"\ge": "greater_equal",
        r"\geq": "greater_equal",
        r"\le": "less_equal",
        r"\leq": "less_equal",
        r"\neq": "not_equal",
        ">=": "greater_equal",
        "<=": "less_equal"
    }

    TRIG_FUNCTIONS = {
        r"\sin", r"\cos", r"\tan", r"\cot", r"\arcsin", r"\arccos", r"\arctan",
        r"\sinh", r"\cosh", r"\tanh", "sin", "cos", "tan"
    }

    COMMON_FUNCTIONS = {
        r"\sqrt", r"\exp", r"\ln", r"\log", r"\lim", r"\sum", r"\prod", r"\int"
    }

    def parse(self, latex_str: str) -> MathTreeNode:
        """Parses LaTeX string into a MathTreeNode tree."""
        tokens = tokenize_latex(latex_str)
        if not tokens:
            return MathTreeNode("EMPTY", node_type="variable")
        return self._parse_expression(tokens)

    def _parse_expression(self, tokens: List[str]) -> MathTreeNode:
        if not tokens:
            return MathTreeNode("EMPTY", node_type="variable")

        # 1. Check for top-level relational operator (Lowest Precedence: =, >, <, \le, \ge)
        depth = 0
        rel_idx = -1
        for i, tok in enumerate(tokens):
            if tok in ("{", "(", "["):
                depth += 1
            elif tok in ("}", ")", "]"):
                depth -= 1
            elif depth == 0 and tok in self.RELATIONAL_OPERATORS:
                rel_idx = i
                break

        if rel_idx != -1:
            rel_op = tokens[rel_idx]
            lhs_tokens = tokens[:rel_idx]
            rhs_tokens = tokens[rel_idx + 1:]
            lhs_node = self._parse_expression(lhs_tokens)
            rhs_node = self._parse_expression(rhs_tokens)
            
            node = MathTreeNode(rel_op, node_type="relation")
            node.add_child(lhs_node, role="lhs")
            node.add_child(rhs_node, role="rhs")
            return node

        # 2. Check for Addition & Subtraction (+, -)
        depth = 0
        add_idx = -1
        for i in range(len(tokens) - 1, -1, -1):
            tok = tokens[i]
            if tok in ("}", ")", "]"):
                depth += 1
            elif tok in ("{", "(", "["):
                depth -= 1
            elif depth == 0 and tok in ("+", "-") and i > 0 and tokens[i-1] not in ("+", "-", "*", "/", "^", "_", "(", "{", "[", ","):
                add_idx = i
                break

        if add_idx != -1:
            op = tokens[add_idx]
            lhs_node = self._parse_expression(tokens[:add_idx])
            rhs_node = self._parse_expression(tokens[add_idx + 1:])
            node = MathTreeNode(op, node_type="operator")
            node.add_child(lhs_node, role="left")
            node.add_child(rhs_node, role="right")
            return node

        # 3. Check for Multiplication & Division (*, /, \cdot, \times)
        depth = 0
        mul_idx = -1
        for i in range(len(tokens) - 1, -1, -1):
            tok = tokens[i]
            if tok in ("}", ")", "]"):
                depth += 1
            elif tok in ("{", "(", "["):
                depth -= 1
            elif depth == 0 and tok in ("*", "/", r"\times", r"\cdot"):
                mul_idx = i
                break

        if mul_idx != -1:
            op = tokens[mul_idx]
            lhs_node = self._parse_expression(tokens[:mul_idx])
            rhs_node = self._parse_expression(tokens[mul_idx + 1:])
            node = MathTreeNode(op, node_type="operator")
            node.add_child(lhs_node, role="left")
            node.add_child(rhs_node, role="right")
            return node

        # 4. Check for Powers / Superscripts (^) and Subscripts (_)
        depth = 0
        pow_idx = -1
        for i in range(len(tokens) - 1, -1, -1):
            tok = tokens[i]
            if tok in ("}", ")", "]"):
                depth += 1
            elif tok in ("{", "(", "["):
                depth -= 1
            elif depth == 0 and tok in ("^", "_"):
                pow_idx = i
                break

        if pow_idx != -1:
            op = tokens[pow_idx]
            base_node = self._parse_expression(tokens[:pow_idx])
            exp_node = self._parse_expression(tokens[pow_idx + 1:])
            node = MathTreeNode(op, node_type="layout")
            role_base = "base"
            role_exp = "exponent" if op == "^" else "subscript"
            node.add_child(base_node, role=role_base)
            node.add_child(exp_node, role=role_exp)
            return node

        # 5. Check for Fractions: \frac{...}{...} or \cfrac{...}{...}
        if tokens[0] in (r"\frac", r"\cfrac"):
            return self._parse_fraction(tokens)

        # 6. Check for Functions: \sin(...), \cos(...), \sqrt{...}
        if tokens[0] in self.TRIG_FUNCTIONS or tokens[0] in self.COMMON_FUNCTIONS:
            return self._parse_function(tokens)

        # 7. Check for Parentheses or Braces wrapping entire expression: { ... } or ( ... )
        if (tokens[0] == "{" and tokens[-1] == "}") or (tokens[0] == "(" and tokens[-1] == ")"):
            if find_matching_brace(tokens, 0) == len(tokens) - 1:
                return self._parse_expression(tokens[1:-1])

        # 8. Unary minus or plus: -x or +x
        if tokens[0] in ("+", "-"):
            operand_node = self._parse_expression(tokens[1:])
            node = MathTreeNode(f"unary_{tokens[0]}", node_type="operator")
            node.add_child(operand_node, role="arg")
            return node

        # 9. Implicit multiplication (e.g. 2 x, x y, a (b+c))
        if len(tokens) > 1:
            # If multiple tokens left without explicit operator, combine them as an implicit product
            first_term_end = 1
            if tokens[0] in ("{", "("):
                first_term_end = find_matching_brace(tokens, 0) + 1
            elif tokens[0] in (r"\frac", r"\cfrac"):
                # Skip two braced groups
                b1 = 1 if len(tokens) > 1 and tokens[1] == "{" else 1
                b1_end = find_matching_brace(tokens, b1)
                b2 = b1_end + 1
                if b2 < len(tokens) and tokens[b2] == "{":
                    first_term_end = find_matching_brace(tokens, b2) + 1
            
            if 0 < first_term_end < len(tokens):
                t1 = self._parse_expression(tokens[:first_term_end])
                t2 = self._parse_expression(tokens[first_term_end:])
                node = MathTreeNode("*", node_type="operator")
                node.add_child(t1, role="left")
                node.add_child(t2, role="right")
                return node

        # 10. Base Atomic Token (Variable or Number)
        val = tokens[0]
        if re.match(r"^\d+(?:\.\d+)?$", val):
            return MathTreeNode(val, node_type="number")
        return MathTreeNode(val, node_type="variable")

    def _parse_fraction(self, tokens: List[str]) -> MathTreeNode:
        """Parses \\frac{num}{den} into a fraction tree node."""
        frac_label = tokens[0]
        idx = 1
        num_tokens = []
        den_tokens = []

        # Numerator group
        if idx < len(tokens) and tokens[idx] == "{":
            closing = find_matching_brace(tokens, idx)
            num_tokens = tokens[idx + 1:closing]
            idx = closing + 1
        elif idx < len(tokens):
            num_tokens = [tokens[idx]]
            idx += 1

        # Denominator group
        if idx < len(tokens) and tokens[idx] == "{":
            closing = find_matching_brace(tokens, idx)
            den_tokens = tokens[idx + 1:closing]
        elif idx < len(tokens):
            den_tokens = [tokens[idx]]

        num_node = self._parse_expression(num_tokens) if num_tokens else MathTreeNode("1", node_type="number")
        den_node = self._parse_expression(den_tokens) if den_tokens else MathTreeNode("1", node_type="number")

        node = MathTreeNode(r"\frac", node_type="layout")
        node.add_child(num_node, role="numerator")
        node.add_child(den_node, role="denominator")
        return node

    def _parse_function(self, tokens: List[str]) -> MathTreeNode:
        """Parses function call \\sin(x) or \\sqrt{x}."""
        fn_label = tokens[0]
        arg_tokens = tokens[1:]

        # Strip surrounding braces or parentheses if any
        if arg_tokens and (arg_tokens[0] in ("{", "(") and arg_tokens[-1] in ("}", ")")):
            if find_matching_brace(arg_tokens, 0) == len(arg_tokens) - 1:
                arg_tokens = arg_tokens[1:-1]

        arg_node = self._parse_expression(arg_tokens) if arg_tokens else MathTreeNode("EMPTY", node_type="variable")
        
        node_type = "trigonometric" if fn_label in self.TRIG_FUNCTIONS else "function"
        node = MathTreeNode(fn_label, node_type=node_type)
        node.add_child(arg_node, role="arg")
        return node


def parse_latex_to_tree(latex_str: str) -> MathTreeNode:
    """Convenience helper to parse a LaTeX string into an Operator / Layout Tree."""
    builder = LaTeXTreeBuilder()
    return builder.parse(latex_str)

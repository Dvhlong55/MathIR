"""
Symbol-pair tuple extraction and structural conflict detection (Tangent-S baseline).
Course: SC203 - Scientific Method (APCS - HCMUS)
Research Group 2: Mathematical Information Retrieval (MathIR)

Implements:
1. Tangent-S symbol-pair tuple extraction: (Parent_Symbol, Child_Symbol, Relationship/Role).
2. Tuple similarity metrics (Jaccard & Dice coefficients).
3. Adversarial conflict detection (Ratio Inversion, Polarity Reversal, Trig Orthogonality)
   to supply Stage 2 Graded Structural Penalty Gate.
"""

from typing import List, Set, Dict, Any, Optional, Tuple
from .tree_parser import MathTreeNode


def extract_symbol_tuples(node: MathTreeNode, tuples: Optional[List[str]] = None) -> List[str]:
    """Recursively extracts Tangent-S symbol pair tuples from a MathTreeNode.
    
    Format: '(Parent_Symbol, Child_Symbol, Role)'
    Example for \\frac{x}{y}:
        '(\\frac, x, NUMERATOR)'
        '(\\frac, y, DENOMINATOR)'
    """
    if tuples is None:
        tuples = []
    
    if not node:
        return tuples

    for child in node.children:
        # Standard Tangent-S tuple: (Parent, Child, Spatial_Role)
        role_label = child.role.upper() if child.role else "CHILD"
        tuple_str = f"({node.label}, {child.label}, {role_label})"
        tuples.append(tuple_str)
        # Recurse down
        extract_symbol_tuples(child, tuples)

    return tuples


def compute_tuple_similarity(
    tuples_a: List[str],
    tuples_b: List[str],
    metric: str = "jaccard"
) -> float:
    """Computes similarity coefficient between two sets of symbol-pair tuples."""
    set_a = set(tuples_a)
    set_b = set(tuples_b)

    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0

    intersection = len(set_a.intersection(set_b))

    if metric == "jaccard":
        union = len(set_a.union(set_b))
        return intersection / union if union > 0 else 0.0
    elif metric == "dice":
        total = len(set_a) + len(set_b)
        return (2.0 * intersection) / total if total > 0 else 0.0
    elif metric == "overlap":
        # Cosine/Geometric overlap used in classic Tangent-S
        denom = (len(set_a) * len(set_b)) ** 0.5
        return intersection / denom if denom > 0 else 0.0
    else:
        raise ValueError(f"Unknown metric '{metric}'. Choose 'jaccard', 'dice', or 'overlap'.")


def _find_nodes_by_type(node: MathTreeNode, target_types: Set[str], found: Optional[List[MathTreeNode]] = None) -> List[MathTreeNode]:
    if found is None:
        found = []
    if node.node_type in target_types:
        found.append(node)
    for child in node.children:
        _find_nodes_by_type(child, target_types, found)
    return found


def _find_nodes_by_label(node: MathTreeNode, target_labels: Set[str], found: Optional[List[MathTreeNode]] = None) -> List[MathTreeNode]:
    if found is None:
        found = []
    if node.label in target_labels:
        found.append(node)
    for child in node.children:
        _find_nodes_by_label(child, target_labels, found)
    return found


def _get_subtree_signature(node: MathTreeNode) -> str:
    """Generates a serialized string signature of a subtree for quick equality comparison."""
    if not node.children:
        return node.label
    children_sigs = ",".join(f"{c.role}:{_get_subtree_signature(c)}" for c in node.children)
    return f"{node.label}({children_sigs})"


def detect_structural_conflicts(tree_a: MathTreeNode, tree_b: MathTreeNode) -> Dict[str, Any]:
    """Inspects structural AST representations for critical adversarial conflict modes.
    
    Identifies:
    1. Polarity Flip (x > 0 vs. x < 0, or a >= 2 vs a <= 2).
    2. Ratio Inversion (\\frac{x}{y} vs. \\frac{y}{x}).
    3. Trig Orthogonality (\\sin(x) vs. \\cos(x)).
    4. Exponent Direction (e^x vs. e^{-x}).
    
    Returns structured diagnosis and a recommended penalty factor \\lambda \\in [0.4, 0.6].
    """
    tuples_a = extract_symbol_tuples(tree_a)
    tuples_b = extract_symbol_tuples(tree_b)
    overlap = compute_tuple_similarity(tuples_a, tuples_b, metric="overlap")
    jaccard = compute_tuple_similarity(tuples_a, tuples_b, metric="jaccard")

    is_polarity_flipped = False
    is_ratio_inverted = False
    is_trig_mismatch = False
    is_exponent_flipped = False

    # 1. Check for Opposite Relational Operators
    OPPOSITE_RELATIONS = {
        ">": "<",
        "<": ">",
        r"\ge": r"\le",
        r"\geq": r"\leq",
        r"\le": r"\ge",
        r"\leq": r"\geq",
        ">=": "<=",
        "<=": ">="
    }
    
    rels_a = _find_nodes_by_type(tree_a, {"relation"})
    rels_b = _find_nodes_by_type(tree_b, {"relation"})

    if rels_a and rels_b:
        for ra in rels_a:
            for rb in rels_b:
                if OPPOSITE_RELATIONS.get(ra.label) == rb.label:
                    is_polarity_flipped = True

    # 2. Check for Fraction / Ratio Inversion (\frac{x}{y} vs \frac{y}{x})
    fracs_a = _find_nodes_by_label(tree_a, {r"\frac", r"\cfrac"})
    fracs_b = _find_nodes_by_label(tree_b, {r"\frac", r"\cfrac"})

    for fa in fracs_a:
        num_a = next((c for c in fa.children if c.role == "numerator"), None)
        den_a = next((c for c in fa.children if c.role == "denominator"), None)
        if not num_a or not den_a:
            continue
        sig_num_a = _get_subtree_signature(num_a)
        sig_den_a = _get_subtree_signature(den_a)

        for fb in fracs_b:
            num_b = next((c for c in fb.children if c.role == "numerator"), None)
            den_b = next((c for c in fb.children if c.role == "denominator"), None)
            if not num_b or not den_b:
                continue
            sig_num_b = _get_subtree_signature(num_b)
            sig_den_b = _get_subtree_signature(den_b)

            # Inverted if numerator of A matches denominator of B AND denominator of A matches numerator of B
            if sig_num_a == sig_den_b and sig_den_a == sig_num_b and sig_num_a != sig_den_a:
                is_ratio_inverted = True

    # 3. Check for Trigonometric Orthogonal Confusion (\sin vs \cos)
    trig_a = _find_nodes_by_type(tree_a, {"trigonometric"})
    trig_b = _find_nodes_by_type(tree_b, {"trigonometric"})

    ORTHO_TRIG = {
        (r"\sin", r"\cos"), (r"\cos", r"\sin"),
        ("sin", "cos"), ("cos", "sin")
    }
    for ta in trig_a:
        for tb in trig_b:
            if (ta.label, tb.label) in ORTHO_TRIG:
                # If they share similar argument, they are functionally orthogonal
                arg_a = _get_subtree_signature(ta.children[0]) if ta.children else ""
                arg_b = _get_subtree_signature(tb.children[0]) if tb.children else ""
                if arg_a == arg_b and len(arg_a) > 0:
                    is_trig_mismatch = True

    # 4. Check for Exponent Sign Inversion (e^x vs e^{-x})
    pows_a = _find_nodes_by_label(tree_a, {"^"})
    pows_b = _find_nodes_by_label(tree_b, {"^"})

    for pa in pows_a:
        for pb in pows_b:
            base_a = _get_subtree_signature(pa.children[0]) if len(pa.children) > 0 else ""
            base_b = _get_subtree_signature(pb.children[0]) if len(pb.children) > 0 else ""
            exp_a = pa.children[1] if len(pa.children) > 1 else None
            exp_b = pb.children[1] if len(pb.children) > 1 else None

            if base_a == base_b and exp_a and exp_b:
                if (exp_a.label == "unary_-" and exp_b.label != "unary_-") or (exp_b.label == "unary_-" and exp_a.label != "unary_-"):
                    is_exponent_flipped = True

    # Compute suggested penalty factor \lambda \in [0.4, 0.6]
    has_critical_conflict = (is_polarity_flipped or is_ratio_inverted or is_trig_mismatch or is_exponent_flipped)
    if is_ratio_inverted or is_trig_mismatch:
        penalty_factor = 0.4
    elif is_polarity_flipped or is_exponent_flipped:
        penalty_factor = 0.5
    else:
        penalty_factor = 1.0

    return {
        "is_polarity_flipped": is_polarity_flipped,
        "is_ratio_inverted": is_ratio_inverted,
        "is_trig_mismatch": is_trig_mismatch,
        "is_exponent_flipped": is_exponent_flipped,
        "has_critical_conflict": has_critical_conflict,
        "penalty_factor": penalty_factor,
        "ast_overlap": round(overlap, 4),
        "ast_jaccard": round(jaccard, 4),
        "tuples_a_count": len(tuples_a),
        "tuples_b_count": len(tuples_b)
    }

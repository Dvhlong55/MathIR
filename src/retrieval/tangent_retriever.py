"""
Tangent-CFT & Tangent-CFTED Baseline Retriever (Structural & Tree-Edit Distance).
Course: SC203 - Scientific Method (APCS - HCMUS)
Research Group 2: Mathematical Information Retrieval (MathIR)

Implements the official structural baseline from Mansouri et al. (ICTIR '19; CLEF ARQMath '20):
1. Symbol-pair tuple indexation & fast coarse retrieval.
2. Exact Tree Edit Distance (TED via Zhang-Shasha / zss) for fine-grained re-ranking.
3. Graded structural conflict penalty integration for Stage 2.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import math

try:
    import zss
except ImportError:
    zss = None

from ..representation.tree_parser import parse_latex_to_tree, MathTreeNode
from ..representation.tuple_extractor import (
    extract_symbol_tuples,
    compute_tuple_similarity,
    detect_structural_conflicts
)


def count_tree_nodes(node: MathTreeNode) -> int:
    """Recursively counts total nodes in tree."""
    return 1 + sum(count_tree_nodes(c) for c in node.children)


class TangentBaseline:
    """Tangent-CFT / Tangent-CFTED Baseline Retrieval Engine."""

    def __init__(self, use_ted: bool = True, max_ted_candidates: int = 50):
        """Initializes retriever.
        
        Args:
            use_ted: Whether to execute Tree Edit Distance re-ranking on top candidates.
            max_ted_candidates: Maximum number of coarse candidates to refine with TED.
        """
        self.use_ted = use_ted and (zss is not None)
        self.max_ted_candidates = max_ted_candidates
        
        # Indexed documents: doc_id -> metadata
        self.corpus: Dict[str, Dict[str, Any]] = {}
        # Inverted index: tuple_str -> set of doc_ids
        self.inverted_index: Dict[str, Set[str]] = {}
        # Cached pre-parsed trees: doc_id -> MathTreeNode
        self.tree_cache: Dict[str, MathTreeNode] = {}

    def index_corpus(self, corpus_items: List[Dict[str, Any]]) -> None:
        """Indexes a collection of corpus formula items."""
        for item in corpus_items:
            doc_id = str(item.get("doc_id") or item.get("formula_id") or f"doc_{len(self.corpus)}")
            latex = item.get("latex", "").strip()
            if not latex:
                continue

            tree = parse_latex_to_tree(latex)
            tuples = extract_symbol_tuples(tree)

            self.corpus[doc_id] = {
                "doc_id": doc_id,
                "latex": latex,
                "tuples": tuples,
                "node_count": count_tree_nodes(tree),
                "context": item.get("context", item.get("context_title", ""))
            }
            self.tree_cache[doc_id] = tree

            for t in tuples:
                self.inverted_index.setdefault(t, set()).add(doc_id)

    def compute_tree_edit_distance(self, tree_a: MathTreeNode, tree_b: MathTreeNode) -> float:
        """Computes Tree Edit Distance using Zhang-Shasha algorithm."""
        if zss is None:
            # Fallback if zss is not installed
            return 0.0

        def get_children(node: MathTreeNode) -> List[MathTreeNode]:
            return node.children

        def get_label(node: MathTreeNode) -> str:
            return f"{node.label}:{node.role}"

        def label_dist(l1: str, l2: str) -> float:
            return 0.0 if l1 == l2 else 1.0

        return float(zss.simple_distance(
            tree_a, tree_b,
            get_children=get_children,
            get_label=get_label,
            label_dist=label_dist
        ))

    def score_pair(self, query_latex: str, candidate_latex: str) -> Dict[str, Any]:
        """Calculates full structural similarity, TED, and conflict diagnostics between two formulas."""
        query_tree = parse_latex_to_tree(query_latex)
        cand_tree = parse_latex_to_tree(candidate_latex)

        q_tuples = extract_symbol_tuples(query_tree)
        c_tuples = extract_symbol_tuples(cand_tree)

        jaccard = compute_tuple_similarity(q_tuples, c_tuples, metric="jaccard")
        overlap = compute_tuple_similarity(q_tuples, c_tuples, metric="overlap")
        conflict = detect_structural_conflicts(query_tree, cand_tree)

        ted_dist = 0.0
        ted_sim = jaccard
        if self.use_ted:
            q_size = count_tree_nodes(query_tree)
            c_size = count_tree_nodes(cand_tree)
            max_size = max(q_size, c_size)
            if max_size > 0:
                ted_dist = self.compute_tree_edit_distance(query_tree, cand_tree)
                ted_sim = max(0.0, 1.0 - (ted_dist / max_size))

        # Final structural score incorporates the penalty gate factor
        penalty = conflict["penalty_factor"]
        base_score = ted_sim if self.use_ted else overlap
        final_score = round(base_score * penalty, 4)

        return {
            "query_latex": query_latex,
            "candidate_latex": candidate_latex,
            "tuple_jaccard": round(jaccard, 4),
            "tuple_overlap": round(overlap, 4),
            "ted_distance": ted_dist,
            "ted_similarity": round(ted_sim, 4),
            "penalty_factor": penalty,
            "final_score": final_score,
            "conflict_diagnostics": conflict
        }

    def retrieve(self, query_latex: str, top_k: int = 100) -> List[Dict[str, Any]]:
        """Two-Stage Retrieval:
        1. Fast Coarse Tuple Scan: Filters candidate formulas via inverted index.
        2. Fine-grained TED & Penalty Gate: Re-ranks top candidates.
        """
        query_tree = parse_latex_to_tree(query_latex)
        query_tuples = extract_symbol_tuples(query_tree)

        if not query_tuples or not self.corpus:
            return []

        # Stage 1: Coarse candidate scoring via tuple overlap
        candidate_scores: Dict[str, int] = {}
        for t in query_tuples:
            for doc_id in self.inverted_index.get(t, []):
                candidate_scores[doc_id] = candidate_scores.get(doc_id, 0) + 1

        if not candidate_scores:
            return []

        # Sort coarse candidates by tuple hit count
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        coarse_top = sorted_candidates[:self.max_ted_candidates]

        # Stage 2: Fine-grained Tree Edit Distance and Penalty verification
        results = []
        for doc_id, hit_count in coarse_top:
            item = self.corpus[doc_id]
            cand_tree = self.tree_cache[doc_id]
            cand_latex = item["latex"]

            score_info = self.score_pair(query_latex, cand_latex)
            results.append({
                "doc_id": doc_id,
                "latex": cand_latex,
                "score": score_info["final_score"],
                "raw_ted_sim": score_info["ted_similarity"],
                "tuple_overlap": score_info["tuple_overlap"],
                "penalty_factor": score_info["penalty_factor"],
                "conflict_info": score_info["conflict_diagnostics"],
                "context": item.get("context", "")
            })

        # Final sort by penalization-calibrated structural score
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

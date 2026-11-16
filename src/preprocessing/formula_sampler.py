"""
Formula sampler and category tagger for MathIR benchmark datasets.
Course: SC203 - Scientific Method (APCS - HCMUS)
Research Group 2: Mathematical Information Retrieval (MathIR)

Provides structured sampling, formula tagging (inequalities, fractions, calculus, trig),
and creates balanced subsets suitable for 2-stage / 3-stage neuro-symbolic benchmarking.
"""

import re
from typing import Dict, List, Any, Optional, Set


def tag_formula_features(latex: str) -> List[str]:
    """Identifies key mathematical structures in a formula for ablation testing.
    
    Helps isolate failure modes analyzed in the Week 5 Benchmark Report:
    - Inversion blindness (fractions)
    - Polarity blindness (inequalities)
    - Trigonometric orthogonality (sin, cos)
    - CAS fragility on ellipsis sequences (1 + 2 + ... + n)
    """
    tags = []
    if not latex:
        return tags
    
    s = latex.lower()
    
    # 1. Relational & Polarity
    if any(op in latex for op in [">", "<", r"\ge", r"\le", r"\geq", r"\leq"]):
        tags.append("inequality")
    if "=" in latex:
        tags.append("equality")
        
    # 2. Fractions & Ratios
    if r"\frac" in latex or r"\cfrac" in latex or "/" in latex:
        tags.append("fraction")
        
    # 3. Trigonometric functions
    if any(fn in s for fn in [r"\sin", r"\cos", r"\tan", r"\cot", "sin(", "cos(", "tan("]):
        tags.append("trigonometric")
        
    # 4. Calculus & Limits
    if any(k in latex for k in [r"\int", r"\lim", r"\partial", r"\sum", r"\prod"]):
        tags.append("calculus")
        
    # 5. Informal ellipsis sequences (CAS crash triggers)
    if any(el in latex for el in ["...", r"\dots", r"\ldots", r"\cdots"]):
        tags.append("ellipsis_sequence")
        
    # 6. Exponentials and Roots
    if "^" in latex or r"\sqrt" in latex or r"\exp" in latex:
        tags.append("power_root")
        
    # 7. Linear Algebra / Matrices
    if any(m in latex for m in ["matrix", "pmatrix", "bmatrix", r"\vec"]):
        tags.append("linear_algebra")
        
    if not tags:
        tags.append("general_algebra")
        
    return tags


class FormulaSampler:
    """Samples and categorizes formulas from benchmark corpora."""

    def __init__(self, corpus: List[Dict[str, Any]], qrels: Optional[Dict[str, Dict[str, float]]] = None):
        """Initializes sampler.
        
        Args:
            corpus: List of corpus dictionaries from MathBenchmarkLoader.
            qrels: Optional dictionary of query-candidate relevance scores.
        """
        self.corpus = corpus
        self.qrels = qrels or {}

    def extract_distinct_formulas(self) -> List[Dict[str, Any]]:
        """Extracts all distinct valid formulas across documents, tagging their features."""
        seen_formulas: Set[str] = set()
        catalog: List[Dict[str, Any]] = []

        # Identify all document IDs that have positive relevance in qrels
        relevant_doc_ids: Set[str] = set()
        for qid, rel_map in self.qrels.items():
            for doc_id, score in rel_map.items():
                if score > 0:
                    relevant_doc_ids.add(doc_id)

        for item in self.corpus:
            doc_id = item.get("doc_id", "")
            is_gold_relevant = doc_id in relevant_doc_ids
            
            # 1. Primary document formula
            candidates = [item.get("latex", "")]
            # 2. Additional inline formulas in context
            candidates.extend(item.get("all_formulas", []))

            for formula in candidates:
                if not formula or len(formula) < 2:
                    continue
                # Normalize spaces
                f_norm = " ".join(formula.split())
                if f_norm in seen_formulas:
                    continue
                seen_formulas.add(f_norm)

                tags = tag_formula_features(f_norm)
                catalog.append({
                    "formula_id": f"f_{len(catalog) + 1:07d}",
                    "doc_id": doc_id,
                    "latex": f_norm,
                    "tags": tags,
                    "is_benchmark_relevant": is_gold_relevant,
                    "context_title": item.get("title", ""),
                    "source_dataset": item.get("source_dataset", "unknown")
                })

        return catalog

    def sample(self, target_size: int = 50000, seed: int = 42) -> List[Dict[str, Any]]:
        """Samples up to target_size formulas, prioritizing gold relevant formulas and category diversity."""
        import random
        rng = random.Random(seed)
        
        catalog = self.extract_distinct_formulas()
        if len(catalog) <= target_size:
            return catalog

        # Always retain gold-standard relevant formulas
        priority = [f for f in catalog if f["is_benchmark_relevant"]]
        remaining = [f for f in catalog if not f["is_benchmark_relevant"]]

        # Stratified sampling over categories
        by_category: Dict[str, List[Dict[str, Any]]] = {}
        for item in remaining:
            primary_tag = item["tags"][0] if item["tags"] else "general"
            by_category.setdefault(primary_tag, []).append(item)

        slots_left = target_size - len(priority)
        selected_remaining = []
        
        # Round-robin allocation across mathematical feature categories
        cats = list(by_category.keys())
        for c in cats:
            rng.shuffle(by_category[c])

        while slots_left > 0 and any(by_category.values()):
            for c in cats:
                if by_category[c] and slots_left > 0:
                    selected_remaining.append(by_category[c].pop())
                    slots_left -= 1

        sampled = priority + selected_remaining
        rng.shuffle(sampled)
        return sampled

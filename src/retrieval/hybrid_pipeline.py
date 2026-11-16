"""
Three-Stage Neuro-Symbolic Hybrid Pipeline for MathIR.
Integrates:
  - Stage 1: Query normalization (VN->EN) & coarse neural recall (MathBERT + FAISS)
  - Stage 2: Structural AST/OPT verification & soft penalty gate (Tangent-S/CFT + Gate)
  - Stage 3: Timeout-guarded SymPy CAS bounded score boost
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from src.retrieval.structural_gate import evaluate_structural_gate


@dataclass
class CandidateItem:
    doc_id: str
    text: str
    neural_score: float
    ast_overlap: Optional[float] = None
    cas_equivalent: Optional[bool] = None
    final_score: float = 0.0
    gate_reasons: List[str] = field(default_factory=list)


class HybridMathIRPipeline:
    """Configurable 3-stage Neuro-Symbolic Hybrid retrieval pipeline."""

    def __init__(
        self,
        lambda_penalty: float = 0.5,
        cas_boost: float = 0.25,
        use_stage2: bool = True,
        use_stage3: bool = True,
    ):
        self.lambda_penalty = lambda_penalty
        self.cas_boost = cas_boost
        self.use_stage2 = use_stage2
        self.use_stage3 = use_stage3

    def score_candidate(self, query: str, item: CandidateItem) -> CandidateItem:
        score = item.neural_score

        if self.use_stage2:
            gate = evaluate_structural_gate(
                query=query,
                candidate=item.text,
                lambda_penalty=self.lambda_penalty,
                ast_overlap=item.ast_overlap,
            )
            score *= gate.penalty_factor
            item.gate_reasons = gate.reasons

        if self.use_stage3 and item.cas_equivalent is True:
            score = min(1.0, score + self.cas_boost)

        item.final_score = score
        return item

    def rerank(
        self,
        query: str,
        candidates: List[CandidateItem],
        normalizer_fn: Optional[Callable[[str], str]] = None,
    ) -> List[CandidateItem]:
        """Rerank Stage 1 candidates through Stage 2 and Stage 3."""
        norm_query = normalizer_fn(query) if normalizer_fn else query
        scored = [self.score_candidate(norm_query, c) for c in candidates]
        scored.sort(key=lambda x: x.final_score, reverse=True)
        return scored

"""Retrieval package for MathIR (Stage 1, Stage 2, Stage 3 & Hybrid)."""

from .tangent_retriever import TangentBaseline
from .faiss_index import MathFAISSIndex
from .stage1_pipeline import Stage1RetrievalPipeline, MathBERTEmbedder, encode_resilient_query
from .symbolic_verifier import (
    TimeoutSymbolicVerifier,
    VerificationResult,
    VerificationStats,
    VerificationStatus,
    apply_bounded_boost,
)

__all__ = [
    "TangentBaseline",
    "MathFAISSIndex",
    "Stage1RetrievalPipeline",
    "MathBERTEmbedder",
    "encode_resilient_query",
    "TimeoutSymbolicVerifier",
    "VerificationResult",
    "VerificationStats",
    "VerificationStatus",
    "apply_bounded_boost",
]

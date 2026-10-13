"""Pipeline configuration for hybrid fusion and reranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FusionMethod = Literal["rrf", "normalized"]


@dataclass(frozen=True, slots=True)
class FusionConfig:
    """Tunables for the hybrid search -> fusion -> rerank pipeline.

    Attributes:
        candidate_limit: Number of hits fetched *per stream* before fusion.
            This bounds the reranker's input size and therefore its latency/cost.
        fusion_method: ``"rrf"`` (rank-based, robust to score scale) or
            ``"normalized"`` (min-max per-stream score blending).
        rrf_k: RRF damping constant; smaller values weight top ranks more.
        vector_weight: Relative weight of the dense stream during fusion.
        lexical_weight: Relative weight of the sparse/BM25 stream during fusion.
        relevance_threshold: Minimum cross-encoder relevance score (0..1) a
            chunk must reach to survive filtering.
        top_n: Maximum number of chunks in the final package sent to the LLM.
    """

    candidate_limit: int = 50
    fusion_method: FusionMethod = "rrf"
    rrf_k: int = 60
    vector_weight: float = 1.0
    lexical_weight: float = 1.0
    relevance_threshold: float = 0.3
    top_n: int = 5

    def __post_init__(self) -> None:
        if self.candidate_limit < 1:
            raise ValueError("candidate_limit must be >= 1")
        if self.rrf_k < 1:
            raise ValueError("rrf_k must be >= 1")
        if not 0.0 <= self.relevance_threshold <= 1.0:
            raise ValueError("relevance_threshold must be within [0.0, 1.0]")
        if self.top_n < 1:
            raise ValueError("top_n must be >= 1")
        if self.vector_weight < 0.0 or self.lexical_weight < 0.0:
            raise ValueError("stream weights must be >= 0")
        if self.vector_weight == 0.0 and self.lexical_weight == 0.0:
            raise ValueError("at least one stream weight must be > 0")
        if self.fusion_method not in ("rrf", "normalized"):
            raise ValueError(f"unknown fusion_method: {self.fusion_method!r}")

    @property
    def weights(self) -> dict[str, float]:
        return {"vector": self.vector_weight, "lexical": self.lexical_weight}

"""Reranker abstraction: swap local cross-encoders for cloud APIs freely."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from models import FusedChunk, RerankedChunk

__all__ = ["Reranker", "RerankerError"]


class RerankerError(RuntimeError):
    """Raised when a reranking backend is misconfigured or rejects a request."""


class Reranker(ABC):
    """Scores fused candidates for exact (query, passage) relevance.

    Contract for implementations:

    * ``relevance_score`` must be calibrated to ``[0.0, 1.0]`` so the pipeline's
      ``relevance_threshold`` means the same thing across backends.
    * Return an **empty list** for empty input without contacting the backend.
    * Return candidates sorted by ``relevance_score`` descending.
    * Failures raise :class:`RerankerError` (the engine degrades to fusion order).
    """

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: Sequence[FusedChunk],
    ) -> list[RerankedChunk]:
        """Score every candidate against ``query`` and return them best-first."""

    async def aclose(self) -> None:
        """Release backend resources (HTTP clients, model handles). No-op by default."""
        return None

    @staticmethod
    def _pack(candidate: FusedChunk, score: float) -> RerankedChunk:
        """Build a :class:`RerankedChunk` with the score clamped to [0, 1]."""
        return RerankedChunk(
            chunk=candidate.chunk,
            relevance_score=max(0.0, min(1.0, float(score))),
            fusion_score=candidate.fusion_score,
            streams=candidate.streams,
        )

    @staticmethod
    def _sort(scored: list[RerankedChunk]) -> list[RerankedChunk]:
        scored.sort(key=lambda item: (-item.relevance_score, item.chunk.id))
        return scored

"""Dependency-free reranker used for demos, tests and rerank-ablation runs.

Scores each chunk by query-term coverage: the fraction of query tokens that
appear in the chunk. Deterministic, instant, and honest about what it is --
swap it for :class:`~rerank.huggingface.CrossEncoderReranker` in
production.
"""

from __future__ import annotations

from collections.abc import Sequence

from embedding import tokenize
from models import FusedChunk, RerankedChunk
from rerank.base import Reranker

__all__ = ["TermOverlapReranker"]


class TermOverlapReranker(Reranker):
    """Approximates relevance with query-token recall over the candidate text."""

    async def rerank(
        self,
        query: str,
        candidates: Sequence[FusedChunk],
    ) -> list[RerankedChunk]:
        if not candidates:
            return []
        query_tokens = set(tokenize(query))
        scored: list[RerankedChunk] = []
        for candidate in candidates:
            if not query_tokens:
                score = 0.0
            else:
                chunk_tokens = set(tokenize(candidate.chunk.text))
                score = len(query_tokens & chunk_tokens) / len(query_tokens)
            scored.append(self._pack(candidate, score))
        return self._sort(scored)

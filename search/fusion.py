"""Score fusion for the dense and sparse result streams.

Two strategies are provided:

* :func:`reciprocal_rank_fusion` -- rank-based, scale-free, and the default.
  score(c) = sum over streams of  weight_s / (k + rank_s(c))
* :func:`normalized_score_fusion` -- min-max normalizes each stream to [0, 1]
  first, then blends with configured weights. Use this when calibrated scores
  (e.g. cosine similarity vs. ts_rank) should contribute proportionally.

Both deduplicate by chunk id and merge the contributing stream labels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from config import FusionMethod
from models import (
    STREAM_LEXICAL,
    STREAM_ORDER,
    STREAM_VECTOR,
    Chunk,
    FusedChunk,
    SearchHit,
    StreamType,
)

DEFAULT_RRF_K = 60

DEFAULT_WEIGHTS: dict[StreamType, float] = {STREAM_VECTOR: 1.0, STREAM_LEXICAL: 1.0}


class _Accumulator:
    """Mutable per-chunk score accumulator shared by both fusion strategies."""

    __slots__ = ("scores", "chunks", "streams")

    def __init__(self) -> None:
        self.scores: dict[str, float] = {}
        self.chunks: dict[str, Chunk] = {}
        self.streams: dict[str, list[StreamType]] = {}

    def add(self, chunk: Chunk, contribution: float, stream: StreamType) -> None:
        if chunk.id not in self.scores:
            self.scores[chunk.id] = 0.0
            self.chunks[chunk.id] = chunk
            self.streams[chunk.id] = []
        self.scores[chunk.id] += contribution
        if stream not in self.streams[chunk.id]:
            self.streams[chunk.id].append(stream)

    def finalize(self) -> list[FusedChunk]:
        fused = [
            FusedChunk(
                chunk=self.chunks[chunk_id],
                fusion_score=score,
                streams=tuple(
                    stream for stream in STREAM_ORDER if stream in self.streams[chunk_id]
                ),
            )
            for chunk_id, score in self.scores.items()
        ]
        fused.sort(key=lambda item: (-item.fusion_score, item.chunk.id))
        return fused


def _resolve_weights(weights: Mapping[StreamType, float] | None) -> dict[StreamType, float]:
    if weights is None:
        return dict(DEFAULT_WEIGHTS)
    resolved = dict(DEFAULT_WEIGHTS)
    resolved.update({name: float(value) for name, value in weights.items()})
    return resolved


def reciprocal_rank_fusion(
    streams: Mapping[StreamType, Sequence[SearchHit]],
    *,
    k: int = DEFAULT_RRF_K,
    weights: Mapping[StreamType, float] | None = None,
) -> list[FusedChunk]:
    """Fuse streams with weighted Reciprocal Rank Fusion.

    A chunk appearing at rank 1 in both streams scores
    ``w_v/(k+1) + w_l/(k+1)``; a chunk seen by only one stream keeps a
    single, smaller contribution -- naturally promoting agreement between the
    dense and lexical retrievers. ``k`` (default 60) dampens the leverage of
    very top ranks; smaller ``k`` sharpens it.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    resolved = _resolve_weights(weights)
    accumulator = _Accumulator()
    for stream_name in STREAM_ORDER:
        hits = streams.get(stream_name)
        if not hits:
            continue
        weight = resolved.get(stream_name, 1.0)
        if weight == 0.0:
            continue
        for hit in hits:
            accumulator.add(hit.chunk, weight / (k + hit.rank), stream_name)
    return accumulator.finalize()


def normalized_score_fusion(
    streams: Mapping[StreamType, Sequence[SearchHit]],
    *,
    weights: Mapping[StreamType, float] | None = None,
) -> list[FusedChunk]:
    """Fuse streams by min-max normalizing each stream's scores to [0, 1].

    Degenerate inputs are handled explicitly:

    * empty stream -> contributes nothing;
    * all scores identical (common when a stream returns a single hit) ->
      every hit in that stream receives 1.0 instead of dividing by zero.
    """
    resolved = _resolve_weights(weights)
    accumulator = _Accumulator()
    for stream_name in STREAM_ORDER:
        hits = streams.get(stream_name)
        if not hits:
            continue
        weight = resolved.get(stream_name, 1.0)
        if weight == 0.0:
            continue
        raw = [hit.score for hit in hits]
        low, high = min(raw), max(raw)
        span = high - low
        for hit in hits:
            normalized = 1.0 if span < 1e-12 else (hit.score - low) / span
            accumulator.add(hit.chunk, weight * normalized, stream_name)
    return accumulator.finalize()


def fuse_streams(
    streams: Mapping[StreamType, Sequence[SearchHit]],
    *,
    method: FusionMethod = "rrf",
    k: int = DEFAULT_RRF_K,
    weights: Mapping[StreamType, float] | None = None,
) -> list[FusedChunk]:
    """Dispatch to the configured fusion strategy and return best-first results."""
    if method == "rrf":
        return reciprocal_rank_fusion(streams, k=k, weights=weights)
    if method == "normalized":
        return normalized_score_fusion(streams, weights=weights)
    raise ValueError(f"unknown fusion method: {method!r}")

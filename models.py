"""Core data contracts shared across every stage of the RAGRAM_Pro pipeline.

The pipeline flows through these immutable types:

    Chunk -> EmbeddedChunk -> SearchHit -> FusedChunk -> RerankedChunk -> FusionResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

StreamType = Literal["vector", "lexical"]

STREAM_VECTOR: StreamType = "vector"
STREAM_LEXICAL: StreamType = "lexical"
STREAM_ORDER: tuple[StreamType, StreamType] = (STREAM_VECTOR, STREAM_LEXICAL)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A unit of retrievable text plus arbitrary provenance metadata."""

    id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """A chunk paired with its dense embedding, ready for storage upserts."""

    chunk: Chunk
    embedding: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SearchHit:
    """One result returned by a single search stream.

    ``rank`` is 1-based within the stream and is required by RRF.
    ``score`` is stream-native (cosine similarity or ts_rank/BM25) and is
    only meaningful relative to other hits in the same stream.
    """

    chunk: Chunk
    stream: StreamType
    score: float
    rank: int


@dataclass(frozen=True, slots=True)
class FusedChunk:
    """A deduplicated chunk carrying the combined evidence of one or both streams."""

    chunk: Chunk
    fusion_score: float
    streams: tuple[StreamType, ...]


@dataclass(frozen=True, slots=True)
class RerankedChunk:
    """A fused chunk scored for exact relevance by the reranking handler."""

    chunk: Chunk
    relevance_score: float
    fusion_score: float
    streams: tuple[StreamType, ...]


@dataclass(frozen=True, slots=True)
class SearchStats:
    """Observability counters for one end-to-end query."""

    vector_hits: int
    lexical_hits: int
    fused_candidates: int
    reranked: int
    kept: int
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class FusionResult:
    """Final output of the hybrid search + rerank pipeline."""

    query: str
    chunks: tuple[RerankedChunk, ...]
    stats: SearchStats

    @property
    def is_empty(self) -> bool:
        return not self.chunks

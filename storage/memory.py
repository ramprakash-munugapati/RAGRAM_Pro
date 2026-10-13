"""In-memory storage engine: cosine scan + BM25, zero external dependencies.

Serves as the demo/test backend and as a reference implementation of the
:class:`~storage.base.StorageEngine` contract.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from models import Chunk, EmbeddedChunk, SearchHit
from storage.base import StorageEngine, StorageError, build_hits
from storage.bm25 import BM25Index


class InMemoryStorageEngine(StorageEngine):
    """Dict-backed store with exact cosine search and an Okapi BM25 lexical index."""

    def __init__(self) -> None:
        self._records: dict[str, EmbeddedChunk] = {}
        self._bm25: BM25Index | None = None
        self._bm25_chunks: list[Chunk] = []

    async def initialize(self) -> None:
        return None

    async def upsert(self, records: Sequence[EmbeddedChunk]) -> int:
        if not records:
            return 0
        expected_dims = len(records[0].embedding)
        for record in records:
            if len(record.embedding) != expected_dims:
                raise StorageError(
                    "embedding dimension mismatch within one upsert batch "
                    f"({len(record.embedding)} vs {expected_dims})"
                )
            self._records[record.chunk.id] = record
        self._rebuild_lexical_index()
        return len(records)

    async def vector_search(self, embedding: Sequence[float], limit: int) -> list[SearchHit]:
        if limit < 1 or not self._records:
            return []
        query = tuple(float(value) for value in embedding)
        scored: list[tuple[Chunk, float]] = []
        for record in self._records.values():
            similarity = _cosine_similarity(query, record.embedding)
            if similarity is not None:
                scored.append((record.chunk, similarity))
        scored.sort(key=lambda pair: (-pair[1], pair[0].id))
        return build_hits(scored[:limit], "vector")

    async def lexical_search(self, query: str, limit: int) -> list[SearchHit]:
        if limit < 1 or self._bm25 is None:
            return []
        ordered = self._bm25.search(query, limit)
        scored = [(self._bm25_chunks[index], score) for index, score in ordered]
        return build_hits(scored, "lexical")

    async def count(self) -> int:
        return len(self._records)

    async def close(self) -> None:
        self._records.clear()
        self._bm25 = None
        self._bm25_chunks = []

    @property
    def _insertion_order(self) -> list[EmbeddedChunk]:
        return list(self._records.values())

    def _rebuild_lexical_index(self) -> None:
        records = self._insertion_order
        if records:
            self._bm25 = BM25Index.from_texts([record.chunk.text for record in records])
            self._bm25_chunks = [record.chunk for record in records]
        else:
            self._bm25 = None
            self._bm25_chunks = []


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right):
        raise StorageError(
            f"query embedding width {len(left)} does not match stored width {len(right)}"
        )
    dot = 0.0
    norm_left = 0.0
    norm_right = 0.0
    for a, b in zip(left, right):
        dot += a * b
        norm_left += a * a
        norm_right += b * b
    if norm_left == 0.0 or norm_right == 0.0:
        return None
    return dot / (math.sqrt(norm_left) * math.sqrt(norm_right))

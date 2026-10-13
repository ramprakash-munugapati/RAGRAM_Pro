"""Storage engine abstraction: what the hybrid coordinator expects from a backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

from models import Chunk, EmbeddedChunk, SearchHit, StreamType


class StorageError(RuntimeError):
    """Raised when a storage backend fails (connection drop, bad schema, ...)."""


def build_hits(items: Iterable[tuple[Chunk, float]], stream: StreamType) -> list[SearchHit]:
    """Wrap ``(chunk, score)`` pairs into ranked :class:`SearchHit` objects.

    Rank is 1-based and strictly follows the backend's ordering (best first),
    which is exactly what Reciprocal Rank Fusion consumes.
    """
    return [
        SearchHit(chunk=chunk, stream=stream, score=float(score), rank=rank)
        for rank, (chunk, score) in enumerate(items, start=1)
    ]


class StorageEngine(ABC):
    """Persistent store for chunks, dense embeddings and lexical indexes.

    Implementations must guarantee:

    * ``vector_search`` / ``lexical_search`` return hits sorted best-first.
    * Empty result sets are returned as ``[]`` -- never ``None`` and never raised.
    * Infrastructure failures raise :class:`StorageError`.
    """

    @abstractmethod
    async def initialize(self) -> None:
        """Create extensions, tables and indexes. Must be idempotent."""

    @abstractmethod
    async def upsert(self, records: Sequence[EmbeddedChunk]) -> int:
        """Insert or update embedded chunks. Returns the number of records written."""

    @abstractmethod
    async def vector_search(self, embedding: Sequence[float], limit: int) -> list[SearchHit]:
        """Cosine-similarity search over stored embeddings."""

    @abstractmethod
    async def lexical_search(self, query: str, limit: int) -> list[SearchHit]:
        """Full-text keyword search (Postgres tsvector or BM25)."""

    @abstractmethod
    async def count(self) -> int:
        """Number of stored chunks."""

    async def close(self) -> None:
        """Release backend resources. Default is a no-op."""

    async def __aenter__(self) -> "StorageEngine":
        await self.initialize()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

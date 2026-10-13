"""Embedder abstraction used by the storage engine and hybrid coordinator.

``Embedder`` is intentionally tiny: the coordinator only needs query vectors,
and the engine only needs document vectors. Swap implementations freely --
only ``dimensions`` must stay consistent with the Postgres ``vector(dims)``
column created at index initialization time.
"""

from __future__ import annotations

import asyncio
import math
import re
import zlib
from abc import ABC, abstractmethod
from collections.abc import Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class EmbeddingError(RuntimeError):
    """Raised when a backend fails to produce embeddings."""


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer shared by demo embedders and BM25."""
    return _TOKEN_RE.findall(text.lower())


class Embedder(ABC):
    """Abstract dense-vector provider."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Width of the produced embedding vectors."""

    @abstractmethod
    async def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Embed a batch of chunks for indexing. Order must match ``texts``."""

    @abstractmethod
    async def embed_query(self, text: str) -> tuple[float, ...]:
        """Embed a single user query for the dense search stream."""


class HashingEmbedder(Embedder):
    """Dependency-free deterministic embedder using the hashing trick.

    Produces L2-normalized bag-of-token (+ bigram) vectors so cosine similarity
    behaves like weighted lexical overlap. Intended for demos, tests and CI --
    production deployments should use ``SentenceTransformerEmbedder``.
    """

    def __init__(
        self,
        dimensions: int = 256,
        *,
        use_bigrams: bool = True,
        seed: int = 0xA5A5,
    ) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be >= 8")
        self._dimensions = dimensions
        self._use_bigrams = use_bigrams
        self._seed = seed

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _bucket(self, token: str) -> int:
        # crc32 is stable across processes (unlike built-in hash()).
        return zlib.crc32(f"{self._seed}:{token}".encode("utf-8")) % self._dimensions

    async def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [self._embed_one(text) for text in texts]

    async def embed_query(self, text: str) -> tuple[float, ...]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> tuple[float, ...]:
        tokens = tokenize(text)
        vector = [0.0] * self._dimensions
        for token in tokens:
            vector[self._bucket(token)] += 1.0
        if self._use_bigrams:
            for left, right in zip(tokens, tokens[1:]):
                vector[self._bucket(f"{left}\u2082{right}")] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return tuple(vector)
        return tuple(value / norm for value in vector)


class SentenceTransformerEmbedder(Embedder):
    """Production embedder backed by a ``sentence-transformers`` model.

    CPU/GPU inference runs in a worker thread so the event loop stays responsive.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        device: str | None = None,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise EmbeddingError(
                "sentence-transformers is not installed. "
                "Install it with: pip install 'ragram-pro[local-rerank]'"
            ) from exc
        self._model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name
        self._batch_size = batch_size
        self._normalize = normalize
        self._dimensions = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_documents(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        try:
            matrix = await asyncio.to_thread(
                self._model.encode,
                list(texts),
                batch_size=self._batch_size,
                normalize_embeddings=self._normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # pragma: no cover - model dependent
            raise EmbeddingError(f"document embedding failed: {exc}") from exc
        return [tuple(float(value) for value in row) for row in matrix]

    async def embed_query(self, text: str) -> tuple[float, ...]:
        vectors = await self.embed_documents([text])
        return vectors[0]

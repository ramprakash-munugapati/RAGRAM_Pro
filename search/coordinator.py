"""Hybrid search coordinator: runs the dense and sparse streams concurrently.

Both streams are dispatched with ``asyncio.gather`` so their latencies overlap
(e.g. an HNSW scan and a GIN tsvector scan execute concurrently against the
same pool). Each stream is individually fault-isolated: if one backend raises
or returns zero rows, the coordinator logs it and continues with the surviving
stream -- a hybrid query can never fail *because* only one retriever worked.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable

from config import FusionConfig
from embedding import Embedder
from models import SearchHit, StreamType
from storage.base import StorageEngine

logger = logging.getLogger("ragram.search.coordinator")


class HybridSearchCoordinator:
    """Executes vector similarity search and lexical keyword search in parallel."""

    def __init__(
        self,
        storage: StorageEngine,
        embedder: Embedder,
        config: FusionConfig,
    ) -> None:
        self._storage = storage
        self._embedder = embedder
        self._config = config

    async def search(self, query: str) -> dict[StreamType, list[SearchHit]]:
        """Run both streams concurrently and return hits keyed by stream name.

        Raises only when the query itself cannot be embedded (fatal: without a
        query vector there is no dense stream at all). Storage-level failures
        are absorbed per stream and surfaced through the log.
        """
        limit = self._config.candidate_limit
        query_embedding = await self._embedder.embed_query(query)

        vector_hits, lexical_hits = await asyncio.gather(
            self._guarded_stream("vector", self._storage.vector_search(query_embedding, limit)),
            self._guarded_stream("lexical", self._storage.lexical_search(query, limit)),
        )

        logger.info(
            "hybrid retrieval complete: vector=%d lexical=%d (limit=%d)",
            len(vector_hits),
            len(lexical_hits),
            limit,
        )
        return {"vector": vector_hits, "lexical": lexical_hits}

    @staticmethod
    async def _guarded_stream(
        stream: StreamType,
        operation: Awaitable[list[SearchHit]],
    ) -> list[SearchHit]:
        """Await one stream's coroutine; on failure log it and return ``[]``."""
        try:
            return await operation
        except Exception:
            logger.exception(
                "search stream %r failed; continuing with the remaining stream(s)",
                stream,
            )
            return []

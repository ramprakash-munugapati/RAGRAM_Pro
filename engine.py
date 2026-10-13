"""``RagramFusionEngine`` -- the end-to-end hybrid search + rerank pipeline.

Flow per query::

    query
      |-- embed_query ------------------------------------.
      |-- vector_search (cosine, pgvector) ----.         |
      |-- lexical_search (tsvector / BM25) ---'-- gather '
      |            (fault-isolated per stream)
      v
    RRF / normalized fusion  ->  deduplicated FusedChunk list
      v
    cross-encoder rerank     ->  relevance_score in [0, 1]
      v
    threshold filter  +  top_n truncation
      v
    FusionResult  ->  (optional) evidence-backed generation prompt
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

from config import FusionConfig
from embedding import Embedder
from models import (
    Chunk,
    EmbeddedChunk,
    FusionResult,
    FusedChunk,
    RerankedChunk,
    SearchStats,
)
from prompts import GenerationPrompt, build_generation_prompt
from rerank.base import Reranker
from search.coordinator import HybridSearchCoordinator
from search.fusion import fuse_streams
from storage.base import StorageEngine

logger = logging.getLogger("ragram.engine")

__all__ = ["RagramFusionEngine"]


class RagramFusionEngine:
    """Coordinates storage, embedding, fusion, reranking and prompt assembly.

    Example::

        engine = RagramFusionEngine(storage, embedder, reranker, FusionConfig(top_n=5))
        await engine.initialize()
        await engine.index(chunks)
        result = await engine.search("what is the escalation path?")
        prompt = await engine.build_prompt("what is the escalation path?")
    """

    def __init__(
        self,
        storage: StorageEngine,
        embedder: Embedder,
        reranker: Reranker,
        config: FusionConfig | None = None,
    ) -> None:
        self._storage = storage
        self._embedder = embedder
        self._reranker = reranker
        self._config = config or FusionConfig()
        self._coordinator = HybridSearchCoordinator(storage, embedder, self._config)
        self._initialized = False

    @property
    def config(self) -> FusionConfig:
        return self._config

    # ------------------------------------------------------------- lifecycle

    async def initialize(self) -> None:
        """Create backend schemas/indexes. Idempotent."""
        await self._storage.initialize()
        self._initialized = True
        logger.info(
            "engine initialized (fusion=%s, threshold=%.2f, top_n=%d, candidate_limit=%d)",
            self._config.fusion_method,
            self._config.relevance_threshold,
            self._config.top_n,
            self._config.candidate_limit,
        )

    async def index(self, chunks: Sequence[Chunk]) -> int:
        """Embed and upsert chunks. Chunks with identical ids overwrite."""
        if not chunks:
            return 0
        if not self._initialized:
            await self.initialize()
        embeddings = await self._embedder.embed_documents([chunk.text for chunk in chunks])
        if len(embeddings) != len(chunks):
            raise RuntimeError(
                f"embedder returned {len(embeddings)} vectors for {len(chunks)} chunks"
            )
        records = [
            EmbeddedChunk(chunk=chunk, embedding=tuple(vector))
            for chunk, vector in zip(chunks, embeddings, strict=True)
        ]
        written = await self._storage.upsert(records)
        logger.info("indexed %d chunks", written)
        return written

    async def aclose(self) -> None:
        await self._reranker.aclose()
        await self._storage.close()

    async def __aenter__(self) -> "RagramFusionEngine":
        await self.initialize()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # ---------------------------------------------------------------- query

    async def search(
        self,
        query: str,
        *,
        top_n: int | None = None,
        relevance_threshold: float | None = None,
    ) -> FusionResult:
        """Run the full hybrid -> fuse -> rerank -> filter pipeline.

        ``top_n`` / ``relevance_threshold`` override the configured defaults
        for this call only. Returns an empty :class:`FusionResult` when no
        chunk survives -- callers should treat that as "no usable evidence".
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        effective_top_n = self._config.top_n if top_n is None else top_n
        effective_threshold = (
            self._config.relevance_threshold if relevance_threshold is None else relevance_threshold
        )
        if effective_top_n < 1:
            raise ValueError("top_n must be >= 1")
        if not 0.0 <= effective_threshold <= 1.0:
            raise ValueError("relevance_threshold must be within [0.0, 1.0]")

        started = time.perf_counter()
        query = query.strip()

        # 1) Parallel dual-stream retrieval (fault-isolated inside the coordinator).
        streams = await self._coordinator.search(query)

        # 2) Rank fusion + deduplication across the surviving streams.
        fused = fuse_streams(
            streams,
            method=self._config.fusion_method,
            k=self._config.rrf_k,
            weights=self._config.weights,
        )

        # 3) Cross-encoder reranking (skipped when nothing was fused).
        reranked = await self._rerank(query, fused)

        # 4) Threshold filter, then top_n truncation.
        kept = [item for item in reranked if item.relevance_score >= effective_threshold]
        kept = kept[:effective_top_n]

        stats = SearchStats(
            vector_hits=len(streams["vector"]),
            lexical_hits=len(streams["lexical"]),
            fused_candidates=len(fused),
            reranked=len(reranked),
            kept=len(kept),
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
        logger.info(
            "query resolved: vector=%d lexical=%d fused=%d reranked=%d kept=%d (%.1f ms)",
            stats.vector_hits,
            stats.lexical_hits,
            stats.fused_candidates,
            stats.reranked,
            stats.kept,
            stats.elapsed_ms,
        )
        return FusionResult(query=query, chunks=tuple(kept), stats=stats)

    async def build_prompt(
        self,
        query: str,
        *,
        top_n: int | None = None,
        relevance_threshold: float | None = None,
    ) -> GenerationPrompt:
        """Search and package the surviving chunks into an LLM-ready prompt."""
        result = await self.search(
            query, top_n=top_n, relevance_threshold=relevance_threshold
        )
        return build_generation_prompt(query, result)

    # -------------------------------------------------------------- internals

    async def _rerank(self, query: str, fused: Sequence[FusedChunk]) -> list[RerankedChunk]:
        if not fused:
            logger.info("no fused candidates; skipping rerank stage")
            return []
        try:
            return await self._reranker.rerank(query, fused)
        except Exception:
            logger.exception(
                "reranker failed; falling back to normalized fusion scores as relevance"
            )
            return _fallback_rerank(list(fused))


def _fallback_rerank(fused: list[FusedChunk]) -> list[RerankedChunk]:
    """Degrade gracefully: min-max the fusion scores into pseudo-relevance scores."""
    scores = [item.fusion_score for item in fused]
    low, high = min(scores), max(scores)
    span = high - low
    reranked = [
        RerankedChunk(
            chunk=item.chunk,
            relevance_score=1.0 if span < 1e-12 else (item.fusion_score - low) / span,
            fusion_score=item.fusion_score,
            streams=item.streams,
        )
        for item in fused
    ]
    reranked.sort(key=lambda item: (-item.relevance_score, item.chunk.id))
    return reranked

"""PostgreSQL storage engine: pgvector cosine search + tsvector full-text search.

Schema (created idempotently by :meth:`PostgresStorageEngine.initialize`)::

    CREATE EXTENSION vector;
    CREATE TABLE ragram_chunks (
        id       TEXT PRIMARY KEY,
        content  TEXT NOT NULL,
        embedding vector(dims) NOT NULL,
        metadata JSONB NOT NULL DEFAULT '{}',
        tsv      tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
    );
    -- GIN index on tsv, HNSW index on embedding (best effort)

Design notes:

* Both search streams hit the same table, so chunks are inherently shared and
  the fusion stage only reconciles *scores*, not identities.
* Connection setup retries transient drops with exponential backoff; every
  public coroutine converts driver errors into :class:`StorageError`, which the
  hybrid coordinator catches per-stream so one failure never kills the query.
* ``asyncpg`` / ``pgvector`` are imported lazily; the rest of the package works
  without them (e.g. with the in-memory backend).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, TypeVar

from models import Chunk, EmbeddedChunk, SearchHit
from storage.base import StorageEngine, StorageError, build_hits

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger("ragram.storage.postgres")

T = TypeVar("T")

_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
_QUOTED_SEGMENT_RE = re.compile(r'"[^"]*"|\'[^\']*\'')
_EDGE_NOISE_RE = re.compile(r"^[\'\"-]+|[\'\"-]+$")


def _to_websearch_or(query: str) -> str:
    """Turn a raw user query into an OR-joined ``websearch_to_tsquery`` string.

    Recall-oriented on purpose: stage-1 retrieval should not require every
    query term to appear in a document (strict AND semantics), because
    precision is enforced downstream by the fusion + rerank stages. This also
    matches the partial-match scoring of the in-memory BM25 backend.

    * Double/single-quoted phrases are preserved as atomic segments.
    * Edge quotes/hyphens are stripped from bare tokens so a stray ``-`` can
      never turn into websearch's negation operator (``a - b`` -> ``!'b'``).
    * Stopwords are dropped by PostgreSQL itself; an all-stopword or empty
      result yields an empty string, i.e. a legitimate zero-match response.
    """
    segments: list[str] = []
    for match in _QUOTED_SEGMENT_RE.finditer(query):
        phrase = match.group(0).strip()
        if len(phrase) > 2:
            segments.append(phrase)
    residual = _QUOTED_SEGMENT_RE.sub(" ", query)
    for token in residual.split():
        token = _EDGE_NOISE_RE.sub("", token)
        if token:
            segments.append(token)
    return " OR ".join(segments)


class PostgresStorageEngine(StorageEngine):
    """Async PostgreSQL backend backed by ``asyncpg`` + the ``pgvector`` extension."""

    def __init__(
        self,
        dsn: str,
        *,
        dimensions: int,
        table: str = "ragram_chunks",
        tsconfig: str = "english",
        pool_min_size: int = 1,
        pool_max_size: int = 10,
        connect_retries: int = 3,
        retry_backoff: float = 0.5,
        command_timeout: float = 30.0,
    ) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be >= 1")
        if not _IDENTIFIER_RE.match(table):
            raise ValueError(
                f"invalid table name {table!r}: only lowercase letters, digits and underscores"
            )
        if not _IDENTIFIER_RE.match(tsconfig):
            raise ValueError(f"invalid tsconfig {tsconfig!r}")
        import importlib.util

        missing = [
            name
            for name in ("asyncpg", "numpy", "pgvector.asyncpg")
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            raise StorageError(
                "PostgresStorageEngine requires asyncpg, numpy and pgvector "
                f"(missing: {', '.join(missing)}). "
                "Install with: pip install 'ragram-pro[postgres]'"
            )

        self._dsn = dsn
        self._dimensions = dimensions
        self._table = table
        self._tsconfig = tsconfig
        self._pool_min_size = pool_min_size
        self._pool_max_size = pool_max_size
        self._connect_retries = max(1, connect_retries)
        self._retry_backoff = retry_backoff
        self._command_timeout = command_timeout
        self._pool: "asyncpg.Pool | None" = None

    # ------------------------------------------------------------------ setup

    async def initialize(self) -> None:
        pool = await self._retry(self._create_pool)
        try:
            async with pool.acquire() as conn:
                await self._create_schema(conn)
        except BaseException:
            await pool.close()  # do not leak the pool when schema setup fails
            raise
        self._pool = pool
        logger.info(
            "postgres storage ready: table=%s dims=%d tsconfig=%s",
            self._table,
            self._dimensions,
            self._tsconfig,
        )

    async def _create_pool(self) -> "asyncpg.Pool":
        import asyncpg
        from pgvector.asyncpg import register_vector

        async def _setup(conn: "asyncpg.Connection") -> None:
            await register_vector(conn)

        try:
            return await asyncpg.create_pool(
                self._dsn,
                min_size=self._pool_min_size,
                max_size=self._pool_max_size,
                command_timeout=self._command_timeout,
                setup=_setup,
            )
        except Exception as exc:
            raise StorageError(f"failed to connect to PostgreSQL: {exc}") from exc

    async def _create_schema(self, conn: "asyncpg.Connection") -> None:
        ddl = f"""
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS {self._table} (
            id        TEXT PRIMARY KEY,
            content   TEXT NOT NULL,
            embedding vector({self._dimensions}) NOT NULL,
            metadata  JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            tsv       tsvector GENERATED ALWAYS AS (to_tsvector('{self._tsconfig}', content)) STORED
        );
        CREATE INDEX IF NOT EXISTS {self._table}_tsv_idx
            ON {self._table} USING GIN (tsv);
        """
        try:
            await conn.execute(ddl)
        except Exception as exc:
            raise StorageError(
                "could not create schema (is the pgvector extension installed and "
                f"does the role have CREATE rights?): {exc}"
            ) from exc
        try:
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self._table}_hnsw_idx "
                f"ON {self._table} USING HNSW (embedding vector_cosine_ops);"
            )
        except Exception as exc:  # HNSW requires PostgreSQL 15+; degrade gracefully.
            logger.warning("HNSW index unavailable, falling back to sequential scan: %s", exc)

    # ------------------------------------------------------------------ write

    async def upsert(self, records: Sequence[EmbeddedChunk]) -> int:
        if not records:
            return 0
        for record in records:
            if len(record.embedding) != self._dimensions:
                raise StorageError(
                    f"embedding width {len(record.embedding)} does not match "
                    f"table vector({self._dimensions}) for chunk {record.chunk.id!r}"
                )
        sql = f"""
            INSERT INTO {self._table} (id, content, embedding, metadata)
            VALUES ($1, $2, $3, $4::jsonb)
            ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata
        """
        parameters = [
            (
                record.chunk.id,
                record.chunk.text,
                _as_pg_vector(record.embedding),
                json.dumps(dict(record.chunk.metadata)),
            )
            for record in records
        ]

        async def _op() -> int:
            pool = self._require_pool()
            async with pool.acquire() as conn:
                async with conn.transaction():
                    await conn.executemany(sql, parameters)
            return len(records)

        written = await self._retry(_op)
        logger.debug("upserted %d chunks into %s", written, self._table)
        return written

    # -------------------------------------------------------------------- read

    async def vector_search(self, embedding: Sequence[float], limit: int) -> list[SearchHit]:
        if limit < 1:
            return []
        if len(embedding) != self._dimensions:
            raise StorageError(
                f"query embedding width {len(embedding)} != table vector({self._dimensions})"
            )
        if not any(abs(float(value)) > 1e-12 for value in embedding):
            # Cosine distance against a zero vector is NaN in pgvector; a
            # zero-valued query vector means "no dense signal", so return an
            # empty stream (matching InMemoryStorageEngine's behavior).
            logger.debug("query embedding is zero-valued; skipping vector stream")
            return []
        sql = f"""
            SELECT id, content, metadata::text AS metadata,
                   1.0 - (embedding <=> $1::vector) AS score
            FROM {self._table}
            ORDER BY embedding <=> $1::vector ASC
            LIMIT $2
        """
        vector = _as_pg_vector(embedding)

        async def _op() -> list["asyncpg.Record"]:
            pool = self._require_pool()
            async with pool.acquire() as conn:
                return await conn.fetch(sql, vector, limit)

        rows = await self._retry(_op)
        return build_hits(_rows_to_chunks(rows), "vector")

    async def lexical_search(self, query: str, limit: int) -> list[SearchHit]:
        if limit < 1:
            return []
        tsquery_text = _to_websearch_or(query)
        if not tsquery_text:
            # Nothing but stopwords/punctuation -- a legitimate zero-match response.
            return []
        sql = f"""
            SELECT c.id, c.content, c.metadata::text AS metadata,
                   ts_rank(c.tsv, q.term) AS score
            FROM {self._table} AS c,
                 websearch_to_tsquery('{self._tsconfig}', $1) AS q(term)
            WHERE c.tsv @@ q.term
            ORDER BY score DESC, c.id ASC
            LIMIT $2
        """

        async def _op() -> list["asyncpg.Record"]:
            pool = self._require_pool()
            async with pool.acquire() as conn:
                return await conn.fetch(sql, tsquery_text, limit)

        rows = await self._retry(_op)
        return build_hits(_rows_to_chunks(rows), "lexical")

    async def count(self) -> int:
        async def _op() -> int:
            pool = self._require_pool()
            async with pool.acquire() as conn:
                return int(await conn.fetchval(f"SELECT count(*) FROM {self._table}"))

        return await self._retry(_op)

    async def close(self) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            await pool.close()

    # --------------------------------------------------------------- internals

    def _require_pool(self) -> "asyncpg.Pool":
        if self._pool is None:
            raise StorageError("storage engine is not initialized; call initialize() first")
        return self._pool

    async def _retry(self, operation: Callable[[], Awaitable[T]]) -> T:
        """Run ``operation`` with retries on transient connection drops."""
        import asyncpg

        connection_errors = (asyncpg.PostgresConnectionError, ConnectionError, OSError)
        last_error: Exception | None = None
        for attempt in range(1, self._connect_retries + 1):
            try:
                return await operation()
            except StorageError:
                raise
            except connection_errors as exc:
                last_error = exc
            except asyncpg.PostgresError as exc:
                raise StorageError(f"database error: {exc}") from exc
            if attempt < self._connect_retries:
                delay = self._retry_backoff * (2 ** (attempt - 1))
                logger.warning(
                    "database operation failed (attempt %d/%d): %s -- retrying in %.2fs",
                    attempt,
                    self._connect_retries,
                    last_error,
                    delay,
                )
                await asyncio.sleep(delay)
        raise StorageError(
            f"database unavailable after {self._connect_retries} attempts: {last_error}"
        ) from last_error


def _as_pg_vector(embedding: Sequence[float]) -> object:
    """Encode a sequence of floats as a pgvector-compatible numpy array."""
    import numpy as np

    return np.asarray(embedding, dtype=np.float32)


def _rows_to_chunks(rows: Sequence["asyncpg.Record"]) -> list[tuple[Chunk, float]]:
    results: list[tuple[Chunk, float]] = []
    for row in rows:
        raw_metadata = row["metadata"]
        try:
            metadata = json.loads(raw_metadata) if raw_metadata else {}
        except (TypeError, ValueError):
            metadata = {"_unparsed_metadata": str(raw_metadata)}
        results.append(
            (Chunk(id=str(row["id"]), text=str(row["content"]), metadata=metadata), float(row["score"]))
        )
    return results

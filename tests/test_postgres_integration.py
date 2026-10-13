"""Live integration test for PostgresStorageEngine (pgvector + tsvector).

Requires a reachable PostgreSQL with the pgvector extension available:

    set DATABASE_URL=postgresql://postgres:password@127.0.0.1:5432/postgres
    python tests/test_postgres_integration.py

Uses a dedicated table (``ragram_it_chunks``) and drops it afterwards.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import FusionConfig  # noqa: E402
from embedding import HashingEmbedder  # noqa: E402
from engine import RagramFusionEngine  # noqa: E402
from models import Chunk  # noqa: E402
from rerank import TermOverlapReranker  # noqa: E402
from storage import PostgresStorageEngine, StorageError  # noqa: E402

DSN = os.environ.get("DATABASE_URL")
TABLE = "ragram_it_chunks"
PASS = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS
    if not condition:
        raise AssertionError(f"FAIL: {name} {detail}")
    PASS += 1
    print(f"  ok: {name}")


CHUNKS = (
    Chunk(
        id="pg-1",
        text="Payment outage escalation path pages the payments on-call engineer via PagerDuty.",
        metadata={"source": "runbook.md", "updated": "2026-08-14"},
    ),
    Chunk(
        id="pg-2",
        text="Refund requests above 5000 USD need dual approval from Finance.",
        metadata={"source": "policy.md"},
    ),
    Chunk(
        id="pg-3",
        text="API rate limit is 1000 requests per minute; excess returns HTTP 429.",
        metadata={},
    ),
)


async def _reset_table() -> None:
    """Drop the scratch table so every run starts from a known state."""
    import asyncpg

    conn = await asyncpg.connect(DSN, timeout=5)
    try:
        await conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
    finally:
        await conn.close()


async def main() -> int:
    if not DSN:
        print("SKIP: DATABASE_URL not set")
        return 0

    embedder = HashingEmbedder(dimensions=64)
    await _reset_table()

    storage = PostgresStorageEngine(DSN, dimensions=embedder.dimensions, table=TABLE)
    await storage.initialize()
    await storage.initialize()  # idempotent
    check("schema initialize is idempotent", True)
    try:
        vectors = await embedder.embed_documents([c.text for c in CHUNKS])
        from models import EmbeddedChunk

        records = [
            EmbeddedChunk(chunk=c, embedding=v) for c, v in zip(CHUNKS, vectors, strict=True)
        ]
        check("upsert returns count", await storage.upsert(records) == 3)
        check("count matches", await storage.count() == 3)

        # Upsert overwrite: same id, new text -> count stable, text updated.
        updated = EmbeddedChunk(
            chunk=Chunk(id="pg-1", text="UPDATED payment outage ladder now pages the incident commander."),
            embedding=vectors[0],
        )
        await storage.upsert([updated])
        check("conflict upsert keeps count", await storage.count() == 3)
        lexical = await storage.lexical_search("incident commander", 5)
        check(
            "updated text is searchable",
            any(h.chunk.id == "pg-1" and "UPDATED" in h.chunk.text for h in lexical),
            str([(h.chunk.id, h.score) for h in lexical]),
        )

        # ------------------------------------------------------- both streams
        query_vec = await embedder.embed_query("payment outage escalation")
        vector_hits = await storage.vector_search(query_vec, 5)
        lexical_hits = await storage.lexical_search("payment outage escalation", 5)
        check("vector stream returns rows", len(vector_hits) > 0)
        check("lexical stream returns rows", len(lexical_hits) > 0)
        check("vector ranks are 1-based ascending", [h.rank for h in vector_hits] == list(range(1, len(vector_hits) + 1)))
        check("vector scores in [-1, 1]", all(-1.0 <= h.score <= 1.01 for h in vector_hits), str([h.score for h in vector_hits]))
        check("lexical scores non-negative", all(h.score >= 0.0 for h in lexical_hits))
        top_lexical = max(lexical_hits, key=lambda h: h.rank)
        check("lexical top hit is the payment chunk", top_lexical.chunk.id == "pg-1", str([(h.chunk.id, h.score) for h in lexical_hits]))

        # Metadata JSONB round-trip (pg-2 untouched; pg-1 overwritten above)
        by_id = {h.chunk.id: h.chunk for h in vector_hits}
        check(
            "metadata JSONB round-trips",
            by_id["pg-2"].metadata.get("source") == "policy.md",
            str(by_id["pg-2"].metadata),
        )
        check("conflict upsert replaces metadata", by_id["pg-1"].metadata == {}, str(by_id["pg-1"].metadata))

        # --------------------------------------------------- edge: no matches
        empty_lexical = await storage.lexical_search("zzz qqq gibberish", 5)
        check("unmatched query -> empty lexical stream", empty_lexical == [])
        empty_lexical2 = await storage.lexical_search(';;; "" ---', 5)
        check("stopword/punctuation-only query -> empty, no crash", empty_lexical2 == [])
        empty_vector = await storage.vector_search([0.0] * embedder.dimensions, 5)
        check(
            "zero query vector -> empty vector stream (no NaN rows)",
            empty_vector == [],
            str([(h.chunk.id, h.score) for h in empty_vector]),
        )

        # -------------------------------------------------- edge: limit zero
        check("limit 0 -> []", await storage.lexical_search("payment", 0) == [])

        # -------------------------------------------------- edge: width mismatch
        try:
            await storage.vector_search([0.0] * 3, 5)
            raise AssertionError("expected StorageError on width mismatch")
        except StorageError:
            check("embedding width mismatch -> StorageError", True)

        # ------------------------------------------------------- full pipeline
        engine = RagramFusionEngine(
            storage,
            embedder,
            TermOverlapReranker(),
            FusionConfig(candidate_limit=10, top_n=2, relevance_threshold=0.2),
        )
        await engine.initialize()
        result = await engine.search("payment outage escalation path")
        check("engine pipeline returns results", not result.is_empty, str(result.stats))
        check("engine top_n respected", len(result.chunks) <= 2)
        check(
            "engine sees both streams over postgres",
            result.stats.vector_hits > 0 and result.stats.lexical_hits > 0,
            str(result.stats),
        )
        check("engine top chunk is payment", result.chunks[0].chunk.id == "pg-1")

        prompt = await engine.build_prompt("payment outage escalation path")
        check("prompt includes postgres chunk", "[C1]" in prompt.user)
        await engine.aclose()
    finally:
        await storage.close()

    # --------------------------------------------------------- error resilience
    broken = PostgresStorageEngine(
        "postgresql://postgres:wrong-password@127.0.0.1:5432/postgres",
        dimensions=8,
        connect_retries=2,
        retry_backoff=0.1,
    )
    try:
        await broken.initialize()
        raise AssertionError("expected StorageError for bad credentials")
    except StorageError as exc:
        check("bad credentials -> StorageError after retries", "attempts" in str(exc) or "connect" in str(exc), str(exc))

    print(f"\nALL {PASS} POSTGRES INTEGRATION CHECKS PASSED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except AssertionError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
    finally:
        # Best-effort cleanup of the scratch table.
        if DSN:

            async def _cleanup() -> None:
                import asyncpg

                conn = await asyncpg.connect(DSN, timeout=5)
                try:
                    await conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
                finally:
                    await conn.close()

            try:
                asyncio.run(_cleanup())
            except Exception:
                pass

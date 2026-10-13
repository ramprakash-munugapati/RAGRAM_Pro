"""Self-contained verification suite for RAGRAM_Pro (stdlib only).

Run with:  python tests/test_pipeline.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import FusionConfig  # noqa: E402
from embedding import HashingEmbedder  # noqa: E402
from engine import RagramFusionEngine  # noqa: E402
from models import Chunk, FusedChunk, FusionResult, SearchHit, SearchStats  # noqa: E402
from prompts import build_generation_prompt  # noqa: E402
from rerank import Reranker, RerankerError, TermOverlapReranker  # noqa: E402
from search import fuse_streams, normalized_score_fusion, reciprocal_rank_fusion  # noqa: E402
from storage import InMemoryStorageEngine  # noqa: E402

PASS = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS
    if not condition:
        raise AssertionError(f"FAIL: {name} {detail}")
    PASS += 1
    print(f"  ok: {name}")


def _hit(chunk_id: str, stream: str, rank: int, score: float = 1.0) -> SearchHit:
    return SearchHit(chunk=Chunk(id=chunk_id, text=f"text {chunk_id}"), stream=stream, score=score, rank=rank)


def test_rrf_math() -> None:
    print("[rrf]")
    streams = {
        "vector": [_hit("a", "vector", 1), _hit("b", "vector", 2)],
        "lexical": [_hit("b", "lexical", 1), _hit("c", "lexical", 2)],
    }
    fused = reciprocal_rank_fusion(streams, k=60)
    by_id = {f.chunk.id: f for f in fused}
    check("dedup to 3 unique chunks", len(fused) == 3)
    check(
        "rrf score = sum 1/(k+rank)",
        abs(by_id["b"].fusion_score - (1 / 62 + 1 / 61)) < 1e-12,
        str(by_id["b"].fusion_score),
    )
    check("both-stream chunk ranks first", fused[0].chunk.id == "b")
    check("b carries both stream labels", set(by_id["b"].streams) == {"vector", "lexical"})
    check("single-stream chunk keeps its label", by_id["a"].streams == ("vector",))


def test_rrf_empty_streams() -> None:
    print("[rrf:empty]")
    check("both empty -> []", reciprocal_rank_fusion({"vector": [], "lexical": []}) == [])
    only_vector = reciprocal_rank_fusion({"vector": [_hit("x", "vector", 1)], "lexical": []})
    check("one empty stream still fuses", len(only_vector) == 1)
    zero_weight = reciprocal_rank_fusion(
        {"vector": [_hit("x", "vector", 1)], "lexical": []},
        weights={"vector": 0.0, "lexical": 1.0},
    )
    check("zero-weight stream contributes nothing", zero_weight == [])


def test_normalized_fusion() -> None:
    print("[normalized]")
    streams = {
        "vector": [_hit("a", "vector", 1, score=0.9), _hit("b", "vector", 2, score=0.5)],
        "lexical": [_hit("c", "lexical", 1, score=5.0)],
    }
    fused = normalized_score_fusion(streams)
    by_id = {f.chunk.id: f for f in fused}
    check("min-max maps top vector hit to 1.0", abs(by_id["a"].fusion_score - 1.0) < 1e-9)
    check("min-max maps lower vector hit to 0.0", abs(by_id["b"].fusion_score - 0.0) < 1e-9)
    check("single-hit stream gets 1.0 (no div-by-zero)", abs(by_id["c"].fusion_score - 1.0) < 1e-9)
    check("tie broken deterministically by id", fused[0].chunk.id in {"a", "c"})


def test_fuse_dispatch() -> None:
    print("[dispatch]")
    check("rrf dispatch", len(fuse_streams({"vector": [_hit("a", "vector", 1)]}, method="rrf")) == 1)
    check(
        "normalized dispatch",
        len(fuse_streams({"vector": [_hit("a", "vector", 1)]}, method="normalized")) == 1,
    )
    try:
        fuse_streams({}, method="magic")  # type: ignore[arg-type]
        raise AssertionError("expected ValueError")
    except ValueError:
        check("unknown method rejected", True)


def test_config_validation() -> None:
    print("[config]")
    for kwargs in (
        {"relevance_threshold": 1.5},
        {"top_n": 0},
        {"candidate_limit": 0},
        {"rrf_k": 0},
        {"vector_weight": 0.0, "lexical_weight": 0.0},
    ):
        try:
            FusionConfig(**kwargs)  # type: ignore[arg-type]
            raise AssertionError(f"expected ValueError for {kwargs}")
        except ValueError:
            pass
    check("invalid configs rejected", True)
    check("defaults load", FusionConfig().fusion_method == "rrf")


class _FailingReranker(Reranker):
    async def rerank(self, query, candidates):
        raise RerankerError("simulated backend outage")


async def test_engine_pipeline() -> None:
    print("[engine]")
    storage = InMemoryStorageEngine()
    embedder = HashingEmbedder(dimensions=128)
    config = FusionConfig(
        candidate_limit=10,
        relevance_threshold=0.2,
        top_n=2,
    )
    chunks = [
        Chunk(id="pay-outage", text="payment outage escalation path pages the on-call engineer", metadata={"source": "runbook"}),
        Chunk(id="refund", text="refund policy duplicate charges refunded within 72 hours"),
        Chunk(id="gdpr", text="gdpr deletion request fulfilled within 30 days"),
        Chunk(id="rate", text="api rate limit 1000 requests per minute"),
    ]
    engine = RagramFusionEngine(storage, embedder, TermOverlapReranker(), config)
    await engine.initialize()
    check("index returns count", await engine.index(chunks) == 4)
    check("re-index upserts without duplicates", await engine.index(chunks) == 4)
    check("store count stable", await storage.count() == 4)

    result = await engine.search("payment outage escalation path")
    check("top_n truncation", len(result.chunks) <= 2, str(len(result.chunks)))
    check("best chunk is the payment one", result.chunks and result.chunks[0].chunk.id == "pay-outage")
    check("stats populated", result.stats.fused_candidates >= 1 and result.stats.elapsed_ms >= 0)
    if result.chunks:
        check(
            "threshold enforced",
            all(c.relevance_score >= 0.2 for c in result.chunks),
        )

    filtered = await engine.search(
        "payment outage escalation path quarterly planning", relevance_threshold=0.99
    )
    check("per-call threshold override empties result", filtered.is_empty)
    check("empty result keeps stats", filtered.stats.vector_hits >= 1)
    truncated = await engine.search("payment outage escalation path", top_n=1)
    check("per-call top_n override truncates", len(truncated.chunks) == 1)

    try:
        await engine.search("   ")
        raise AssertionError("expected ValueError")
    except ValueError:
        check("blank query rejected", True)
    zero = await storage.vector_search([0.0] * embedder.dimensions, 5)
    check("memory: zero query vector -> empty vector stream", zero == [])
    await engine.aclose()


async def test_empty_store() -> None:
    print("[empty-store]")
    engine = RagramFusionEngine(
        InMemoryStorageEngine(), HashingEmbedder(), TermOverlapReranker(), FusionConfig()
    )
    await engine.initialize()
    result = await engine.search("anything at all")
    check("zero chunks -> empty result, no crash", result.is_empty)
    check("stats all zero", result.stats.fused_candidates == 0 and result.stats.kept == 0)
    prompt = await engine.build_prompt("anything at all")
    check("prompt still built", "no evidence retrieved" in prompt.user)
    check("messages shape", [m["role"] for m in prompt.to_messages()] == ["system", "user"])
    await engine.aclose()


async def test_reranker_fallback() -> None:
    print("[reranker-fallback]")
    engine = RagramFusionEngine(
        InMemoryStorageEngine(), HashingEmbedder(), _FailingReranker(), FusionConfig(top_n=3, relevance_threshold=0.0)
    )
    await engine.initialize()
    await engine.index([Chunk(id=f"c{i}", text=f"unique content number {i}") for i in range(5)])
    result = await engine.search("unique content number")
    check("failed reranker degrades to fusion scores", not result.is_empty)
    check("fallback still respects top_n", len(result.chunks) <= 3)
    await engine.aclose()


async def test_one_stream_down() -> None:
    print("[stream-isolation]")
    from storage.base import StorageEngine

    class VectorOnlyStorage(StorageEngine):
        def __init__(self) -> None:
            self.inner = InMemoryStorageEngine()

        async def initialize(self):
            await self.inner.initialize()

        async def upsert(self, records):
            return await self.inner.upsert(records)

        async def vector_search(self, embedding, limit):
            return await self.inner.vector_search(embedding, limit)

        async def lexical_search(self, query, limit):
            raise ConnectionError("simulated tsvector index corruption")

        async def count(self):
            return await self.inner.count()

    engine = RagramFusionEngine(
        VectorOnlyStorage(), HashingEmbedder(), TermOverlapReranker(), FusionConfig(top_n=3, relevance_threshold=0.0)
    )
    await engine.initialize()
    await engine.index([Chunk(id="only", text="only relevant chunk here")])
    result = await engine.search("relevant chunk")
    check("lexical crash absorbed; vector stream survives", result.stats.lexical_hits == 0 and not result.is_empty)
    check("chunks come from vector stream only", result.chunks[0].streams == ("vector",))
    await engine.aclose()


async def test_prompt_package() -> None:
    print("[prompts]")
    from prompts import build_context_block

    fused = [
        FusedChunk(chunk=Chunk(id="a", text="alpha " * 500, metadata={"source": "s1", "note": "x" * 200}), fusion_score=0.05, streams=("vector",)),
        FusedChunk(chunk=Chunk(id="b", text="beta", metadata={}), fusion_score=0.03, streams=("lexical",)),
    ]
    reranked = TermOverlapReranker()
    items = await reranked.rerank("alpha", fused)
    context = build_context_block(items, max_chars_per_chunk=64)
    check("chunk labels present", "[C1]" in context and "[C2]" in context)
    check("long chunks truncated", "...[truncated]" in context)
    check("metadata provenance rendered", "source=s1" in context)
    result = FusionResult(query="alpha", chunks=tuple(items), stats=SearchStats(1, 1, 2, 2, 2, 1.0))
    prompt = build_generation_prompt("alpha", result)
    check("grounding contract in system prompt", "GROUNDING CONTRACT" in prompt.system)
    check("insufficiency policy in system prompt", "Insufficient evidence" in prompt.system)


def test_envfile() -> None:
    print("[envfile]")
    from envfile import env_float, env_int, env_str, load_env

    keys = (
        "RAGRAM_TEST_EXPORTED",
        "RAGRAM_TEST_QUOTED",
        "RAGRAM_TEST_SINGLE",
        "RAGRAM_TEST_INLINE",
        "RAGRAM_TEST_NUMERIC",
        "RAGRAM_TEST_FLOATY",
        "RAGRAM_TEST_PRESET",
        "RAGRAM_TEST_BLANK",
        "RAGRAM_TEST_BAD",
    )
    saved = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.env"
            path.write_text(
                "# comment line\n"
                "export RAGRAM_TEST_EXPORTED=exported\n"
                'RAGRAM_TEST_QUOTED="has spaces and # hash"\n'
                "RAGRAM_TEST_SINGLE='raw # value'\n"
                "RAGRAM_TEST_INLINE=value  # trailing comment\n"
                "RAGRAM_TEST_NUMERIC=42\n"
                "RAGRAM_TEST_FLOATY=0.75\n"
                "RAGRAM_TEST_PRESET=from-file\n"
                "RAGRAM_TEST_EMPTY=\n"
                "not an assignment\n",
                encoding="utf-8",
            )
            os.environ["RAGRAM_TEST_PRESET"] = "from-shell"
            loaded = load_env(path)
            check(
                "assignments parsed, blanks skipped",
                "RAGRAM_TEST_EXPORTED" in loaded and "RAGRAM_TEST_EMPTY" not in loaded,
            )
            check("comment + junk lines skipped", len(loaded) == 7)
            check("export prefix stripped", env_str("RAGRAM_TEST_EXPORTED") == "exported")
            check(
                "double quotes keep inner hash",
                env_str("RAGRAM_TEST_QUOTED") == "has spaces and # hash",
            )
            check("single quotes preserved", env_str("RAGRAM_TEST_SINGLE") == "raw # value")
            check("inline comment stripped", env_str("RAGRAM_TEST_INLINE") == "value")
            check("shell env wins by default", env_str("RAGRAM_TEST_PRESET") == "from-shell")
            check("env_int parses", env_int("RAGRAM_TEST_NUMERIC", 0) == 42)
            check("env_float parses", env_float("RAGRAM_TEST_FLOATY", 0.0) == 0.75)
            check("unset var -> default", env_str("RAGRAM_TEST_BLANK", "d") == "d")

            load_env(path, override=True)
            check("override=True replaces shell env", env_str("RAGRAM_TEST_PRESET") == "from-file")

            os.environ["RAGRAM_TEST_BAD"] = "abc"
            try:
                env_int("RAGRAM_TEST_BAD", 7)
                rejected = False
            except ValueError:
                rejected = True
            check("env_int rejects non-numeric", rejected)

            try:
                load_env(Path(tmp) / "missing.env")
                raised = False
            except FileNotFoundError:
                raised = True
            check("missing explicit path raises FileNotFoundError", raised)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


async def main() -> int:
    test_rrf_math()
    test_rrf_empty_streams()
    test_normalized_fusion()
    test_fuse_dispatch()
    test_config_validation()
    test_envfile()
    await test_engine_pipeline()
    await test_empty_store()
    await test_reranker_fallback()
    await test_one_stream_down()
    await test_prompt_package()
    print(f"\nALL {PASS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except AssertionError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)

"""RAGRAM_Pro demo: hybrid index initialization, mock production load, full pipeline query.

Runs dependency-free by default (in-memory storage + term-overlap reranker):

    python app.py
    python app.py --reranker hf                # local BAAI/bge-reranker-base
    python app.py --reranker cohere            # managed API (needs COHERE_API_KEY)
    python app.py --storage postgres           # pgvector + tsvector (needs DATABASE_URL)
    python app.py --query "when is a SEV1 declared?"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from config import FusionConfig
from embedding import Embedder, HashingEmbedder, SentenceTransformerEmbedder
from engine import RagramFusionEngine
from models import Chunk
from rerank import CohereReranker, CrossEncoderReranker, Reranker, TermOverlapReranker
from storage import InMemoryStorageEngine, PostgresStorageEngine, StorageEngine

LOGGER = logging.getLogger("ragram.app")

# --------------------------------------------------------------------------- #
# Mock production rows: an enterprise ops/policy knowledge base.              #
# --------------------------------------------------------------------------- #

MOCK_ROWS: tuple[Chunk, ...] = (
    Chunk(
        id="INC-PAY-042",
        text=(
            "Payment outage escalation ladder: a Severity-1 payment outage pages the Payments "
            "on-call engineer immediately via PagerDuty. If there is no mitigation within 10 "
            "minutes, the Incident Commander is engaged, and the VP of Engineering must be "
            "notified within 30 minutes of declaration. The payments lead owns customer-facing "
            "status updates until resolution."
        ),
        metadata={"source": "runbooks/incident-response.md", "department": "SRE", "updated": "2026-08-14"},
    ),
    Chunk(
        id="POL-SEV-001",
        text=(
            "Severity definitions: SEV1 is a total customer-facing outage or payment processing "
            "failure affecting more than 5 percent of transactions. SEV2 is degraded core "
            "functionality with a workaround. SEV3 is a minor defect with no revenue impact. "
            "SEV1 requires an Incident Commander and a 15-minute response target."
        ),
        metadata={"source": "policies/severity-matrix.md", "department": "SRE", "updated": "2026-07-02"},
    ),
    Chunk(
        id="OPS-ONCALL-011",
        text=(
            "On-call rotation: payments and platform services are paged through a weekly "
            "follow-the-sun rotation. Handoff occurs Monday 10:00 UTC. Escalation skips one "
            "level after two unanswered pages, then reaches the engineering manager on duty."
        ),
        metadata={"source": "runbooks/oncall.md", "department": "SRE", "updated": "2026-06-20"},
    ),
    Chunk(
        id="OPS-COMMS-007",
        text=(
            "Incident communication protocol: during a SEV1 the status page must be updated "
            "within 15 minutes of declaration and every 30 minutes thereafter. Customer support "
            "receives a templated holding statement from the Incident Commander; no engineer "
            "posts externally without approval."
        ),
        metadata={"source": "runbooks/comms.md", "department": "Support", "updated": "2026-05-30"},
    ),
    Chunk(
        id="SLA-ENT-003",
        text=(
            "Enterprise SLA: 99.95 percent monthly uptime for the checkout API, a 15-minute "
            "response target for Severity-1 incidents, and a 4-hour response target for "
            "Severity-2. Credits of 10 percent of monthly fees apply per 30 minutes of breach."
        ),
        metadata={"source": "contracts/sla-enterprise.md", "department": "Legal", "updated": "2026-04-11"},
    ),
    Chunk(
        id="FIN-REFUND-020",
        text=(
            "Refund policy: duplicate charges are refunded automatically within 72 hours. "
            "Manual refund requests submitted by customers are reviewed by Finance within two "
            "business days. Refunds above 5,000 USD require dual approval from Finance and the "
            "payments lead."
        ),
        metadata={"source": "policies/refunds.md", "department": "Finance", "updated": "2026-03-08"},
    ),
    Chunk(
        id="SEC-RETENTION-014",
        text=(
            "Data retention: application logs are retained for 90 days, financial transaction "
            "records for 7 years as required by audit, and deleted backups are purged within "
            "35 days. Encryption keys rotate every 90 days."
        ),
        metadata={"source": "policies/data-retention.md", "department": "Security", "updated": "2026-02-19"},
    ),
    Chunk(
        id="SEC-GDPR-009",
        text=(
            "GDPR deletion requests: verified data-subject deletion requests are fulfilled "
            "within 30 days across production, analytics and backups. The privacy officer logs "
            "each request and issues a signed deletion certificate to the requester."
        ),
        metadata={"source": "policies/gdpr.md", "department": "Security", "updated": "2026-01-27"},
    ),
    Chunk(
        id="API-RATE-005",
        text=(
            "API rate limits: the public API allows 1,000 requests per minute per API key. "
            "Exceeding the limit returns HTTP 429 with a Retry-After header. Sustained abuse "
            "triggers automatic key suspension and a notification to the account owner."
        ),
        metadata={"source": "docs/api-rate-limits.md", "department": "Platform", "updated": "2026-09-05"},
    ),
    Chunk(
        id="OPS-POSTMORTEM-002",
        text=(
            "Post-incident process: every SEV1 requires a blameless postmortem within five "
            "business days, owned by the Incident Commander. Action items carry named owners "
            "and due dates, and are reviewed in the weekly reliability sync."
        ),
        metadata={"source": "runbooks/postmortem.md", "department": "SRE", "updated": "2026-07-25"},
    ),
)

DEFAULT_QUERIES: tuple[str, ...] = (
    "What is the escalation path when the payment service goes down?",
    "How fast must we respond to a Severity-1 incident?",
)


# --------------------------------------------------------------------------- #
# Component factories                                                          #
# --------------------------------------------------------------------------- #


def build_storage(choice: str, *, dimensions: int) -> StorageEngine:
    if choice == "memory":
        return InMemoryStorageEngine()
    if choice == "postgres":
        dsn = os.environ.get("DATABASE_URL")
        if not dsn:
            raise SystemExit("--storage postgres requires DATABASE_URL (postgresql://...)")
        return PostgresStorageEngine(dsn, dimensions=dimensions)
    raise SystemExit(f"unknown storage backend: {choice!r}")


def build_embedder(choice: str) -> Embedder:
    if choice == "hashing":
        return HashingEmbedder(dimensions=256)
    if choice == "sbert":
        return SentenceTransformerEmbedder()
    raise SystemExit(f"unknown embedder: {choice!r}")


def build_reranker(choice: str) -> Reranker:
    if choice == "noop":
        LOGGER.warning("using TermOverlapReranker (demo only); pass --reranker hf for production")
        return TermOverlapReranker()
    if choice == "hf":
        return CrossEncoderReranker("BAAI/bge-reranker-base")
    if choice == "cohere":
        return CohereReranker()
    raise SystemExit(f"unknown reranker: {choice!r}")


# --------------------------------------------------------------------------- #
# Pipeline execution                                                           #
# --------------------------------------------------------------------------- #


def _preview(text: str, width: int = 110) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 3] + "..."


async def run_query(engine: RagramFusionEngine, query: str) -> None:
    print(f"\n{'=' * 100}")
    print(f"QUERY: {query}")
    print("=" * 100)

    result = await engine.search(query)
    stats = result.stats
    print(
        f"[stats] vector={stats.vector_hits} lexical={stats.lexical_hits} "
        f"fused={stats.fused_candidates} reranked={stats.reranked} "
        f"kept={stats.kept} elapsed={stats.elapsed_ms:.1f}ms"
    )

    if result.is_empty:
        print("[warn] no chunk survived threshold filtering -- LLM will get the empty-evidence prompt")
        return

    for position, item in enumerate(result.chunks, start=1):
        streams = "+".join(item.streams)
        print(
            f"\n  [{position}] {item.chunk.id}  relevance={item.relevance_score:.3f}  "
            f"fusion={item.fusion_score:.4f}  streams={streams}"
        )
        source = item.chunk.metadata.get("source", "-")
        updated = item.chunk.metadata.get("updated", "-")
        print(f"      source={source}  updated={updated}")
        print(f"      {_preview(item.chunk.text)}")


async def amain(args: argparse.Namespace) -> int:
    embedder = build_embedder(args.embedder)
    storage = build_storage(args.storage, dimensions=embedder.dimensions)
    reranker = build_reranker(args.reranker)
    config = FusionConfig(
        candidate_limit=args.candidate_limit,
        fusion_method=args.fusion,
        rrf_k=args.rrf_k,
        relevance_threshold=args.threshold,
        top_n=args.top_n,
    )

    queries = tuple(args.query or ()) or DEFAULT_QUERIES

    async with RagramFusionEngine(storage, embedder, reranker, config) as engine:
        indexed = await engine.index(MOCK_ROWS)
        stored = await storage.count()
        print(f"[init] indexed {indexed} mock production rows (store now holds {stored})")
        print(
            f"[init] fusion={config.fusion_method}(k={config.rrf_k}) "
            f"threshold={config.relevance_threshold} top_n={config.top_n} "
            f"candidate_limit={config.candidate_limit}"
        )

        for query in queries:
            await run_query(engine, query)

        # ------------------------------------------------------------- #
        # The exact prompt that would be sent to the LLM.                #
        # ------------------------------------------------------------- #
        prompt = await engine.build_prompt(queries[0])
        print(f"\n{'=' * 100}")
        print(f"GENERATION PROMPT FOR: {queries[0]}")
        print("=" * 100)
        print("\n--- system ---")
        print(prompt.system)
        print("\n--- user ---")
        print(prompt.user)
        print(f"\n[info] to_messages() -> {len(prompt.to_messages())} chat messages")

    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAGRAM_Pro hybrid search + reranking demo")
    parser.add_argument(
        "--storage",
        choices=("memory", "postgres"),
        default="memory",
        help="storage backend (postgres needs DATABASE_URL)",
    )
    parser.add_argument(
        "--embedder",
        choices=("hashing", "sbert"),
        default="hashing",
        help="hashing = dependency-free demo embedder, sbert = sentence-transformers",
    )
    parser.add_argument(
        "--reranker",
        choices=("noop", "hf", "cohere"),
        default="noop",
        help="noop = term overlap (demo), hf = local cross-encoder, cohere = managed API",
    )
    parser.add_argument("--fusion", choices=("rrf", "normalized"), default="rrf")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--candidate-limit", type=int, default=25)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--top-n", type=int, default=4)
    parser.add_argument("--query", action="append", help="repeat for multiple queries")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args(argv)
    try:
        return asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

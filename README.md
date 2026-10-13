# RAGRAM_Pro

**Hybrid search (dense + sparse) with Reciprocal Rank Fusion and cross-encoder reranking for precision-critical RAG.**

RAGRAM_Pro is built for production retrieval where plain vector search is not precise enough. It runs two
search streams in parallel, fuses their ranks, scores every candidate with a cross-encoder, then filters
and truncates the result so the LLM only ever sees the strongest, most relevant context.

```
query ──┬── vector stream   (pgvector cosine / in-memory cosine)  ──┐
        └── lexical stream  (Postgres tsvector / Okapi BM25)     ──┴─► RRF or normalized fusion
                                                                        │  dedup + merge streams
                                                                        ▼
                                                          cross-encoder rerank (HF / Cohere)
                                                                        │  relevance in [0, 1]
                                                                        ▼
                                                     threshold filter ─► top_n ─► LLM prompt
```

## Features

- **Hybrid coordinator** — vector similarity and keyword full-text search execute concurrently
  (`asyncio.gather`), each stream fault-isolated: one backend failing or returning zero rows never fails the query.
- **Score fusion** — weighted Reciprocal Rank Fusion (default, `k=60`) or per-stream min-max
  normalization, deduplicated by chunk id with contributing-stream labels preserved.
- **Pluggable reranking** — `Reranker` abstraction with three backends:
  local HuggingFace cross-encoder (`BAAI/bge-reranker-base`), managed Cohere `/v2/rerank` API,
  or a dependency-free term-overlap scorer for demos/tests. Scores are calibrated to `[0, 1]`
  so `relevance_threshold` means the same thing everywhere; a failing reranker degrades gracefully
  to fusion-order scores instead of killing the query.
- **Two storage engines** behind one `StorageEngine` interface:
  `PostgresStorageEngine` (pgvector cosine + generated `tsvector` column with GIN/HNSW indexes,
  connection retries with backoff) and `InMemoryStorageEngine` (exact cosine + built-in Okapi BM25).
- **Prompt packaging** — a strict grounding system prompt plus numbered, provenance-tagged evidence
  blocks (`[C1]`, `[C2]`, ...) ready for any chat API via `to_messages()`.
- **Full typing, structured logging, observable stats** — every query reports
  `vector_hits / lexical_hits / fused_candidates / reranked / kept / elapsed_ms`.

## Layout

```
RAGRAM_Pro/
├── app.py              # demo: hybrid index init, mock rows, full pipeline query
├── engine.py           # RagramFusionEngine — the end-to-end pipeline
├── config.py           # FusionConfig (threshold, top_n, rrf_k, weights, ...)
├── models.py           # Chunk → SearchHit → FusedChunk → RerankedChunk contracts
├── embedding.py        # Embedder ABC + HashingEmbedder / SentenceTransformerEmbedder
├── prompts.py          # system prompt + evidence-block builders
├── storage/            # StorageEngine ABC, Postgres (pgvector+tsvector), memory (BM25)
├── search/             # HybridSearchCoordinator + RRF/normalized fusion
├── rerank/             # Reranker ABC + HuggingFace / Cohere / term-overlap backends
└── tests/              # stdlib unit suite + live Postgres integration suite
```

---

## Install

```powershell
# Core pipeline (in-memory storage, BM25, fusion, prompts) — zero dependencies
pip install -e .

# Pick the backends you actually deploy:
pip install -e ".[postgres]"      # PostgresStorageEngine   (asyncpg, pgvector)
pip install -e ".[cohere]"        # CohereReranker           (httpx)
pip install -e ".[local]"         # CrossEncoderReranker + SentenceTransformerEmbedder
pip install -e ".[all]"
```

| Extra        | Enables                                          |
|--------------|--------------------------------------------------|
| *(none)*     | `InMemoryStorageEngine`, `HashingEmbedder`, `TermOverlapReranker`, fusion, prompts |
| `postgres`   | `PostgresStorageEngine` (pgvector + tsvector)    |
| `cohere`     | `CohereReranker` (managed `/v2/rerank`)          |
| `local`      | `CrossEncoderReranker` (HF cross-encoder), `SentenceTransformerEmbedder` |

Missing extras raise a clear `StorageError` / `RerankerError` with the exact install command —
they never crash unrelated code paths.

---

## Quickstart

```powershell
# Core pipeline is dependency-free — run the demo immediately:
python app.py

# With a local cross-encoder reranker (downloads BAAI/bge-reranker-base once):
pip install sentence-transformers
python app.py --reranker hf

# Against a real PostgreSQL + pgvector:
pip install asyncpg pgvector
$env:DATABASE_URL = 'postgresql://user:pass@localhost:5432/dbname'
python app.py --storage postgres
```

```bash
# bash equivalents
export DATABASE_URL='postgresql://user:pass@localhost:5432/dbname'
python app.py --storage postgres
```

Minimal programmatic usage:

```python
import asyncio
from config import FusionConfig
from embedding import HashingEmbedder
from engine import RagramFusionEngine
from models import Chunk
from rerank import TermOverlapReranker
from storage import InMemoryStorageEngine

async def main() -> None:
    engine = RagramFusionEngine(
        storage=InMemoryStorageEngine(),
        embedder=HashingEmbedder(),
        reranker=TermOverlapReranker(),
        config=FusionConfig(top_n=3, relevance_threshold=0.3),
    )
    async with engine:
        await engine.index([Chunk(id="c1", text="Payment outage pages the on-call engineer.")])
        result = await engine.search("payment outage escalation")
        for hit in result.chunks:
            print(hit.relevance_score, hit.chunk.id)

asyncio.run(main())
```

---

## How the pipeline works

```python
async with RagramFusionEngine(storage, embedder, reranker, config) as engine:
    await engine.index(chunks)                       # 1. embed + upsert
    result = await engine.search(query)              # 2..6 below
    prompt = await engine.build_prompt(query)        # same pipeline, packaged for the LLM
```

Per query, `engine.search()` executes:

1. **Embed the query** — `embedder.embed_query(query)`.
2. **Dual-stream retrieval** — `storage.vector_search()` (cosine) and
   `storage.lexical_search()` (tsvector/BM25) run **concurrently**; either may return
   `[]` or fail without failing the query (see [Edge cases](#edge-case--failure-behavior)).
3. **Fusion** — weighted RRF (default) or min-max normalization deduplicates by chunk id
   and produces `FusedChunk` entries sorted best-first.
4. **Rerank** — `reranker.rerank()` scores every fused candidate against the raw query
   (`relevance_score ∈ [0, 1]`).
5. **Filter** — drop chunks with `relevance_score < relevance_threshold`.
6. **Truncate** — keep the first `top_n` chunks → `FusionResult`.

### Configuration reference (`config.py`)

```python
from config import FusionConfig

config = FusionConfig(
    candidate_limit=50,          # hits fetched PER STREAM before fusion (bounds rerank cost)
    fusion_method="rrf",         # "rrf" | "normalized"
    rrf_k=60,                    # RRF damping; smaller = sharper top-rank weighting
    vector_weight=1.0,           # dense stream weight during fusion
    lexical_weight=1.0,          # sparse stream weight during fusion
    relevance_threshold=0.3,     # min reranker score [0..1] to survive filtering
    top_n=5,                     # max chunks in the final package
)
```

Validation is eager (`ValueError` on construction): thresholds outside `[0, 1]`,
`top_n < 1`, `rrf_k < 1`, or both weights `0` are rejected immediately.

Per-call overrides (do not mutate the config):

```python
result = await engine.search(query, top_n=3, relevance_threshold=0.5)
```

**Choosing fusion:**

- `rrf` (default) — rank-only, ignores score scales. Use when the two streams' scores are
  not comparable (cosine similarity vs. ts_rank), i.e. almost always with Postgres.
- `normalized` — min-max normalizes each stream to `[0, 1]`, then blends with the weights.
  Use when you *do* trust score magnitudes (e.g. two calibrated backends) and want score
  gaps to influence the merge.

---

## Storage backends

### In-memory (default for demos/tests)

```python
from storage import InMemoryStorageEngine

storage = InMemoryStorageEngine()   # exact cosine + built-in Okapi BM25
```

No setup, no dependencies. Index state is lost on exit.

### PostgreSQL + pgvector (production)

```python
import os
from storage import PostgresStorageEngine

storage = PostgresStorageEngine(
    os.environ["DATABASE_URL"],          # postgresql://user:pass@host:5432/dbname
    dimensions=384,                      # MUST equal embedder.dimensions
    table="ragram_chunks",               # default; validated as a safe identifier
    tsconfig="english",                  # text search config for stemming/stopwords
    pool_max_size=10,
    connect_retries=3,                   # exponential backoff on dropped connections
)
```

`await storage.initialize()` (done automatically by `RagramFusionEngine.initialize()`)
idempotently creates everything it needs:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE ragram_chunks (
    id        TEXT PRIMARY KEY,
    content   TEXT NOT NULL,
    embedding vector(dims) NOT NULL,
    metadata  JSONB NOT NULL DEFAULT '{}'::jsonb,
    tsv       tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);
CREATE INDEX ... USING GIN (tsv);
CREATE INDEX ... USING HNSW (embedding vector_cosine_ops);   -- best effort (PG15+)
```

Notes:

- The role needs rights to create the `vector` extension (or have it pre-installed).
  If HNSW is unavailable (PostgreSQL < 15) the engine logs a warning and falls back to a
  sequential scan instead of failing.
- Lexical matching builds an **OR** query (`websearch_to_tsquery`) so any matching term
  contributes — recall-oriented, consistent with BM25, precision handled by the reranker.
- `upsert()` overwrites on id conflict (text, embedding and metadata all replaced).
- A zero-valued query vector returns an empty vector stream instead of NaN rows.
- Connection failures retry with backoff, then raise `StorageError`.

### Custom storage

Subclass `StorageEngine` (`storage/base.py`) and implement five methods:
`initialize`, `upsert`, `vector_search`, `lexical_search`, `count`
(+ optionally `close`). Contract: best-first ordering, `[]` for no matches,
`StorageError` for infrastructure failures.

---

## Embedders

```python
from embedding import HashingEmbedder, SentenceTransformerEmbedder

demo   = HashingEmbedder(dimensions=256)                          # dependency-free
real   = SentenceTransformerEmbedder()                            # all-MiniLM-L6-v2 (384-dim)
large  = SentenceTransformerEmbedder("BAAI/bge-m3", device="cuda")
```

- `embedder.dimensions` must match the Postgres `vector(dims)` column.
- The embedder is used for **documents** (`embed_documents`) at index time and for the
  **query** (`embed_query`) at search time.
- Swap in any custom backend by subclassing `Embedder` (two async methods + `dimensions`).

---

## Rerankers

All backends implement one method and return candidates sorted by `relevance_score`
(clamped to `[0, 1]`), so `relevance_threshold` behaves identically across them.

```python
from rerank import CrossEncoderReranker, CohereReranker, TermOverlapReranker

# A) Local HuggingFace cross-encoder (on-prem)
local = CrossEncoderReranker("BAAI/bge-reranker-base")   # logits -> sigmoid -> [0, 1]

# B) Managed Cohere API (needs COHERE_API_KEY or api_key=...)
cloud = CohereReranker(model="rerank-v3.5")

# C) Dependency-free term-overlap scorer (demos/tests only)
fast = TermOverlapReranker()
```

Switching backends is a one-line change — the pipeline only sees the `Reranker` ABC:

```python
engine = RagramFusionEngine(storage, embedder, reranker=CrossEncoderReranker(), config=config)
```

### Custom reranker

```python
from rerank import Reranker
from models import FusedChunk, RerankedChunk

class MyReranker(Reranker):
    async def rerank(self, query: str, candidates: list[FusedChunk]) -> list[RerankedChunk]:
        if not candidates:                      # contract: empty in -> empty out
            return []
        scored = [self._pack(c, my_model.score(query, c.chunk.text)) for c in candidates]
        return self._sort(scored)               # descending relevance

    async def aclose(self) -> None:             # optional: release clients/models
        ...
```

If any reranker raises, `RagramFusionEngine` logs the failure and **falls back to
min-max-normalized fusion scores**, so retrieval still returns ranked results.

---

## Working with results

```python
result = await engine.search("what is the escalation path for a payment outage?")

result.is_empty                 # True when no chunk survived thresholding
result.query                    # normalized query text

for item in result.chunks:      # RerankedChunk, best-first, already thresholded + top_n
    item.chunk.id               # str
    item.chunk.text             # str
    item.chunk.metadata         # dict (source, updated, department, ...)
    item.relevance_score        # reranker score in [0, 1]  (threshold applied on this)
    item.fusion_score           # raw RRF/normalized score
    item.streams                # ("vector", "lexical"), ("vector",), or ("lexical",)

s = result.stats
s.vector_hits                   # rows from the dense stream
s.lexical_hits                  # rows from the sparse stream
s.fused_candidates              # unique chunks after fusion/dedup
s.reranked                      # chunks the reranker scored
s.kept                          # after threshold + top_n
s.elapsed_ms                    # wall-clock for the whole query
```

Log line per query:

```
hybrid retrieval complete: vector=10 lexical=4 (limit=25)
query resolved: vector=10 lexical=4 fused=10 reranked=10 kept=3 (8.4 ms)
```

---

## Building the LLM prompt

```python
from prompts import build_generation_prompt, SYSTEM_PROMPT, build_context_block

prompt = await engine.build_prompt("what is the escalation path for a payment outage?")

prompt.system          # strict grounding system prompt (cite [C#], conflict policy,
                       # deterministic "Insufficient evidence ..." refusal)
prompt.user            # "### Retrieved Evidence" + [C1]/[C2] blocks + "### User Question"
prompt.to_messages()   # [{"role": "system", ...}, {"role": "user", ...}] — any chat API
prompt.result           # the underlying FusionResult (stats, scores, provenance)
```

Rendered evidence block:

```
[C1] relevance=0.912 | source=runbooks/incident-response.md | updated=2026-08-14
Payment outage escalation ladder: a Severity-1 payment outage pages ...
```

Customization:

```python
# Long chunks are truncated per-block (default 1600 chars):
build_context_block(result.chunks, max_chars_per_chunk=800)

# Full control — compose your own template around the helpers:
from prompts import build_user_prompt
my_user_message = f"Answer using ONLY this evidence:\n{build_context_block(result.chunks)}\n\nQ: {q}"
```

When zero chunks survive, `prompt.user` contains an explicit
`(no evidence retrieved ...)` notice so the model follows its insufficiency policy
instead of hallucinating.

---

## Edge-case behavior

| Situation | Behavior |
|---|---|
| Lexical stream returns 0 rows | Fusion runs on the vector stream alone; `lexical_hits=0` in stats |
| Vector stream returns 0 rows | Fusion runs on the lexical stream alone |
| A storage stream **raises** (e.g. dropped connection) | Logged (`logger.exception`), stream treated as `[]`, query continues |
| Empty index / no matches anywhere | `FusionResult` with `is_empty=True`, stats all zero — no exception |
| Reranker raises (API down, model missing) | Logged; falls back to normalized fusion scores as relevance |
| Query embeds to a zero vector (e.g. punctuation-only) | Empty vector stream (no NaN cosine rows) |
| Query has only stopwords/punctuation (Postgres) | Legitimate empty lexical result, no SQL error |
| Blank / whitespace query | `ValueError` before any backend call |
| `relevance_threshold` filters everything | Empty result + prompt still buildable (insufficiency notice) |

Fault isolation lives in `search/coordinator.py` (`_guarded_stream`) — each stream is
individually absorbed, so hybrid queries can never fail *because* one retriever did.

---

## Demo reference (`app.py`)

```powershell
python app.py                       # memory storage + term-overlap reranker, zero deps
python app.py --reranker hf         # local BAAI/bge-reranker-base
python app.py --reranker cohere     # managed API (set $env:COHERE_API_KEY)
python app.py --storage postgres    # real pgvector + tsvector (set $env:DATABASE_URL)
python app.py --fusion normalized --rrf-k 30 --threshold 0.5 --top-n 3
python app.py --query "when is a SEV1 declared?" --query "who approves refunds above 5000 USD"
```

| Flag | Values | Default |
|---|---|---|
| `--storage` | `memory`, `postgres` | `memory` |
| `--embedder` | `hashing`, `sbert` | `hashing` |
| `--reranker` | `noop`, `hf`, `cohere` | `noop` |
| `--fusion` | `rrf`, `normalized` | `rrf` |
| `--rrf-k` | int | `60` |
| `--candidate-limit` | int | `25` |
| `--threshold` | float 0..1 | `0.3` |
| `--top-n` | int | `4` |
| `--query` | repeatable | two built-in queries |

The demo prints per-stream stats, the ranked result table, and the exact generation
prompt (system + user) that would be sent to the LLM.

---

## Tests

```powershell
python tests/test_pipeline.py                                    # 42 stdlib checks, no services needed
$env:DATABASE_URL = 'postgresql://user:pass@localhost:5432/db'
python tests/test_postgres_integration.py                        # 24 live Postgres checks
python -m pyflakes .                                             # lint
```

Both suites exit non-zero on failure, so they slot directly into CI.

---

## Production checklist

- [ ] `SentenceTransformerEmbedder` (or your own) instead of `HashingEmbedder`
- [ ] `PostgresStorageEngine` with `dimensions == embedder.dimensions`
- [ ] `CrossEncoderReranker` or `CohereReranker` instead of `TermOverlapReranker`
- [ ] Tune `candidate_limit` (rerank cost/latency) and `relevance_threshold` (precision)
      on a labelled query set — start at `0.3`, cross-encoder scores cluster near 0.5
- [ ] Keep `top_n` small (3–6): reranked chunks are already the strongest evidence
- [ ] Wire `prompt.to_messages()` into your chat API and log `result.stats` per query
- [ ] Set up log handlers for the `ragram` logger namespace
      (`ragram.engine`, `ragram.search.coordinator`, `ragram.storage.postgres`, ...)

## Requirements

Python ≥ 3.10. Core has **zero dependencies**; every backend is an optional extra
(`postgres`, `cohere`, `local` — see `pyproject.toml`).

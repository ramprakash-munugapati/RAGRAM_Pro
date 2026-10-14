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

**Contents:** [Who this is for](#who-this-is-for) · [Install](#install) ·
[Quickstart](#quickstart) · [Bringing your own data](#bringing-your-own-data) ·
[How the pipeline works](#how-the-pipeline-works) · 
[Detailed ingest / query flow](#detailed-ingest--query-flow) ·
[Storage backends](#storage-backends) · [Embedders](#embedders) · [Rerankers](#rerankers) ·
[Working with results](#working-with-results) · [Building the LLM prompt](#building-the-llm-prompt) ·
[Integrations](#integrations) · [Edge-case behavior](#edge-case-behavior) ·
[Demo reference](#demo-reference) · [Environment configuration](#environment-configuration) ·
[Tests](#tests) · [Troubleshooting](#troubleshooting) · [Production checklist](#production-checklist) ·
[Learning guide](docs/learn.md)

**Want to explore the codebase itself?**
[GitIngest](https://gitingest.com/ramprakash-munugapati/ragram_pro) — the whole repo as a
single LLM-friendly text file (paste into your AI assistant) ·
[GitDiagram](https://gitdiagram.com/ramprakash-munugapati/ragram_pro) — auto-generated
architecture diagram of the source tree.

## Who this is for

You already know the basics of RAG: split documents into chunks, embed them, run a
similarity search, paste the winners into a prompt. If that pipeline already answers
your questions reliably, you don't need this project.

RAGRAM_Pro is for the next step — when plain vector search quietly misses:

- **Exact tokens matter.** Error codes (`ECONNRESET`), order IDs (`INV-20491`),
  function names, config keys, version strings (`v2.3.1-rc1`). Embeddings blur these;
  the lexical stream finds them literally.
- **Closest ≠ correct.** Cosine similarity ranks by meaning, not by whether the chunk
  actually answers the question. A paragraph about "payment timeouts" can outrank the
  one that names the escalation ladder.
- **Users type keywords, not prose.** Real queries are often short and noun-heavy —
  exactly where sparse retrieval wins.

Hybrid search fixes the first and third; the cross-encoder reranker fixes the second by
reading the query and each candidate together and scoring real relevance in `[0, 1]`.
The dense stream still carries its share — paraphrases and synonym-only matches that
contain none of your query words.

**When you can skip it:** tiny corpora where plain vector search already scores well on
your own test questions, or latency budgets too tight for a rerank pass (bounded here by
`tune down candidate_limit`). Honest cost: two retrieval paths and one reranker model/API
to operate instead of one vector index.

**New to hybrid search, fusion or reranking?** The step-by-step refresher —
embeddings, BM25, RRF with a worked example, cross-encoders, tuning — lives in
[docs/learn.md](docs/learn.md).

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
- **Trending-tool integrations** — LangChain `BaseRetriever`, LlamaIndex `BaseRetriever`, and
  grounded generation backends for OpenAI / Anthropic / Ollama, all lazily imported so the
  core stays dependency-free (`integrations/`).
- **Externalized configuration** — `.env`-driven properties (backends, models, fusion knobs,
  API keys) through the stdlib `envfile.py` loader; shell/CI variables always win.
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
├── envfile.py          # stdlib .env loader + env_str/env_int/env_float helpers
├── .env.example        # documented property template (copy to gitignored .env)
├── storage/            # StorageEngine ABC, Postgres (pgvector+tsvector), memory (BM25)
├── search/             # HybridSearchCoordinator + RRF/normalized fusion
├── rerank/             # Reranker ABC + HuggingFace / Cohere / term-overlap backends
├── integrations/       # LangChain + LlamaIndex retrievers, LLM generation backends
├── examples/           # fastapi_app.py REST demo (driven by .env)
└── tests/              # stdlib unit suite + integrations suite + live Postgres suite
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
pip install -e ".[langchain]"     # RagramRetriever          (langchain-core)
pip install -e ".[llamaindex]"    # RagramLlamaIndexRetriever (llama-index-core)
pip install -e ".[openai]"        # OpenAIGenerator          (openai)
pip install -e ".[anthropic]"     # AnthropicGenerator       (anthropic)
pip install -e ".[api]"           # examples/fastapi_app.py  (fastapi, uvicorn)
pip install -e ".[all]"
```

| Extra        | Enables                                          |
|--------------|--------------------------------------------------|
| *(none)*     | `InMemoryStorageEngine`, `HashingEmbedder`, `TermOverlapReranker`, fusion, prompts, `OllamaGenerator` |
| `postgres`   | `PostgresStorageEngine` (pgvector + tsvector)    |
| `cohere`     | `CohereReranker` (managed `/v2/rerank`)          |
| `local`      | `CrossEncoderReranker` (HF cross-encoder), `SentenceTransformerEmbedder` |
| `langchain`  | `RagramRetriever` (LangChain `BaseRetriever`)    |
| `llamaindex` | `RagramLlamaIndexRetriever` (LlamaIndex `BaseRetriever`) |
| `openai`     | `OpenAIGenerator` (chat completions)             |
| `anthropic`  | `AnthropicGenerator` (Messages API)              |
| `api`        | runs `examples/fastapi_app.py` (FastAPI + uvicorn) |

Missing extras raise a clear `StorageError` / `RerankerError` with the exact install command —
they never crash unrelated code paths.

---

## Quickstart

```powershell
# Core pipeline is dependency-free — run the demo immediately:
python app.py

# Optional: externalize defaults (DATABASE_URL, API keys, RAGRAM_* knobs) into .env:
copy .env.example .env

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

Everything is async by design. If your app doesn't already run an event loop, drive
calls with `asyncio.run()` exactly as above — the LangChain adapter bridges sync and
async callers for you automatically.

---

## Bringing your own data

RAGRAM_Pro does **not** chunk documents — you do, with whatever splitter you already
use (LangChain `RecursiveCharacterTextSplitter`, LlamaIndex `SentenceSplitter`, or a
few lines of your own). You hand the engine a list of `Chunk` objects; everything else
in this README works unchanged.

Three rules:

1. **`id` is the identity.** Indexing the same id again overwrites that chunk
   completely (text, embedding, metadata), so use a *stable* id derived from the
   source — `"policies/refunds.md#0"`, a content hash, your row's primary key.
   Random/sequential ids make re-ingestion duplicate instead of update.
2. **Changing your chunk sizes? Start a fresh table.** There is no delete API: if you
   re-chunk with different boundaries, the old ids no longer match and stale rows
   linger. `DROP TABLE ragram_chunks` (or point `table=` at a new name) and re-index.
3. **`metadata` is provenance, not payload.** Whatever you put there
   (`source`, `updated`, `department`, ...) is rendered into the evidence headers the
   LLM sees (`[C1] relevance=0.91 | source=... | updated=...`) and copied into
   LangChain/LlamaIndex metadata. Keys pass through untouched.

End-to-end with your own text and a local LLM:

```python
import asyncio
import os

from config import FusionConfig
from embedding import SentenceTransformerEmbedder      # 384-dim
from engine import RagramFusionEngine
from integrations import OllamaGenerator                # swap for OpenAI/Anthropic
from models import Chunk
from rerank import CrossEncoderReranker
from storage import PostgresStorageEngine

def chunk_my_documents(files: dict[str, str]) -> list[Chunk]:
    """Your splitter goes here — this is the part RAGRAM deliberately leaves to you."""
    chunks = []
    for source, text in files.items():
        for i, part in enumerate(split(text, size=800, overlap=150)):   # pseudo-code
            chunks.append(Chunk(id=f"{source}#{i}", text=part,
                                metadata={"source": source}))
    return chunks

async def main() -> None:
    engine = RagramFusionEngine(
        storage=PostgresStorageEngine(os.environ["DATABASE_URL"], dimensions=384),
        embedder=SentenceTransformerEmbedder(),         # dimensions MUST match 384
        reranker=CrossEncoderReranker(),                # pip install 'ragram-pro[local]'
        config=FusionConfig(top_n=4, relevance_threshold=0.3),
    )
    async with engine:
        await engine.index(chunk_my_documents(load_files()))

        prompt = await engine.build_prompt("how do I roll back a failed deploy?")
        answer = await OllamaGenerator().generate(prompt)   # or your own chat call
        print(answer.answer)

asyncio.run(main())
```

Prefer a framework? Skip the plumbing entirely — feed your splitter's output to
`engine.index()` the same way, or use the LangChain / LlamaIndex retrievers under
[Integrations](#integrations), which run the whole pipeline behind their standard
`retrieve()` interface.

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
   `[]` or fail without failing the query (see [Edge cases](#edge-case-behavior)).
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

## Detailed ingest / query flow

### Ingest (index) flow

```
chunks: Sequence[Chunk]          # Chunk(id, text, metadata) — id is the identity
   │
   ▼  engine.index(chunks)                                   engine.py:91
   ├── empty batch ───────────────────────► return 0 (no backend touched)
   ├── first call? ──► engine.initialize() ──► storage.initialize()     (idempotent:
   │                                                   │                schema/index
   │                                                   │                creation)
   ▼                                                   ▼
embedder.embed_documents([c.text, ...])      # one batched call, one vector per chunk
   │  len(vectors) != len(chunks) ──► RuntimeError (nothing written)
   ▼
zip(chunk, vector, strict=True) ──► EmbeddedChunk records
   │
   ▼  storage.upsert(records)                                storage backend
   ├── Postgres: INSERT INTO chunks (id, content, embedding, metadata)
   │             ON CONFLICT (id) DO UPDATE SET content, embedding, metadata
   │             • `tsv` is a GENERATED column → tsvector + GIN index refresh
   │             • HNSW index entry updated for the new embedding
   │             • connection failures retried with backoff → StorageError
   └── memory:  dict[id] = record (overwrite) + BM25 index rebuilt lazily
   │
   ▼
returns written count (int)
```

Ingestion invariants:

- **Idempotent by id** — re-indexing the same `Chunk.id` replaces text, embedding and
  metadata wholesale; ingestion can be replayed safely after a source document changes.
- **Embed before write** — the storage layer never sees text without a vector, so the
  dense and lexical representations can never drift apart within a row.
- **Validation order** — empty input returns `0` before any backend call; a misconfigured
  embedder (vector-count mismatch) fails *before* `upsert`, leaving the store untouched.
- **Schema lives in storage** — `initialize()` creates the Postgres table/tsvector/HNSW
  objects (or no-ops for memory) and is called automatically by `index()` and `async with`.

### Query flow

```
engine.search(query, top_n?, relevance_threshold?)                        engine.py:123
   │
   ├─ 0. validate ── blank query → ValueError; resolve per-call overrides against
   │                 FusionConfig; range-check top_n >= 1, threshold ∈ [0, 1]
   │
   ├─ 1. embed query ── embedder.embed_query(query)                        coordinator.py:45
   │        zero vector → dense stream short-circuits to [] (no NaN cosine rows)
   │
   ├─ 2. dual-stream retrieval ── asyncio.gather(...)                      coordinator.py:47
   │        ├─ vector:  storage.vector_search(qvec, candidate_limit)
   │        │            Postgres: ORDER BY embedding <=> $1::vector (cosine, HNSW)
   │        │            memory:   exact cosine over all rows
   │        └─ lexical: storage.lexical_search(query, candidate_limit)
   │                     Postgres: websearch_to_tsquery(OR-joined) + ts_rank, GIN
   │                     memory:   Okapi BM25
   │        each stream wrapped in _guarded_stream ── exception → log + []    coordinator.py:60
   │        latencies overlap; each stream independently capped at candidate_limit
   │
   ├─ 3. fusion ── fuse_streams(streams, method, k, weights)                fusion.py:137
   │        rrf:        score(c) = Σ_streams  weight_s / (k + rank_s(c))     k = 60
   │        normalized: min-max each stream to [0,1], then Σ weight_s · norm_s(c)
   │        dedup by chunk id, merge contributing-stream labels,
   │        sort by (-fusion_score, chunk.id) ──► list[FusedChunk]
   │
   ├─ 4. rerank ── reranker.rerank(raw_query, fused)                        engine.py:202
   │        cross-encoder / Cohere / term-overlap ──► relevance_score ∈ [0, 1]
   │        empty fused list → skip reranker entirely
   │        any exception → log + _fallback_rerank (min-max of fusion scores,
   │        engine.py:215) so retrieval still returns ranked results
   │
   ├─ 5. filter ── keep relevance_score >= effective_threshold              engine.py:165
   ├─ 6. truncate ── keep[:effective_top_n]                                 engine.py:166
   │
   ├─ 7. stats ── SearchStats(vector_hits, lexical_hits, fused_candidates,
   │                          reranked, kept, elapsed_ms)                   engine.py:168
   ▼
FusionResult(query, chunks, stats)
   │
   └─ build_prompt(query) ── build_generation_prompt() ──► GenerationPrompt
          system = grounding contract, user = [C1]…[Cn] evidence blocks      prompts.py
```

Flow invariants:

- **Ranks are 1-based** — rank 1 is the best hit, so the top RRF term is
  `weight / (k + 1)`; ties after fusion are broken deterministically by chunk id.
- **Reranker sees the raw query + chunk text**, never backend scores — its `[0, 1]`
  output is directly comparable across streams, which is why thresholding happens
  *after* reranking (cosine and ts_rank scales never meet).
- **`candidate_limit` bounds stage 2, not the final result** — it caps how many hits
  each stream contributes (rerank cost/latency knob); `top_n` caps stage 6.
- **Stats tell the drop-off story** — `vector_hits + lexical_hits → fused_candidates`
  (dedup) → `reranked` → `kept` (threshold + top_n) in one struct per query.
- **Partial failure is normal, not an error** — stages 1–3 degrade to whatever the
  surviving stream(s) returned; only stages 0 and a total embedder failure raise.

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
- `HashingEmbedder` is for demos and tests only — its vectors are hash buckets, not
  semantic meaning. Production results need a real model (`SentenceTransformerEmbedder`
  or your own `Embedder` subclass).
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

## Integrations

Adapters for trending RAG/LLM tooling live in `integrations/`. Framework names are
resolved **lazily** (PEP 562): `import integrations` never pulls in LangChain or
LlamaIndex, and touching a missing framework raises an `ImportError` naming the exact
install extra. Generation backends import their SDKs inside the constructors, so the
package itself stays dependency-free.

### LangChain retriever — `pip install 'ragram-pro[langchain]'`

```python
from integrations.langchain import RagramRetriever

retriever = RagramRetriever(engine=engine, top_n=5, relevance_threshold=0.4)

docs = retriever.invoke("what is the escalation path?")       # sync (bridges to async)
docs = await retriever.ainvoke("payment outage escalation")   # native async

chain = prompt | retriever | StrOutputParser()                # works in any LCEL chain
```

Each chunk becomes a `Document` whose metadata carries `chunk_id`,
`relevance_score`, `fusion_score`, `streams` plus the original provenance fields.
`invoke()` runs the hybrid pipeline on a private event loop when the caller has none,
so it is safe even inside Jupyter/asyncio apps.

### LlamaIndex retriever — `pip install 'ragram-pro[llamaindex]'`

```python
from integrations.llamaindex import RagramLlamaIndexRetriever

retriever = RagramLlamaIndexRetriever(engine, top_n=5, relevance_threshold=0.4)
nodes = retriever.retrieve("what is the escalation path?")    # list[NodeWithScore]
query_engine = retriever.as_query_engine(llm=llm)             # full RAG in one line
```

Every chunk becomes a `TextNode` scored by the reranker relevance, so LlamaIndex
similarity thresholds and response synthesizers behave normally.

### Generation backends

All backends consume the exact `system`/`user` messages from `engine.build_prompt()`,
so the grounding contract travels with the request:

```python
from integrations import AnthropicGenerator, OllamaGenerator, OpenAIGenerator

# OpenAI — pip install 'ragram-pro[openai]'  (OPENAI_API_KEY, OPENAI_MODEL)
openai_gen = OpenAIGenerator()                    # default model: gpt-4o-mini

# Anthropic — pip install 'ragram-pro[anthropic]'  (ANTHROPIC_API_KEY, ANTHROPIC_MODEL)
claude = AnthropicGenerator()                     # default model: claude-sonnet-4-5

# Ollama — stdlib only, fully local  (OLLAMA_HOST, OLLAMA_MODEL)
local = OllamaGenerator()                         # default model: llama3.1

# One-call pipeline: hybrid search -> prompt -> grounded answer
result = await local.answer(engine, "what is the escalation path for a payment outage?")
print(result.answer, result.usage)

# Or package first, generate later / swap backends:
prompt = await engine.build_prompt("payment outage escalation")
result = await claude.generate(prompt)
```

`GroundedGenerator` is the ABC — subclass it to plug in any other chat API. Backend
failures raise `GenerationError` with the install hint or the missing variable name.

### FastAPI example — `pip install 'ragram-pro[api]'`

```powershell
pip install fastapi uvicorn
uvicorn examples.fastapi_app:app --reload     # or: python examples/fastapi_app.py
```

```
GET  /health   -> status + indexed chunk count
POST /search   -> {"query", "chunks": [...], "stats": {...}}
POST /prompt   -> {"system", "user", "chunks", "stats"}   ready for any chat API
```

The example is configured entirely from `.env` (storage, embedder, reranker, fusion
knobs, `API_HOST`/`API_PORT`) — see [Environment configuration](#environment-configuration).

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

## Demo reference

The repo-root `app.py` demo (driven by CLI flags and `.env`):

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

Every flag's default can also be set through the environment (`RAGRAM_STORAGE`,
`RAGRAM_TOP_N`, ...) — see [Environment configuration](#environment-configuration).
Command-line flags always override the environment.

The demo prints per-stream stats, the ranked result table, and the exact generation
prompt (system + user) that would be sent to the LLM.

---

## Environment configuration

`python app.py` and `examples/fastapi_app.py` load a `.env` file at startup through
the dependency-free `envfile.py` loader (no python-dotenv). Copy the documented
template and edit it:

```powershell
copy .env.example .env      # .env is gitignored; .env.example is committed
python app.py               # defaults now come from .env
```

Resolution rules (`envfile.load_env`):

1. Explicit `path` argument → `$RAGRAM_ENV_FILE` → `./.env` → repo-root `.env`
   (first hit wins; a missing *explicit* file raises, auto-discovery is silent).
2. Real shell/CI environment variables **always win** over file values.
3. Blank values (`KEY=`) are treated as *unset*, so placeholders never mask defaults.
4. Syntax: `KEY=value`, `export KEY=value`, quoted values, `#` comments.

The library itself only ever reads `os.environ` — call `load_env()` in your entrypoint
(or export the variables in CI) and everything below picks them up:

| Variable | Default | Used by |
|---|---|---|
| `DATABASE_URL` | — | `PostgresStorageEngine` DSN |
| `RAGRAM_STORAGE` | `memory` | demo/example backend: `memory` \| `postgres` |
| `RAGRAM_EMBEDDER` | `hashing` | demo/example: `hashing` \| `sbert` |
| `RAGRAM_EMBEDDER_DIMENSIONS` | `256` | `HashingEmbedder(dimensions=...)` |
| `RAGRAM_EMBEDDER_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | `SentenceTransformerEmbedder` |
| `RAGRAM_RERANKER` | `noop` | demo/example: `noop` \| `hf` \| `cohere` |
| `RAGRAM_RERANKER_MODEL` | `BAAI/bge-reranker-base` | `CrossEncoderReranker` |
| `RAGRAM_FUSION` | `rrf` | `FusionConfig.fusion_method` |
| `RAGRAM_RRF_K` | `60` | `FusionConfig.rrf_k` |
| `RAGRAM_CANDIDATE_LIMIT` | `25` | `FusionConfig.candidate_limit` |
| `RAGRAM_THRESHOLD` | `0.3` | `FusionConfig.relevance_threshold` |
| `RAGRAM_TOP_N` | `4` | `FusionConfig.top_n` |
| `COHERE_API_KEY` / `COHERE_MODEL` | — / `rerank-v3.5` | `CohereReranker` |
| `OPENAI_API_KEY` / `OPENAI_MODEL` / `OPENAI_BASE_URL` | — / `gpt-4o-mini` | `OpenAIGenerator` |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | — / `claude-sonnet-4-5` | `AnthropicGenerator` |
| `OLLAMA_HOST` / `OLLAMA_MODEL` | `http://localhost:11434` / `llama3.1` | `OllamaGenerator` |
| `API_HOST` / `API_PORT` | `127.0.0.1` / `8000` | `examples/fastapi_app.py` direct run |

Programmatic use in your own entrypoint:

```python
from envfile import env_float, env_int, env_str, load_env

load_env()                              # populate os.environ from .env
top_n = env_int("RAGRAM_TOP_N", 4)      # typed getters with defaults
threshold = env_float("RAGRAM_THRESHOLD", 0.3)
```

---

## Tests

```powershell
python tests/test_pipeline.py                                    # 55 stdlib checks, no services needed
python tests/test_integrations.py                                # 47 integration checks (SDKs skipped if absent)
$env:DATABASE_URL = 'postgresql://user:pass@localhost:5432/db'
python tests/test_postgres_integration.py                        # 24 live Postgres checks
python -m pyflakes .                                             # lint
```

All suites exit non-zero on failure, so they slot directly into CI.

---

## Troubleshooting

Start every investigation with `result.stats` and the log line — they show exactly
where results dropped (`vector_hits + lexical_hits → fused → reranked → kept`).

| Symptom | What's happening | Fix |
|---|---|---|
| `result.is_empty` but the index has data | Look at stats: `vector_hits`/`lexical_hits` both `0` → query terms don't match (stopwords, language mismatch); hits `> 0` but `kept = 0` → threshold too high | Lower `relevance_threshold` per call, or `0` to see everything the reranker scored |
| Postgres: `different vector dimensions` | `PostgresStorageEngine(dimensions=...)` ≠ embedder dimensions, or you changed the embedding model | Set `dimensions=embedder.dimensions`; changing models means a new table + full re-index |
| Postgres: `extension "vector" does not exist` / permission denied | pgvector isn't installed or the role can't `CREATE EXTENSION` | `CREATE EXTENSION vector` as a superuser (or ask your admin), then retry |
| Search works but feels slow | Either no HNSW index (sequential scan on PG < 15) or rerank cost | Check for the HNSW warning in logs; lower `candidate_limit` (rerank cost scales with it) |
| Exact IDs / codes still miss | The lexical stream isn't matching: `tsconfig` stemming doesn't fit your content's language | Pass a matching `tsconfig` to `PostgresStorageEngine` (e.g. `simple` for code/IDs) |
| Old chunks linger after re-chunking | Ids are identities; there is no delete API | `DROP TABLE ragram_chunks` (or new `table=`) and re-index |
| Results look random | `HashingEmbedder` (hash buckets, not semantics) in use | `pip install sentence-transformers`, then `--embedder sbert` / `SentenceTransformerEmbedder()` |
| Log: `reranker failed; falling back...` | Backend down or missing (HF model not downloaded, `COHERE_API_KEY` unset) | Install the model/extra or set the key; retrieval still returns fusion-ordered results meanwhile |
| Threshold behaves "differently" per reranker | It does: term-overlap = % of query words present (0.3 → "at least 30% of query terms appear"); cross-encoder = calibrated relevance (junk ≈ 0, good ≈ 0.5–0.9) | Tune against your own questions using `stats.kept` and spot-checked scores; start at `0.3` |

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
(`postgres`, `cohere`, `local`, `langchain`, `llamaindex`, `openai`, `anthropic`,
`api` — see `pyproject.toml`).

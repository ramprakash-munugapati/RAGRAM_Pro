# Learning guide: the concepts behind RAGRAM_Pro

This document is for **understanding**, not for looking things up — the README is
the reference; this is the refresher. Every concept below is explained with a small
worked example, then tied to the exact place it lives in this pipeline, so theory
never floats free of code.

Read in order the first time; skim by heading afterwards.

---

## 1. RAG in 60 seconds

The baseline you already know:

```
your documents ──► split into chunks ──► embed each chunk to a vector
                                              │
user question ──► embed to a vector ───────────┴──► similarity search
                                                          │
                                              top-k chunks ──► stuff into the LLM prompt
```

It works. It also fails in four predictable ways — and each failure is what the
rest of this document (and this project) addresses:

| # | Failure | Concept that fixes it |
|---|---|---|
| 1 | Exact tokens (IDs, codes, error names) don't match by meaning | [Sparse retrieval](#3-sparse-retrieval-matching-words-not-meaning) |
| 2 | The *closest* chunk isn't the one that *answers* the question | [Reranking](#5-reranking-judging-relevance-properly) |
| 3 | Two different matchers each catch things the other misses | [Fusion](#4-why-two-streams-fusion) |
| 4 | The LLM happily answers from memory when evidence is weak | [Prompt packaging](../README.md#building-the-llm-prompt) |

---

## 2. Dense retrieval: embeddings and cosine similarity

**The idea.** A language model maps text to a point in high-dimensional space
(384–1024 numbers for common models) so that *similar meanings land near each
other*. "Page the on-call engineer" and "alert the engineer on duty" end up close
even though they share almost no words.

**The comparison.** Vectors are compared by **cosine similarity** — the cosine of
the angle between them:

```
cos(A, B) = A · B / (|A| × |B|)
```

Only *direction* matters, not length (a 30-word chunk and a 500-word chunk about
the same thing score similarly). The result sits in `[-1, 1]`; for real embedding
models, unrelated text lands near 0 and near-paraphrases climb toward 1.

**What dense retrieval is good at:**
- paraphrases and synonyms — zero shared keywords, still matches
- fuzzy, conversational questions ("what do we do when payments catch fire")

**What it blurs:**
- exact strings: `ECONNRESET`, `INV-20491`, `v2.3.1-rc1` — embeddings treat these
  as just more tokens; nearest neighbor by meaning can quietly miss them
- rare proper nouns and internal jargon underrepresented in the model's training

**The trap:** cosine ranks by *meaning proximity*, not by *answering the question*.
A chunk about "payment timeouts" can be semantically nearer to "payment outage"
than the chunk that actually names the escalation ladder.

**In RAGRAM:**
- the same model embeds documents at index time and the query at search time —
  see [Embedders](../README.md#embedders)
- `HashingEmbedder` (the zero-dependency default) produces hash buckets, not
  meaning — it exists so the demo runs, never for real retrieval
- `dimensions` must equal the Postgres `vector(dims)` column, or nothing is
  comparable

---

## 3. Sparse retrieval: matching words, not meaning

**The idea.** Count words — but weight them smartly. The evolution in one breath:

1. **Term frequency** — more occurrences ⇒ more relevant. Problem: the word
   "the" appears everywhere.
2. **TF-IDF** — multiply by *inverse document frequency*: a term that appears in
   few documents is a stronger signal. "postmortem" outvotes "incident".
3. **BM25** (the modern default, built into this project's memory engine) —
   TF-IDF plus two refinements: *diminishing returns* (5 mentions beat 2, but
   not 2.5×) and *length normalization* (short documents don't win just by being
   short).

**Postgres does the same job differently:**

| Piece | What it is |
|---|---|
| `tsvector` | the document pre-converted to normalized word forms ("lexemes") — RAGRAM makes this a **generated column**, so it's computed automatically on write |
| GIN index | an *inverted index*: for each word, the list of documents containing it — turns "which docs contain X" into a lookup instead of a scan |
| `ts_rank` | the score — a lightweight relevance number like TF-IDF |
| `tsconfig` | the language rules: stopwords and **stemming** ("running", "ran", "runs" → `run`). This is also sparse retrieval's main failure mode: English stemming on non-English text matches nothing |

**Where sparse wins:** IDs, error codes, config keys, version strings, product
names, acronyms — anything where the *exact characters* are the meaning.

**In RAGRAM:** the lexical stream builds an **OR** query
(`websearch_to_tsquery`) — *any* matching term contributes. That's deliberately
recall-oriented: catching too much is the reranker's problem to fix. See
[Storage backends](../README.md#storage-backends).

---

## 4. Why two streams? Fusion

Dense finds paraphrases; sparse finds exact tokens. Each has blind spots the
other covers — so run both, then **merge two ranked lists into one**.

### Reciprocal Rank Fusion (RRF)

The default merger, and the trick is that it uses **only ranks, not scores**.
Cosine values and `ts_rank` values live on incomparable scales — RRF throws the
scales away:

```
score(chunk) = Σ over streams   weight_stream / (k + rank_stream(chunk))
```

**Worked example** (`k = 60`, weights 1.0):

| Chunk | vector rank | lexical rank | calculation | RRF score |
|---|---|---|---|---|
| A | 1 | — | `1/61` | 0.0164 |
| B | 2 | — | `1/62` | 0.0161 |
| C | 3 | 1 | `1/63 + 1/61` | **0.0323** |
| D | — | 2 | `1/62` | 0.0161 |

Final order: **C → A → B → D** (B and D tie; chunk id breaks it deterministically).

Read what happened:

- **C wins despite being only #3 in the dense stream** — appearing in *both*
  streams is worth more than a single #1. Agreement between independent matchers
  is evidence.
- **A (a solo #1) beats B (a solo #2)** — rank still matters within a stream.
- **B and D scored identically** — a #2 from either stream contributes the same.

**What `k` does:** it dampens how much the top ranks dominate. At `k = 60`, rank 1
(`1/61`) and rank 2 (`1/62`) are nearly equal — so *whether a stream found the
chunk* matters more than *where*. Drop `k` to a small number and rank 1 pulls far
ahead of rank 2 (at `k = 1`: `1/2` vs `1/3`). Smaller `k` ⇒ sharper top-rank
weighting.

**The alternative — `normalized` fusion:** min-max each stream's raw scores to
`[0, 1]` (best hit → 1.0, worst → 0.0), then blend with `vector_weight` /
`lexical_weight`. Use this when you *trust* score magnitudes and want gaps to
influence the merge; use RRF (default) when you don't.

**Deduplication:** a chunk found by both streams appears once, with its scores
combined and `streams` labelled `("vector", "lexical")`.

**In RAGRAM:** `FusionConfig(fusion_method=..., rrf_k=..., vector_weight=...,
lexical_weight=...)` — see
[Configuration reference](../README.md#how-the-pipeline-works) and the formulas'
home in [search/fusion.py](../README.md#detailed-ingest--query-flow).

---

## 5. Reranking: judging relevance properly

Fusion merges two lists, but it only knows **positions** — not whether a chunk
actually *answers* the question. The reranker fixes failure #2 from section 1.

### Bi-encoder vs cross-encoder

```
  BI-ENCODER (retrieval — runs thousands of times)     CROSS-ENCODER (rerank — runs dozens)

  query ──►[model]──► q-vector  ┐                      ┌ query  ┐
                                 ├► cosine ─► score                      ┘
  chunk ──►[model]──► c-vector  ┘                      │
                                                       ├►[joint transformer]──► score in [0,1]
  (two separate passes — vectors can be                chunk ┘
   computed once and cached)                           (query and chunk read TOGETHER,
                                                        token by token — no caching)
```

- The **bi-encoder** *is* your embedding model. Separable, cheap, cacheable —
  and blind: it never actually reads the query and chunk side by side.
- The **cross-encoder** reads the pair jointly, so it can notice the difference
  between a chunk that mentions "payment" and one that answers "escalation path".
  Far more accurate, far too slow to run over your whole corpus — which is
  exactly why it runs *after* fusion narrowed candidates to a few dozen.

The raw model logit passes through a sigmoid, so scores arrive calibrated in
`[0, 1]`. Because they're on one shared scale — unlike cosine vs `ts_rank` —
**thresholding happens after reranking**, never before.

### Failure behavior

If the reranker itself fails (model missing, API down), RAGRAM logs it and
**min-max normalizes the fusion scores** into pseudo-relevance scores: you lose
precision, not the query. See [Edge-case behavior](../README.md#edge-case-behavior).

### Backend note

`TermOverlapReranker` (the dependency-free demo) scores = *fraction of query
words present in the chunk*. Honest, instant, shallow — `0.3` literally means
"at least 30% of your query terms appear". A cross-encoder's `0.3` means
something quite different: junk scores near 0, solid matches cluster 0.5–0.9.
Same knob, different scales — which is why
[Troubleshooting](../README.md#troubleshooting) says to re-tune when you swap
backends.

**In RAGRAM:** [Rerankers](../README.md#rerankers) — swap backends with one
argument; the pipeline only sees the `Reranker` interface.

---

## 6. Indexes, latency and cost

**HNSW** (used for the Postgres vector column, PostgreSQL 15+) — *Hierarchical
Navigable Small World* graph: a proximity graph over your vectors with long-range
"highway" edges on top. Search hops from a few entry points toward the answer in
roughly `O(log n)` steps instead of scanning every row. It's **approximate** —
it can miss a true nearest neighbor now and then — the classic, usually good,
speed-for-tiny-recall trade. Without it (older Postgres), search falls back to an
exact sequential scan: correct, but `O(n)` per query.

**GIN** (for the `tsvector` column) — an inverted index: per word, a posting list
of documents. "Find docs containing any of these words" becomes a merge of short
lists.

**Where the time actually goes** per query:

```
embed query        ~1–10 ms   (model-dependent)
two stream scans   ~1–50 ms   (indexes do the heavy lifting)
fusion             <1 ms      (pure math on a few dozen rows)
RERANK             the big one — cost ≈ candidates × model/API latency
```

Which gives the two knobs their meaning:

| Knob | What it bounds | Move it when... |
|---|---|---|
| `candidate_limit` | how many hits **each stream** contributes → rerank workload (at most `2 × candidate_limit` before dedup) | rerank latency dominates: turn **down**; you trade some recall |
| `top_n` | how many chunks reach the prompt | token budget: turn **down** — reranked chunks are already the best |

**In RAGRAM:** the HNSW warning appears in logs if the index couldn't be created;
`stats.elapsed_ms` measures the whole funnel. See
[Detailed ingest / query flow](../README.md#detailed-ingest--query-flow).

---

## 7. Tuning: precision, recall and reading stats

Plain-language definitions for this pipeline:

- **Recall** — of the chunks that *could* answer the question, how many survived
  retrieval and thresholding? (Bad recall: right answer never reaches the LLM.)
- **Precision** — of the chunks in the final prompt, how many actually help?
  (Bad precision: the LLM wastes attention on near-misses — or worse, cites them.)

Every knob trades one for the other, and every query reports the funnel:

```
vector_hits + lexical_hits  →  fused_candidates  →  reranked  →  kept
        retrieval                dedup across        scored      threshold
       (recall here)              streams          (ordering)    + top_n
                                                                             (both)
```

**Reading a bad query:**

1. `vector_hits = lexical_hits = 0` → nothing matched at all: stopwords, wrong
   language/stemming, or empty index.
2. Hits exist, `kept = 0` → the reranker scored everything below the threshold:
   lower `relevance_threshold` (try `0` to *see* the scores first).
3. `kept` small but you expected more → `top_n` truncating, or threshold shaving
   the tail.

**Tuning method (do this once, it pays forever):**

1. Write 15–20 questions **with known answers** from *your* corpus — include the
   hard cases: exact IDs, paraphrases, jargon.
2. Run them; check: did the chunk you wanted survive to `kept`? (recall)
   Are the survivors actually good? (precision)
3. Move one knob at a time — threshold first, then `candidate_limit`, then `k`.

Start at `relevance_threshold = 0.3`; raise it if junk reaches the prompt, lower
it if correct chunks vanish. Full symptom table:
[Troubleshooting](../README.md#troubleshooting).

---

## 8. Try it (with the demo)

The demo corpus is 10 ops/policy rows — tiny, but every knob is observable. Run
from the repo root; each command ends with a stats funnel you can read.

```powershell
# 1. Baseline — watch the funnel: vector=? lexical=? fused=? reranked=? kept=?
python app.py

# 2. Threshold is the precision dial:
python app.py --threshold 0        # kept jumps — every candidate survives
python app.py --threshold 0.9      # kept collapses — only near-certain chunks

# 3. Fusion is tunable, not magic:
python app.py --fusion normalized --rrf-k 10
#    small k → rank 1 dominates; normalized → score gaps start mattering

# 4. Exact-ID query — the lexical stream's home turf:
python app.py --query "FIN-REFUND-020"

# 5. Honest dense behavior needs a real embedder (downloads once):
pip install sentence-transformers
python app.py --embedder sbert --reranker hf --threshold 0.3
#    now both streams carry meaning and the scores are calibrated relevance
```

**What to look for:** `kept` is where precision lives; the gap between
`fused_candidates` and `reranked` shows deduplication at work; id-style queries
should light up `lexical_hits` while `vector_hits` stays flat.

---

## 9. Glossary

| Term | One line |
|---|---|
| **Chunk** | The unit of retrieval: a piece of text plus an id and metadata; you create them, RAGRAM stores them |
| **Embedding** | A vector of numbers representing text so similar meanings sit nearby |
| **Cosine similarity** | Angle-based similarity between two vectors; ignores magnitude, cares about direction |
| **Dense retrieval** | Similarity search over embeddings — meaning-based |
| **Sparse retrieval** | Word-matching search (BM25 / full-text) — literal, token-based |
| **BM25** | The classic ranking function: term frequency, rare-term weighting, length normalization |
| **IDF** | *Inverse document frequency* — rare words are stronger evidence than common ones |
| **`tsvector`** | Postgres' normalized, stemmed representation of document text |
| **`ts_rank`** | Postgres' relevance score for a full-text match |
| **`tsconfig`** | Language rules (stopwords + stemming) for full-text search |
| **Stemming** | Reducing words to a root ("runs" → "run") so inflections match |
| **GIN index** | Inverted index making "which docs contain this word" instant |
| **HNSW** | Approximate nearest-neighbor graph index; `O(log n)` vector search |
| **Stream** | One retrieval path: `"vector"` (dense) or `"lexical"` (sparse) |
| **Fusion** | Merging both streams' ranked lists into one |
| **RRF** | Reciprocal Rank Fusion: `Σ weight / (k + rank)` — rank-only, scale-free |
| **Bi-encoder** | Model embeddinging query and document *separately* (your embedder); fast, cacheable |
| **Cross-encoder** | Model scoring query and document *together*; slow, accurate — the reranker |
| **Reranker** | The stage re-scoring candidates by true query↔chunk relevance in `[0, 1]` |
| **`relevance_threshold`** | Minimum reranker score to keep a chunk — the precision dial |
| **`candidate_limit`** | Per-stream retrieval cap — bounds rerank cost |
| **`top_n`** | Final chunk count sent to the LLM — bounds prompt size |
| **Precision / Recall** | Of what you sent, how much helped / of what could help, how much you sent |
| **Provenance metadata** | Your per-chunk fields (`source`, `updated`, ...) shown to the LLM and frameworks |

---

## Where to go next

- See it end-to-end: [Detailed ingest / query flow](../README.md#detailed-ingest--query-flow)
- Wire it into your app: [Bringing your own data](../README.md#bringing-your-own-data) ·
  [Integrations](../README.md#integrations)
- Ship it: [Production checklist](../README.md#production-checklist) ·
  [Troubleshooting](../README.md#troubleshooting)

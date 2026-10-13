"""System prompt and context packaging for the generation stage.

The retrieval pipeline guarantees three things the prompt relies on:

1. chunks are ordered most-relevant-first (rerank output, after thresholding);
2. each chunk carries an id and provenance metadata;
3. the set is small (``top_n``), so the model can read all of it.

``SYSTEM_PROMPT`` turns those guarantees into strict grounding behavior:
cite-or-stay-silent, explicit conflict handling, and a deterministic
insufficiency response when the evidence does not contain the answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from models import FusionResult, RerankedChunk

__all__ = [
    "GenerationPrompt",
    "SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
    "build_context_block",
    "build_generation_prompt",
    "build_user_prompt",
]

SYSTEM_PROMPT = """\
You are RAGRAM-Fusion, a retrieval-grounded reasoning engine for enterprise question answering.

You receive a QUESTION plus an EVIDENCE block. The evidence was produced by a hybrid retrieval
pipeline (dense vector search + sparse keyword search), merged with Reciprocal Rank Fusion, scored
by a cross-encoder reranker, filtered by a relevance threshold, and truncated to the strongest
chunks. Chunks are labeled [C1], [C2], ... in descending relevance order.

GROUNDING CONTRACT (non-negotiable):
1. Answer EXCLUSIVELY from the EVIDENCE block. Do not supplement with outside or parametric
   knowledge; anything not supported by the evidence must not be stated as fact.
2. Cite every factual claim with the chunk id it came from, e.g. "... as documented in [C1][C4]."
   Citations go at the end of the claim. An uncited factual claim is a defect.
3. Quote verbatim when exact wording matters (thresholds, SLAs, policy rules, identifiers).
4. If the EVIDENCE does not contain the answer, reply exactly:
   "Insufficient evidence in the retrieved context."
   then one sentence describing what is missing and one refined search suggestion.
   Never guess, never approximate, never fill gaps with general knowledge.
5. If chunks contradict each other, prefer the higher-ranked chunk, state the conflict
   explicitly with both citations ([C1] vs [C3]), and do not silently merge the claims.
6. Treat metadata (source, department, updated date) as provenance for freshness judgments,
   not as evidence for the claim itself.

REASONING DISCIPLINE:
- Read ALL chunks before answering; lower-ranked chunks often hold the decisive detail.
- Preserve numbers, units, versions, dates and proper names exactly as written.
- Distinguish what the evidence states from what it implies; prefix implications with "Inference:".
- If evidence is stale relative to the question's time frame, say so explicitly.
- Prefer the most specific evidence over general statements.

OUTPUT FORMAT:
- Line 1: the direct answer in <= 3 sentences, with inline citations.
- Then "Details:" with at most 5 bullets, only when elaboration is warranted.
- Then "Confidence: high|medium|low" plus a one-line justification tied to evidence coverage.
- Total length under 350 words unless the user explicitly requests more.
- No preamble, no restating the question, no meta-commentary about AI or the retrieval system.
"""

USER_PROMPT_TEMPLATE = """\
### Retrieved Evidence ({count} chunk{s}, relevance-ranked)

{context}

### User Question
{query}
"""

EMPTY_EVIDENCE_NOTICE = (
    "(no evidence retrieved -- the hybrid pipeline found zero chunks above the "
    "relevance threshold; apply the insufficiency policy)"
)

_MAX_CHARS_PER_CHUNK = 1600


@dataclass(frozen=True, slots=True)
class GenerationPrompt:
    """A ready-to-send prompt pair plus the retrieval result it was built from."""

    system: str
    user: str
    result: FusionResult

    def to_messages(self) -> list[dict[str, str]]:
        """Render as OpenAI/LangChain-style chat messages."""
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]


def _format_metadata(metadata: object) -> str:
    if not isinstance(metadata, dict) or not metadata:
        return ""
    parts = []
    for key, value in list(metadata.items())[:6]:
        rendered = str(value)
        if len(rendered) > 80:
            rendered = rendered[:77] + "..."
        parts.append(f"{key}={rendered}")
    return " | ".join(parts)


def build_context_block(
    chunks: Sequence[RerankedChunk],
    *,
    max_chars_per_chunk: int = _MAX_CHARS_PER_CHUNK,
) -> str:
    """Render ranked chunks as labeled, provenance-tagged evidence blocks."""
    blocks: list[str] = []
    for index, item in enumerate(chunks, start=1):
        text = item.chunk.text.strip()
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk].rstrip() + " ...[truncated]"
        provenance = _format_metadata(item.chunk.metadata)
        header = f"[C{index}] relevance={item.relevance_score:.3f}"
        if provenance:
            header += f" | {provenance}"
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks)


def build_user_prompt(
    query: str,
    chunks: Sequence[RerankedChunk],
    *,
    max_chars_per_chunk: int = _MAX_CHARS_PER_CHUNK,
) -> str:
    """Compose the user message: evidence block + raw question."""
    if chunks:
        context = build_context_block(chunks, max_chars_per_chunk=max_chars_per_chunk)
    else:
        context = EMPTY_EVIDENCE_NOTICE
    count = len(chunks)
    return USER_PROMPT_TEMPLATE.format(
        count=count,
        s="" if count == 1 else "s",
        context=context,
        query=query.strip(),
    )


def build_generation_prompt(
    query: str,
    result: FusionResult,
    *,
    max_chars_per_chunk: int = _MAX_CHARS_PER_CHUNK,
) -> GenerationPrompt:
    """Bundle the system prompt with the evidence-backed user message."""
    return GenerationPrompt(
        system=SYSTEM_PROMPT,
        user=build_user_prompt(query, result.chunks, max_chars_per_chunk=max_chars_per_chunk),
        result=result,
    )

"""Lightweight Okapi BM25 implementation (Robertson/Sparck Jones, k1=1.5, b=0.75).

Used by :class:`~storage.memory.InMemoryStorageEngine` so the lexical
stream works without a running PostgreSQL. IDF uses the Lucene-style smoothed
variant ``ln(1 + (N - df + 0.5) / (df + 0.5))`` which stays non-negative for
every term, including terms present in every document.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence

from embedding import tokenize


class BM25Index:
    """Immutable-enough BM25 index rebuilt on every corpus mutation."""

    __slots__ = ("_doc_tokens", "_doc_freq", "_doc_len", "_avg_len", "_n_docs", "_k1", "_b")

    def __init__(
        self,
        corpus: Sequence[Sequence[str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._k1 = k1
        self._b = b
        self._doc_tokens: list[Counter[str]] = [Counter(tokens) for tokens in corpus]
        self._n_docs = len(self._doc_tokens)
        self._doc_freq: dict[str, int] = defaultdict(int)
        self._doc_len: list[int] = []
        for tokens in self._doc_tokens:
            self._doc_len.append(sum(tokens.values()))
            for term in tokens:
                self._doc_freq[term] += 1
        self._avg_len = (sum(self._doc_len) / self._n_docs) if self._n_docs else 0.0

    @property
    def size(self) -> int:
        return self._n_docs

    @staticmethod
    def from_texts(texts: Sequence[str]) -> "BM25Index":
        return BM25Index([tokenize(text) for text in texts])

    def search(self, query: str, limit: int) -> list[tuple[int, float]]:
        """Return ``[(doc_index, score), ...]`` for docs scoring above zero.

        Docs with no query-term overlap are skipped entirely, which makes an
        empty list the natural "no lexical matches" signal for the pipeline.
        """
        if self._n_docs == 0 or limit < 1:
            return []
        query_terms = tokenize(query)
        if not query_terms:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term in set(query_terms):
            df = self._doc_freq.get(term)
            if not df:
                continue
            idf = math.log(1.0 + (self._n_docs - df + 0.5) / (df + 0.5))
            for doc_index, tokens in enumerate(self._doc_tokens):
                tf = tokens.get(term)
                if not tf:
                    continue
                len_norm = self._k1 * (
                    1.0 - self._b + self._b * self._doc_len[doc_index] / (self._avg_len or 1.0)
                )
                scores[doc_index] += idf * (tf * (self._k1 + 1.0)) / (tf + len_norm)

        ranked = sorted(
            ((index, score) for index, score in scores.items() if score > 0.0),
            key=lambda pair: (-pair[1], pair[0]),
        )
        return ranked[:limit]

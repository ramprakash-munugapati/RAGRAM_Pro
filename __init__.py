"""RAGRAM_Pro: hybrid search (dense + sparse) with RRF fusion and cross-encoder reranking.

Flat project layout: this file doubles as a bundled facade so the whole API can
also be imported as a package from the parent directory (``import RAGRAM_Pro``),
while day-to-day code imports the flat modules directly
(``from engine import RagramFusionEngine``).
"""

from .config import FusionConfig
from .embedding import (
    Embedder,
    EmbeddingError,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    tokenize,
)
from .engine import RagramFusionEngine
from .models import (
    Chunk,
    EmbeddedChunk,
    FusionResult,
    FusedChunk,
    RerankedChunk,
    SearchHit,
    SearchStats,
)
from .prompts import (
    GenerationPrompt,
    SYSTEM_PROMPT,
    build_context_block,
    build_generation_prompt,
    build_user_prompt,
)
from .rerank import (
    CohereReranker,
    CrossEncoderReranker,
    Reranker,
    RerankerError,
    TermOverlapReranker,
)
from .search import (
    HybridSearchCoordinator,
    fuse_streams,
    normalized_score_fusion,
    reciprocal_rank_fusion,
)
from .storage import (
    BM25Index,
    InMemoryStorageEngine,
    PostgresStorageEngine,
    StorageEngine,
    StorageError,
)

__version__ = "8.0.0"

__all__ = [
    "BM25Index",
    "CohereReranker",
    "Chunk",
    "CrossEncoderReranker",
    "Embedder",
    "EmbeddingError",
    "EmbeddedChunk",
    "FusionConfig",
    "FusionResult",
    "FusedChunk",
    "GenerationPrompt",
    "HashingEmbedder",
    "HybridSearchCoordinator",
    "InMemoryStorageEngine",
    "PostgresStorageEngine",
    "Reranker",
    "RerankerError",
    "RerankedChunk",
    "RagramFusionEngine",
    "SearchHit",
    "SearchStats",
    "SentenceTransformerEmbedder",
    "SYSTEM_PROMPT",
    "StorageEngine",
    "StorageError",
    "TermOverlapReranker",
    "build_context_block",
    "build_generation_prompt",
    "build_user_prompt",
    "fuse_streams",
    "normalized_score_fusion",
    "reciprocal_rank_fusion",
    "tokenize",
    "__version__",
]

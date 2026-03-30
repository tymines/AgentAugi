"""Gemini embedding provider for PlanCache semantic similarity.

Wraps Google's ``models/gemini-embedding-001`` via the ``google-genai`` SDK.
Maintains an in-memory text→vector cache to minimise API round-trips.

Usage (standalone)::

    from evoagentx.core.gemini_embedder import GeminiEmbedder, make_gemini_plan_cache

    embedder = GeminiEmbedder()                          # reads GEMINI_API_KEY from env
    vec = embedder("search the web for Paris weather")   # List[float], 3072 dims

    cache = make_gemini_plan_cache(similarity_threshold=0.80)

Package requirement::

    pip install google-genai
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

from .plan_cache import PlanCache

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "models/gemini-embedding-001"
_EMBEDDING_DIMS = 3072  # gemini-embedding-001 output size


class GeminiEmbedder:
    """Callable that maps text → float vector via Gemini embedding API.

    Args:
        api_key: Gemini API key.  Falls back to ``GEMINI_API_KEY`` then
            ``GOOGLE_API_KEY`` environment variables.
        model: Gemini embedding model name.  Defaults to
            ``models/gemini-embedding-001`` (3 072 dims).
        cache_size: Max number of (text, embedding) pairs kept in memory.
            When the cache is full, the oldest half is evicted.

    Raises:
        ValueError: If no API key is available.
        ImportError: If ``google-genai`` is not installed.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = _DEFAULT_MODEL,
        cache_size: int = 1000,
    ) -> None:
        self._api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not self._api_key:
            raise ValueError(
                "Gemini API key not found. Set the GEMINI_API_KEY environment "
                "variable, or pass api_key= explicitly."
            )
        self._model = model
        self._cache: Dict[str, List[float]] = {}
        self._cache_order: List[str] = []  # insertion-order tracking for eviction
        self._cache_size = cache_size
        self._client = self._build_client()
        logger.debug("GeminiEmbedder: ready — model=%s cache_size=%d", model, cache_size)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(self, text: str) -> List[float]:
        """Embed *text* and return a float vector.

        Results are cached by exact text to avoid redundant API calls.

        Args:
            text: The string to embed.

        Returns:
            Embedding vector as ``List[float]`` (length 3 072 for the default
            model).
        """
        if text in self._cache:
            return self._cache[text]

        result = self._client.models.embed_content(
            model=self._model,
            contents=text,
        )
        vector: List[float] = list(result.embeddings[0].values)

        # Evict oldest half on overflow
        if len(self._cache) >= self._cache_size:
            evict_count = max(1, len(self._cache_order) // 2)
            for key in self._cache_order[:evict_count]:
                self._cache.pop(key, None)
            self._cache_order = self._cache_order[evict_count:]

        self._cache[text] = vector
        self._cache_order.append(text)
        return vector

    @property
    def model(self) -> str:
        """The Gemini model name used for embedding."""
        return self._model

    @property
    def cache_size(self) -> int:
        """Number of embeddings currently held in the in-memory cache."""
        return len(self._cache)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_client(self):  # type: ignore[return]
        try:
            from google import genai  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for Gemini embeddings. "
                "Install it with: pip install google-genai"
            ) from exc
        return genai.Client(api_key=self._api_key)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_gemini_plan_cache(
    api_key: Optional[str] = None,
    model: str = _DEFAULT_MODEL,
    similarity_threshold: float = 0.80,
    max_templates: int = 500,
    ttl_seconds: Optional[float] = None,
    cost_tracker=None,
    embedder_cache_size: int = 1000,
) -> PlanCache:
    """Create a :class:`~evoagentx.core.plan_cache.PlanCache` backed by Gemini embeddings.

    Shorthand for::

        embedder = GeminiEmbedder(api_key=api_key, model=model)
        cache = PlanCache(embed_fn=embedder, similarity_threshold=0.80, ...)

    If ``google-genai`` is not installed or no API key is available, raises
    ``ImportError`` / ``ValueError`` rather than falling back silently, so that
    misconfiguration is caught early.

    Args:
        api_key: Gemini API key.  Reads ``GEMINI_API_KEY`` from env if omitted.
        model: Gemini embedding model name.
        similarity_threshold: Cosine similarity threshold for a cache hit
            (0–1, default 0.80).  Higher than the Jaccard default of 0.75
            because cosine distances are more discriminating.
        max_templates: Maximum number of cached plan templates.
        ttl_seconds: Optional per-template TTL.  ``None`` = no expiry.
        cost_tracker: Optional :class:`~evoagentx.core.cost_tracker.CostTracker`.
        embedder_cache_size: Max in-memory embeddings cached by the embedder.

    Returns:
        Configured :class:`~evoagentx.core.plan_cache.PlanCache` instance.
    """
    embedder = GeminiEmbedder(
        api_key=api_key,
        model=model,
        cache_size=embedder_cache_size,
    )
    return PlanCache(
        embed_fn=embedder,
        similarity_threshold=similarity_threshold,
        max_templates=max_templates,
        ttl_seconds=ttl_seconds,
        cost_tracker=cost_tracker,
    )


__all__ = ["GeminiEmbedder", "make_gemini_plan_cache", "_DEFAULT_MODEL", "_EMBEDDING_DIMS"]

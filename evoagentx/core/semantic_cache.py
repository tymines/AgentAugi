"""Embedding-based semantic cache for LLM responses.

Goes beyond exact-match or Jaccard-based caching (Phase 0's CachingLLM) by
using embedding similarity to identify *semantically equivalent* prompts that
differ in surface form but request the same information.

Design
------
``SemanticCache`` is a standalone cache that stores (query, embedding, response)
triples.  It provides:

- **Exact match** — checked first, O(1) dict lookup.
- **Semantic match** — linear scan over stored embeddings using cosine similarity.
  Returns the best hit above ``similarity_threshold``.
- **Graceful degradation** — if the embedding function raises, falls back to exact
  match only (no silent failure, just a cache miss).
- **TTL eviction** — entries older than ``ttl_seconds`` are expired on next access.
- **LRU-style eviction** — when the cache is full, the least-recently-used entry is
  removed (tracked via access timestamp).
- **Drop-in wrap** — ``SemanticCache.wrap(generate_fn)`` returns a cached version of
  any callable ``(prompt: str) -> str``, serving as a drop-in upgrade for Phase 0's
  CachingLLM which uses exact + Jaccard matching.

Typical usage
-------------
    >>> from evoagentx.core.semantic_cache import SemanticCache
    >>> cache = SemanticCache(embed_fn=my_embed, similarity_threshold=0.95)
    >>> cached_generate = cache.wrap(my_llm_generate)
    >>> response = cached_generate("What is 2 + 2?")   # cache miss → calls LLM
    >>> response = cached_generate("What does 2 + 2 equal?")  # cache hit
    >>> print(cache.stats)

The cache is designed for EvoPrompt and similar optimizers where many candidate
prompts are near-duplicates — a 0.95 cosine threshold catches rephrased variants
while avoiding false positives on distinct queries.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .logging import logger


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """One stored cache record.

    Attributes:
        query: Original query string.
        embedding: Optional embedding vector of the query.  None if the
            embedding function was unavailable when the entry was created.
        response: Stored LLM response.
        created_at: Unix timestamp when the entry was stored.
        last_accessed_at: Unix timestamp of most recent hit.
        hit_count: Number of times this entry served a cache hit.
    """

    query: str
    embedding: Optional[List[float]]
    response: str
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    hit_count: int = 0

    def touch(self) -> None:
        """Update last_accessed_at and increment hit_count."""
        self.last_accessed_at = time.time()
        self.hit_count += 1


@dataclass
class CacheStats:
    """Cumulative statistics for a ``SemanticCache`` instance.

    Attributes:
        total_queries: All ``get()`` calls since construction or last reset.
        exact_hits: Queries served from the exact-match layer.
        semantic_hits: Queries served from the semantic-similarity layer.
        misses: Queries that found no hit and required an LLM call.
        evictions: Number of entries evicted (TTL or LRU).
        cache_size: Current number of entries in the cache.
    """

    total_queries: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    evictions: int = 0
    cache_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of queries served from cache, in [0, 1]."""
        if self.total_queries == 0:
            return 0.0
        return (self.exact_hits + self.semantic_hits) / self.total_queries

    def __str__(self) -> str:
        return (
            f"CacheStats(queries={self.total_queries}, "
            f"exact_hits={self.exact_hits}, semantic_hits={self.semantic_hits}, "
            f"misses={self.misses}, hit_rate={self.hit_rate:.1%}, "
            f"size={self.cache_size}, evictions={self.evictions})"
        )


# ---------------------------------------------------------------------------
# Internal math helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Cosine similarity between two vectors, clamped to [0, 1].

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        Cosine similarity in [0, 1].  Returns 0.0 on zero vectors.

    Raises:
        ValueError: If vectors have different lengths.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Embedding dimension mismatch: {len(vec_a)} vs {len(vec_b)}"
        )
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------

class SemanticCache:
    """Embedding-based semantic cache for LLM call deduplication.

    Caches LLM responses and retrieves them for semantically equivalent queries
    without repeating API calls.  Two lookup layers are tried in order:

    1. **Exact match** — ``query == stored_query``.
    2. **Semantic match** — cosine similarity between embeddings ≥ threshold.

    Args:
        embed_fn: Callable ``(query: str) -> List[float]``.  If ``None``, only
            exact matching is performed.
        similarity_threshold: Cosine similarity threshold for a semantic hit
            (default 0.95).  Higher values = stricter matching.
        max_size: Maximum number of entries.  When exceeded, the least-recently-
            used entry is evicted (default 1000).
        ttl_seconds: Time-to-live for entries in seconds.  Expired entries are
            removed on access.  ``None`` (default) means entries never expire.

    Note:
        The semantic search is currently a linear scan (O(N)).  For caches with
        >10k entries consider replacing with an ANN index such as HNSW.
    """

    def __init__(
        self,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        similarity_threshold: float = 0.95,
        max_size: int = 1000,
        ttl_seconds: Optional[float] = None,
    ) -> None:
        if not (0.0 < similarity_threshold <= 1.0):
            raise ValueError(
                f"similarity_threshold must be in (0, 1], got {similarity_threshold}"
            )
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError(
                f"ttl_seconds must be positive or None, got {ttl_seconds}"
            )

        self._embed_fn = embed_fn
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds

        # Storage: ordered by insertion time for LRU eviction
        self._entries: Dict[str, CacheEntry] = {}  # key = query string

        # Stats
        self._stats = CacheStats()

        logger.debug(
            "SemanticCache: threshold=%.3f max_size=%d ttl=%s embed=%s",
            similarity_threshold,
            max_size,
            ttl_seconds,
            "enabled" if embed_fn else "disabled",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, query: str) -> Optional[str]:
        """Look up a query in the cache.

        Tries exact match first, then semantic match.  Updates hit statistics
        and the entry's ``last_accessed_at`` timestamp on a hit.

        Args:
            query: The LLM prompt / query string.

        Returns:
            Cached response string on a hit, or ``None`` on a miss.
        """
        self._stats.total_queries += 1

        # --- Exact match ---
        entry = self._entries.get(query)
        if entry is not None:
            if not self._is_expired(entry):
                entry.touch()
                self._stats.exact_hits += 1
                logger.debug("SemanticCache: exact hit for query (len=%d)", len(query))
                return entry.response
            # Expired — remove and continue to miss
            self._remove(query)

        # --- Semantic match ---
        if self._embed_fn is not None:
            try:
                query_emb = self._embed_fn(query)
                hit_entry = self._find_semantic_match(query_emb)
                if hit_entry is not None:
                    hit_entry.touch()
                    self._stats.semantic_hits += 1
                    logger.debug(
                        "SemanticCache: semantic hit for query (len=%d)", len(query)
                    )
                    return hit_entry.response
            except Exception as exc:
                logger.warning(
                    "SemanticCache: embed_fn failed, skipping semantic lookup — %s", exc
                )

        # --- Miss ---
        self._stats.misses += 1
        return None

    def put(self, query: str, response: str) -> None:
        """Store a query-response pair in the cache.

        If the cache is full, evicts the least-recently-used entry first.
        Attempts to compute and store the embedding; if the embedding fails,
        the entry is stored without an embedding (exact-match only).

        Args:
            query: The original query string.
            response: The LLM response to store.
        """
        # Evict if full
        if len(self._entries) >= self.max_size:
            self._evict_lru()

        # Compute embedding (best-effort)
        embedding: Optional[List[float]] = None
        if self._embed_fn is not None:
            try:
                embedding = self._embed_fn(query)
            except Exception as exc:
                logger.warning(
                    "SemanticCache: embed_fn failed during put — "
                    "entry stored without embedding: %s",
                    exc,
                )

        self._entries[query] = CacheEntry(
            query=query,
            embedding=embedding,
            response=response,
        )
        self._stats.cache_size = len(self._entries)
        logger.debug("SemanticCache: stored entry (size=%d)", len(self._entries))

    def wrap(
        self, generate_fn: Callable[[str], str]
    ) -> Callable[[str], str]:
        """Return a cached version of a single-string generate function.

        This is the primary drop-in upgrade path for Phase 0's ``CachingLLM``.
        Any callable that takes a prompt string and returns a response string
        can be wrapped.

        Args:
            generate_fn: Original ``(prompt: str) -> str`` callable.

        Returns:
            A new callable with the same signature that checks the cache before
            calling ``generate_fn`` and stores new responses after generation.

        Example::

            >>> fast_llm = cache.wrap(my_llm_generate)
            >>> result = fast_llm("What is the capital of France?")
        """
        def cached(prompt: str) -> str:
            hit = self.get(prompt)
            if hit is not None:
                return hit
            response = generate_fn(prompt)
            self.put(prompt, response)
            return response

        return cached

    def wrap_messages(
        self,
        generate_fn: Callable[[List[Dict[str, str]]], str],
        key_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    ) -> Callable[[List[Dict[str, str]]], str]:
        """Return a cached version of a messages-list generate function.

        Serialises the message list to a string key for cache lookup.  Useful
        for wrapping ``BaseLLM.single_generate(messages)`` patterns.

        Args:
            generate_fn: ``(messages: List[dict]) -> str`` callable.
            key_fn: Optional custom function to convert a messages list to a
                cache key string.  Defaults to concatenating role+content pairs.

        Returns:
            Cached callable with the same signature.
        """
        def default_key(messages: List[Dict[str, str]]) -> str:
            return "\n".join(
                f"{m.get('role', '')}:{m.get('content', '')}" for m in messages
            )

        key_func = key_fn or default_key

        def cached_messages(messages: List[Dict[str, str]]) -> str:
            key = key_func(messages)
            hit = self.get(key)
            if hit is not None:
                return hit
            response = generate_fn(messages)
            self.put(key, response)
            return response

        return cached_messages

    def invalidate(self, query: str) -> bool:
        """Remove a specific query from the cache.

        Args:
            query: Query string to remove.

        Returns:
            True if the entry existed and was removed, False otherwise.
        """
        if query in self._entries:
            self._remove(query)
            return True
        return False

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._entries.clear()
        self._stats.cache_size = 0
        logger.debug("SemanticCache: cleared all entries.")

    def reset_stats(self) -> None:
        """Reset cumulative statistics while preserving cached entries."""
        self._stats = CacheStats(cache_size=len(self._entries))

    @property
    def stats(self) -> CacheStats:
        """Current cumulative cache statistics (read-only snapshot)."""
        self._stats.cache_size = len(self._entries)
        return self._stats

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Return True if the entry has exceeded its TTL."""
        if self.ttl_seconds is None:
            return False
        return (time.time() - entry.created_at) > self.ttl_seconds

    def _remove(self, key: str) -> None:
        """Remove an entry by key and update stats."""
        if key in self._entries:
            del self._entries[key]
            self._stats.evictions += 1
            self._stats.cache_size = len(self._entries)

    def _evict_lru(self) -> None:
        """Evict the least-recently-used entry."""
        if not self._entries:
            return
        lru_key = min(
            self._entries, key=lambda k: self._entries[k].last_accessed_at
        )
        logger.debug(
            "SemanticCache: evicting LRU entry '%s...'", lru_key[:40]
        )
        self._remove(lru_key)

    def _find_semantic_match(
        self, query_embedding: List[float]
    ) -> Optional[CacheEntry]:
        """Scan the cache for the best semantic match above the threshold.

        Args:
            query_embedding: Embedding of the incoming query.

        Returns:
            Best matching ``CacheEntry`` or None if no entry meets the threshold.
        """
        best_entry: Optional[CacheEntry] = None
        best_sim: float = self.similarity_threshold - 1e-9  # must strictly exceed

        now = time.time()
        for entry in list(self._entries.values()):
            # Skip entries without embeddings or expired entries
            if entry.embedding is None:
                continue
            if self.ttl_seconds is not None:
                if (now - entry.created_at) > self.ttl_seconds:
                    # Lazy expiry during scan — remove in bulk after
                    continue

            try:
                sim = _cosine_similarity(query_embedding, entry.embedding)
            except ValueError:
                # Dimension mismatch — skip (may happen if embed model changed)
                continue

            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        return best_entry

    # ------------------------------------------------------------------
    # Context manager support for scoped usage
    # ------------------------------------------------------------------

    def __enter__(self) -> "SemanticCache":
        return self

    def __exit__(self, *_: Any) -> None:
        pass  # No cleanup required; entries persist unless explicitly cleared


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_semantic_cache(
    embed_fn: Optional[Callable[[str], List[float]]] = None,
    similarity_threshold: float = 0.95,
    max_size: int = 1000,
    ttl_seconds: Optional[float] = None,
) -> SemanticCache:
    """Convenience factory for creating a ``SemanticCache`` instance.

    Args:
        embed_fn: Optional embedding function.
        similarity_threshold: Cosine similarity threshold for cache hits.
        max_size: Maximum cache capacity (LRU eviction beyond this).
        ttl_seconds: Optional TTL in seconds; None = no expiry.

    Returns:
        Configured ``SemanticCache`` instance.
    """
    return SemanticCache(
        embed_fn=embed_fn,
        similarity_threshold=similarity_threshold,
        max_size=max_size,
        ttl_seconds=ttl_seconds,
    )


__all__ = [
    "SemanticCache",
    "CacheEntry",
    "CacheStats",
    "build_semantic_cache",
]

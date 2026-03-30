"""Caching wrapper for BaseLLM that eliminates redundant API calls.

Wraps any :class:`~evoagentx.models.base_model.BaseLLM` instance and adds a
transparent two-level cache:

* **Exact cache** – SHA-256 hash of the serialised messages list.  Hits return
  immediately without touching the upstream LLM.
* **Near-duplicate detection** – optional Jaccard similarity check on token
  sets.  Useful when the same prompt arrives with trivial whitespace changes.

Cache statistics (hits, misses, hit-rate) are accumulated per instance and can
be printed at any time.

Usage
-----
    >>> from evoagentx.core.caching_llm import CachingLLM
    >>> cached_llm = CachingLLM(inner=my_llm, ttl_seconds=3600, max_size=10_000)
    >>> output = cached_llm.generate(prompt="Explain gravity", parse_mode="str")
    >>> print(cached_llm.stats())

The wrapper is a drop-in replacement wherever a ``BaseLLM`` is expected; it
passes all method calls through to ``inner`` when a cache miss occurs.

Disk Persistence
----------------
Pass a ``cache_dir`` path to enable disk-backed caching (JSON files, one per
cache entry).  This lets cache entries survive process restarts.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union

from .logging import logger
from .cost_tracker import estimate_messages_tokens


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """A single cached LLM response."""

    key: str
    response: str                    # raw string from single_generate
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0
    input_tokens_estimate: int = 0


# ---------------------------------------------------------------------------
# Statistics container
# ---------------------------------------------------------------------------

@dataclass
class CacheStats:
    """Snapshot of caching efficiency metrics."""

    hits: int
    misses: int
    evictions: int
    size: int

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def __str__(self) -> str:
        return (
            f"CacheStats | hits={self.hits} misses={self.misses} "
            f"hit_rate={self.hit_rate:.1%} size={self.size} evictions={self.evictions}"
        )


# ---------------------------------------------------------------------------
# Internal LRU-ish store
# ---------------------------------------------------------------------------

class _CacheStore:
    """Thread-safe in-memory cache store with optional TTL and max-size eviction.

    Eviction policy: when at capacity, remove the entry that was least recently
    accessed (LRU) using insertion-order tracking via ``dict`` ordering.
    """

    def __init__(self, max_size: int, ttl_seconds: Optional[float]) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._store: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._evictions = 0

    def get(self, key: str) -> Optional[CacheEntry]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if self._ttl is not None and (time.time() - entry.created_at) > self._ttl:
                del self._store[key]
                self._evictions += 1
                return None
            entry.hit_count += 1
            # Move to end for LRU tracking (dict preserves insertion order in Python 3.7+)
            self._store[key] = self._store.pop(key)
            return entry

    def put(self, entry: CacheEntry) -> None:
        with self._lock:
            if entry.key in self._store:
                self._store[entry.key] = self._store.pop(entry.key)  # refresh
                return
            while len(self._store) >= self._max_size:
                oldest_key = next(iter(self._store))
                del self._store[oldest_key]
                self._evictions += 1
            self._store[entry.key] = entry

    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def evictions(self) -> int:
        return self._evictions


# ---------------------------------------------------------------------------
# Key computation
# ---------------------------------------------------------------------------

def _messages_cache_key(messages: List[Dict]) -> str:
    """Return a deterministic SHA-256 hex digest for a messages list.

    The messages are serialised to JSON with sorted keys to make the hash
    independent of dict insertion order.
    """
    canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity on word-token sets of two strings."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# CachingLLM
# ---------------------------------------------------------------------------

class CachingLLM:
    """Drop-in caching wrapper for any :class:`BaseLLM` implementation.

    Parameters
    ----------
    inner:
        The wrapped ``BaseLLM`` instance.
    ttl_seconds:
        Cache entry lifetime in seconds.  ``None`` disables TTL (entries live
        until eviction by max-size policy or manual ``clear()``).
    max_size:
        Maximum number of entries kept in-memory.  LRU eviction fires when at
        capacity.
    cache_dir:
        If set, cache entries are also written to / loaded from this directory
        as JSON files (one file per entry, named by cache key).
    near_duplicate_threshold:
        When > 0, a Jaccard similarity check is performed against stored
        message texts.  If any stored entry scores above this threshold the
        cached response is returned.  Set to ``0`` to disable (default).
    """

    def __init__(
        self,
        inner: Any,                              # BaseLLM subclass instance
        ttl_seconds: Optional[float] = None,
        max_size: int = 5_000,
        cache_dir: Optional[str] = None,
        near_duplicate_threshold: float = 0.0,
    ) -> None:
        self._inner = inner
        self._store = _CacheStore(max_size=max_size, ttl_seconds=ttl_seconds)
        self._cache_dir = cache_dir
        self._near_dup_threshold = near_duplicate_threshold

        self._hits = 0
        self._misses = 0
        self._stat_lock = threading.Lock()

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self._load_disk_cache()

        # expose config so callers can inspect
        self.config = inner.config

    # ------------------------------------------------------------------
    # Cache key helpers
    # ------------------------------------------------------------------

    def _find_near_duplicate(self, messages: List[Dict]) -> Optional[str]:
        """Return cached response for a near-duplicate prompt, or ``None``."""
        if self._near_dup_threshold <= 0:
            return None
        query_text = " ".join(
            m.get("content", "") if isinstance(m.get("content"), str) else ""
            for m in messages
        )
        # Only compare against stored keys; not ideal for large caches but fine
        # for the sizes we target in optimization loops.
        for entry in list(self._store._store.values()):
            # entry.key is the sha256 — we need the original text.  We embed a
            # short content fingerprint in the entry metadata at insertion time.
            entry_text = getattr(entry, "_content_text", "")
            if entry_text and _jaccard_similarity(query_text, entry_text) >= self._near_dup_threshold:
                logger.debug(f"CachingLLM near-duplicate hit for key={entry.key[:8]}...")
                return entry.response
        return None

    # ------------------------------------------------------------------
    # Core single_generate hook
    # ------------------------------------------------------------------

    def _cached_single_generate(self, messages: List[Dict], **kwargs) -> str:
        """Check cache before delegating to ``inner.single_generate``."""
        key = _messages_cache_key(messages)
        entry = self._store.get(key)

        if entry is not None:
            with self._stat_lock:
                self._hits += 1
            logger.debug(f"CachingLLM cache hit key={key[:8]}... hits={entry.hit_count}")
            return entry.response

        # Near-duplicate check
        near_dup_response = self._find_near_duplicate(messages)
        if near_dup_response is not None:
            with self._stat_lock:
                self._hits += 1
            return near_dup_response

        # Cache miss — call inner LLM
        with self._stat_lock:
            self._misses += 1

        response = self._inner.single_generate(messages, **kwargs)

        content_text = " ".join(
            m.get("content", "") if isinstance(m.get("content"), str) else ""
            for m in messages
        )
        new_entry = CacheEntry(
            key=key,
            response=response,
            input_tokens_estimate=estimate_messages_tokens(messages),
        )
        object.__setattr__(new_entry, "_content_text", content_text)
        self._store.put(new_entry)

        if self._cache_dir:
            self._write_disk_entry(new_entry)

        return response

    async def _cached_single_generate_async(self, messages: List[Dict], **kwargs) -> str:
        """Async variant of the cached single_generate."""
        key = _messages_cache_key(messages)
        entry = self._store.get(key)

        if entry is not None:
            with self._stat_lock:
                self._hits += 1
            return entry.response

        with self._stat_lock:
            self._misses += 1

        response = await self._inner.single_generate_async(messages, **kwargs)

        new_entry = CacheEntry(
            key=key,
            response=response,
            input_tokens_estimate=estimate_messages_tokens(messages),
        )
        self._store.put(new_entry)
        if self._cache_dir:
            self._write_disk_entry(new_entry)

        return response

    # ------------------------------------------------------------------
    # BaseLLM interface delegation
    # ------------------------------------------------------------------

    def generate(self, *args, **kwargs):
        """Generate with caching by overriding single_generate at call time."""
        # Temporarily patch the inner LLM's single_generate to go through cache
        original = self._inner.single_generate
        self._inner.single_generate = self._cached_single_generate
        try:
            result = self._inner.generate(*args, **kwargs)
        finally:
            self._inner.single_generate = original
        return result

    async def async_generate(self, *args, **kwargs):
        """Async generate with caching."""
        original_async = self._inner.single_generate_async
        self._inner.single_generate_async = self._cached_single_generate_async
        try:
            result = await self._inner.async_generate(*args, **kwargs)
        finally:
            self._inner.single_generate_async = original_async
        return result

    def single_generate(self, messages: List[Dict], **kwargs) -> str:
        """Cached single_generate — primary hot path."""
        return self._cached_single_generate(messages, **kwargs)

    async def single_generate_async(self, messages: List[Dict], **kwargs) -> str:
        return await self._cached_single_generate_async(messages, **kwargs)

    def batch_generate(self, batch_messages: List[List[Dict]], **kwargs) -> List[str]:
        """Cached batch_generate — checks cache per-message."""
        return [self._cached_single_generate(msgs, **kwargs) for msgs in batch_messages]

    async def batch_generate_async(self, batch_messages: List[List[Dict]], **kwargs) -> List[str]:
        import asyncio
        tasks = [self._cached_single_generate_async(msgs, **kwargs) for msgs in batch_messages]
        return await asyncio.gather(*tasks)

    def formulate_messages(self, *args, **kwargs):
        return self._inner.formulate_messages(*args, **kwargs)

    def parse_generated_text(self, *args, **kwargs):
        return self._inner.parse_generated_text(*args, **kwargs)

    def parse_generated_texts(self, *args, **kwargs):
        return self._inner.parse_generated_texts(*args, **kwargs)

    def init_model(self):
        return self._inner.init_model()

    def __getattr__(self, name: str):
        """Fall through to inner LLM for any attributes not explicitly wrapped."""
        return getattr(self._inner, name)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def stats(self) -> CacheStats:
        """Return a :class:`CacheStats` snapshot."""
        with self._stat_lock:
            h, m = self._hits, self._misses
        return CacheStats(
            hits=h,
            misses=m,
            evictions=self._store.evictions(),
            size=self._store.size(),
        )

    def clear(self) -> None:
        """Remove all entries from the in-memory cache."""
        with self._store._lock:
            self._store._store.clear()
        with self._stat_lock:
            self._hits = 0
            self._misses = 0

    # ------------------------------------------------------------------
    # Disk persistence
    # ------------------------------------------------------------------

    def _write_disk_entry(self, entry: CacheEntry) -> None:
        """Persist a cache entry to ``cache_dir`` as a JSON file."""
        if not self._cache_dir:
            return
        path = os.path.join(self._cache_dir, f"{entry.key}.json")
        try:
            payload = {
                "key": entry.key,
                "response": entry.response,
                "created_at": entry.created_at,
                "hit_count": entry.hit_count,
                "input_tokens_estimate": entry.input_tokens_estimate,
            }
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
        except OSError as exc:
            logger.warning(f"CachingLLM: failed to write disk cache entry: {exc}")

    def _load_disk_cache(self) -> None:
        """Load all JSON entries from ``cache_dir`` into the in-memory store."""
        if not self._cache_dir or not os.path.isdir(self._cache_dir):
            return
        loaded = 0
        for fname in os.listdir(self._cache_dir):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(self._cache_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                entry = CacheEntry(
                    key=payload["key"],
                    response=payload["response"],
                    created_at=payload.get("created_at", time.time()),
                    hit_count=payload.get("hit_count", 0),
                    input_tokens_estimate=payload.get("input_tokens_estimate", 0),
                )
                self._store.put(entry)
                loaded += 1
            except (OSError, KeyError, json.JSONDecodeError) as exc:
                logger.warning(f"CachingLLM: skipping corrupt cache file {fname}: {exc}")
        if loaded:
            logger.info(f"CachingLLM: loaded {loaded} entries from disk cache at {self._cache_dir}")

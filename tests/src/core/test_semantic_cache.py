"""Unit tests for evoagentx.core.semantic_cache."""

import math
import time
import unittest
from typing import List, Optional

from evoagentx.core.semantic_cache import (
    CacheEntry,
    CacheStats,
    SemanticCache,
    build_semantic_cache,
    _cosine_similarity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _embed(text: str) -> List[float]:
    """Deterministic fake embedding: character frequency (26-dim)."""
    counts = [0.0] * 26
    for ch in text.lower():
        if "a" <= ch <= "z":
            counts[ord(ch) - ord("a")] += 1.0
    total = max(1.0, sum(counts))
    return [c / total for c in counts]


def _failing_embed(text: str) -> List[float]:
    raise RuntimeError("Embed failed intentionally.")


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity(unittest.TestCase):

    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(_cosine_similarity(v, v), 1.0)

    def test_orthogonal_vectors(self):
        self.assertAlmostEqual(_cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_zero_vector_returns_zero(self):
        self.assertAlmostEqual(_cosine_similarity([0.0], [1.0]), 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            _cosine_similarity([1.0, 2.0], [1.0])

    def test_clamped_to_unit(self):
        # Due to floating point, results should stay in [0, 1]
        v = [0.5, 0.5]
        result = _cosine_similarity(v, v)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)


# ---------------------------------------------------------------------------
# CacheEntry
# ---------------------------------------------------------------------------

class TestCacheEntry(unittest.TestCase):

    def test_touch_increments_hit_count(self):
        entry = CacheEntry(query="q", embedding=None, response="r")
        self.assertEqual(entry.hit_count, 0)
        entry.touch()
        self.assertEqual(entry.hit_count, 1)
        entry.touch()
        self.assertEqual(entry.hit_count, 2)

    def test_touch_updates_last_accessed(self):
        entry = CacheEntry(query="q", embedding=None, response="r")
        before = entry.last_accessed_at
        time.sleep(0.01)
        entry.touch()
        self.assertGreater(entry.last_accessed_at, before)


# ---------------------------------------------------------------------------
# CacheStats
# ---------------------------------------------------------------------------

class TestCacheStats(unittest.TestCase):

    def test_hit_rate_zero_queries(self):
        stats = CacheStats()
        self.assertAlmostEqual(stats.hit_rate, 0.0)

    def test_hit_rate_all_hits(self):
        stats = CacheStats(total_queries=10, exact_hits=5, semantic_hits=5)
        self.assertAlmostEqual(stats.hit_rate, 1.0)

    def test_hit_rate_partial(self):
        stats = CacheStats(total_queries=10, exact_hits=3, semantic_hits=2, misses=5)
        self.assertAlmostEqual(stats.hit_rate, 0.5)

    def test_str_representation(self):
        stats = CacheStats(total_queries=5, exact_hits=3, semantic_hits=1, misses=1)
        s = str(stats)
        self.assertIn("queries=5", s)
        self.assertIn("hit_rate=80.0%", s)


# ---------------------------------------------------------------------------
# SemanticCache — construction
# ---------------------------------------------------------------------------

class TestSemanticCacheConstruction(unittest.TestCase):

    def test_default_construction(self):
        cache = SemanticCache()
        self.assertEqual(len(cache), 0)

    def test_invalid_threshold_zero(self):
        with self.assertRaises(ValueError):
            SemanticCache(similarity_threshold=0.0)

    def test_invalid_threshold_above_one(self):
        with self.assertRaises(ValueError):
            SemanticCache(similarity_threshold=1.1)

    def test_invalid_max_size(self):
        with self.assertRaises(ValueError):
            SemanticCache(max_size=0)

    def test_invalid_ttl(self):
        with self.assertRaises(ValueError):
            SemanticCache(ttl_seconds=-1.0)

    def test_context_manager(self):
        with SemanticCache() as cache:
            self.assertIsInstance(cache, SemanticCache)


# ---------------------------------------------------------------------------
# SemanticCache — exact match
# ---------------------------------------------------------------------------

class TestSemanticCacheExactMatch(unittest.TestCase):

    def test_miss_returns_none(self):
        cache = SemanticCache()
        self.assertIsNone(cache.get("unseen query"))

    def test_put_then_exact_get(self):
        cache = SemanticCache()
        cache.put("hello world", "response A")
        result = cache.get("hello world")
        self.assertEqual(result, "response A")

    def test_stats_after_hit(self):
        cache = SemanticCache()
        cache.put("q", "r")
        cache.get("q")
        self.assertEqual(cache.stats.exact_hits, 1)
        self.assertEqual(cache.stats.misses, 0)

    def test_stats_after_miss(self):
        cache = SemanticCache()
        cache.get("not stored")
        self.assertEqual(cache.stats.misses, 1)
        self.assertEqual(cache.stats.exact_hits, 0)

    def test_multiple_entries(self):
        cache = SemanticCache()
        cache.put("q1", "r1")
        cache.put("q2", "r2")
        self.assertEqual(cache.get("q1"), "r1")
        self.assertEqual(cache.get("q2"), "r2")
        self.assertEqual(len(cache), 2)


# ---------------------------------------------------------------------------
# SemanticCache — semantic match
# ---------------------------------------------------------------------------

class TestSemanticCacheSemanticMatch(unittest.TestCase):

    def test_semantic_hit_on_similar_query(self):
        """Two texts with the same vocabulary should hit the semantic cache."""
        cache = SemanticCache(embed_fn=_embed, similarity_threshold=0.80)
        query_a = "the capital of france is paris"
        query_b = "paris is the capital of france"  # Same words, different order
        cache.put(query_a, "Paris")
        result = cache.get(query_b)
        self.assertEqual(result, "Paris")
        self.assertEqual(cache.stats.semantic_hits, 1)

    def test_no_semantic_hit_on_different_query(self):
        """Completely different queries should not hit the semantic cache."""
        cache = SemanticCache(embed_fn=_embed, similarity_threshold=0.95)
        cache.put("quantum mechanics wavefunction", "physics response")
        result = cache.get("cooking pasta al dente")
        self.assertIsNone(result)

    def test_embed_failure_falls_back_to_miss(self):
        """When embed_fn raises, the cache should return a miss without crashing."""
        cache = SemanticCache(embed_fn=_failing_embed, similarity_threshold=0.95)
        cache.put("known query", "known response")
        # A different query triggers embed which fails
        result = cache.get("another query")
        self.assertIsNone(result)

    def test_no_embed_fn_only_exact(self):
        """Without embed_fn, only exact matching occurs."""
        cache = SemanticCache(embed_fn=None, similarity_threshold=0.95)
        cache.put("the capital of france is paris", "Paris")
        result = cache.get("paris is the capital of france")
        self.assertIsNone(result)  # Different string → miss


# ---------------------------------------------------------------------------
# SemanticCache — TTL
# ---------------------------------------------------------------------------

class TestSemanticCacheTTL(unittest.TestCase):

    def test_entry_expires(self):
        cache = SemanticCache(ttl_seconds=0.05)
        cache.put("q", "r")
        self.assertEqual(cache.get("q"), "r")  # Fresh hit
        time.sleep(0.08)
        result = cache.get("q")  # Should be expired
        self.assertIsNone(result)

    def test_entry_not_expired_before_ttl(self):
        cache = SemanticCache(ttl_seconds=10.0)
        cache.put("q", "r")
        self.assertEqual(cache.get("q"), "r")


# ---------------------------------------------------------------------------
# SemanticCache — LRU eviction
# ---------------------------------------------------------------------------

class TestSemanticCacheLRUEviction(unittest.TestCase):

    def test_lru_entry_evicted_when_full(self):
        cache = SemanticCache(max_size=3)
        cache.put("q1", "r1")
        cache.put("q2", "r2")
        cache.put("q3", "r3")
        self.assertEqual(len(cache), 3)

        # Access q1 and q3 to make q2 the LRU
        cache.get("q1")
        cache.get("q3")

        # Adding q4 should evict q2 (LRU)
        cache.put("q4", "r4")
        self.assertEqual(len(cache), 3)
        self.assertIsNone(cache.get("q2"))
        self.assertIsNotNone(cache.get("q1"))
        self.assertIsNotNone(cache.get("q3"))
        self.assertIsNotNone(cache.get("q4"))

    def test_eviction_increments_counter(self):
        cache = SemanticCache(max_size=1)
        cache.put("q1", "r1")
        cache.put("q2", "r2")  # Evicts q1
        self.assertGreater(cache.stats.evictions, 0)


# ---------------------------------------------------------------------------
# SemanticCache — invalidate and clear
# ---------------------------------------------------------------------------

class TestSemanticCacheInvalidation(unittest.TestCase):

    def test_invalidate_existing(self):
        cache = SemanticCache()
        cache.put("q", "r")
        removed = cache.invalidate("q")
        self.assertTrue(removed)
        self.assertIsNone(cache.get("q"))

    def test_invalidate_nonexistent_returns_false(self):
        cache = SemanticCache()
        self.assertFalse(cache.invalidate("not here"))

    def test_clear_removes_all(self):
        cache = SemanticCache()
        cache.put("q1", "r1")
        cache.put("q2", "r2")
        cache.clear()
        self.assertEqual(len(cache), 0)
        self.assertIsNone(cache.get("q1"))

    def test_reset_stats(self):
        cache = SemanticCache()
        cache.put("q", "r")
        cache.get("q")
        self.assertEqual(cache.stats.exact_hits, 1)
        cache.reset_stats()
        self.assertEqual(cache.stats.exact_hits, 0)
        self.assertEqual(cache.stats.total_queries, 0)


# ---------------------------------------------------------------------------
# SemanticCache — wrap()
# ---------------------------------------------------------------------------

class TestSemanticCacheWrap(unittest.TestCase):

    def test_wrap_returns_correct_response(self):
        cache = SemanticCache()
        call_count = [0]

        def my_llm(prompt: str) -> str:
            call_count[0] += 1
            return f"response to: {prompt}"

        fast_llm = cache.wrap(my_llm)
        r1 = fast_llm("What is 2+2?")
        r2 = fast_llm("What is 2+2?")  # Cache hit

        self.assertEqual(r1, "response to: What is 2+2?")
        self.assertEqual(r2, r1)
        self.assertEqual(call_count[0], 1)  # LLM called only once

    def test_wrap_multiple_distinct_prompts(self):
        cache = SemanticCache()
        call_count = [0]

        def my_llm(prompt: str) -> str:
            call_count[0] += 1
            return f"answer:{prompt}"

        fast_llm = cache.wrap(my_llm)
        fast_llm("query A")
        fast_llm("query B")
        fast_llm("query A")  # Hit

        self.assertEqual(call_count[0], 2)
        self.assertEqual(cache.stats.exact_hits, 1)


# ---------------------------------------------------------------------------
# SemanticCache — wrap_messages()
# ---------------------------------------------------------------------------

class TestSemanticCacheWrapMessages(unittest.TestCase):

    def test_wrap_messages_caches_by_content(self):
        cache = SemanticCache()
        call_count = [0]

        def my_llm(messages):
            call_count[0] += 1
            return "response"

        fast_llm = cache.wrap_messages(my_llm)
        msgs = [{"role": "user", "content": "hello"}]
        fast_llm(msgs)
        fast_llm(msgs)  # Second call should hit cache

        self.assertEqual(call_count[0], 1)


# ---------------------------------------------------------------------------
# build_semantic_cache factory
# ---------------------------------------------------------------------------

class TestBuildSemanticCache(unittest.TestCase):

    def test_creates_instance(self):
        cache = build_semantic_cache(embed_fn=_embed, similarity_threshold=0.9)
        self.assertIsInstance(cache, SemanticCache)
        self.assertAlmostEqual(cache.similarity_threshold, 0.9)

    def test_default_factory(self):
        cache = build_semantic_cache()
        self.assertIsInstance(cache, SemanticCache)


if __name__ == "__main__":
    unittest.main()

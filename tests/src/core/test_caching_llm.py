"""Tests for CachingLLM (Phase 0)."""
from __future__ import annotations

import asyncio
import tempfile
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


class _FakeLLM:
    """Minimal stub satisfying the CachingLLM inner-LLM interface."""

    def __init__(self, responses: Optional[Dict[str, str]] = None) -> None:
        self.call_count = 0
        self._responses = responses or {}
        self.config = MagicMock()
        self.config.llm_type = "fake"

    def single_generate(self, messages: List[Dict], **kwargs) -> str:
        self.call_count += 1
        key = messages[-1].get("content", "") if messages else ""
        return self._responses.get(key, f"response_for_{key}")

    async def single_generate_async(self, messages: List[Dict], **kwargs) -> str:
        return self.single_generate(messages, **kwargs)

    def batch_generate(self, batch_messages, **kwargs) -> List[str]:
        return [self.single_generate(msgs, **kwargs) for msgs in batch_messages]

    async def batch_generate_async(self, batch_messages, **kwargs):
        return [self.single_generate(msgs, **kwargs) for msgs in batch_messages]

    def formulate_messages(self, prompts, system_messages=None):
        return [[{"role": "user", "content": p}] for p in prompts]

    def generate(self, prompt=None, messages=None, parse_mode=None, **kwargs):
        if messages:
            msgs = messages if isinstance(messages[0], dict) else messages[0]
        else:
            msgs = [{"role": "user", "content": str(prompt)}]
        text = self.single_generate(msgs, **kwargs)
        result = MagicMock()
        result.content = text
        return result

    async def async_generate(self, prompt=None, messages=None, **kwargs):
        return self.generate(prompt=prompt, messages=messages, **kwargs)

    def parse_generated_text(self, text, **kwargs):
        return text

    def parse_generated_texts(self, texts, **kwargs):
        return texts

    def init_model(self):
        pass


def _make_messages(text: str) -> List[Dict]:
    return [{"role": "user", "content": text}]


class TestCachingLLM:

    def _make_cached(self, responses=None, **kwargs):
        from evoagentx.core.caching_llm import CachingLLM
        inner = _FakeLLM(responses)
        return CachingLLM(inner=inner, **kwargs)

    def test_cache_hit_on_repeat(self):
        cached = self._make_cached()
        msgs = _make_messages("hello")
        r1 = cached.single_generate(msgs)
        r2 = cached.single_generate(msgs)
        assert r1 == r2
        assert cached._inner.call_count == 1

    def test_cache_miss_on_different_messages(self):
        cached = self._make_cached()
        r1 = cached.single_generate(_make_messages("hello"))
        r2 = cached.single_generate(_make_messages("world"))
        assert r1 != r2
        assert cached._inner.call_count == 2

    def test_stats_hit_rate(self):
        cached = self._make_cached()
        msgs = _make_messages("test prompt")
        cached.single_generate(msgs)   # miss
        cached.single_generate(msgs)   # hit
        cached.single_generate(msgs)   # hit
        stats = cached.stats()
        assert stats.hits == 2
        assert stats.misses == 1
        assert abs(stats.hit_rate - 2 / 3) < 0.001

    def test_ttl_expiry(self):
        from evoagentx.core.caching_llm import CachingLLM
        inner = _FakeLLM()
        cached = CachingLLM(inner=inner, ttl_seconds=0.01)
        msgs = _make_messages("expire test")
        cached.single_generate(msgs)
        time.sleep(0.05)
        cached.single_generate(msgs)
        assert inner.call_count == 2

    def test_max_size_eviction(self):
        from evoagentx.core.caching_llm import CachingLLM
        inner = _FakeLLM()
        cached = CachingLLM(inner=inner, max_size=2)
        cached.single_generate(_make_messages("a"))
        cached.single_generate(_make_messages("b"))
        cached.single_generate(_make_messages("c"))   # evicts "a"
        assert inner.call_count == 3
        cached.single_generate(_make_messages("a"))   # should miss
        assert inner.call_count == 4
        assert cached.stats().evictions >= 1

    def test_disk_cache_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from evoagentx.core.caching_llm import CachingLLM
            inner1 = _FakeLLM()
            c1 = CachingLLM(inner=inner1, cache_dir=tmpdir)
            msgs = _make_messages("persistent query")
            c1.single_generate(msgs)
            assert inner1.call_count == 1

            inner2 = _FakeLLM()
            c2 = CachingLLM(inner=inner2, cache_dir=tmpdir)
            c2.single_generate(msgs)
            assert inner2.call_count == 0

    def test_batch_generate_caches_each(self):
        cached = self._make_cached()
        batch = [_make_messages("x"), _make_messages("y"), _make_messages("x")]
        results = cached.batch_generate(batch)
        assert len(results) == 3
        assert results[0] == results[2]
        assert cached._inner.call_count == 2

    def test_clear_resets_stats(self):
        cached = self._make_cached()
        cached.single_generate(_make_messages("q"))
        cached.single_generate(_make_messages("q"))
        cached.clear()
        stats = cached.stats()
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.size == 0

    def test_near_duplicate_detection(self):
        from evoagentx.core.caching_llm import CachingLLM
        inner = _FakeLLM()
        cached = CachingLLM(inner=inner, near_duplicate_threshold=0.7)
        msgs_a = _make_messages("what is the population of france")
        msgs_b = _make_messages("what is the population of france today")
        cached.single_generate(msgs_a)
        cached.single_generate(msgs_b)
        assert inner.call_count == 1

    def test_async_single_generate(self):
        async def _run():
            cached = self._make_cached()
            msgs = _make_messages("async test")
            r1 = await cached.single_generate_async(msgs)
            r2 = await cached.single_generate_async(msgs)
            assert r1 == r2
            assert cached._inner.call_count == 1
        asyncio.run(_run())

    def test_getattr_fallthrough(self):
        cached = self._make_cached()
        assert cached.call_count == 0

    def test_messages_cache_key_determinism(self):
        from evoagentx.core.caching_llm import _messages_cache_key
        msgs = [{"role": "user", "content": "hello"}, {"role": "system", "content": "sys"}]
        k1 = _messages_cache_key(msgs)
        k2 = _messages_cache_key(list(reversed(msgs)))
        k3 = _messages_cache_key(msgs)
        assert k1 == k3
        assert k1 != k2

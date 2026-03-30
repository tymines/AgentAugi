"""Unit tests for evoagentx.core.streaming."""

import asyncio
import unittest
from typing import AsyncIterator, List

from evoagentx.core.streaming import (
    StreamConfig,
    StreamPipeline,
    StreamStats,
    collect_stream,
    drop_empty_tokens,
    stream_to_list,
    strip_whitespace,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _tokens(*chunks: str) -> AsyncIterator[str]:
    """Async generator that yields each string in *chunks*."""
    for chunk in chunks:
        await asyncio.sleep(0)
        yield chunk


def _run(coro):
    """Run a coroutine in a new event loop (for use inside unittest)."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Tests for StreamConfig defaults
# ---------------------------------------------------------------------------

class TestStreamConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = StreamConfig()
        self.assertEqual(cfg.buffer_size, 64)
        self.assertEqual(cfg.filters, [])
        self.assertEqual(cfg.transforms, [])
        self.assertIsNone(cfg.on_token)
        self.assertIsNone(cfg.on_complete)


# ---------------------------------------------------------------------------
# Tests for StreamPipeline
# ---------------------------------------------------------------------------

class TestStreamPipeline(unittest.TestCase):

    # --- basic streaming ---

    def test_yields_all_tokens(self):
        async def run():
            pipeline = StreamPipeline()
            result = []
            async for tok in pipeline.stream(_tokens("a", "b", "c")):
                result.append(tok)
            return result

        tokens = _run(run())
        self.assertEqual(tokens, ["a", "b", "c"])

    def test_accumulated_text_after_stream(self):
        async def run():
            pipeline = StreamPipeline()
            async for _ in pipeline.stream(_tokens("Hello", " ", "world")):
                pass
            return pipeline.accumulated_text

        text = _run(run())
        self.assertEqual(text, "Hello world")

    def test_is_done_after_stream(self):
        async def run():
            pipeline = StreamPipeline()
            async for _ in pipeline.stream(_tokens("x")):
                pass
            return pipeline.is_done

        self.assertTrue(_run(run()))

    def test_is_not_done_before_stream(self):
        pipeline = StreamPipeline()
        self.assertFalse(pipeline.is_done)

    # --- empty stream ---

    def test_empty_stream(self):
        async def run():
            pipeline = StreamPipeline()
            tokens = []
            async for tok in pipeline.stream(_tokens()):
                tokens.append(tok)
            return tokens, pipeline.accumulated_text

        tokens, text = _run(run())
        self.assertEqual(tokens, [])
        self.assertEqual(text, "")

    # --- filters ---

    def test_filter_drops_tokens(self):
        async def run():
            cfg = StreamConfig(filters=[drop_empty_tokens])
            pipeline = StreamPipeline(config=cfg)
            result = []
            async for tok in pipeline.stream(_tokens("a", "", "  ", "b")):
                result.append(tok)
            return result

        tokens = _run(run())
        self.assertEqual(tokens, ["a", "b"])

    def test_filter_updates_dropped_count(self):
        async def run():
            cfg = StreamConfig(filters=[drop_empty_tokens])
            pipeline = StreamPipeline(config=cfg)
            async for _ in pipeline.stream(_tokens("", "", "ok")):
                pass
            return pipeline.stats.dropped_count

        dropped = _run(run())
        self.assertEqual(dropped, 2)

    def test_multiple_filters(self):
        # Drop empty AND tokens starting with 'x'
        async def run():
            cfg = StreamConfig(filters=[
                drop_empty_tokens,
                lambda t: t.startswith("x"),
            ])
            pipeline = StreamPipeline(config=cfg)
            result = []
            async for tok in pipeline.stream(_tokens("hello", "", "xbad", "world")):
                result.append(tok)
            return result

        tokens = _run(run())
        self.assertEqual(tokens, ["hello", "world"])

    # --- transforms ---

    def test_transform_applied(self):
        async def run():
            cfg = StreamConfig(transforms=[str.upper])
            pipeline = StreamPipeline(config=cfg)
            result = []
            async for tok in pipeline.stream(_tokens("hello", "world")):
                result.append(tok)
            return result

        tokens = _run(run())
        self.assertEqual(tokens, ["HELLO", "WORLD"])

    def test_multiple_transforms_applied_in_order(self):
        # First strip, then upper
        async def run():
            cfg = StreamConfig(transforms=[strip_whitespace, str.upper])
            pipeline = StreamPipeline(config=cfg)
            result = []
            async for tok in pipeline.stream(_tokens("  hello  ", "  world  ")):
                result.append(tok)
            return result

        tokens = _run(run())
        self.assertEqual(tokens, ["HELLO", "WORLD"])

    # --- callbacks ---

    def test_on_token_callback_called_per_token(self):
        seen = []

        async def run():
            cfg = StreamConfig(on_token=seen.append)
            pipeline = StreamPipeline(config=cfg)
            async for _ in pipeline.stream(_tokens("a", "b", "c")):
                pass

        _run(run())
        self.assertEqual(seen, ["a", "b", "c"])

    def test_on_complete_callback_called_with_full_text(self):
        result = []

        async def run():
            cfg = StreamConfig(on_complete=result.append)
            pipeline = StreamPipeline(config=cfg)
            async for _ in pipeline.stream(_tokens("Hello", " world")):
                pass

        _run(run())
        self.assertEqual(result, ["Hello world"])

    # --- stats ---

    def test_stats_token_count(self):
        async def run():
            pipeline = StreamPipeline()
            async for _ in pipeline.stream(_tokens("a", "b", "c")):
                pass
            return pipeline.stats.token_count

        self.assertEqual(_run(run()), 3)

    def test_stats_elapsed_positive(self):
        async def run():
            pipeline = StreamPipeline()
            async for _ in pipeline.stream(_tokens("a", "b")):
                pass
            return pipeline.stats.elapsed

        elapsed = _run(run())
        self.assertGreaterEqual(elapsed, 0.0)

    def test_stats_tokens_per_second(self):
        async def run():
            pipeline = StreamPipeline()
            async for _ in pipeline.stream(_tokens("a", "b", "c")):
                pass
            return pipeline.stats.tokens_per_second

        tps = _run(run())
        self.assertGreater(tps, 0.0)

    # --- pipeline chaining ---

    def test_add_filter_chaining(self):
        pipeline = StreamPipeline()
        ret = pipeline.add_filter(drop_empty_tokens)
        self.assertIs(ret, pipeline)

    def test_add_transform_chaining(self):
        pipeline = StreamPipeline()
        ret = pipeline.add_transform(str.upper)
        self.assertIs(ret, pipeline)

    # --- sync generator wrapper ---

    def test_stream_sync_generator(self):
        def gen():
            yield "x"
            yield "y"
            yield "z"

        async def run():
            pipeline = StreamPipeline()
            result = []
            async for tok in pipeline.stream_sync(gen()):
                result.append(tok)
            return result

        tokens = _run(run())
        self.assertEqual(tokens, ["x", "y", "z"])

    # --- backpressure / buffer_size ---

    def test_small_buffer_still_works(self):
        """A buffer_size of 1 should still produce all tokens."""
        async def run():
            cfg = StreamConfig(buffer_size=1)
            pipeline = StreamPipeline(config=cfg)
            result = []
            async for tok in pipeline.stream(_tokens("a", "b", "c", "d", "e")):
                result.append(tok)
            return result

        tokens = _run(run())
        self.assertEqual(tokens, ["a", "b", "c", "d", "e"])

    # --- pipeline reset between uses ---

    def test_pipeline_resets_on_second_use(self):
        async def run():
            pipeline = StreamPipeline()
            async for _ in pipeline.stream(_tokens("first")):
                pass
            text1 = pipeline.accumulated_text
            async for _ in pipeline.stream(_tokens("second")):
                pass
            text2 = pipeline.accumulated_text
            return text1, text2

        t1, t2 = _run(run())
        self.assertEqual(t1, "first")
        self.assertEqual(t2, "second")


# ---------------------------------------------------------------------------
# Tests for convenience helpers
# ---------------------------------------------------------------------------

class TestConvenienceHelpers(unittest.TestCase):

    def test_collect_stream(self):
        async def run():
            return await collect_stream(_tokens("hello", " ", "world"))

        self.assertEqual(_run(run()), "hello world")

    def test_collect_stream_with_filter(self):
        async def run():
            cfg = StreamConfig(filters=[drop_empty_tokens])
            return await collect_stream(_tokens("a", "", "b"), config=cfg)

        self.assertEqual(_run(run()), "ab")

    def test_stream_to_list(self):
        async def run():
            return await stream_to_list(_tokens("x", "y", "z"))

        self.assertEqual(_run(run()), ["x", "y", "z"])

    def test_drop_empty_tokens_helper(self):
        self.assertTrue(drop_empty_tokens(""))
        self.assertTrue(drop_empty_tokens("   "))
        self.assertFalse(drop_empty_tokens("hello"))
        self.assertFalse(drop_empty_tokens(" a "))

    def test_strip_whitespace_helper(self):
        self.assertEqual(strip_whitespace("  hello  "), "hello")
        self.assertEqual(strip_whitespace("ok"), "ok")


if __name__ == "__main__":
    unittest.main()

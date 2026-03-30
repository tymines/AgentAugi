"""
Tests for evoagentx.core.speculative_executor (PASTE).

Coverage
--------
- ToolPrediction / SpeculationResult / ToolCallRecord dataclasses
- SpeculativeConfig defaults and custom values
- _pattern_match: empty / short / repeating sequences
- _confidence_from_pattern: value ranges and ordering
- predict_next_tool: filters, confidence gate, LLM fallback
- speculatively_execute: start, skip (low conf / cap / unknown tool / duplicate)
- resolve: cache hit, tool-name miss, args mismatch
- Discard-on-mismatch cancels competing speculations
- Stats tracking: hits, misses, wasted ms, latency saved
- max_parallel_speculations cap
- Timeout enforcement during speculative execution
- CostTracker integration
- Confidence threshold filtering end-to-end
"""

from __future__ import annotations

import asyncio
import time
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

from evoagentx.core.speculative_executor import (
    SpeculationResult,
    SpeculativeConfig,
    SpeculativeExecutor,
    ToolCallRecord,
    ToolPrediction,
)
from evoagentx.core.cost_tracker import CostTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run a coroutine synchronously."""
    return asyncio.run(coro)


def _history(*tool_names: str) -> List[ToolCallRecord]:
    """Create a minimal history from an ordered sequence of tool names."""
    return [ToolCallRecord(tool=name) for name in tool_names]


def _sync_tool(return_value: Any = "ok") -> Any:
    def fn(**kwargs):
        return return_value
    return fn


def _async_tool(return_value: Any = "ok") -> Any:
    async def fn(**kwargs):
        return return_value
    return fn


def _slow_async_tool(delay: float, return_value: Any = "ok") -> Any:
    async def fn(**kwargs):
        await asyncio.sleep(delay)
        return return_value
    return fn


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------

class TestToolCallRecord(unittest.TestCase):
    def test_minimal_creation(self):
        r = ToolCallRecord(tool="search")
        self.assertEqual(r.tool, "search")
        self.assertEqual(r.args, {})
        self.assertIsNone(r.result)
        self.assertEqual(r.duration_ms, 0.0)

    def test_full_creation(self):
        r = ToolCallRecord(tool="parse", args={"text": "hello"}, result=42, duration_ms=50.0)
        self.assertEqual(r.tool, "parse")
        self.assertEqual(r.args, {"text": "hello"})
        self.assertEqual(r.result, 42)
        self.assertEqual(r.duration_ms, 50.0)


class TestToolPrediction(unittest.TestCase):
    def test_defaults(self):
        p = ToolPrediction(predicted_tool="summarise")
        self.assertEqual(p.predicted_tool, "summarise")
        self.assertEqual(p.predicted_args, {})
        self.assertEqual(p.confidence, 0.0)
        self.assertEqual(p.basis, "")

    def test_full_creation(self):
        p = ToolPrediction(
            predicted_tool="summarise",
            predicted_args={"text": "..."},
            confidence=0.85,
            basis="pattern(ctx_len=2)",
        )
        self.assertEqual(p.predicted_tool, "summarise")
        self.assertEqual(p.predicted_args, {"text": "..."})
        self.assertAlmostEqual(p.confidence, 0.85)
        self.assertIn("ctx_len=2", p.basis)


class TestSpeculationResult(unittest.TestCase):
    def test_was_used(self):
        pred = ToolPrediction(predicted_tool="parse", confidence=0.9)
        sr = SpeculationResult(prediction=pred, result="data", was_used=True, latency_saved_ms=120.0)
        self.assertTrue(sr.was_used)
        self.assertEqual(sr.result, "data")
        self.assertAlmostEqual(sr.latency_saved_ms, 120.0)

    def test_not_used(self):
        pred = ToolPrediction(predicted_tool="other", confidence=0.8)
        sr = SpeculationResult(prediction=pred, result=None, was_used=False)
        self.assertFalse(sr.was_used)
        self.assertIsNone(sr.result)


class TestSpeculativeConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = SpeculativeConfig()
        self.assertEqual(cfg.max_speculative_depth, 1)
        self.assertAlmostEqual(cfg.confidence_threshold, 0.7)
        self.assertEqual(cfg.max_parallel_speculations, 2)
        self.assertTrue(cfg.discard_on_mismatch)
        self.assertIsNone(cfg.timeout_per_speculation)

    def test_custom_values(self):
        cfg = SpeculativeConfig(
            max_speculative_depth=3,
            confidence_threshold=0.5,
            max_parallel_speculations=4,
            discard_on_mismatch=False,
            timeout_per_speculation=2.0,
        )
        self.assertEqual(cfg.max_speculative_depth, 3)
        self.assertAlmostEqual(cfg.confidence_threshold, 0.5)
        self.assertEqual(cfg.max_parallel_speculations, 4)
        self.assertFalse(cfg.discard_on_mismatch)
        self.assertAlmostEqual(cfg.timeout_per_speculation, 2.0)


# ---------------------------------------------------------------------------
# Pattern matching tests
# ---------------------------------------------------------------------------

class TestPatternMatch(unittest.TestCase):
    def _make(self, depth: int = 3) -> SpeculativeExecutor:
        return SpeculativeExecutor(
            config=SpeculativeConfig(
                max_speculative_depth=depth,
                confidence_threshold=0.0,  # accept all confidence levels
            )
        )

    def test_empty_history(self):
        ex = self._make()
        self.assertIsNone(ex._pattern_match([]))

    def test_single_item_history(self):
        ex = self._make()
        # Only one item — no pattern possible
        self.assertIsNone(ex._pattern_match(_history("A")))

    def test_two_distinct_items_no_repetition(self):
        # [A, B] — A appears once, context [A] → nothing after
        # context [B] (last item) — no occurrence of B before
        ex = self._make()
        result = ex._pattern_match(_history("A", "B"))
        # B is the last item; context [B] has no prior occurrence → no prediction
        self.assertIsNone(result)

    def test_simple_two_repetition(self):
        # [A, B, A] → after A we have B (observed once)
        ex = self._make()
        result = ex._pattern_match(_history("A", "B", "A"))
        self.assertIsNotNone(result)
        tool, args, conf, basis = result
        self.assertEqual(tool, "B")
        self.assertEqual(args, {})
        self.assertGreater(conf, 0.0)

    def test_three_step_chain(self):
        # [A, B, C, A, B] → context [A, B] → predicts C
        ex = self._make()
        result = ex._pattern_match(_history("A", "B", "C", "A", "B"))
        self.assertIsNotNone(result)
        tool, _, conf, _ = result
        self.assertEqual(tool, "C")

    def test_longer_pattern_preferred(self):
        # [A, B, C, A, B, C, A, B] → context [A, B] predicts C with higher
        # confidence than context [B] predicting C
        ex = self._make(depth=3)
        result = ex._pattern_match(_history("A", "B", "C", "A", "B", "C", "A", "B"))
        self.assertIsNotNone(result)
        tool, _, conf, basis = result
        self.assertEqual(tool, "C")
        # ctx_len=2 should give conf >= 0.83
        self.assertGreaterEqual(conf, 0.8)

    def test_ambiguous_pattern_lowers_confidence(self):
        # [A, B, A, C, A] — after A: B once, C once → freq_ratio=0.5
        ex = self._make()
        result = ex._pattern_match(_history("A", "B", "A", "C", "A"))
        self.assertIsNotNone(result)
        _, _, conf, _ = result
        # 0.5 freq_ratio → confidence < 0.75
        self.assertLess(conf, 0.75)

    def test_alternating_returns_prediction(self):
        # [A, B, A, B, A] → after A: always B; after B: always A
        ex = self._make()
        result = ex._pattern_match(_history("A", "B", "A", "B", "A"))
        self.assertIsNotNone(result)
        tool, _, conf, _ = result
        self.assertEqual(tool, "B")
        self.assertGreaterEqual(conf, 0.7)


# ---------------------------------------------------------------------------
# Confidence calculation tests
# ---------------------------------------------------------------------------

class TestConfidenceFromPattern(unittest.TestCase):
    def _exec(self) -> SpeculativeExecutor:
        return SpeculativeExecutor()

    def test_zero_total_returns_zero(self):
        ex = self._exec()
        self.assertEqual(ex._confidence_from_pattern(1, 0, 0), 0.0)

    def test_perfect_single_observation(self):
        # freq=1, total=1, length=1 → 1.0 * (0.5 + 0.5*0.5) = 0.75
        ex = self._exec()
        conf = ex._confidence_from_pattern(1, 1, 1)
        self.assertAlmostEqual(conf, 0.75)

    def test_perfect_longer_pattern(self):
        # freq=1, total=1, length=2 → 1.0 * (0.5 + 0.5*0.667) ≈ 0.833
        ex = self._exec()
        conf = ex._confidence_from_pattern(2, 1, 1)
        self.assertAlmostEqual(conf, 0.8333, places=3)

    def test_half_frequency(self):
        # freq=1, total=2, length=1 → 0.5 * 0.75 = 0.375
        ex = self._exec()
        conf = ex._confidence_from_pattern(1, 1, 2)
        self.assertAlmostEqual(conf, 0.375)

    def test_longer_pattern_higher_confidence(self):
        ex = self._exec()
        c1 = ex._confidence_from_pattern(1, 1, 1)
        c2 = ex._confidence_from_pattern(2, 1, 1)
        c3 = ex._confidence_from_pattern(3, 1, 1)
        self.assertLess(c1, c2)
        self.assertLess(c2, c3)

    def test_bounded_to_one(self):
        ex = self._exec()
        for length in range(1, 10):
            conf = ex._confidence_from_pattern(length, 1, 1)
            self.assertLessEqual(conf, 1.0)
            self.assertGreaterEqual(conf, 0.0)


# ---------------------------------------------------------------------------
# predict_next_tool tests
# ---------------------------------------------------------------------------

class TestPredictNextTool(unittest.TestCase):
    def _exec(self, threshold: float = 0.0) -> SpeculativeExecutor:
        return SpeculativeExecutor(
            config=SpeculativeConfig(confidence_threshold=threshold)
        )

    def test_empty_history_returns_none(self):
        ex = self._exec()
        self.assertIsNone(ex.predict_next_tool([], ["search", "parse"]))

    def test_prediction_filtered_by_available_tools(self):
        # Pattern says "B" is next, but B is not in available_tools
        ex = self._exec()
        history = _history("A", "B", "A")
        result = ex.predict_next_tool(history, available_tools=["C", "D"])
        self.assertIsNone(result)

    def test_prediction_within_available_tools(self):
        ex = self._exec()
        history = _history("search", "parse", "search")
        result = ex.predict_next_tool(history, available_tools=["parse", "summarise"])
        self.assertIsNotNone(result)
        self.assertEqual(result.predicted_tool, "parse")

    def test_confidence_threshold_blocks_prediction(self):
        # Threshold=0.9 means only very confident patterns pass
        ex = SpeculativeExecutor(
            config=SpeculativeConfig(confidence_threshold=0.9)
        )
        # [A, B, A] gives confidence 0.75 — below 0.9
        history = _history("A", "B", "A")
        self.assertIsNone(ex.predict_next_tool(history, ["A", "B", "C"]))

    def test_llm_predictor_fallback(self):
        ex = self._exec(threshold=0.8)
        # History too short for high-confidence pattern match
        history = _history("A", "B")
        called = []

        def my_predictor(hist, tools):
            called.append(True)
            return ToolPrediction(predicted_tool="B", confidence=0.9, basis="llm")

        result = ex.predict_next_tool(history, ["A", "B"], llm_predictor=my_predictor)
        self.assertTrue(called)
        self.assertIsNotNone(result)
        self.assertEqual(result.predicted_tool, "B")

    def test_llm_predictor_filtered_by_available_tools(self):
        ex = self._exec(threshold=0.0)
        history = _history("A")

        def bad_predictor(hist, tools):
            return ToolPrediction(predicted_tool="UNKNOWN", confidence=0.99)

        result = ex.predict_next_tool(history, ["A", "B"], llm_predictor=bad_predictor)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# speculatively_execute tests
# ---------------------------------------------------------------------------

class TestSpeculativelyExecute(unittest.TestCase):
    def test_starts_background_task(self):
        async def _test():
            ex = SpeculativeExecutor()
            pred = ToolPrediction(predicted_tool="parse", confidence=0.8)
            started = await ex.speculatively_execute(pred, {"parse": _sync_tool("parsed")})
            self.assertTrue(started)
            self.assertIn("parse", ex._active)
            # Cleanup
            await ex._cancel_all_active()

        _run(_test())

    def test_skips_unknown_tool(self):
        async def _test():
            ex = SpeculativeExecutor()
            pred = ToolPrediction(predicted_tool="missing", confidence=0.9)
            started = await ex.speculatively_execute(pred, {"parse": _sync_tool()})
            self.assertFalse(started)

        _run(_test())

    def test_skips_low_confidence(self):
        async def _test():
            ex = SpeculativeExecutor(config=SpeculativeConfig(confidence_threshold=0.8))
            pred = ToolPrediction(predicted_tool="parse", confidence=0.5)
            started = await ex.speculatively_execute(pred, {"parse": _sync_tool()})
            self.assertFalse(started)

        _run(_test())

    def test_skips_duplicate_tool(self):
        async def _test():
            ex = SpeculativeExecutor()
            pred = ToolPrediction(predicted_tool="parse", confidence=0.9)
            registry = {"parse": _slow_async_tool(0.1)}
            await ex.speculatively_execute(pred, registry)
            # Second call for same tool should be skipped
            started_again = await ex.speculatively_execute(pred, registry)
            self.assertFalse(started_again)
            await ex._cancel_all_active()

        _run(_test())

    def test_respects_max_parallel_speculations(self):
        async def _test():
            ex = SpeculativeExecutor(
                config=SpeculativeConfig(max_parallel_speculations=2)
            )
            registry = {
                "a": _slow_async_tool(1.0),
                "b": _slow_async_tool(1.0),
                "c": _slow_async_tool(1.0),
            }
            pred_a = ToolPrediction(predicted_tool="a", confidence=0.9)
            pred_b = ToolPrediction(predicted_tool="b", confidence=0.9)
            pred_c = ToolPrediction(predicted_tool="c", confidence=0.9)

            await ex.speculatively_execute(pred_a, registry)
            await ex.speculatively_execute(pred_b, registry)
            # Third should be rejected — cap reached
            started_c = await ex.speculatively_execute(pred_c, registry)
            self.assertFalse(started_c)
            self.assertEqual(len(ex._active), 2)
            await ex._cancel_all_active()

        _run(_test())

    def test_async_tool_executes(self):
        async def _test():
            ex = SpeculativeExecutor()
            pred = ToolPrediction(predicted_tool="fetch", confidence=0.9)
            await ex.speculatively_execute(pred, {"fetch": _async_tool("html")})
            result = await ex.resolve("fetch", {})
            self.assertIsNotNone(result)
            self.assertTrue(result.was_used)
            self.assertEqual(result.result, "html")

        _run(_test())


# ---------------------------------------------------------------------------
# resolve tests
# ---------------------------------------------------------------------------

class TestResolve(unittest.TestCase):
    def test_cache_hit_matching_tool(self):
        async def _test():
            ex = SpeculativeExecutor()
            pred = ToolPrediction(predicted_tool="parse", confidence=0.85)
            await ex.speculatively_execute(pred, {"parse": _async_tool("parsed_data")})
            result = await ex.resolve("parse", {})
            self.assertIsNotNone(result)
            self.assertTrue(result.was_used)
            self.assertEqual(result.result, "parsed_data")

        _run(_test())

    def test_cache_miss_no_speculation(self):
        async def _test():
            ex = SpeculativeExecutor()
            result = await ex.resolve("summarise", {})
            self.assertIsNone(result)

        _run(_test())

    def test_miss_wrong_tool_name(self):
        async def _test():
            ex = SpeculativeExecutor()
            pred = ToolPrediction(predicted_tool="parse", confidence=0.9)
            await ex.speculatively_execute(pred, {"parse": _async_tool("parsed")})
            # LLM actually calls "summarise" — miss
            result = await ex.resolve("summarise", {})
            self.assertIsNone(result)
            # Active should be cleared (discard_on_mismatch=True)
            self.assertEqual(len(ex._active), 0)

        _run(_test())

    def test_miss_args_mismatch(self):
        async def _test():
            ex = SpeculativeExecutor()
            # Prediction has specific args
            pred = ToolPrediction(
                predicted_tool="parse",
                predicted_args={"mode": "strict"},
                confidence=0.9,
            )
            await ex.speculatively_execute(pred, {"parse": _async_tool("ok")})
            # Actual call uses different args
            result = await ex.resolve("parse", {"mode": "lenient"})
            self.assertIsNone(result)

        _run(_test())

    def test_hit_empty_predicted_args_ignores_actual_args(self):
        async def _test():
            ex = SpeculativeExecutor()
            # Prediction has no args (pattern-based)
            pred = ToolPrediction(predicted_tool="parse", confidence=0.85)
            await ex.speculatively_execute(pred, {"parse": _async_tool("result")})
            # Actual call has args — should still be a hit (empty predicted_args = no constraint)
            result = await ex.resolve("parse", {"text": "any_text"})
            self.assertIsNotNone(result)
            self.assertTrue(result.was_used)

        _run(_test())

    def test_discard_on_mismatch_cancels_others(self):
        async def _test():
            ex = SpeculativeExecutor(
                config=SpeculativeConfig(
                    max_parallel_speculations=2,
                    discard_on_mismatch=True,
                )
            )
            registry = {
                "parse": _slow_async_tool(0.5, "parsed"),
                "summarise": _slow_async_tool(0.5, "summary"),
            }
            pred_parse = ToolPrediction(predicted_tool="parse", confidence=0.9)
            pred_sum = ToolPrediction(predicted_tool="summarise", confidence=0.9)
            await ex.speculatively_execute(pred_parse, registry)
            await ex.speculatively_execute(pred_sum, registry)
            self.assertEqual(len(ex._active), 2)

            # LLM decides to call "parse" — other speculation should be cancelled
            result = await ex.resolve("parse", {})
            self.assertIsNotNone(result)
            self.assertTrue(result.was_used)
            self.assertEqual(len(ex._active), 0)

        _run(_test())

    def test_latency_saved_positive_when_task_precomputed(self):
        async def _test():
            ex = SpeculativeExecutor()
            pred = ToolPrediction(predicted_tool="fast", confidence=0.9)
            tool_fn = _async_tool("done")
            await ex.speculatively_execute(pred, {"fast": tool_fn})
            # Give the task time to complete before resolving
            await asyncio.sleep(0.05)
            result = await ex.resolve("fast", {})
            self.assertIsNotNone(result)
            self.assertGreaterEqual(result.latency_saved_ms, 0.0)

        _run(_test())


# ---------------------------------------------------------------------------
# Stats tests
# ---------------------------------------------------------------------------

class TestStats(unittest.TestCase):
    def test_initial_stats_are_zero(self):
        ex = SpeculativeExecutor()
        s = ex.stats()
        self.assertEqual(s["hits"], 0)
        self.assertEqual(s["misses"], 0)
        self.assertEqual(s["total"], 0)
        self.assertEqual(s["hit_rate"], 0.0)
        self.assertEqual(s["miss_rate"], 1.0)
        self.assertEqual(s["total_latency_saved_ms"], 0.0)
        self.assertEqual(s["avg_latency_saved_ms"], 0.0)
        self.assertEqual(s["total_wasted_ms"], 0.0)

    def test_hit_increments_stats(self):
        async def _test():
            ex = SpeculativeExecutor()
            pred = ToolPrediction(predicted_tool="parse", confidence=0.9)
            await ex.speculatively_execute(pred, {"parse": _async_tool("ok")})
            await ex.resolve("parse", {})
            s = ex.stats()
            self.assertEqual(s["hits"], 1)
            self.assertEqual(s["misses"], 0)
            self.assertEqual(s["total"], 1)
            self.assertAlmostEqual(s["hit_rate"], 1.0)

        _run(_test())

    def test_miss_increments_stats(self):
        async def _test():
            ex = SpeculativeExecutor()
            result = await ex.resolve("does_not_exist", {})
            self.assertIsNone(result)
            s = ex.stats()
            self.assertEqual(s["hits"], 0)
            self.assertEqual(s["misses"], 1)
            self.assertAlmostEqual(s["miss_rate"], 1.0)

        _run(_test())

    def test_hit_rate_mixed(self):
        async def _test():
            ex = SpeculativeExecutor()
            # Two hits
            for _ in range(2):
                pred = ToolPrediction(predicted_tool="parse", confidence=0.9)
                await ex.speculatively_execute(pred, {"parse": _async_tool("ok")})
                await ex.resolve("parse", {})
            # One miss
            await ex.resolve("nonexistent", {})
            s = ex.stats()
            self.assertEqual(s["hits"], 2)
            self.assertEqual(s["misses"], 1)
            self.assertAlmostEqual(s["hit_rate"], 2 / 3)

        _run(_test())

    def test_reset_stats(self):
        async def _test():
            ex = SpeculativeExecutor()
            pred = ToolPrediction(predicted_tool="t", confidence=0.9)
            await ex.speculatively_execute(pred, {"t": _async_tool()})
            await ex.resolve("t", {})
            ex.reset_stats()
            s = ex.stats()
            self.assertEqual(s["hits"], 0)
            self.assertEqual(s["misses"], 0)

        _run(_test())


# ---------------------------------------------------------------------------
# Timeout tests
# ---------------------------------------------------------------------------

class TestTimeout(unittest.TestCase):
    def test_timeout_causes_miss(self):
        async def _test():
            ex = SpeculativeExecutor(
                config=SpeculativeConfig(timeout_per_speculation=0.01)
            )
            pred = ToolPrediction(predicted_tool="slow", confidence=0.9)
            await ex.speculatively_execute(pred, {"slow": _slow_async_tool(1.0)})
            result = await ex.resolve("slow", {})
            # The task timed out, so resolve returns None
            self.assertIsNone(result)
            self.assertEqual(ex.stats()["misses"], 1)

        _run(_test())


# ---------------------------------------------------------------------------
# CostTracker integration tests
# ---------------------------------------------------------------------------

class TestCostTrackerIntegration(unittest.TestCase):
    def test_records_speculative_call(self):
        async def _test():
            tracker = CostTracker()
            ex = SpeculativeExecutor(cost_tracker=tracker)
            pred = ToolPrediction(
                predicted_tool="parse",
                predicted_args={},
                confidence=0.9,
                basis="test",
            )
            await ex.speculatively_execute(pred, {"parse": _async_tool("ok")})
            await ex._cancel_all_active()

            records = tracker.records()
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].provider, "speculative")
            self.assertEqual(records[0].model, "parse")
            meta = records[0].metadata
            self.assertEqual(meta["type"], "speculative_tool")
            self.assertAlmostEqual(meta["confidence"], 0.9)

        _run(_test())

    def test_no_tracker_does_not_raise(self):
        async def _test():
            ex = SpeculativeExecutor(cost_tracker=None)
            pred = ToolPrediction(predicted_tool="t", confidence=0.9)
            # Should not raise even without a tracker
            await ex.speculatively_execute(pred, {"t": _async_tool()})
            await ex._cancel_all_active()

        _run(_test())


if __name__ == "__main__":
    unittest.main()

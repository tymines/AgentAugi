"""Unit tests for evoagentx.core.parallel_executor."""

import asyncio
import time
import unittest

from evoagentx.core.parallel_executor import (
    CallResult,
    CostBudgetExceeded,
    ExecutionResult,
    ParallelExecutor,
    ToolCall,
    _DependencyGraph,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sync_double(x: int) -> int:
    return x * 2


def _sync_add(a: int, b: int) -> int:
    return a + b


def _sync_error():
    raise RuntimeError("deliberate error")


async def _async_greet(name: str) -> str:
    await asyncio.sleep(0)
    return f"hello {name}"


async def _async_slow(seconds: float) -> str:
    await asyncio.sleep(seconds)
    return "done"


# ---------------------------------------------------------------------------
# Tests for _DependencyGraph
# ---------------------------------------------------------------------------

class TestDependencyGraph(unittest.TestCase):

    def test_no_deps_single_wave(self):
        calls = [
            ToolCall("a", _sync_double, args=[1]),
            ToolCall("b", _sync_double, args=[2]),
        ]
        graph = _DependencyGraph(calls)
        self.assertEqual(len(graph.waves), 1)
        self.assertCountEqual(graph.waves[0], ["a", "b"])

    def test_linear_deps_multiple_waves(self):
        calls = [
            ToolCall("a", _sync_double, args=[1]),
            ToolCall("b", _sync_double, args=[2], depends_on=["a"]),
            ToolCall("c", _sync_double, args=[3], depends_on=["b"]),
        ]
        graph = _DependencyGraph(calls)
        self.assertEqual(len(graph.waves), 3)
        self.assertEqual(graph.waves[0], ["a"])
        self.assertEqual(graph.waves[1], ["b"])
        self.assertEqual(graph.waves[2], ["c"])

    def test_diamond_dependency(self):
        calls = [
            ToolCall("root", _sync_double, args=[1]),
            ToolCall("left", _sync_double, args=[2], depends_on=["root"]),
            ToolCall("right", _sync_double, args=[3], depends_on=["root"]),
            ToolCall("join", _sync_double, args=[4], depends_on=["left", "right"]),
        ]
        graph = _DependencyGraph(calls)
        self.assertEqual(graph.waves[0], ["root"])
        self.assertCountEqual(graph.waves[1], ["left", "right"])
        self.assertEqual(graph.waves[2], ["join"])

    def test_duplicate_name_raises(self):
        calls = [
            ToolCall("a", _sync_double, args=[1]),
            ToolCall("a", _sync_double, args=[2]),
        ]
        with self.assertRaises(ValueError):
            _DependencyGraph(calls)

    def test_unknown_dependency_raises(self):
        calls = [ToolCall("a", _sync_double, args=[1], depends_on=["nonexistent"])]
        with self.assertRaises(ValueError):
            _DependencyGraph(calls)

    def test_cycle_raises(self):
        calls = [
            ToolCall("a", _sync_double, args=[1], depends_on=["b"]),
            ToolCall("b", _sync_double, args=[2], depends_on=["a"]),
        ]
        with self.assertRaises(ValueError):
            _DependencyGraph(calls)


# ---------------------------------------------------------------------------
# Tests for ParallelExecutor
# ---------------------------------------------------------------------------

class TestParallelExecutor(unittest.TestCase):

    def setUp(self):
        self.executor = ParallelExecutor(max_concurrency=4, default_timeout=5.0)

    # --- empty batch ---

    def test_empty_batch(self):
        result = self.executor.execute_sync([])
        self.assertIsInstance(result, ExecutionResult)
        self.assertEqual(len(result.results), 0)
        self.assertTrue(result.success)

    # --- sync callables ---

    def test_sync_callable_succeeds(self):
        calls = [ToolCall("double", _sync_double, args=[5])]
        result = self.executor.execute_sync(calls)
        self.assertTrue(result.success)
        self.assertEqual(result.outputs["double"], 10)

    def test_multiple_sync_callables(self):
        calls = [
            ToolCall("d2", _sync_double, args=[2]),
            ToolCall("d3", _sync_double, args=[3]),
        ]
        result = self.executor.execute_sync(calls)
        self.assertTrue(result.success)
        self.assertEqual(result.outputs["d2"], 4)
        self.assertEqual(result.outputs["d3"], 6)

    # --- async callables ---

    def test_async_callable_succeeds(self):
        calls = [ToolCall("greet", _async_greet, args=["world"])]
        result = self.executor.execute_sync(calls)
        self.assertTrue(result.success)
        self.assertEqual(result.outputs["greet"], "hello world")

    # --- error handling ---

    def test_error_recorded_without_raising(self):
        calls = [
            ToolCall("ok", _sync_double, args=[1]),
            ToolCall("fail", _sync_error),
        ]
        result = self.executor.execute_sync(calls)
        self.assertFalse(result.success)
        self.assertIn("fail", result.failed_calls)
        self.assertIn("ok", result.outputs)
        fail_res = next(r for r in result.results if r.name == "fail")
        self.assertIsInstance(fail_res.error, RuntimeError)

    # --- timeout ---

    def test_timeout_marks_call_as_failed(self):
        executor = ParallelExecutor(max_concurrency=4, default_timeout=0.05)
        calls = [ToolCall("slow", _async_slow, args=[5.0])]
        result = executor.execute_sync(calls)
        self.assertFalse(result.success)
        self.assertIn("slow", result.failed_calls)
        slow_res = next(r for r in result.results if r.name == "slow")
        self.assertTrue(slow_res.timed_out)

    def test_per_call_timeout_overrides_default(self):
        # Very tight per-call timeout
        calls = [ToolCall("slow", _async_slow, args=[5.0], timeout=0.05)]
        result = self.executor.execute_sync(calls)
        self.assertIn("slow", result.failed_calls)

    # --- dependencies ---

    def test_dependency_respected(self):
        """Ensure dependent call sees predecessor in results."""
        shared = {}

        def write_value():
            shared["key"] = 42

        def read_value():
            return shared.get("key")

        calls = [
            ToolCall("write", write_value),
            ToolCall("read", read_value, depends_on=["write"]),
        ]
        result = self.executor.execute_sync(calls)
        self.assertTrue(result.success)
        self.assertEqual(result.outputs["read"], 42)

    # --- cost tracking ---

    def test_cost_tracked(self):
        calls = [
            ToolCall("a", _sync_double, args=[1], cost_estimate=1.5),
            ToolCall("b", _sync_double, args=[2], cost_estimate=2.5),
        ]
        result = self.executor.execute_sync(calls)
        self.assertAlmostEqual(result.total_cost, 4.0)

    def test_cost_budget_exceeded_raises(self):
        executor = ParallelExecutor(cost_budget=1.0)
        calls = [ToolCall("a", _sync_double, args=[1], cost_estimate=2.0)]
        with self.assertRaises(CostBudgetExceeded):
            executor.execute_sync(calls)

    # --- parallelism speedup ---

    def test_independent_calls_run_concurrently(self):
        """Two 0.1s async sleeps should finish in < 0.15s when parallelised."""
        calls = [
            ToolCall("s1", _async_slow, args=[0.1]),
            ToolCall("s2", _async_slow, args=[0.1]),
        ]
        start = time.perf_counter()
        result = self.executor.execute_sync(calls)
        elapsed = time.perf_counter() - start
        self.assertTrue(result.success)
        self.assertLess(elapsed, 0.20, "Independent calls should run concurrently")

    # --- CallResult.ok property ---

    def test_call_result_ok_true_on_success(self):
        calls = [ToolCall("x", _sync_double, args=[3])]
        result = self.executor.execute_sync(calls)
        cr = result.results[0]
        self.assertTrue(cr.ok)

    def test_call_result_ok_false_on_error(self):
        calls = [ToolCall("x", _sync_error)]
        result = self.executor.execute_sync(calls)
        cr = result.results[0]
        self.assertFalse(cr.ok)

    # --- ExecutionResult.get convenience ---

    def test_execution_result_get_missing_returns_default(self):
        result = self.executor.execute_sync([ToolCall("a", _sync_double, args=[1])])
        self.assertIsNone(result.get("nonexistent"))
        self.assertEqual(result.get("nonexistent", "fallback"), "fallback")


if __name__ == "__main__":
    unittest.main()

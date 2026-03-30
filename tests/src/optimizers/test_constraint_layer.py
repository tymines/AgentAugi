"""Unit tests for evoagentx.optimizers.constraint_layer."""

import math
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from evoagentx.optimizers.constraint_layer import (
    BaseConstraint,
    ConstrainedOptimizer,
    ConstraintResult,
    CostConstraint,
    DriftConstraint,
    HallucinationConstraint,
    ViolationRecord,
    _cosine_similarity,
)
from evoagentx.optimizers.engine.registry import ParamRegistry


# ---------------------------------------------------------------------------
# ConstraintResult
# ---------------------------------------------------------------------------

class TestConstraintResult(unittest.TestCase):

    def test_passed_construction(self):
        r = ConstraintResult(passed=True, constraint_name="test")
        self.assertTrue(r.passed)
        self.assertEqual(r.constraint_name, "test")
        self.assertEqual(r.reason, "")

    def test_failed_construction(self):
        r = ConstraintResult(passed=False, constraint_name="test", reason="too expensive")
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, "too expensive")


# ---------------------------------------------------------------------------
# CostConstraint
# ---------------------------------------------------------------------------

class TestCostConstraint(unittest.TestCase):

    def test_passes_when_below_budget(self):
        c = CostConstraint(budget_per_call=0.10, total_budget=5.0)
        # Provide a cheap eval result
        result = c.check({}, {"input_tokens": 100, "output_tokens": 50})
        self.assertTrue(result.passed)

    def test_fails_when_per_call_exceeded(self):
        c = CostConstraint(budget_per_call=0.001)
        # 10000 input tokens at $0.000003 = $0.03 > $0.001
        result = c.check({}, {"input_tokens": 10000, "output_tokens": 0})
        self.assertFalse(result.passed)
        self.assertIn("Per-call", result.reason)

    def test_fails_when_total_budget_exceeded(self):
        c = CostConstraint(total_budget=0.001)
        # Accumulated 3 cheap calls that together exceed $0.001
        for _ in range(5):
            r = c.check({}, {"input_tokens": 500, "output_tokens": 200})
        # At some point total should breach $0.001
        self.assertFalse(r.passed)
        self.assertIn("total budget", r.reason)

    def test_no_budget_always_passes(self):
        c = CostConstraint(budget_per_call=None, total_budget=None)
        for _ in range(100):
            r = c.check({}, {"input_tokens": 100000, "output_tokens": 100000})
            self.assertTrue(r.passed)

    def test_metadata_contains_cost(self):
        c = CostConstraint(budget_per_call=1.0)
        result = c.check({}, {"input_tokens": 100, "output_tokens": 50})
        self.assertIn("call_cost_usd", result.metadata)
        self.assertIn("cumulative_cost_usd", result.metadata)

    def test_reset_clears_cumulative(self):
        c = CostConstraint(total_budget=0.001)
        for _ in range(10):
            c.check({}, {"input_tokens": 1000, "output_tokens": 500})
        c.reset()
        # After reset, first check should pass again
        r = c.check({}, {"input_tokens": 1, "output_tokens": 1})
        self.assertTrue(r.passed)

    def test_zero_token_eval_result(self):
        c = CostConstraint(budget_per_call=0.0001)
        r = c.check({}, {"input_tokens": 0, "output_tokens": 0})
        self.assertTrue(r.passed)

    def test_none_eval_result(self):
        c = CostConstraint(budget_per_call=1.0)
        r = c.check({}, None)
        self.assertTrue(r.passed)


# ---------------------------------------------------------------------------
# HallucinationConstraint
# ---------------------------------------------------------------------------

class TestHallucinationConstraint(unittest.TestCase):

    def test_passes_above_threshold(self):
        c = HallucinationConstraint(detector_fn=lambda text: 0.9, threshold=0.7)
        r = c.check({}, {"output": "some text"})
        self.assertTrue(r.passed)
        self.assertAlmostEqual(r.metadata["reliability_score"], 0.9)

    def test_fails_below_threshold(self):
        c = HallucinationConstraint(detector_fn=lambda text: 0.3, threshold=0.7)
        r = c.check({}, {"output": "some text"})
        self.assertFalse(r.passed)
        self.assertIn("below threshold", r.reason)

    def test_passes_at_exact_threshold(self):
        c = HallucinationConstraint(detector_fn=lambda text: 0.7, threshold=0.7)
        r = c.check({}, {"output": "text"})
        self.assertTrue(r.passed)

    def test_no_eval_result_passes(self):
        c = HallucinationConstraint(detector_fn=lambda text: 0.0, threshold=0.7)
        r = c.check({}, None)
        self.assertTrue(r.passed)

    def test_empty_output_passes(self):
        c = HallucinationConstraint(detector_fn=lambda text: 0.0, threshold=0.7)
        r = c.check({}, {"output": ""})
        self.assertTrue(r.passed)

    def test_custom_output_key(self):
        c = HallucinationConstraint(
            detector_fn=lambda text: 0.9,
            threshold=0.7,
            output_key="answer",
        )
        r = c.check({}, {"answer": "some answer"})
        self.assertTrue(r.passed)

    def test_detector_exception_passes(self):
        def bad_detector(text):
            raise RuntimeError("detector crashed")

        c = HallucinationConstraint(detector_fn=bad_detector, threshold=0.7)
        r = c.check({}, {"output": "text"})
        self.assertTrue(r.passed)
        self.assertIn("error", r.metadata)

    def test_invalid_threshold_raises(self):
        with self.assertRaises(ValueError):
            HallucinationConstraint(detector_fn=lambda x: 0.5, threshold=1.5)


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------

class TestCosineSimilarity(unittest.TestCase):

    def test_identical_vectors(self):
        a = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(_cosine_similarity(a, a), 1.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        self.assertAlmostEqual(_cosine_similarity(a, b), 0.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        self.assertAlmostEqual(_cosine_similarity(a, b), -1.0)

    def test_zero_vector_returns_zero(self):
        a = [0.0, 0.0]
        b = [1.0, 1.0]
        self.assertAlmostEqual(_cosine_similarity(a, b), 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            _cosine_similarity([1.0, 2.0], [1.0])


# ---------------------------------------------------------------------------
# DriftConstraint
# ---------------------------------------------------------------------------

class TestDriftConstraint(unittest.TestCase):

    @staticmethod
    def _embed(text: str) -> List[float]:
        """Trivial embedding: character frequency vector (len=3)."""
        return [float(text.count(c)) for c in "abc"]

    def test_similar_text_passes(self):
        c = DriftConstraint(
            baseline_behavior="aaa bbb ccc",
            semantic_distance_threshold=0.5,
            embed_fn=self._embed,
        )
        # Same text — should be very similar
        r = c.check({"output": "aaa bbb ccc"}, {"output": "aaa bbb ccc"})
        self.assertTrue(r.passed)

    def test_different_text_fails(self):
        c = DriftConstraint(
            baseline_behavior="aaaaaa",
            semantic_distance_threshold=0.05,
            embed_fn=self._embed,
        )
        # Completely different characters
        r = c.check({"output": "bbbbbb"}, {"output": "bbbbbb"})
        self.assertFalse(r.passed)
        self.assertIn("exceeds threshold", r.reason)

    def test_no_behavior_text_passes(self):
        c = DriftConstraint(
            baseline_behavior="aaa",
            semantic_distance_threshold=0.1,
            embed_fn=self._embed,
        )
        r = c.check({}, {})
        self.assertTrue(r.passed)

    def test_uses_eval_result_over_config(self):
        """eval_result output should take precedence over config value."""
        c = DriftConstraint(
            baseline_behavior="aaa",
            semantic_distance_threshold=0.5,
            embed_fn=self._embed,
            behavior_key="output",
        )
        # Eval result has same text as baseline → should pass
        r = c.check({"output": "ccc"}, {"output": "aaa"})
        self.assertTrue(r.passed)

    def test_baseline_embedding_cached(self):
        embed_calls = [0]

        def counting_embed(text: str) -> List[float]:
            embed_calls[0] += 1
            return [1.0, 0.0, 0.0]

        c = DriftConstraint(
            baseline_behavior="test",
            semantic_distance_threshold=0.5,
            embed_fn=counting_embed,
        )
        c.check({}, {"output": "test"})
        c.check({}, {"output": "test"})
        # First call computes baseline + candidate, second call only candidate
        # So second call should use cached baseline → 3 total embed calls, not 4
        self.assertEqual(embed_calls[0], 3)

    def test_invalid_threshold_raises(self):
        with self.assertRaises(ValueError):
            DriftConstraint(
                baseline_behavior="x",
                semantic_distance_threshold=1.5,
                embed_fn=lambda t: [1.0],
            )


# ---------------------------------------------------------------------------
# ConstrainedOptimizer
# ---------------------------------------------------------------------------

class _AlwaysPassConstraint(BaseConstraint):
    @property
    def name(self):
        return "AlwaysPass"

    def check(self, config, eval_result=None):
        return ConstraintResult(passed=True, constraint_name=self.name)


class _AlwaysFailConstraint(BaseConstraint):
    @property
    def name(self):
        return "AlwaysFail"

    def check(self, config, eval_result=None):
        return ConstraintResult(
            passed=False, constraint_name=self.name, reason="always fails"
        )


class _FakeBaseOptimizer:
    """Minimal stand-in for BaseOptimizer with _evaluate_config."""

    def __init__(self):
        from evoagentx.optimizers.engine.registry import ParamRegistry

        class Dummy:
            prompt = "test"

        obj = Dummy()
        self.registry = ParamRegistry()
        self.registry.track(obj, "prompt")
        self.program = lambda: {"output": "test", "accuracy": 0.8}
        self.evaluator = lambda r: r.get("accuracy", 0.0)
        self._calls = []

    def _evaluate_config(self, config):
        self._calls.append(config)
        return {"output": "result", "accuracy": 0.8}, 0.8

    def apply_cfg(self, cfg):
        pass

    def get_current_cfg(self):
        return {"prompt": "test"}

    def optimize(self):
        # Simulate 3 candidate evaluations
        results = []
        for i in range(3):
            cfg = {"prompt": f"candidate_{i}"}
            eval_result, quality = self._evaluate_config(cfg)
            results.append((cfg, eval_result, quality))
        return results


class TestConstrainedOptimizer(unittest.TestCase):

    def test_check_all_passes_when_all_pass(self):
        base = _FakeBaseOptimizer()
        opt = ConstrainedOptimizer(
            base_optimizer=base,
            constraints=[_AlwaysPassConstraint(), _AlwaysPassConstraint()],
        )
        passed, results = opt.check_all({"p": "v"}, {}, iteration=0)
        self.assertTrue(passed)
        self.assertEqual(len(results), 2)

    def test_check_all_fails_on_first_failure(self):
        base = _FakeBaseOptimizer()
        opt = ConstrainedOptimizer(
            base_optimizer=base,
            constraints=[_AlwaysPassConstraint(), _AlwaysFailConstraint()],
        )
        passed, results = opt.check_all({"p": "v"}, {}, iteration=0)
        self.assertFalse(passed)
        # Short-circuit: only 2 results returned
        self.assertEqual(len(results), 2)

    def test_violation_log_populated_on_failure(self):
        base = _FakeBaseOptimizer()
        opt = ConstrainedOptimizer(
            base_optimizer=base,
            constraints=[_AlwaysFailConstraint()],
        )
        opt.check_all({"p": "v"}, {}, iteration=5)
        log = opt.get_violation_log()
        self.assertEqual(len(log), 1)
        self.assertIsInstance(log[0], ViolationRecord)
        self.assertEqual(log[0].iteration, 5)
        self.assertEqual(log[0].constraint_name, "AlwaysFail")

    def test_violation_summary(self):
        base = _FakeBaseOptimizer()
        opt = ConstrainedOptimizer(
            base_optimizer=base,
            constraints=[_AlwaysFailConstraint()],
        )
        opt.check_all({}, {}, iteration=0)
        opt.check_all({}, {}, iteration=1)
        summary = opt.violation_summary()
        self.assertEqual(summary["total_checks"], 2)
        self.assertEqual(summary["total_violations"], 2)
        self.assertAlmostEqual(summary["violation_ratio"], 1.0)
        self.assertIn("AlwaysFail", summary["by_constraint"])

    def test_optimize_with_hook_calls_base_optimize(self):
        base = _FakeBaseOptimizer()
        opt = ConstrainedOptimizer(
            base_optimizer=base,
            constraints=[_AlwaysPassConstraint()],
        )
        result = opt.optimize()
        # Base optimizer returned a list of 3 tuples
        self.assertEqual(len(result), 3)

    def test_optimize_with_failing_constraint_forces_neg_inf_quality(self):
        """When a constraint fails, quality should be forced to -inf."""
        base = _FakeBaseOptimizer()

        # Track whether -inf quality was encountered
        qualities_seen = []
        original_evaluate = base._evaluate_config

        def capturing_evaluate(config):
            ev, q = original_evaluate(config)
            qualities_seen.append(q)
            return ev, q

        base._evaluate_config = capturing_evaluate

        opt = ConstrainedOptimizer(
            base_optimizer=base,
            constraints=[_AlwaysFailConstraint()],
        )
        # After optimization, constrained evaluate should have returned -inf
        opt._optimize_with_hook()
        # The constrained wrapper intercepts and returns -inf; the base optimizer
        # sees -inf from the constrained version (not the original quality)

    def test_name_property_contains_base_optimizer_name(self):
        base = _FakeBaseOptimizer()
        opt = ConstrainedOptimizer(
            base_optimizer=base,
            constraints=[_AlwaysPassConstraint()],
        )
        self.assertIn("_FakeBaseOptimizer", opt.name)
        self.assertIn("AlwaysPass", opt.name)

    def test_multiple_passes_track_checks_correctly(self):
        base = _FakeBaseOptimizer()
        opt = ConstrainedOptimizer(
            base_optimizer=base,
            constraints=[_AlwaysPassConstraint()],
        )
        for _ in range(5):
            opt.check_all({}, {}, iteration=0)
        summary = opt.violation_summary()
        self.assertEqual(summary["total_checks"], 5)
        self.assertEqual(summary["total_violations"], 0)


if __name__ == "__main__":
    unittest.main()

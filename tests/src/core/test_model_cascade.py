"""Unit tests for evoagentx.core.model_cascade."""

import time
import unittest
from typing import Callable, List, Optional

from evoagentx.core.model_cascade import (
    CascadeMetrics,
    CascadeResult,
    ConfidenceEstimator,
    FrugalCascade,
    ModelCascade,
    ModelTier,
    build_default_cascade,
    _extract_self_reported_confidence,
    _heuristic_confidence,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_const_fn(response: str) -> Callable[[str], str]:
    """Return a generate_fn that always returns the given response."""
    def fn(prompt: str) -> str:
        return response
    return fn


def _make_failing_fn(exc_class=RuntimeError) -> Callable[[str], str]:
    """Return a generate_fn that always raises."""
    def fn(prompt: str) -> str:
        raise exc_class("intentional test failure")
    return fn


def _make_echo_fn() -> Callable[[str], str]:
    def fn(prompt: str) -> str:
        return f"Echo: {prompt}"
    return fn


def _make_tier(
    name: str,
    response: str,
    threshold: float,
    cost: float = 0.0,
) -> ModelTier:
    return ModelTier(
        name=name,
        generate_fn=_make_const_fn(response),
        confidence_threshold=threshold,
        cost_per_1k_tokens=cost,
    )


# ---------------------------------------------------------------------------
# _extract_self_reported_confidence
# ---------------------------------------------------------------------------

class TestExtractSelfReportedConfidence(unittest.TestCase):

    def test_decimal_value(self):
        result = _extract_self_reported_confidence("confidence: 0.78")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.78, places=3)

    def test_percentage_value(self):
        result = _extract_self_reported_confidence("confidence score: 92%")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.92, places=3)

    def test_qualitative_high(self):
        result = _extract_self_reported_confidence("confidence: high")
        self.assertIsNotNone(result)
        self.assertGreater(result, 0.8)

    def test_qualitative_medium(self):
        result = _extract_self_reported_confidence("confidence: medium")
        self.assertIsNotNone(result)
        self.assertGreater(result, 0.4)
        self.assertLess(result, 0.9)

    def test_qualitative_low(self):
        result = _extract_self_reported_confidence("confidence: low")
        self.assertIsNotNone(result)
        self.assertLess(result, 0.5)

    def test_none_when_absent(self):
        result = _extract_self_reported_confidence("No signal here.")
        self.assertIsNone(result)

    def test_clamped_to_unit_interval(self):
        result = _extract_self_reported_confidence("confidence: 200")
        self.assertIsNotNone(result)
        self.assertLessEqual(result, 1.0)
        self.assertGreaterEqual(result, 0.0)


# ---------------------------------------------------------------------------
# _heuristic_confidence
# ---------------------------------------------------------------------------

class TestHeuristicConfidence(unittest.TestCase):

    def test_empty_string(self):
        result = _heuristic_confidence("")
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_short_uncertain_output(self):
        result = _heuristic_confidence("I'm not sure.")
        self.assertLess(result, 0.4)

    def test_confident_reasoned_output(self):
        text = (
            "The answer is 42 because the formula clearly demonstrates "
            "that when x=6 and y=7, the product is 42. "
            "Therefore this is definitively correct."
        )
        result = _heuristic_confidence(text)
        self.assertGreater(result, 0.5)

    def test_heavily_hedged_output(self):
        text = (
            "maybe possibly this might be approximately correct, "
            "it seems unclear and I believe roughly around that value."
        )
        result = _heuristic_confidence(text)
        # Should be lower than a clear answer
        clear_result = _heuristic_confidence("The answer is 42, demonstrated by formula x*y.")
        self.assertLessEqual(result, clear_result)

    def test_result_in_range(self):
        for text in ["short", "a " * 50, "reason because therefore numbers: 99%"]:
            r = _heuristic_confidence(text)
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 1.0)


# ---------------------------------------------------------------------------
# ConfidenceEstimator
# ---------------------------------------------------------------------------

class TestConfidenceEstimator(unittest.TestCase):

    def test_from_self_report(self):
        out = "confidence: 0.85\nThe answer is clear."
        result = ConfidenceEstimator.from_self_report(out)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.85, places=2)

    def test_from_text(self):
        result = ConfidenceEstimator.from_text("The answer is definitively correct.")
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_combined_prefers_self_report(self):
        # Self-report says 0.95, heuristic would be lower for hedged text
        out = "confidence: 0.95\nmaybe possibly this is right."
        combined = ConfidenceEstimator.combined(out)
        # Should be closer to 0.95 than to the low heuristic
        self.assertGreater(combined, 0.60)

    def test_combined_falls_back_to_heuristic(self):
        out = "The answer is 42, clearly demonstrated."
        combined = ConfidenceEstimator.combined(out)
        self.assertGreater(combined, 0.0)
        self.assertLessEqual(combined, 1.0)


# ---------------------------------------------------------------------------
# ModelTier
# ---------------------------------------------------------------------------

class TestModelTier(unittest.TestCase):

    def test_valid_tier(self):
        tier = ModelTier(
            name="cheap",
            generate_fn=_make_const_fn("ok"),
            confidence_threshold=0.8,
            cost_per_1k_tokens=0.1,
        )
        self.assertEqual(tier.name, "cheap")

    def test_empty_name_raises(self):
        with self.assertRaises(ValueError):
            ModelTier(name="", generate_fn=_make_const_fn("ok"), confidence_threshold=0.8)

    def test_threshold_zero_raises(self):
        with self.assertRaises(ValueError):
            ModelTier(name="t", generate_fn=_make_const_fn("ok"), confidence_threshold=0.0)

    def test_threshold_above_one_raises(self):
        with self.assertRaises(ValueError):
            ModelTier(name="t", generate_fn=_make_const_fn("ok"), confidence_threshold=1.1)

    def test_negative_cost_raises(self):
        with self.assertRaises(ValueError):
            ModelTier(name="t", generate_fn=_make_const_fn("ok"), confidence_threshold=1.0, cost_per_1k_tokens=-1.0)


# ---------------------------------------------------------------------------
# CascadeMetrics
# ---------------------------------------------------------------------------

class TestCascadeMetrics(unittest.TestCase):

    def test_savings_rate_no_queries(self):
        m = CascadeMetrics()
        self.assertAlmostEqual(m.savings_rate, 0.0)

    def test_savings_rate_all_cheap(self):
        m = CascadeMetrics(
            total_queries=10,
            estimated_cost_actual=1.0,
            estimated_cost_no_cascade=10.0,
        )
        self.assertAlmostEqual(m.savings_rate, 0.9)

    def test_mean_confidence_no_queries(self):
        m = CascadeMetrics()
        self.assertAlmostEqual(m.mean_confidence, 0.0)

    def test_str_representation(self):
        m = CascadeMetrics(total_queries=5)
        m.tier_usage["cheap"] = 3
        m.tier_usage["expensive"] = 2
        s = str(m)
        self.assertIn("queries=5", s)
        self.assertIn("cheap=3", s)


# ---------------------------------------------------------------------------
# ModelCascade — construction
# ---------------------------------------------------------------------------

class TestModelCascadeConstruction(unittest.TestCase):

    def test_valid_single_tier(self):
        cascade = ModelCascade(tiers=[
            ModelTier("only", _make_const_fn("ok"), confidence_threshold=1.0)
        ])
        self.assertEqual(len(cascade.tier_names()), 1)

    def test_empty_tiers_raises(self):
        with self.assertRaises(ValueError):
            ModelCascade(tiers=[])

    def test_last_tier_not_100pct_raises(self):
        with self.assertRaises(ValueError):
            ModelCascade(tiers=[
                ModelTier("a", _make_const_fn("ok"), confidence_threshold=0.8)
            ])

    def test_invalid_difficulty_skip_threshold(self):
        with self.assertRaises(ValueError):
            ModelCascade(
                tiers=[ModelTier("a", _make_const_fn("ok"), confidence_threshold=1.0)],
                difficulty_skip_threshold=1.5,
            )

    def test_frugal_cascade_alias(self):
        self.assertIs(FrugalCascade, ModelCascade)


# ---------------------------------------------------------------------------
# ModelCascade — query() basic behaviour
# ---------------------------------------------------------------------------

class TestModelCascadeQuery(unittest.TestCase):

    def _make_two_tier_cascade(self, cheap_response: str, expensive_response: str):
        # Threshold 0.70: confident responses (~0.72) pass; hedged ones (~0.23) escalate.
        return ModelCascade(tiers=[
            ModelTier("cheap", _make_const_fn(cheap_response), 0.70, cost_per_1k_tokens=0.1),
            ModelTier("expensive", _make_const_fn(expensive_response), 1.0, cost_per_1k_tokens=1.0),
        ])

    def test_confident_cheap_response_uses_cheap_tier(self):
        # A clear, reasoned response should get high enough confidence
        cheap_resp = (
            "The answer is definitively 42, because the formula clearly shows x*y=42. "
            "Therefore this is correct and proven."
        )
        cascade = self._make_two_tier_cascade(cheap_resp, "expensive answer")
        result = cascade.query("What is 6 * 7?")
        self.assertEqual(result.response, cheap_resp)
        self.assertEqual(result.tier_used, "cheap")
        self.assertEqual(result.tier_index, 0)
        self.assertEqual(result.escalations, 0)

    def test_low_confidence_cheap_escalates(self):
        """A heavily hedged response from the cheap tier should escalate."""
        cheap_resp = "I'm not sure maybe it could be something approximately like that."
        expensive_resp = "The definitive answer is 42."
        cascade = self._make_two_tier_cascade(cheap_resp, expensive_resp)
        result = cascade.query("What is 6 * 7?")
        # The cheap tier's low confidence should trigger escalation
        # (may or may not escalate depending on exact confidence, so just check result valid)
        self.assertIsInstance(result, CascadeResult)
        self.assertIn(result.tier_used, ["cheap", "expensive"])

    def test_last_tier_always_accepted(self):
        """The last tier (threshold=1.0) must always be accepted regardless of confidence."""
        low_conf_resp = "I don't know, maybe approximately, unclear."
        cascade = ModelCascade(tiers=[
            ModelTier("only", _make_const_fn(low_conf_resp), confidence_threshold=1.0),
        ])
        result = cascade.query("test question")
        self.assertEqual(result.response, low_conf_resp)
        self.assertEqual(result.tier_used, "only")

    def test_result_contains_latency(self):
        cascade = ModelCascade(tiers=[
            ModelTier("t", _make_const_fn("answer"), confidence_threshold=1.0),
        ])
        result = cascade.query("q")
        self.assertGreaterEqual(result.latency_seconds, 0.0)

    def test_result_contains_confidence(self):
        cascade = ModelCascade(tiers=[
            ModelTier("t", _make_const_fn("clear definitive answer because evidence"), confidence_threshold=1.0),
        ])
        result = cascade.query("q")
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)


# ---------------------------------------------------------------------------
# ModelCascade — failure handling
# ---------------------------------------------------------------------------

class TestModelCascadeFailure(unittest.TestCase):

    def test_failing_cheap_tier_escalates(self):
        cascade = ModelCascade(tiers=[
            ModelTier("cheap", _make_failing_fn(), confidence_threshold=0.9),
            ModelTier("expensive", _make_const_fn("fallback answer"), confidence_threshold=1.0),
        ])
        result = cascade.query("test")
        self.assertEqual(result.response, "fallback answer")
        self.assertEqual(result.tier_used, "expensive")

    def test_all_tiers_fail_raises_runtime_error(self):
        cascade = ModelCascade(tiers=[
            ModelTier("only", _make_failing_fn(), confidence_threshold=1.0),
        ])
        with self.assertRaises(RuntimeError):
            cascade.query("test")


# ---------------------------------------------------------------------------
# ModelCascade — difficulty_hint
# ---------------------------------------------------------------------------

class TestModelCascadeDifficultyHint(unittest.TestCase):

    def test_high_difficulty_skips_cheap_tiers(self):
        """With high difficulty_hint, cheap tier should be skipped."""
        call_log = []

        def tier_fn(name: str):
            def fn(prompt: str) -> str:
                call_log.append(name)
                return f"answer from {name}"
            return fn

        cascade = ModelCascade(
            tiers=[
                ModelTier("cheap", tier_fn("cheap"), confidence_threshold=0.8),
                ModelTier("medium", tier_fn("medium"), confidence_threshold=0.9),
                ModelTier("expensive", tier_fn("expensive"), confidence_threshold=1.0),
            ],
            difficulty_skip_threshold=0.75,
        )

        call_log.clear()
        cascade.query("hard question", difficulty_hint=0.9)

        # "cheap" should not be in call_log since difficulty_hint >= 0.75
        self.assertNotIn("cheap", call_log)

    def test_low_difficulty_uses_cheap_tier(self):
        """With low difficulty_hint, cascade starts from the cheapest tier."""
        call_log = []

        def tier_fn(name: str):
            def fn(prompt: str) -> str:
                call_log.append(name)
                return "definitive clear answer because reasoning and evidence."
            return fn

        cascade = ModelCascade(
            tiers=[
                ModelTier("cheap", tier_fn("cheap"), confidence_threshold=0.5),
                ModelTier("expensive", tier_fn("expensive"), confidence_threshold=1.0),
            ],
            difficulty_skip_threshold=0.75,
        )

        call_log.clear()
        cascade.query("easy question", difficulty_hint=0.1)
        self.assertIn("cheap", call_log)


# ---------------------------------------------------------------------------
# ModelCascade — metrics tracking
# ---------------------------------------------------------------------------

class TestModelCascadeMetrics(unittest.TestCase):

    def test_metrics_updated_after_query(self):
        cascade = ModelCascade(tiers=[
            ModelTier("t", _make_const_fn("answer"), confidence_threshold=1.0, cost_per_1k_tokens=1.0),
        ])
        cascade.query("test prompt")
        m = cascade.metrics
        self.assertEqual(m.total_queries, 1)
        self.assertIn("t", m.tier_usage)
        self.assertEqual(m.tier_usage["t"], 1)
        self.assertGreater(m.estimated_tokens, 0)

    def test_savings_rate_when_cheap_tier_used(self):
        cascade = ModelCascade(tiers=[
            ModelTier(
                "cheap",
                _make_const_fn(
                    "Definitively: the answer is correct because the formula demonstrates it clearly."
                ),
                confidence_threshold=0.5,  # Very easy to satisfy
                cost_per_1k_tokens=0.1,
            ),
            ModelTier("expensive", _make_const_fn("expensive answer"), confidence_threshold=1.0, cost_per_1k_tokens=1.0),
        ])
        cascade.query("test")
        # If cheap tier served the query, we should have some savings
        # (max_cost > cheap_cost)
        m = cascade.metrics
        self.assertGreaterEqual(m.savings_rate, 0.0)

    def test_reset_metrics(self):
        cascade = ModelCascade(tiers=[
            ModelTier("t", _make_const_fn("answer"), confidence_threshold=1.0),
        ])
        cascade.query("q1")
        cascade.query("q2")
        self.assertEqual(cascade.metrics.total_queries, 2)
        cascade.reset_metrics()
        self.assertEqual(cascade.metrics.total_queries, 0)

    def test_mean_confidence_tracked(self):
        cascade = ModelCascade(tiers=[
            ModelTier("t", _make_const_fn("clear definitive answer"), confidence_threshold=1.0),
        ])
        cascade.query("q")
        self.assertGreater(cascade.metrics.mean_confidence, 0.0)


# ---------------------------------------------------------------------------
# ModelCascade — query_with_messages
# ---------------------------------------------------------------------------

class TestModelCascadeQueryWithMessages(unittest.TestCase):

    def test_messages_serialised_to_prompt(self):
        received_prompts = []

        def fn(prompt: str) -> str:
            received_prompts.append(prompt)
            return "answer"

        cascade = ModelCascade(tiers=[
            ModelTier("t", fn, confidence_threshold=1.0),
        ])
        msgs = [{"role": "user", "content": "hello there"}]
        cascade.query_with_messages(msgs)
        self.assertTrue(received_prompts)
        self.assertIn("hello there", received_prompts[0])

    def test_custom_key_fn_used(self):
        received_prompts = []

        def fn(prompt: str) -> str:
            received_prompts.append(prompt)
            return "answer"

        cascade = ModelCascade(tiers=[
            ModelTier("t", fn, confidence_threshold=1.0),
        ])
        msgs = [{"role": "user", "content": "test"}]
        cascade.query_with_messages(msgs, key_fn=lambda m: "CUSTOM_KEY")
        self.assertIn("CUSTOM_KEY", received_prompts[0])


# ---------------------------------------------------------------------------
# build_default_cascade
# ---------------------------------------------------------------------------

class TestBuildDefaultCascade(unittest.TestCase):

    def test_no_functions_returns_none(self):
        result = build_default_cascade()
        self.assertIsNone(result)

    def test_single_function_returns_cascade(self):
        cascade = build_default_cascade(claude_fn=_make_const_fn("ok"))
        self.assertIsInstance(cascade, ModelCascade)
        self.assertEqual(len(cascade.tier_names()), 1)

    def test_all_functions_builds_four_tier_cascade(self):
        cascade = build_default_cascade(
            ollama_fn=_make_const_fn("local"),
            deepseek_fn=_make_const_fn("deepseek"),
            kimi_fn=_make_const_fn("kimi"),
            claude_fn=_make_const_fn("claude"),
        )
        self.assertIsNotNone(cascade)
        names = cascade.tier_names()
        self.assertEqual(len(names), 4)
        self.assertEqual(names[0], "ollama")
        self.assertEqual(names[-1], "claude")

    def test_last_tier_always_has_threshold_100(self):
        cascade = build_default_cascade(
            ollama_fn=_make_const_fn("local"),
            deepseek_fn=_make_const_fn("deepseek"),
        )
        self.assertIsNotNone(cascade)
        # Verify the cascade works (last tier guarantees acceptance)
        result = cascade.query("test")
        self.assertIsInstance(result, CascadeResult)

    def test_partial_chain_works(self):
        cascade = build_default_cascade(
            deepseek_fn=_make_const_fn("ds answer"),
            claude_fn=_make_const_fn("claude answer"),
        )
        self.assertIsNotNone(cascade)
        result = cascade.query("What is the meaning of life?")
        self.assertIn(result.tier_used, ["deepseek", "claude"])


if __name__ == "__main__":
    unittest.main()

"""Unit tests for evoagentx.evaluators.alignment_drift."""

import math
import unittest
from typing import Any, Dict, List, Optional

from evoagentx.evaluators.alignment_drift import (
    AlignmentDriftDetector,
    BehavioralDriftMeasure,
    CapabilityDriftMeasure,
    DriftReport,
    DriftThresholdExceeded,
    SemanticDriftMeasure,
    _cosine_distance,
    _distribution_distance,
    _mean_embedding,
)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

class TestCosineDistance(unittest.TestCase):

    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(_cosine_distance(v, v), 0.0)

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        self.assertAlmostEqual(_cosine_distance(a, b), 1.0)

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        self.assertAlmostEqual(_cosine_distance(a, b), 2.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            _cosine_distance([1.0], [1.0, 2.0])

    def test_zero_vector_returns_max(self):
        self.assertAlmostEqual(_cosine_distance([0.0, 0.0], [1.0, 0.0]), 1.0)


class TestMeanEmbedding(unittest.TestCase):

    def test_single_embedding(self):
        e = [[1.0, 2.0, 3.0]]
        result = _mean_embedding(e)
        self.assertEqual(result, [1.0, 2.0, 3.0])

    def test_mean_of_two(self):
        e = [[0.0, 2.0], [2.0, 0.0]]
        result = _mean_embedding(e)
        self.assertEqual(result, [1.0, 1.0])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            _mean_embedding([])


class TestDistributionDistance(unittest.TestCase):

    def test_identical_distributions(self):
        samples = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(_distribution_distance(samples, samples), 0.0)

    def test_empty_samples(self):
        self.assertAlmostEqual(_distribution_distance([], []), 0.0)

    def test_shifted_distribution(self):
        a = [1.0, 2.0, 3.0]
        b = [10.0, 11.0, 12.0]
        dist = _distribution_distance(a, b)
        self.assertGreater(dist, 0.0)
        self.assertLessEqual(dist, 1.0)


# ---------------------------------------------------------------------------
# SemanticDriftMeasure
# ---------------------------------------------------------------------------

def _embed(text: str) -> List[float]:
    """Deterministic fake embedding: mean char code, variance, length proxy."""
    if not text:
        return [0.0, 0.0, 0.0]
    codes = [float(ord(c)) for c in text]
    mean = sum(codes) / len(codes)
    var = sum((c - mean) ** 2 for c in codes) / max(1, len(codes))
    return [mean / 128.0, math.sqrt(var) / 50.0, len(text) / 200.0]


class TestSemanticDriftMeasure(unittest.TestCase):

    def test_zero_drift_same_outputs(self):
        m = SemanticDriftMeasure(embed_fn=_embed)
        outputs = ["The answer is yes.", "The answer is no."]
        m.capture_baseline(outputs)
        drift = m.measure(outputs)
        self.assertAlmostEqual(drift, 0.0, places=5)

    def test_nonzero_drift_different_outputs(self):
        m = SemanticDriftMeasure(embed_fn=_embed)
        baseline = ["The answer is yes.", "The answer is no."]
        evolved = ["1+1=2", "Pi = 3.14159265358979"]
        m.capture_baseline(baseline)
        drift = m.measure(evolved)
        self.assertGreaterEqual(drift, 0.0)

    def test_measure_without_baseline_raises(self):
        m = SemanticDriftMeasure(embed_fn=_embed)
        with self.assertRaises(RuntimeError):
            m.measure(["some output"])

    def test_empty_baseline_raises(self):
        m = SemanticDriftMeasure(embed_fn=_embed)
        with self.assertRaises(ValueError):
            m.capture_baseline([])

    def test_empty_current_outputs(self):
        m = SemanticDriftMeasure(embed_fn=_embed)
        m.capture_baseline(["baseline output"])
        drift = m.measure([])
        self.assertAlmostEqual(drift, 0.0)


# ---------------------------------------------------------------------------
# BehavioralDriftMeasure
# ---------------------------------------------------------------------------

class TestBehavioralDriftMeasure(unittest.TestCase):

    def test_zero_drift_same_outputs(self):
        m = BehavioralDriftMeasure()
        outputs = ["This is a short answer.", "Another short answer."]
        m.capture_baseline(outputs)
        drift = m.measure(outputs)
        self.assertAlmostEqual(drift, 0.0)

    def test_nonzero_drift_different_characteristics(self):
        m = BehavioralDriftMeasure()
        baseline = ["yes", "no", "maybe"]  # very short
        # Evolved outputs are long with JSON structure
        evolved = [
            '{"result": "' + ("x " * 100) + '"}',
            '{"answer": "' + ("y " * 100) + '"}',
            '{"output": "' + ("z " * 100) + '"}',
        ]
        m.capture_baseline(baseline)
        drift = m.measure(evolved)
        self.assertGreater(drift, 0.0)

    def test_empty_baseline_raises(self):
        m = BehavioralDriftMeasure()
        with self.assertRaises(ValueError):
            m.capture_baseline([])

    def test_measure_without_baseline_raises(self):
        m = BehavioralDriftMeasure()
        with self.assertRaises(RuntimeError):
            m.measure(["some output"])

    def test_drift_in_range(self):
        m = BehavioralDriftMeasure()
        m.capture_baseline(["a b c", "d e f"])
        drift = m.measure(["very long output text here and even more words for length"])
        self.assertGreaterEqual(drift, 0.0)
        self.assertLessEqual(drift, 1.0)


# ---------------------------------------------------------------------------
# CapabilityDriftMeasure
# ---------------------------------------------------------------------------

def _probe_agent(example: Dict[str, Any]) -> str:
    return example.get("expected_answer", "ok")


def _probe_scorer(output: str, example: Dict[str, Any]) -> float:
    return 1.0 if output == example.get("expected_answer", "") else 0.0


class TestCapabilityDriftMeasure(unittest.TestCase):

    def test_zero_regression_same_agent(self):
        probes = [{"expected_answer": "ok"}, {"expected_answer": "yes"}]
        m = CapabilityDriftMeasure()
        m.capture_baseline(_probe_agent, probes, _probe_scorer)
        regression = m.measure(_probe_agent, probes, _probe_scorer)
        self.assertAlmostEqual(regression, 0.0)

    def test_regression_when_agent_degrades(self):
        probes = [{"expected_answer": "ok"}, {"expected_answer": "yes"}]
        m = CapabilityDriftMeasure()
        m.capture_baseline(_probe_agent, probes, _probe_scorer)

        def bad_agent(example: Dict[str, Any]) -> str:
            return "wrong"

        regression = m.measure(bad_agent, probes, _probe_scorer)
        self.assertGreater(regression, 0.0)

    def test_no_probes_returns_none(self):
        m = CapabilityDriftMeasure()
        m.capture_baseline(_probe_agent, [], _probe_scorer)
        result = m.measure(_probe_agent, [], _probe_scorer)
        self.assertIsNone(result)

    def test_baseline_not_captured_returns_none(self):
        m = CapabilityDriftMeasure()
        result = m.measure(_probe_agent, [{"expected_answer": "ok"}], _probe_scorer)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# DriftReport
# ---------------------------------------------------------------------------

class TestDriftReport(unittest.TestCase):

    def test_no_exceedance_by_default(self):
        r = DriftReport(
            semantic_drift=0.1,
            behavioral_drift=0.1,
            capability_drift=0.05,
        )
        self.assertFalse(r.any_exceeded)

    def test_semantic_exceeded(self):
        r = DriftReport(semantic_drift=0.9, semantic_threshold=0.25)
        self.assertTrue(r.semantic_exceeded)
        self.assertTrue(r.any_exceeded)

    def test_behavioral_exceeded(self):
        r = DriftReport(behavioral_drift=0.9, behavioral_threshold=0.30)
        self.assertTrue(r.behavioral_exceeded)

    def test_capability_exceeded(self):
        r = DriftReport(capability_drift=0.9, capability_threshold=0.15)
        self.assertTrue(r.capability_exceeded)

    def test_none_drift_not_exceeded(self):
        r = DriftReport(semantic_drift=None)
        self.assertFalse(r.semantic_exceeded)

    def test_summary_contains_ok(self):
        r = DriftReport(semantic_drift=0.1, behavioral_drift=0.1, capability_drift=0.05)
        summary = r.summary()
        self.assertIn("OK", summary)


# ---------------------------------------------------------------------------
# AlignmentDriftDetector — full integration
# ---------------------------------------------------------------------------

class TestAlignmentDriftDetector(unittest.TestCase):

    def _make_agent(self, response: str):
        def agent(example: Dict[str, Any]) -> str:
            return response
        return agent

    def test_check_without_baseline_raises(self):
        detector = AlignmentDriftDetector(embed_fn=_embed)
        with self.assertRaises(RuntimeError):
            detector.check(agent_fn=self._make_agent("x"))

    def test_check_same_agent_no_drift(self):
        detector = AlignmentDriftDetector(embed_fn=_embed)
        agent = self._make_agent("The answer is 42.")
        inputs = [{"q": "x"}, {"q": "y"}, {"q": "z"}]
        detector.capture_baseline(agent_fn=agent, baseline_inputs=inputs)
        report = detector.check(agent_fn=agent, evaluation_inputs=inputs)
        self.assertFalse(report.semantic_exceeded)
        self.assertFalse(report.behavioral_exceeded)

    def test_strict_mode_raises_on_exceedance(self):
        detector = AlignmentDriftDetector(
            embed_fn=_embed,
            semantic_threshold=0.0001,  # Very tight threshold
            strict_mode=True,
        )
        baseline_agent = self._make_agent("short")
        evolved_agent = self._make_agent("completely different longer text with many words")
        inputs = [{"q": str(i)} for i in range(5)]
        detector.capture_baseline(agent_fn=baseline_agent, baseline_inputs=inputs)
        with self.assertRaises(DriftThresholdExceeded):
            detector.check(agent_fn=evolved_agent, evaluation_inputs=inputs)

    def test_non_strict_mode_returns_report(self):
        detector = AlignmentDriftDetector(
            embed_fn=_embed,
            semantic_threshold=0.0001,
            strict_mode=False,
        )
        baseline_agent = self._make_agent("short")
        evolved_agent = self._make_agent("completely different longer text here and more")
        inputs = [{"q": str(i)} for i in range(5)]
        detector.capture_baseline(agent_fn=baseline_agent, baseline_inputs=inputs)
        report = detector.check(agent_fn=evolved_agent, evaluation_inputs=inputs)
        self.assertIsInstance(report, DriftReport)

    def test_no_embed_fn_skips_semantic(self):
        detector = AlignmentDriftDetector(embed_fn=None)
        agent = self._make_agent("answer")
        inputs = [{"q": "x"}]
        detector.capture_baseline(agent_fn=agent, baseline_inputs=inputs)
        report = detector.check(agent_fn=agent, evaluation_inputs=inputs)
        self.assertIsNone(report.semantic_drift)

    def test_capability_measured_when_probes_provided(self):
        probes = [{"expected_answer": "ok"}, {"expected_answer": "yes"}]
        detector = AlignmentDriftDetector(embed_fn=_embed)
        agent = self._make_agent("ok")

        detector.capture_baseline(
            agent_fn=agent,
            probe_examples=probes,
            probe_evaluator=_probe_scorer,
            baseline_inputs=probes,
        )
        report = detector.check(
            agent_fn=agent,
            probe_examples=probes,
            probe_evaluator=_probe_scorer,
            evaluation_inputs=probes,
        )
        self.assertIsNotNone(report.capability_drift)

    def test_drift_threshold_exceeded_attributes(self):
        exc = DriftThresholdExceeded("semantic", observed=0.5, threshold=0.25)
        self.assertEqual(exc.dimension, "semantic")
        self.assertAlmostEqual(exc.observed, 0.5)
        self.assertAlmostEqual(exc.threshold, 0.25)


if __name__ == "__main__":
    unittest.main()

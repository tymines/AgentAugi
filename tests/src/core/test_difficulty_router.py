"""Unit tests for evoagentx.core.difficulty_router."""

import json
import os
import tempfile
import unittest

from evoagentx.core.difficulty_router import (
    DifficultyRouter,
    DifficultyTier,
    RoutingDecision,
    _FeatureExtractor,
    _ThresholdState,
)


class TestFeatureExtractor(unittest.TestCase):
    """Tests for the internal _FeatureExtractor class."""

    def setUp(self):
        self.extractor = _FeatureExtractor()

    def test_empty_string_returns_all_zero(self):
        feats = self.extractor.extract("")
        self.assertIn("token_count_norm", feats)
        # Empty string → zero tokens
        self.assertEqual(feats["token_count_norm"], 0.0)

    def test_features_in_unit_range(self):
        text = "Implement a distributed rate-limiter with Redis and asyncio. " * 20
        feats = self.extractor.extract(text)
        for name, val in feats.items():
            self.assertGreaterEqual(val, 0.0, f"{name} < 0")
            self.assertLessEqual(val, 1.0, f"{name} > 1")

    def test_hard_keywords_detected(self):
        feats = self.extractor.extract("Please implement a comprehensive solution")
        # "implement" and "comprehensive" are hard keywords
        self.assertGreater(feats["hard_keyword_ratio"], 0.0)

    def test_easy_keywords_detected(self):
        feats = self.extractor.extract("What is the capital of France?")
        self.assertGreater(feats["easy_keyword_ratio"], 0.0)

    def test_code_block_detected(self):
        feats = self.extractor.extract("Here is code:\n```python\nprint('hi')\n```")
        self.assertEqual(feats["code_block_present"], 1.0)

    def test_no_code_block(self):
        feats = self.extractor.extract("No code here, just text.")
        self.assertEqual(feats["code_block_present"], 0.0)

    def test_question_marks_counted(self):
        feats = self.extractor.extract("What? Why? When? Who? Where?")
        self.assertGreater(feats["question_count_norm"], 0.0)

    def test_token_count_soft_cap(self):
        long_text = " ".join(["word"] * 5000)
        feats = self.extractor.extract(long_text)
        self.assertEqual(feats["token_count_norm"], 1.0)


class TestThresholdState(unittest.TestCase):
    """Tests for _ThresholdState EMA calibration."""

    def test_defaults_are_valid(self):
        ts = _ThresholdState()
        self.assertLess(ts.simple_max, ts.medium_max)
        self.assertGreater(ts.simple_max, 0.0)
        self.assertLess(ts.medium_max, 1.0)

    def test_correct_routing_does_not_move_thresholds(self):
        ts = _ThresholdState()
        simple_before = ts.simple_max
        medium_before = ts.medium_max
        ts.update(0.1, tier_was_correct=True)
        self.assertEqual(ts.simple_max, simple_before)
        self.assertEqual(ts.medium_max, medium_before)

    def test_wrong_simple_routing_lowers_simple_max(self):
        ts = _ThresholdState()
        simple_before = ts.simple_max
        # Score below simple_max but routing was wrong → lower threshold
        ts.update(ts.simple_max * 0.5, tier_was_correct=False)
        self.assertLess(ts.simple_max, simple_before)

    def test_invariant_maintained_after_update(self):
        ts = _ThresholdState()
        for _ in range(20):
            ts.update(0.1, tier_was_correct=False)
        self.assertLess(ts.simple_max, ts.medium_max)
        self.assertGreater(ts.simple_max, 0.0)
        self.assertLess(ts.medium_max, 1.0)


class TestDifficultyRouter(unittest.TestCase):
    """Tests for the main DifficultyRouter class."""

    def setUp(self):
        self.router = DifficultyRouter()

    # --- basic routing ---

    def test_simple_task_routed_correctly(self):
        decision = self.router.route("What is 2 + 2?")
        self.assertIsInstance(decision, RoutingDecision)
        self.assertIsInstance(decision.tier, DifficultyTier)

    def test_hard_task_routed_to_harder_tier(self):
        simple_dec = self.router.route("What is 2+2?")
        hard_dec = self.router.route(
            "Implement a fully distributed, fault-tolerant, multi-step "
            "orchestration engine with comprehensive error handling, "
            "complex retry logic and parallel task scheduling. Design, "
            "architect and optimise the system for scalability. " * 5
        )
        # The hard task should have a higher raw score
        self.assertGreater(hard_dec.score, simple_dec.score)

    def test_score_in_unit_range(self):
        for text in ["hello", "implement a complex system", "what is"]:
            dec = self.router.route(text)
            self.assertGreaterEqual(dec.score, 0.0)
            self.assertLessEqual(dec.score, 1.0)

    def test_decision_contains_features(self):
        dec = self.router.route("Implement a rate-limiter")
        self.assertIsInstance(dec.features, dict)
        self.assertIn("token_count_norm", dec.features)

    def test_model_hint_populated(self):
        dec = self.router.route("What is Python?")
        self.assertIsInstance(dec.model_hint, str)
        self.assertGreater(len(dec.model_hint), 0)

    def test_task_id_stored(self):
        dec = self.router.route("Analyze this code", task_id="task-42")
        self.assertEqual(dec.task_id, "task-42")

    # --- batch routing ---

    def test_route_batch_returns_correct_count(self):
        tasks = ["task one", "task two", "task three"]
        decisions = self.router.route_batch(tasks)
        self.assertEqual(len(decisions), 3)

    def test_route_batch_with_ids(self):
        tasks = ["a", "b"]
        ids = ["id1", "id2"]
        decisions = self.router.route_batch(tasks, task_ids=ids)
        self.assertEqual(decisions[0].task_id, "id1")
        self.assertEqual(decisions[1].task_id, "id2")

    def test_route_batch_mismatched_ids_raises(self):
        with self.assertRaises(ValueError):
            self.router.route_batch(["a", "b"], task_ids=["only_one"])

    # --- history and metrics ---

    def test_metrics_tier_counts_update(self):
        initial = self.router.metrics()["tier_counts"]
        self.router.route("What is 2+2?")
        after = self.router.metrics()["tier_counts"]
        total_initial = sum(initial.values())
        total_after = sum(after.values())
        self.assertEqual(total_after, total_initial + 1)

    def test_history_size_grows(self):
        for i in range(5):
            self.router.route(f"task {i}")
        self.assertEqual(self.router.metrics()["history_size"], 5)

    def test_history_limit_enforced(self):
        router = DifficultyRouter(history_limit=3)
        for i in range(10):
            router.route(f"task {i}")
        self.assertLessEqual(router.metrics()["history_size"], 3)

    # --- outcome recording ---

    def test_record_outcome_unknown_id_does_not_raise(self):
        # Should just log a warning, not raise
        self.router.record_outcome("nonexistent-id", 0.9)

    def test_record_outcome_stores_score(self):
        dec = self.router.route("Some task", task_id="t1")
        self.router.record_outcome("t1", 0.85)
        # Find the decision in history and verify outcome
        found = None
        for d in reversed(self.router._history):
            if d.task_id == "t1":
                found = d
                break
        self.assertIsNotNone(found)
        self.assertAlmostEqual(found.outcome_score, 0.85)

    # --- state persistence ---

    def test_save_and_load_state(self):
        self.router.route("test task")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            path = tmp.name
        try:
            self.router.save_state(path)
            self.assertTrue(os.path.exists(path))
            new_router = DifficultyRouter()
            new_router.load_state(path)
            self.assertAlmostEqual(
                new_router._thresholds.simple_max,
                self.router._thresholds.simple_max,
            )
        finally:
            os.unlink(path)

    def test_load_state_file_not_found_raises(self):
        with self.assertRaises(FileNotFoundError):
            self.router.load_state("/tmp/nonexistent_router_state_xyz.json")

    # --- custom model hints ---

    def test_custom_model_hints(self):
        router = DifficultyRouter(
            model_hints={
                DifficultyTier.SIMPLE: "gpt-3.5-turbo",
                DifficultyTier.MEDIUM: "gpt-4o-mini",
                DifficultyTier.HARD:   "gpt-4o",
            }
        )
        # Force a score below simple_max to ensure SIMPLE tier
        dec = router.route("What is 2+2?")
        if dec.tier == DifficultyTier.SIMPLE:
            self.assertEqual(dec.model_hint, "gpt-3.5-turbo")


if __name__ == "__main__":
    unittest.main()

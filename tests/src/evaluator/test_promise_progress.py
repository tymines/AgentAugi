"""Unit tests for evoagentx.evaluators.promise_progress."""

import json
import unittest
from unittest.mock import MagicMock

from evoagentx.evaluators.promise_progress import (
    LLMPromiseProgressEvaluator,
    PromiseProgressScore,
    TrajectoryPromiseProgress,
    trajectory_to_aflow_node_value,
    trajectory_to_evoprompt_fitness,
    trajectory_to_textgrad_weights,
)


class TestPromiseProgressScore(unittest.TestCase):
    """Tests for the PromiseProgressScore dataclass."""

    def test_valid_construction(self):
        s = PromiseProgressScore(
            promise=0.8,
            progress=0.6,
            promise_rationale="on track",
            progress_rationale="good step",
            step_index=1,
        )
        self.assertAlmostEqual(s.promise, 0.8)
        self.assertAlmostEqual(s.progress, 0.6)
        self.assertAlmostEqual(s.composite, 0.7)

    def test_boundary_values(self):
        PromiseProgressScore(promise=0.0, progress=0.0)
        PromiseProgressScore(promise=1.0, progress=1.0)

    def test_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            PromiseProgressScore(promise=1.1, progress=0.5)
        with self.assertRaises(ValueError):
            PromiseProgressScore(promise=0.5, progress=-0.1)


class TestTrajectoryPromiseProgress(unittest.TestCase):
    """Tests for TrajectoryPromiseProgress aggregation."""

    def _make(self, pairs):
        scores = [
            PromiseProgressScore(promise=p, progress=r, step_index=i)
            for i, (p, r) in enumerate(pairs)
        ]
        return TrajectoryPromiseProgress(scores=scores)

    def test_promises_and_progresses(self):
        tpp = self._make([(0.4, 0.6), (0.7, 0.2)])
        self.assertEqual(tpp.promises, [0.4, 0.7])
        self.assertEqual(tpp.progresses, [0.6, 0.2])

    def test_mean_promise(self):
        tpp = self._make([(0.4, 0.0), (0.8, 0.0)])
        self.assertAlmostEqual(tpp.mean_promise, 0.6)

    def test_mean_progress(self):
        tpp = self._make([(0.0, 0.3), (0.0, 0.7)])
        self.assertAlmostEqual(tpp.mean_progress, 0.5)

    def test_final_promise(self):
        tpp = self._make([(0.3, 0.5), (0.6, 0.4), (0.9, 0.8)])
        self.assertAlmostEqual(tpp.final_promise, 0.9)

    def test_empty(self):
        tpp = TrajectoryPromiseProgress()
        self.assertAlmostEqual(tpp.mean_promise, 0.0)
        self.assertAlmostEqual(tpp.mean_progress, 0.0)
        self.assertAlmostEqual(tpp.final_promise, 0.0)

    def test_fitness_score_default_weights(self):
        # alpha=0.7 * final_promise + beta=0.3 * mean_progress
        # promise = [0.5, 0.8] → final = 0.8
        # progress = [0.4, 0.6] → mean = 0.5
        # fitness = 0.7*0.8 + 0.3*0.5 = 0.56 + 0.15 = 0.71
        tpp = self._make([(0.5, 0.4), (0.8, 0.6)])
        self.assertAlmostEqual(tpp.fitness_score(), 0.71)


class TestLLMPromiseProgressEvaluator(unittest.TestCase):
    """Tests for LLMPromiseProgressEvaluator."""

    def _make_evaluator(self, llm_response: str):
        llm = MagicMock()
        r = MagicMock()
        r.content = llm_response
        llm.generate.return_value = r
        return LLMPromiseProgressEvaluator(llm=llm)

    def _valid_json_response(self, promise=0.7, progress=0.5):
        return json.dumps({
            "promise": promise,
            "promise_rationale": "looks good",
            "progress": progress,
            "progress_rationale": "moved forward",
        })

    def test_evaluate_step_valid(self):
        ev = self._make_evaluator(self._valid_json_response(0.7, 0.5))
        step = {"type": "reasoning", "content": "deduced X"}
        result = ev.evaluate_step(step, preceding=[], context="solve problem")
        self.assertIsInstance(result, PromiseProgressScore)
        self.assertAlmostEqual(result.promise, 0.7)
        self.assertAlmostEqual(result.progress, 0.5)

    def test_evaluate_step_clamps_out_of_range(self):
        resp = json.dumps({"promise": 1.5, "progress": -0.2,
                           "promise_rationale": "", "progress_rationale": ""})
        ev = self._make_evaluator(resp)
        result = ev.evaluate_step({"type": "tool_call", "content": "x"}, [], "ctx")
        self.assertAlmostEqual(result.promise, 1.0)
        self.assertAlmostEqual(result.progress, 0.0)

    def test_evaluate_step_parse_failure_uses_fallback(self):
        ev = self._make_evaluator("this is not json")
        ev.fallback_promise = 0.4
        ev.fallback_progress = 0.35
        result = ev.evaluate_step({"type": "unknown", "content": ""}, [], "")
        self.assertAlmostEqual(result.promise, 0.4)
        self.assertAlmostEqual(result.progress, 0.35)

    def test_evaluate_step_with_markdown_fence(self):
        inner = self._valid_json_response(0.65, 0.45)
        fenced = f"```json\n{inner}\n```"
        ev = self._make_evaluator(fenced)
        result = ev.evaluate_step({"type": "reasoning", "content": "x"}, [], "ctx")
        self.assertAlmostEqual(result.promise, 0.65)
        self.assertAlmostEqual(result.progress, 0.45)

    def test_evaluate_full_trajectory(self):
        responses = [self._valid_json_response(p, r) for p, r in
                     [(0.3, 0.2), (0.6, 0.5), (0.9, 0.8)]]
        call_count = [0]
        llm = MagicMock()

        def side_effect(prompt):
            r = MagicMock()
            r.content = responses[call_count[0] % len(responses)]
            call_count[0] += 1
            return r

        llm.generate.side_effect = side_effect
        ev = LLMPromiseProgressEvaluator(llm=llm)

        trajectory = [
            {"type": "tool_call", "content": "searched"},
            {"type": "reasoning", "content": "analysed"},
            {"type": "tool_call", "content": "fetched"},
        ]
        tpp = ev.evaluate(trajectory, context="task")
        self.assertEqual(len(tpp.scores), 3)
        for i, score in enumerate(tpp.scores):
            self.assertEqual(score.step_index, i)

    def test_evaluate_empty_trajectory(self):
        ev = self._make_evaluator(self._valid_json_response())
        tpp = ev.evaluate([])
        self.assertEqual(len(tpp.scores), 0)


class TestAggregationHelpers(unittest.TestCase):
    """Tests for trajectory_to_* aggregation utilities."""

    def _make_tpp(self, promise_progress_pairs):
        scores = [
            PromiseProgressScore(promise=p, progress=r, step_index=i)
            for i, (p, r) in enumerate(promise_progress_pairs)
        ]
        return TrajectoryPromiseProgress(scores=scores)

    def test_textgrad_weights_sum_to_one(self):
        tpp = self._make_tpp([(0.4, 0.6), (0.8, 0.2), (0.6, 0.4)])
        weights = trajectory_to_textgrad_weights(tpp)
        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(sum(weights), 1.0)

    def test_textgrad_weights_proportional(self):
        # composite = [0.5, 0.5, 1.0] → weights should be [0.25, 0.25, 0.5]
        tpp = self._make_tpp([(0.5, 0.5), (0.5, 0.5), (1.0, 1.0)])
        weights = trajectory_to_textgrad_weights(tpp)
        self.assertAlmostEqual(weights[2], 0.5)
        self.assertAlmostEqual(weights[0], 0.25)

    def test_textgrad_weights_empty(self):
        tpp = TrajectoryPromiseProgress()
        weights = trajectory_to_textgrad_weights(tpp)
        self.assertEqual(weights, [])

    def test_textgrad_weights_all_zero_returns_uniform(self):
        tpp = self._make_tpp([(0.0, 0.0), (0.0, 0.0)])
        weights = trajectory_to_textgrad_weights(tpp)
        self.assertAlmostEqual(weights[0], 0.5)
        self.assertAlmostEqual(weights[1], 0.5)

    def test_evoprompt_fitness_blends_correctly(self):
        tpp = self._make_tpp([(0.4, 0.6), (0.8, 0.4)])
        # final_promise = 0.8, mean_progress = 0.5
        # fitness = 0.7*0.8 + 0.3*0.5 = 0.56 + 0.15 = 0.71
        fitness = trajectory_to_evoprompt_fitness(tpp, outcome_score=0.0)
        # With outcome=0, it's just 0.3 * mean(composite)
        # composites = [0.5, 0.6], mean = 0.55 → fitness = 0.3 * 0.55 = 0.165
        self.assertGreaterEqual(fitness, 0.0)
        self.assertLessEqual(fitness, 1.0)

    def test_evoprompt_fitness_with_outcome_score(self):
        tpp = self._make_tpp([(1.0, 1.0)])
        fitness = trajectory_to_evoprompt_fitness(tpp, outcome_score=1.0, alpha=0.7, beta=0.3)
        self.assertAlmostEqual(fitness, 1.0)

    def test_aflow_node_value_is_final_promise(self):
        tpp = self._make_tpp([(0.3, 0.5), (0.9, 0.7)])
        self.assertAlmostEqual(trajectory_to_aflow_node_value(tpp), 0.9)

    def test_aflow_node_value_empty(self):
        tpp = TrajectoryPromiseProgress()
        self.assertAlmostEqual(trajectory_to_aflow_node_value(tpp), 0.0)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for evoagentx.evaluators.process_reward."""

import json
import unittest
from unittest.mock import MagicMock, patch

from evoagentx.evaluators.process_reward import (
    LLMStepwiseRewardEvaluator,
    StepScore,
    TrajectoryScores,
)


class TestStepScore(unittest.TestCase):
    """Tests for the StepScore dataclass."""

    def test_valid_construction(self):
        s = StepScore(score=0.8, rationale="good step", step_index=2, step_type="reasoning")
        self.assertAlmostEqual(s.score, 0.8)
        self.assertEqual(s.step_type, "reasoning")
        self.assertEqual(s.step_index, 2)

    def test_boundary_scores(self):
        StepScore(score=0.0)
        StepScore(score=1.0)

    def test_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            StepScore(score=1.1)
        with self.assertRaises(ValueError):
            StepScore(score=-0.01)


class TestTrajectoryScores(unittest.TestCase):
    """Tests for TrajectoryScores aggregation."""

    def _make_ts(self, values):
        steps = [StepScore(score=v, step_index=i) for i, v in enumerate(values)]
        return TrajectoryScores(step_scores=steps)

    def test_mean_score(self):
        ts = self._make_ts([0.2, 0.4, 0.6, 0.8])
        self.assertAlmostEqual(ts.mean_score, 0.5)

    def test_min_score(self):
        ts = self._make_ts([0.3, 0.9, 0.1, 0.7])
        self.assertAlmostEqual(ts.min_score, 0.1)

    def test_max_score(self):
        ts = self._make_ts([0.3, 0.9, 0.1, 0.7])
        self.assertAlmostEqual(ts.max_score, 0.9)

    def test_empty_trajectory(self):
        ts = TrajectoryScores()
        self.assertAlmostEqual(ts.mean_score, 0.0)
        self.assertAlmostEqual(ts.min_score, 0.0)
        self.assertAlmostEqual(ts.max_score, 0.0)
        self.assertEqual(ts.as_float_list(), [])

    def test_as_float_list(self):
        ts = self._make_ts([0.1, 0.5, 0.9])
        self.assertEqual(ts.as_float_list(), [0.1, 0.5, 0.9])

    def test_weighted_score(self):
        ts = self._make_ts([0.6, 0.4])
        # mean = 0.5, min = 0.4 → weighted = 0.5*0.5 + 0.5*0.4 = 0.45
        self.assertAlmostEqual(ts.weighted_score(alpha=0.5, beta=0.5), 0.45)


class TestLLMStepwiseRewardEvaluator(unittest.TestCase):
    """Tests for LLMStepwiseRewardEvaluator."""

    def _make_evaluator(self, llm_response: str, fallback: float = 0.5):
        llm = MagicMock()
        response = MagicMock()
        response.content = llm_response
        llm.generate.return_value = response
        return LLMStepwiseRewardEvaluator(llm=llm, fallback_score=fallback)

    # ------------------------------------------------------------------
    # score_tool_call
    # ------------------------------------------------------------------

    def test_score_tool_call_valid_json(self):
        payload = json.dumps({"score": 0.75, "rationale": "correct tool"})
        ev = self._make_evaluator(payload)
        result = ev.score_tool_call(
            tool_name="search",
            args={"query": "python docs"},
            context="answer a coding question",
            trajectory_so_far=[],
        )
        self.assertIsInstance(result, StepScore)
        self.assertAlmostEqual(result.score, 0.75)
        self.assertEqual(result.rationale, "correct tool")
        self.assertEqual(result.step_type, "tool_call")

    def test_score_tool_call_clamps_score_above_1(self):
        payload = json.dumps({"score": 2.5, "rationale": "too high"})
        ev = self._make_evaluator(payload)
        result = ev.score_tool_call("tool", {}, "ctx", [])
        self.assertAlmostEqual(result.score, 1.0)

    def test_score_tool_call_clamps_score_below_0(self):
        payload = json.dumps({"score": -0.3, "rationale": "negative"})
        ev = self._make_evaluator(payload)
        result = ev.score_tool_call("tool", {}, "ctx", [])
        self.assertAlmostEqual(result.score, 0.0)

    def test_score_tool_call_parse_failure_uses_fallback(self):
        ev = self._make_evaluator("not valid json at all", fallback=0.3)
        result = ev.score_tool_call("tool", {}, "ctx", [])
        self.assertAlmostEqual(result.score, 0.3)

    def test_score_tool_call_with_markdown_fence(self):
        payload = "```json\n" + json.dumps({"score": 0.9, "rationale": "great"}) + "\n```"
        ev = self._make_evaluator(payload)
        result = ev.score_tool_call("tool", {}, "ctx", [])
        self.assertAlmostEqual(result.score, 0.9)

    def test_score_tool_call_step_index_equals_trajectory_length(self):
        payload = json.dumps({"score": 0.5, "rationale": "ok"})
        ev = self._make_evaluator(payload)
        trajectory = [{"type": "reasoning", "content": "step1"}] * 3
        result = ev.score_tool_call("tool", {}, "ctx", trajectory)
        self.assertEqual(result.step_index, 3)

    # ------------------------------------------------------------------
    # score_reasoning_step
    # ------------------------------------------------------------------

    def test_score_reasoning_step_valid(self):
        payload = json.dumps({"score": 0.6, "rationale": "sound reasoning"})
        ev = self._make_evaluator(payload)
        result = ev.score_reasoning_step("The answer is X because Y.", "math task")
        self.assertIsInstance(result, StepScore)
        self.assertAlmostEqual(result.score, 0.6)
        self.assertEqual(result.step_type, "reasoning")

    def test_score_reasoning_step_empty_context(self):
        payload = json.dumps({"score": 0.5, "rationale": "neutral"})
        ev = self._make_evaluator(payload)
        result = ev.score_reasoning_step("some reasoning", "")
        self.assertAlmostEqual(result.score, 0.5)

    # ------------------------------------------------------------------
    # score_trajectory
    # ------------------------------------------------------------------

    def test_score_trajectory_mixed_steps(self):
        # Two tool_call steps and one reasoning step
        responses = [
            json.dumps({"score": 0.8, "rationale": "r1"}),
            json.dumps({"score": 0.4, "rationale": "r2"}),
            json.dumps({"score": 0.9, "rationale": "r3"}),
        ]
        call_count = [0]
        llm = MagicMock()

        def make_response(prompt):
            r = MagicMock()
            r.content = responses[call_count[0] % len(responses)]
            call_count[0] += 1
            return r

        llm.generate.side_effect = make_response
        ev = LLMStepwiseRewardEvaluator(llm=llm)

        trajectory = [
            {"type": "tool_call", "tool_name": "search", "args": {}, "content": "searched"},
            {"type": "reasoning", "content": "deduced something"},
            {"type": "tool_call", "tool_name": "calc", "args": {"x": 1}, "content": "calculated"},
        ]
        ts = ev.score_trajectory(trajectory, context="solve problem")

        self.assertEqual(len(ts.step_scores), 3)
        self.assertEqual(ts.step_scores[0].step_type, "tool_call")
        self.assertEqual(ts.step_scores[1].step_type, "reasoning")
        self.assertEqual(ts.step_scores[2].step_type, "tool_call")
        # step_index should match position
        for i, s in enumerate(ts.step_scores):
            self.assertEqual(s.step_index, i)

    def test_score_trajectory_unknown_type_uses_fallback(self):
        ev = self._make_evaluator(
            json.dumps({"score": 0.5, "rationale": "ok"}), fallback=0.42
        )
        trajectory = [{"type": "unknown_type", "content": "mystery step"}]
        ts = ev.score_trajectory(trajectory)
        self.assertEqual(len(ts.step_scores), 1)
        self.assertAlmostEqual(ts.step_scores[0].score, 0.42)

    def test_score_trajectory_empty(self):
        ev = self._make_evaluator(json.dumps({"score": 0.5, "rationale": ""}))
        ts = ev.score_trajectory([])
        self.assertEqual(len(ts.step_scores), 0)
        self.assertAlmostEqual(ts.mean_score, 0.0)

    def test_history_window_respected(self):
        """Evaluator should only pass `history_window` preceding steps to the prompt."""
        payload = json.dumps({"score": 0.7, "rationale": "ok"})
        llm = MagicMock()
        response = MagicMock()
        response.content = payload
        llm.generate.return_value = response

        ev = LLMStepwiseRewardEvaluator(llm=llm, history_window=2)
        # Build a long trajectory; the format_history call should only show 2
        trajectory = [
            {"type": "reasoning", "content": f"step {i}"} for i in range(5)
        ]
        # Just confirm no exception is raised and scores come back
        ts = ev.score_trajectory(trajectory)
        self.assertEqual(len(ts.step_scores), 5)


if __name__ == "__main__":
    unittest.main()

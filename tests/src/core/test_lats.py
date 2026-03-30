"""Unit tests for evoagentx.core.lats (LATS runtime planning)."""

import json
import math
import unittest
from unittest.mock import MagicMock, call, patch

import sys
import os

# Ensure repo root is on path when running tests directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from evoagentx.core.lats import (
    LATS,
    LATSConfig,
    LATSNode,
    LATSResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_llm(responses: list) -> MagicMock:
    """Return a MagicMock LLM that cycles through ``responses``."""
    counter = [0]

    def side_effect(prompt):
        resp = MagicMock()
        resp.content = responses[counter[0] % len(responses)]
        counter[0] += 1
        return resp

    llm = MagicMock()
    llm.generate.side_effect = side_effect
    return llm


_ACTIONS_JSON = json.dumps(["action A", "action B", "action C"])
_VALUE_JSON = json.dumps({"value": 0.7})


# ---------------------------------------------------------------------------
# LATSNode tests
# ---------------------------------------------------------------------------

class TestLATSNode(unittest.TestCase):
    """Tests for LATSNode UCB1, mean_value, and action_sequence."""

    def test_mean_value_before_any_visit(self):
        node = LATSNode(state="", depth=0)
        self.assertAlmostEqual(node.mean_value, 0.5)

    def test_mean_value_after_visits(self):
        node = LATSNode(state="")
        node.visit_count = 4
        node.total_value = 3.2
        self.assertAlmostEqual(node.mean_value, 0.8)

    def test_ucb1_unvisited_returns_inf(self):
        node = LATSNode(state="")
        # An unvisited node should return +inf so it is always selected first.
        score = node.ucb1(parent_visits=10, exploration_constant=math.sqrt(2))
        self.assertEqual(score, float("inf"))

    def test_ucb1_structure(self):
        """UCB1 = mean + c * sqrt(ln(parent_n) / n)."""
        node = LATSNode(state="")
        node.visit_count = 4
        node.total_value = 2.0
        c = math.sqrt(2)
        expected = 0.5 + c * math.sqrt(math.log(16) / 4)
        self.assertAlmostEqual(node.ucb1(parent_visits=16, exploration_constant=c), expected, places=6)

    def test_is_leaf_no_children(self):
        node = LATSNode(state="")
        self.assertTrue(node.is_leaf())

    def test_is_leaf_with_children(self):
        parent = LATSNode(state="")
        child = LATSNode(state="x", parent=parent)
        parent.children.append(child)
        self.assertFalse(parent.is_leaf())

    def test_action_sequence_root(self):
        root = LATSNode(state="", action="")
        self.assertEqual(root.action_sequence(), [])

    def test_action_sequence_depth_two(self):
        root = LATSNode(state="", action="")
        child = LATSNode(state="s1", action="step1", parent=root, depth=1)
        grandchild = LATSNode(state="s2", action="step2", parent=child, depth=2)
        self.assertEqual(grandchild.action_sequence(), ["step1", "step2"])


# ---------------------------------------------------------------------------
# LATSConfig tests
# ---------------------------------------------------------------------------

class TestLATSConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = LATSConfig()
        self.assertEqual(cfg.max_depth, 5)
        self.assertEqual(cfg.max_width, 3)
        self.assertEqual(cfg.n_simulations, 20)
        self.assertAlmostEqual(cfg.exploration_constant, math.sqrt(2))
        self.assertEqual(cfg.simulation_depth, 3)
        self.assertAlmostEqual(cfg.value_threshold, 0.9)

    def test_custom_values(self):
        cfg = LATSConfig(max_depth=2, n_simulations=5, max_width=2)
        self.assertEqual(cfg.max_depth, 2)
        self.assertEqual(cfg.n_simulations, 5)
        self.assertEqual(cfg.max_width, 2)


# ---------------------------------------------------------------------------
# LATS._parse_actions tests
# ---------------------------------------------------------------------------

class TestLATSParseActions(unittest.TestCase):

    def setUp(self):
        self.lats = LATS(llm=MagicMock(), config=LATSConfig())

    def test_valid_json_array(self):
        raw = '["act1", "act2", "act3"]'
        result = self.lats._parse_actions(raw)
        self.assertEqual(result, ["act1", "act2", "act3"])

    def test_json_with_markdown_fence(self):
        raw = '```json\n["a", "b"]\n```'
        result = self.lats._parse_actions(raw)
        self.assertEqual(result, ["a", "b"])

    def test_fallback_on_invalid_json(self):
        raw = "do this thing"
        result = self.lats._parse_actions(raw)
        # Falls back to raw string as single action
        self.assertEqual(result, ["do this thing"])

    def test_empty_response(self):
        result = self.lats._parse_actions("")
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# LATS._parse_value tests
# ---------------------------------------------------------------------------

class TestLATSParseValue(unittest.TestCase):

    def setUp(self):
        self.lats = LATS(llm=MagicMock(), config=LATSConfig())

    def test_json_object(self):
        raw = '{"value": 0.82}'
        self.assertAlmostEqual(self.lats._parse_value(raw), 0.82)

    def test_bare_float(self):
        self.assertAlmostEqual(self.lats._parse_value("0.6"), 0.6)

    def test_clamps_above_one(self):
        self.assertAlmostEqual(self.lats._parse_value("1.5"), 1.0)

    def test_clamps_below_zero(self):
        self.assertAlmostEqual(self.lats._parse_value("-0.3"), 0.0)

    def test_fallback_on_garbage(self):
        self.assertAlmostEqual(self.lats._parse_value("not a number"), 0.5)

    def test_markdown_fence_stripped(self):
        raw = "```json\n{\"value\": 0.75}\n```"
        self.assertAlmostEqual(self.lats._parse_value(raw), 0.75)


# ---------------------------------------------------------------------------
# LATS._apply_action tests
# ---------------------------------------------------------------------------

class TestLATSApplyAction(unittest.TestCase):

    def test_initial_state_empty(self):
        result = LATS._apply_action("", "search for docs")
        self.assertIn("search for docs", result)

    def test_appends_to_existing_state(self):
        state = "Step 1: read docs"
        result = LATS._apply_action(state, "write code")
        self.assertIn("Step 1: read docs", result)
        self.assertIn("write code", result)


# ---------------------------------------------------------------------------
# LATS.search integration tests (mocked LLM)
# ---------------------------------------------------------------------------

class TestLATSSearch(unittest.TestCase):

    def _make_lats(self, action_response=_ACTIONS_JSON, value_response=_VALUE_JSON, n_sims=4):
        llm = _make_llm([action_response, value_response])
        config = LATSConfig(
            max_depth=3,
            max_width=2,
            n_simulations=n_sims,
            value_threshold=0.95,  # high threshold so we don't exit early by default
        )
        return LATS(llm=llm, config=config)

    def test_returns_lats_result(self):
        lats = self._make_lats()
        result = lats.search(task="Plan a trip", initial_state="")
        self.assertIsInstance(result, LATSResult)

    def test_result_has_action_sequence(self):
        lats = self._make_lats()
        result = lats.search(task="Plan a trip")
        self.assertIsInstance(result.action_sequence, list)

    def test_result_best_value_in_range(self):
        lats = self._make_lats()
        result = lats.search(task="Plan a trip")
        self.assertGreaterEqual(result.best_value, 0.0)
        self.assertLessEqual(result.best_value, 1.0)

    def test_simulations_run_equals_n_simulations_when_no_early_exit(self):
        n = 3
        lats = self._make_lats(n_sims=n)
        result = lats.search(task="task")
        self.assertEqual(result.simulations_run, n)

    def test_nodes_expanded_positive(self):
        lats = self._make_lats(n_sims=4)
        result = lats.search(task="task")
        self.assertGreater(result.nodes_expanded, 0)

    def test_early_exit_when_high_value(self):
        # LLM always returns value=0.99 → should trigger early exit
        high_value = json.dumps({"value": 0.99})
        llm = _make_llm([_ACTIONS_JSON, high_value])
        config = LATSConfig(
            max_depth=3, max_width=2, n_simulations=50,
            value_threshold=0.95,
        )
        lats = LATS(llm=llm, config=config)
        result = lats.search(task="easy task")
        self.assertTrue(result.converged_early)
        self.assertLess(result.simulations_run, 50)

    def test_elapsed_seconds_positive(self):
        lats = self._make_lats(n_sims=2)
        result = lats.search(task="x")
        self.assertGreaterEqual(result.elapsed_seconds, 0.0)

    def test_llm_failure_produces_fallback(self):
        """When LLM always returns empty string, search should still complete."""
        llm = MagicMock()
        resp = MagicMock()
        resp.content = ""
        llm.generate.return_value = resp
        lats = LATS(llm=llm, config=LATSConfig(n_simulations=3, max_depth=2))
        result = lats.search(task="task")
        self.assertIsInstance(result, LATSResult)
        self.assertAlmostEqual(result.best_value, 0.5)

    def test_with_reward_evaluator(self):
        """Uses process reward evaluator instead of rollout simulation."""
        llm = _make_llm([_ACTIONS_JSON])
        reward_eval = MagicMock()
        traj_scores = MagicMock()
        traj_scores.mean_score = 0.88
        reward_eval.score_trajectory.return_value = traj_scores

        lats = LATS(
            llm=llm,
            config=LATSConfig(n_simulations=2, max_depth=2, max_width=1),
            reward_evaluator=reward_eval,
        )
        result = lats.search(task="task with reward")
        self.assertIsInstance(result, LATSResult)

    def test_cost_tracker_called(self):
        """CostTracker.record is called for each LLM call."""
        llm = _make_llm([_ACTIONS_JSON, _VALUE_JSON])
        tracker = MagicMock()
        lats = LATS(
            llm=llm,
            config=LATSConfig(n_simulations=1, max_depth=2, max_width=1),
            cost_tracker=tracker,
            provider="test",
            model_name="test-model",
        )
        lats.search(task="task")
        self.assertTrue(tracker.record.called)

    def test_max_depth_respected(self):
        """Tree should not expand beyond max_depth."""
        # With max_depth=1, all children are at depth 1 and should not expand
        llm = _make_llm([_ACTIONS_JSON, _VALUE_JSON])
        config = LATSConfig(max_depth=1, max_width=2, n_simulations=10)
        lats = LATS(llm=llm, config=config)
        result = lats.search(task="shallow task")
        # action_sequence length should be at most max_depth
        self.assertLessEqual(len(result.action_sequence), 1)


# ---------------------------------------------------------------------------
# Backpropagation unit test
# ---------------------------------------------------------------------------

class TestLATSBackprop(unittest.TestCase):

    def test_backprop_increments_all_ancestors(self):
        lats = LATS(llm=MagicMock(), config=LATSConfig())
        root = LATSNode(state="")
        child = LATSNode(state="s1", parent=root, depth=1)
        grandchild = LATSNode(state="s2", parent=child, depth=2)

        lats._backpropagate(grandchild, 0.8)

        self.assertEqual(grandchild.visit_count, 1)
        self.assertAlmostEqual(grandchild.total_value, 0.8)
        self.assertEqual(child.visit_count, 1)
        self.assertAlmostEqual(child.total_value, 0.8)
        self.assertEqual(root.visit_count, 1)
        self.assertAlmostEqual(root.total_value, 0.8)

    def test_backprop_accumulates_over_multiple_visits(self):
        lats = LATS(llm=MagicMock(), config=LATSConfig())
        root = LATSNode(state="")
        child = LATSNode(state="s", parent=root, depth=1)

        lats._backpropagate(child, 0.6)
        lats._backpropagate(child, 0.4)

        self.assertEqual(child.visit_count, 2)
        self.assertAlmostEqual(child.total_value, 1.0)
        self.assertAlmostEqual(child.mean_value, 0.5)


if __name__ == "__main__":
    unittest.main()

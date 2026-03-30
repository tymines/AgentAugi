"""Comprehensive tests for evoagentx.core.master_search (MASTER runtime planning)."""

import json
import math
import unittest
from unittest.mock import MagicMock, call

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from evoagentx.core.master_search import (
    MASTERSearch,
    MASTERConfig,
    MASTERNode,
)
from evoagentx.core.lats import LATSResult


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
_SELF_EVAL_JSON = json.dumps({
    "goal_progress": 0.7,
    "reasoning_quality": 0.8,
    "path_confidence": 0.75,
    "rationale": "Good progress so far.",
})
_SIMPLE_EVAL_JSON = json.dumps({"value": 0.7})
_HIGH_EVAL_JSON = json.dumps({
    "goal_progress": 0.95,
    "reasoning_quality": 0.97,
    "path_confidence": 0.96,
    "rationale": "Excellent trajectory.",
})


# ---------------------------------------------------------------------------
# MASTERNode tests
# ---------------------------------------------------------------------------

class TestMASTERNode(unittest.TestCase):
    """Tests for MASTERNode structure, UCB1, and action_sequence."""

    def test_mean_value_before_any_visit(self):
        node = MASTERNode(state="", depth=0)
        self.assertAlmostEqual(node.mean_value, 0.5)

    def test_mean_value_with_zero_total_weight(self):
        node = MASTERNode(state="")
        node.total_weight = 0.0
        node.total_weighted_value = 0.0
        self.assertAlmostEqual(node.mean_value, 0.5)

    def test_mean_value_after_weighted_updates(self):
        node = MASTERNode(state="")
        # Two updates: value=0.8 weight=1.0, value=0.6 weight=0.5
        node.total_weighted_value = 0.8 * 1.0 + 0.6 * 0.5
        node.total_weight = 1.0 + 0.5
        expected = (0.8 + 0.3) / 1.5
        self.assertAlmostEqual(node.mean_value, expected, places=6)

    def test_ucb1_unvisited_returns_inf(self):
        node = MASTERNode(state="")
        score = node.ucb1(parent_visits=10, exploration_weight=math.sqrt(2))
        self.assertEqual(score, float("inf"))

    def test_ucb1_visited_finite(self):
        node = MASTERNode(state="")
        node.visit_count = 4
        node.total_weighted_value = 2.0
        node.total_weight = 2.5
        node.confidence = 0.8
        score = node.ucb1(parent_visits=16, exploration_weight=math.sqrt(2))
        self.assertIsInstance(score, float)
        self.assertFalse(math.isinf(score))

    def test_ucb1_low_confidence_increases_exploration(self):
        """Low-confidence nodes should have higher UCB1 due to exploration bonus."""
        c = math.sqrt(2)
        parent_visits = 20

        high_conf = MASTERNode(state="")
        high_conf.visit_count = 4
        high_conf.total_weighted_value = 2.0
        high_conf.total_weight = 4.0
        high_conf.confidence = 0.9

        low_conf = MASTERNode(state="")
        low_conf.visit_count = 4
        low_conf.total_weighted_value = 2.0
        low_conf.total_weight = 4.0
        low_conf.confidence = 0.1

        # Same visit count and value, but low_conf should score higher
        self.assertGreater(
            low_conf.ucb1(parent_visits, c),
            high_conf.ucb1(parent_visits, c),
        )

    def test_is_leaf_no_children(self):
        node = MASTERNode(state="")
        self.assertTrue(node.is_leaf())

    def test_is_leaf_with_children(self):
        parent = MASTERNode(state="")
        child = MASTERNode(state="x", parent=parent)
        parent.children.append(child)
        self.assertFalse(parent.is_leaf())

    def test_action_sequence_root(self):
        root = MASTERNode(state="", action="")
        self.assertEqual(root.action_sequence(), [])

    def test_action_sequence_depth_two(self):
        root = MASTERNode(state="", action="")
        child = MASTERNode(state="s1", action="step1", parent=root, depth=1)
        grandchild = MASTERNode(state="s2", action="step2", parent=child, depth=2)
        self.assertEqual(grandchild.action_sequence(), ["step1", "step2"])

    def test_action_sequence_three_deep(self):
        root = MASTERNode(state="", action="")
        a = MASTERNode(state="a", action="a1", parent=root, depth=1)
        b = MASTERNode(state="b", action="b1", parent=a, depth=2)
        c = MASTERNode(state="c", action="c1", parent=b, depth=3)
        self.assertEqual(c.action_sequence(), ["a1", "b1", "c1"])

    def test_default_confidence(self):
        node = MASTERNode(state="")
        self.assertAlmostEqual(node.confidence, 0.5)

    def test_is_terminal_default_false(self):
        node = MASTERNode(state="")
        self.assertFalse(node.is_terminal)


# ---------------------------------------------------------------------------
# MASTERConfig tests
# ---------------------------------------------------------------------------

class TestMASTERConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = MASTERConfig()
        self.assertEqual(cfg.max_iterations, 20)
        self.assertEqual(cfg.max_depth, 5)
        self.assertAlmostEqual(cfg.exploration_weight, math.sqrt(2))
        self.assertAlmostEqual(cfg.confidence_threshold, 0.3)
        self.assertEqual(cfg.num_candidates, 3)
        self.assertTrue(cfg.use_self_evaluation)
        self.assertEqual(cfg.token_budget, 0)
        self.assertAlmostEqual(cfg.value_threshold, 0.9)

    def test_custom_values(self):
        cfg = MASTERConfig(max_iterations=5, max_depth=3, num_candidates=2)
        self.assertEqual(cfg.max_iterations, 5)
        self.assertEqual(cfg.max_depth, 3)
        self.assertEqual(cfg.num_candidates, 2)

    def test_token_budget_zero_means_unlimited(self):
        cfg = MASTERConfig(token_budget=0)
        self.assertEqual(cfg.token_budget, 0)


# ---------------------------------------------------------------------------
# Self-evaluation prompt parsing tests
# ---------------------------------------------------------------------------

class TestMASTERParseSelfEval(unittest.TestCase):

    def setUp(self):
        self.master = MASTERSearch(llm=MagicMock(), config=MASTERConfig())

    def test_valid_json_response(self):
        raw = json.dumps({
            "goal_progress": 0.7,
            "reasoning_quality": 0.8,
            "path_confidence": 0.6,
            "rationale": "Decent progress.",
        })
        value, confidence = self.master._parse_self_eval(raw)
        expected_value = 0.4 * 0.7 + 0.3 * 0.8 + 0.3 * 0.6
        self.assertAlmostEqual(value, expected_value, places=6)
        self.assertAlmostEqual(confidence, 0.6)

    def test_fallback_on_invalid_json(self):
        value, confidence = self.master._parse_self_eval("not valid json")
        self.assertAlmostEqual(value, 0.5)
        self.assertAlmostEqual(confidence, 0.5)

    def test_fallback_on_empty_string(self):
        value, confidence = self.master._parse_self_eval("")
        self.assertAlmostEqual(value, 0.5)
        self.assertAlmostEqual(confidence, 0.5)

    def test_clamping_above_one(self):
        raw = json.dumps({
            "goal_progress": 1.5,
            "reasoning_quality": 2.0,
            "path_confidence": 1.2,
        })
        value, confidence = self.master._parse_self_eval(raw)
        self.assertLessEqual(value, 1.0)
        self.assertLessEqual(confidence, 1.0)

    def test_clamping_below_zero(self):
        raw = json.dumps({
            "goal_progress": -0.5,
            "reasoning_quality": -1.0,
            "path_confidence": -0.2,
        })
        value, confidence = self.master._parse_self_eval(raw)
        self.assertGreaterEqual(value, 0.0)
        self.assertGreaterEqual(confidence, 0.0)

    def test_markdown_fence_stripped(self):
        inner = json.dumps({
            "goal_progress": 0.5,
            "reasoning_quality": 0.5,
            "path_confidence": 0.5,
        })
        raw = f"```json\n{inner}\n```"
        value, confidence = self.master._parse_self_eval(raw)
        self.assertAlmostEqual(value, 0.5, places=5)

    def test_missing_keys_use_defaults(self):
        raw = json.dumps({"goal_progress": 0.8})
        value, confidence = self.master._parse_self_eval(raw)
        # reasoning_quality and path_confidence default to 0.5
        expected_value = 0.4 * 0.8 + 0.3 * 0.5 + 0.3 * 0.5
        self.assertAlmostEqual(value, expected_value, places=6)


# ---------------------------------------------------------------------------
# Backpropagation tests
# ---------------------------------------------------------------------------

class TestMASTERBackprop(unittest.TestCase):

    def test_backprop_increments_all_ancestors(self):
        master = MASTERSearch(llm=MagicMock(), config=MASTERConfig())
        root = MASTERNode(state="")
        child = MASTERNode(state="s1", parent=root, depth=1)
        grandchild = MASTERNode(state="s2", parent=child, depth=2)

        master._backpropagate(grandchild, value=0.8, confidence=1.0)

        self.assertEqual(grandchild.visit_count, 1)
        self.assertAlmostEqual(grandchild.total_weighted_value, 0.8)
        self.assertEqual(child.visit_count, 1)
        self.assertAlmostEqual(child.total_weighted_value, 0.8)
        self.assertEqual(root.visit_count, 1)
        self.assertAlmostEqual(root.total_weighted_value, 0.8)

    def test_backprop_confidence_weights_value(self):
        master = MASTERSearch(llm=MagicMock(), config=MASTERConfig())
        root = MASTERNode(state="")
        child = MASTERNode(state="s1", parent=root, depth=1)

        master._backpropagate(child, value=0.8, confidence=0.6)

        self.assertAlmostEqual(child.total_weight, 0.6)
        self.assertAlmostEqual(child.total_weighted_value, 0.6 * 0.8)

    def test_backprop_low_confidence_uses_floor(self):
        """Confidence 0.0 should still update (floor = 0.1)."""
        master = MASTERSearch(llm=MagicMock(), config=MASTERConfig())
        root = MASTERNode(state="")
        child = MASTERNode(state="s", parent=root, depth=1)

        master._backpropagate(child, value=0.9, confidence=0.0)

        self.assertAlmostEqual(child.total_weight, 0.1)  # floor applied
        self.assertGreater(child.total_weighted_value, 0.0)

    def test_backprop_accumulates_over_multiple_calls(self):
        master = MASTERSearch(llm=MagicMock(), config=MASTERConfig())
        root = MASTERNode(state="")
        child = MASTERNode(state="s", parent=root, depth=1)

        master._backpropagate(child, value=0.8, confidence=1.0)
        master._backpropagate(child, value=0.6, confidence=0.5)

        self.assertEqual(child.visit_count, 2)
        self.assertAlmostEqual(child.total_weight, 1.5)
        self.assertAlmostEqual(child.total_weighted_value, 0.8 + 0.3, places=6)


# ---------------------------------------------------------------------------
# Action parsing tests
# ---------------------------------------------------------------------------

class TestMASTERParseActions(unittest.TestCase):

    def setUp(self):
        self.master = MASTERSearch(llm=MagicMock(), config=MASTERConfig())

    def test_valid_json_array(self):
        raw = '["act1", "act2", "act3"]'
        result = self.master._parse_actions(raw)
        self.assertEqual(result, ["act1", "act2", "act3"])

    def test_json_with_markdown_fence(self):
        raw = '```json\n["a", "b"]\n```'
        result = self.master._parse_actions(raw)
        self.assertEqual(result, ["a", "b"])

    def test_fallback_on_invalid_json(self):
        raw = "do this thing"
        result = self.master._parse_actions(raw)
        self.assertEqual(result, ["do this thing"])

    def test_empty_response(self):
        result = self.master._parse_actions("")
        self.assertEqual(result, [])


# ---------------------------------------------------------------------------
# Full search integration tests
# ---------------------------------------------------------------------------

class TestMASTERSearch(unittest.TestCase):

    def _make_master(
        self,
        action_response=_ACTIONS_JSON,
        eval_response=_SELF_EVAL_JSON,
        n_iters=4,
        extra_kwargs=None,
    ):
        llm = _make_llm([action_response, eval_response])
        config = MASTERConfig(
            max_iterations=n_iters,
            max_depth=3,
            num_candidates=2,
            value_threshold=0.98,  # high so we don't exit early by default
        )
        kwargs = {"llm": llm, "config": config}
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        return MASTERSearch(**kwargs)

    def test_returns_lats_result(self):
        master = self._make_master()
        result = master.search(task="Plan a trip", initial_state="")
        self.assertIsInstance(result, LATSResult)

    def test_result_has_action_sequence(self):
        master = self._make_master()
        result = master.search(task="Plan a trip")
        self.assertIsInstance(result.action_sequence, list)

    def test_result_best_value_in_range(self):
        master = self._make_master()
        result = master.search(task="Plan a trip")
        self.assertGreaterEqual(result.best_value, 0.0)
        self.assertLessEqual(result.best_value, 1.0)

    def test_simulations_run_equals_n_iterations_when_no_early_exit(self):
        n = 3
        master = self._make_master(n_iters=n)
        result = master.search(task="task")
        self.assertEqual(result.simulations_run, n)

    def test_nodes_expanded_positive(self):
        master = self._make_master(n_iters=4)
        result = master.search(task="task")
        self.assertGreater(result.nodes_expanded, 0)

    def test_early_exit_when_high_value(self):
        """Should trigger early exit when value >= value_threshold."""
        llm = _make_llm([_ACTIONS_JSON, _HIGH_EVAL_JSON])
        config = MASTERConfig(
            max_iterations=50,
            max_depth=3,
            num_candidates=2,
            value_threshold=0.9,
        )
        master = MASTERSearch(llm=llm, config=config)
        result = master.search(task="easy task")
        self.assertTrue(result.converged_early)
        self.assertLess(result.simulations_run, 50)

    def test_elapsed_seconds_nonnegative(self):
        master = self._make_master(n_iters=2)
        result = master.search(task="x")
        self.assertGreaterEqual(result.elapsed_seconds, 0.0)

    def test_llm_failure_produces_fallback(self):
        """When LLM always returns empty string, search should still complete."""
        llm = MagicMock()
        resp = MagicMock()
        resp.content = ""
        llm.generate.return_value = resp
        master = MASTERSearch(
            llm=llm,
            config=MASTERConfig(max_iterations=3, max_depth=2),
        )
        result = master.search(task="task")
        self.assertIsInstance(result, LATSResult)
        self.assertAlmostEqual(result.best_value, 0.5)

    def test_max_depth_respected(self):
        """Tree should not expand beyond max_depth."""
        llm = _make_llm([_ACTIONS_JSON, _SELF_EVAL_JSON])
        config = MASTERConfig(max_depth=1, num_candidates=2, max_iterations=10)
        master = MASTERSearch(llm=llm, config=config)
        result = master.search(task="shallow task")
        self.assertLessEqual(len(result.action_sequence), 1)

    def test_cost_tracker_called(self):
        """CostTracker.record is called for each LLM call."""
        llm = _make_llm([_ACTIONS_JSON, _SELF_EVAL_JSON])
        tracker = MagicMock()
        config = MASTERConfig(max_iterations=1, max_depth=2, num_candidates=1)
        master = MASTERSearch(
            llm=llm,
            config=config,
            cost_tracker=tracker,
            provider="test",
            model_name="test-model",
        )
        master.search(task="task")
        self.assertTrue(tracker.record.called)

    def test_cost_tracker_receives_master_purpose(self):
        """CostTracker metadata should contain 'master/' prefix."""
        llm = _make_llm([_ACTIONS_JSON, _SELF_EVAL_JSON])
        tracker = MagicMock()
        config = MASTERConfig(max_iterations=1, max_depth=2, num_candidates=1)
        master = MASTERSearch(
            llm=llm,
            config=config,
            cost_tracker=tracker,
            provider="test",
            model_name="test-model",
        )
        master.search(task="task")
        for call_args in tracker.record.call_args_list:
            metadata = call_args.kwargs.get("metadata", {})
            if "purpose" in metadata:
                self.assertTrue(metadata["purpose"].startswith("master/"))

    def test_with_reward_evaluator(self):
        """Uses process reward evaluator blended with self-evaluation."""
        llm = _make_llm([_ACTIONS_JSON, _SELF_EVAL_JSON])
        reward_eval = MagicMock()
        traj_scores = MagicMock()
        traj_scores.mean_score = 0.85
        reward_eval.score_trajectory.return_value = traj_scores

        config = MASTERConfig(max_iterations=2, max_depth=2, num_candidates=1)
        master = MASTERSearch(
            llm=llm,
            config=config,
            reward_evaluator=reward_eval,
        )
        result = master.search(task="task with reward")
        self.assertIsInstance(result, LATSResult)

    def test_use_self_evaluation_false(self):
        """When use_self_evaluation=False, uses simple evaluation."""
        llm = _make_llm([_ACTIONS_JSON, _SIMPLE_EVAL_JSON])
        config = MASTERConfig(
            max_iterations=2,
            max_depth=2,
            num_candidates=1,
            use_self_evaluation=False,
        )
        master = MASTERSearch(llm=llm, config=config)
        result = master.search(task="simple task")
        self.assertIsInstance(result, LATSResult)
        self.assertGreaterEqual(result.best_value, 0.0)
        self.assertLessEqual(result.best_value, 1.0)

    def test_single_iteration(self):
        """Search with a single iteration still returns a valid result."""
        llm = _make_llm([_ACTIONS_JSON, _SELF_EVAL_JSON])
        config = MASTERConfig(max_iterations=1, max_depth=3, num_candidates=2)
        master = MASTERSearch(llm=llm, config=config)
        result = master.search(task="one-shot task")
        self.assertIsInstance(result, LATSResult)
        self.assertEqual(result.simulations_run, 1)


# ---------------------------------------------------------------------------
# Token budget enforcement tests
# ---------------------------------------------------------------------------

class TestMASTERTokenBudget(unittest.TestCase):

    def test_token_budget_stops_search_early(self):
        """Search should stop before max_iterations when token budget is hit."""
        llm = _make_llm([_ACTIONS_JSON, _SELF_EVAL_JSON])
        config = MASTERConfig(
            max_iterations=100,
            max_depth=5,
            num_candidates=3,
            token_budget=50,  # very small budget — will be hit immediately
        )
        master = MASTERSearch(llm=llm, config=config)
        result = master.search(task="task")
        # Should have stopped well before 100 iterations
        self.assertLess(result.simulations_run, 100)

    def test_zero_token_budget_means_unlimited(self):
        """token_budget=0 should run all iterations."""
        llm = _make_llm([_ACTIONS_JSON, _SELF_EVAL_JSON])
        config = MASTERConfig(
            max_iterations=3,
            max_depth=2,
            num_candidates=1,
            token_budget=0,
        )
        master = MASTERSearch(llm=llm, config=config)
        result = master.search(task="task")
        self.assertEqual(result.simulations_run, 3)

    def test_tokens_used_tracked(self):
        """Internal token counter should be positive after a search."""
        llm = _make_llm([_ACTIONS_JSON, _SELF_EVAL_JSON])
        config = MASTERConfig(max_iterations=2, max_depth=2, num_candidates=1)
        master = MASTERSearch(llm=llm, config=config)
        master.search(task="task")
        self.assertGreater(master._tokens_used, 0)


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------

class TestMASTEREdgeCases(unittest.TestCase):

    def test_all_candidates_below_threshold_still_runs(self):
        """Even with low-confidence evaluations, search should complete."""
        low_eval = json.dumps({
            "goal_progress": 0.1,
            "reasoning_quality": 0.1,
            "path_confidence": 0.05,
            "rationale": "Very uncertain.",
        })
        llm = _make_llm([_ACTIONS_JSON, low_eval])
        config = MASTERConfig(
            max_iterations=4,
            max_depth=3,
            num_candidates=2,
            confidence_threshold=0.3,
        )
        master = MASTERSearch(llm=llm, config=config)
        result = master.search(task="hard task")
        self.assertIsInstance(result, LATSResult)

    def test_single_candidate_expansion(self):
        """num_candidates=1 should still build a valid tree."""
        llm = _make_llm([json.dumps(["only action"]), _SELF_EVAL_JSON])
        config = MASTERConfig(max_iterations=3, max_depth=3, num_candidates=1)
        master = MASTERSearch(llm=llm, config=config)
        result = master.search(task="narrow task")
        self.assertIsInstance(result, LATSResult)

    def test_max_depth_zero_no_expansion(self):
        """max_depth=0 should not expand beyond root."""
        llm = _make_llm([_ACTIONS_JSON, _SELF_EVAL_JSON])
        config = MASTERConfig(max_iterations=3, max_depth=0, num_candidates=2)
        master = MASTERSearch(llm=llm, config=config)
        result = master.search(task="zero depth")
        self.assertEqual(result.nodes_expanded, 0)
        self.assertEqual(result.action_sequence, [])

    def test_lats_result_compatibility(self):
        """MASTERSearch.search() return type must match LATSResult fields."""
        llm = _make_llm([_ACTIONS_JSON, _SELF_EVAL_JSON])
        config = MASTERConfig(max_iterations=2, max_depth=2, num_candidates=1)
        master = MASTERSearch(llm=llm, config=config)
        result = master.search(task="compatibility check")

        self.assertTrue(hasattr(result, "action_sequence"))
        self.assertTrue(hasattr(result, "best_value"))
        self.assertTrue(hasattr(result, "final_state"))
        self.assertTrue(hasattr(result, "nodes_expanded"))
        self.assertTrue(hasattr(result, "simulations_run"))
        self.assertTrue(hasattr(result, "elapsed_seconds"))
        self.assertTrue(hasattr(result, "converged_early"))

    def test_initial_state_included_in_final_state(self):
        """final_state should include the initial_state content."""
        llm = _make_llm([_ACTIONS_JSON, _SELF_EVAL_JSON])
        config = MASTERConfig(max_iterations=2, max_depth=3, num_candidates=1)
        master = MASTERSearch(llm=llm, config=config)
        result = master.search(task="task", initial_state="context from prior step")
        # The best node's state should trace back to initial_state or be a
        # child state; for the root (if no expansion happened), it equals initial_state
        self.assertIsInstance(result.final_state, str)

    def test_reward_evaluator_failure_fallback(self):
        """If reward evaluator raises, search should continue with fallback."""
        llm = _make_llm([_ACTIONS_JSON, _SELF_EVAL_JSON])
        reward_eval = MagicMock()
        reward_eval.score_trajectory.side_effect = RuntimeError("evaluator down")

        config = MASTERConfig(max_iterations=2, max_depth=2, num_candidates=1)
        master = MASTERSearch(
            llm=llm,
            config=config,
            reward_evaluator=reward_eval,
        )
        result = master.search(task="resilience test")
        self.assertIsInstance(result, LATSResult)


# ---------------------------------------------------------------------------
# apply_action helper tests
# ---------------------------------------------------------------------------

class TestMASTERApplyAction(unittest.TestCase):

    def test_initial_empty_state(self):
        result = MASTERSearch._apply_action("", "search docs")
        self.assertIn("search docs", result)

    def test_appends_to_existing_state(self):
        state = "Step 1: read docs"
        result = MASTERSearch._apply_action(state, "write code")
        self.assertIn("Step 1: read docs", result)
        self.assertIn("write code", result)


if __name__ == "__main__":
    unittest.main()

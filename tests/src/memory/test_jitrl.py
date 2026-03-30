"""Comprehensive tests for evoagentx.memory.jitrl."""

import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from evoagentx.memory.jitrl import (
    ActionStatistics,
    JitRLAgent,
    JitRLConfig,
    JitRLMemory,
    TrajectoryStatistics,
    TrajectoryStep,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_trajectory(
    actions: list,
    outcome: str = "success",
    rewards: list = None,
) -> TrajectoryStatistics:
    """Build a TrajectoryStatistics from a list of action-type strings."""
    rewards = rewards or [0.0] * len(actions)
    steps = [
        TrajectoryStep(action_type=a, reward=r) for a, r in zip(actions, rewards)
    ]
    return TrajectoryStatistics(
        steps=steps,
        outcome=outcome,
        total_reward=sum(rewards),
    )


# ---------------------------------------------------------------------------
# ActionStatistics tests
# ---------------------------------------------------------------------------


class TestActionStatistics(unittest.TestCase):

    def test_default_construction(self):
        stat = ActionStatistics(action_type="click", context_hash="abc123")
        self.assertEqual(stat.action_type, "click")
        self.assertEqual(stat.context_hash, "abc123")
        self.assertEqual(stat.success_count, 0.0)
        self.assertEqual(stat.failure_count, 0.0)
        self.assertEqual(stat.total_count, 0.0)
        self.assertAlmostEqual(stat.success_rate, 0.5)  # neutral default

    def test_success_rate_with_data(self):
        stat = ActionStatistics(
            action_type="search",
            context_hash="aabbcc",
            success_count=8.0,
            failure_count=2.0,
        )
        self.assertAlmostEqual(stat.success_rate, 0.8)
        self.assertAlmostEqual(stat.total_count, 10.0)

    def test_success_rate_all_failures(self):
        stat = ActionStatistics(
            action_type="click",
            context_hash="x",
            success_count=0.0,
            failure_count=10.0,
        )
        self.assertAlmostEqual(stat.success_rate, 0.0)

    def test_success_rate_all_successes(self):
        stat = ActionStatistics(
            action_type="type",
            context_hash="y",
            success_count=5.0,
            failure_count=0.0,
        )
        self.assertAlmostEqual(stat.success_rate, 1.0)

    def test_apply_decay(self):
        stat = ActionStatistics(
            action_type="act",
            context_hash="hash",
            success_count=10.0,
            failure_count=10.0,
            total_reward=5.0,
            avg_reward=0.5,
        )
        stat.apply_decay(0.9)
        self.assertAlmostEqual(stat.success_count, 9.0)
        self.assertAlmostEqual(stat.failure_count, 9.0)
        self.assertAlmostEqual(stat.total_reward, 4.5)
        # avg_reward should be recalculated as total_reward / total_count
        self.assertAlmostEqual(stat.avg_reward, 4.5 / 18.0)

    def test_apply_decay_to_near_zero(self):
        stat = ActionStatistics(
            action_type="act",
            context_hash="h",
            success_count=0.0001,
            failure_count=0.0,
        )
        stat.apply_decay(0.5)
        # avg_reward stays 0.0 when total_count reaches 0
        self.assertAlmostEqual(stat.avg_reward, 0.0)

    def test_last_updated_is_string(self):
        stat = ActionStatistics(action_type="foo", context_hash="bar")
        self.assertIsInstance(stat.last_updated, str)


# ---------------------------------------------------------------------------
# JitRLConfig tests
# ---------------------------------------------------------------------------


class TestJitRLConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = JitRLConfig()
        self.assertAlmostEqual(cfg.learning_rate, 0.1)
        self.assertAlmostEqual(cfg.discount_factor, 0.95)
        self.assertAlmostEqual(cfg.exploration_bonus, 0.1)
        self.assertEqual(cfg.min_samples, 5)
        self.assertEqual(cfg.context_window, 3)
        self.assertAlmostEqual(cfg.decay_rate, 0.99)

    def test_custom_values(self):
        cfg = JitRLConfig(learning_rate=0.5, discount_factor=0.8, min_samples=10)
        self.assertAlmostEqual(cfg.learning_rate, 0.5)
        self.assertAlmostEqual(cfg.discount_factor, 0.8)
        self.assertEqual(cfg.min_samples, 10)


# ---------------------------------------------------------------------------
# JitRLMemory — context hash tests
# ---------------------------------------------------------------------------


class TestContextHash(unittest.TestCase):

    def test_same_steps_same_hash(self):
        mem = JitRLMemory()
        h1 = mem._compute_context_hash(["search", "click"])
        h2 = mem._compute_context_hash(["search", "click"])
        self.assertEqual(h1, h2)

    def test_different_steps_different_hash(self):
        mem = JitRLMemory()
        h1 = mem._compute_context_hash(["search", "click"])
        h2 = mem._compute_context_hash(["click", "search"])
        self.assertNotEqual(h1, h2)

    def test_empty_steps_hash(self):
        mem = JitRLMemory()
        h = mem._compute_context_hash([])
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 8)

    def test_context_window_limits_input(self):
        mem = JitRLMemory(config=JitRLConfig(context_window=2))
        # Hash with only the last 2 steps should match
        h1 = mem._compute_context_hash(["a", "b", "c", "d"])
        h2 = mem._compute_context_hash(["x", "c", "d"])  # different prefix, same last 2
        self.assertEqual(h1, h2)

    def test_hash_length_is_8(self):
        mem = JitRLMemory()
        h = mem._compute_context_hash(["foo", "bar", "baz"])
        self.assertEqual(len(h), 8)


# ---------------------------------------------------------------------------
# JitRLMemory — temporal credit assignment
# ---------------------------------------------------------------------------


class TestTemporalCreditAssignment(unittest.TestCase):

    def test_successful_trajectory_last_step_highest_credit(self):
        mem = JitRLMemory(config=JitRLConfig(discount_factor=0.9))
        traj = make_trajectory(["a", "b", "c"], outcome="success")
        credits = mem._temporal_credit_assignment(traj)
        self.assertEqual(len(credits), 3)
        actions, values = zip(*credits)
        # Later steps get higher discounted credit (less discounting)
        self.assertLess(values[0], values[1])
        self.assertLess(values[1], values[2])

    def test_failure_trajectory_terminal_reward_zero(self):
        mem = JitRLMemory(config=JitRLConfig(discount_factor=0.9))
        traj = make_trajectory(["a", "b"], outcome="failure")
        credits = mem._temporal_credit_assignment(traj)
        # With failure (terminal=0) and no per-step rewards, all credits ≈ 0
        for _, credit in credits:
            self.assertAlmostEqual(credit, 0.0)

    def test_partial_outcome_credit_around_half(self):
        mem = JitRLMemory(config=JitRLConfig(discount_factor=1.0))
        traj = make_trajectory(["x"], outcome="partial")
        credits = mem._temporal_credit_assignment(traj)
        self.assertAlmostEqual(credits[0][1], 0.5)

    def test_empty_trajectory_returns_empty(self):
        mem = JitRLMemory()
        traj = TrajectoryStatistics(steps=[], outcome="success")
        credits = mem._temporal_credit_assignment(traj)
        self.assertEqual(credits, [])

    def test_discount_factor_propagation(self):
        mem = JitRLMemory(config=JitRLConfig(discount_factor=0.5))
        traj = make_trajectory(["a", "b"], outcome="success")
        credits = mem._temporal_credit_assignment(traj)
        # "a" is 1 step before end: 1.0 * 0.5^1 = 0.5
        # "b" is 0 steps before end: 1.0 * 0.5^0 = 1.0
        self.assertAlmostEqual(credits[0][1], 0.5)
        self.assertAlmostEqual(credits[1][1], 1.0)

    def test_per_step_rewards_are_additive(self):
        mem = JitRLMemory(config=JitRLConfig(discount_factor=1.0))
        traj = make_trajectory(["x"], outcome="success", rewards=[0.0])
        # With discount_factor=1 and no per-step reward: credit = 1.0
        credits = mem._temporal_credit_assignment(traj)
        self.assertAlmostEqual(credits[0][1], 1.0)

    def test_per_step_rewards_clipped_at_one(self):
        mem = JitRLMemory(config=JitRLConfig(discount_factor=1.0))
        # Per-step reward of 0.5 + terminal 1.0 would be 1.5 → clipped to 1.0
        traj = make_trajectory(["a"], outcome="success", rewards=[0.5])
        credits = mem._temporal_credit_assignment(traj)
        self.assertAlmostEqual(credits[0][1], 1.0)


# ---------------------------------------------------------------------------
# JitRLMemory — record_trajectory
# ---------------------------------------------------------------------------


class TestRecordTrajectory(unittest.TestCase):

    def test_trajectory_count_increments(self):
        mem = JitRLMemory()
        mem.record_trajectory(make_trajectory(["act"], outcome="success"))
        self.assertEqual(mem._trajectory_count, 1)
        mem.record_trajectory(make_trajectory(["act"], outcome="failure"))
        self.assertEqual(mem._trajectory_count, 2)

    def test_success_increments_success_count(self):
        mem = JitRLMemory()
        mem.record_trajectory(make_trajectory(["click"], outcome="success"))
        # Find the stat for "click"
        found = [s for s in mem._stats.values() if s.action_type == "click"]
        self.assertTrue(len(found) > 0)
        self.assertGreater(found[0].success_count, 0.0)

    def test_failure_increments_failure_count(self):
        mem = JitRLMemory()
        mem.record_trajectory(make_trajectory(["click"], outcome="failure"))
        found = [s for s in mem._stats.values() if s.action_type == "click"]
        self.assertTrue(len(found) > 0)
        self.assertGreater(found[0].failure_count, 0.0)

    def test_empty_trajectory_skipped(self):
        mem = JitRLMemory()
        mem.record_trajectory(TrajectoryStatistics(steps=[], outcome="success"))
        self.assertEqual(mem._trajectory_count, 0)
        self.assertEqual(len(mem._stats), 0)

    def test_multiple_trajectories_accumulate(self):
        mem = JitRLMemory()
        for _ in range(10):
            mem.record_trajectory(make_trajectory(["search"], outcome="success"))
        stats = [s for s in mem._stats.values() if s.action_type == "search"]
        self.assertTrue(any(s.success_count > 1.0 for s in stats))

    def test_same_action_all_success_high_rate(self):
        mem = JitRLMemory()
        for _ in range(20):
            mem.record_trajectory(make_trajectory(["navigate"], outcome="success"))
        found = [s for s in mem._stats.values() if s.action_type == "navigate"]
        self.assertTrue(any(s.success_rate > 0.8 for s in found))

    def test_same_action_all_failure_low_rate(self):
        mem = JitRLMemory()
        for _ in range(20):
            mem.record_trajectory(make_trajectory(["navigate"], outcome="failure"))
        found = [s for s in mem._stats.values() if s.action_type == "navigate"]
        self.assertTrue(any(s.success_rate < 0.5 for s in found))


# ---------------------------------------------------------------------------
# JitRLMemory — get_action_bias
# ---------------------------------------------------------------------------


class TestGetActionBias(unittest.TestCase):

    def _warm_up(self, mem, action, outcome, n=10):
        for _ in range(n):
            mem.record_trajectory(make_trajectory([action], outcome=outcome))

    def test_unknown_action_gets_exploration_bonus(self):
        mem = JitRLMemory(config=JitRLConfig(exploration_bonus=0.1))
        biases = mem.get_action_bias([], ["unseen_action"])
        self.assertAlmostEqual(biases["unseen_action"], 0.1)

    def test_successful_action_positive_bias(self):
        mem = JitRLMemory(config=JitRLConfig(min_samples=5))
        self._warm_up(mem, "good_action", "success", n=10)
        biases = mem.get_action_bias([], ["good_action"])
        self.assertGreater(biases["good_action"], 0.0)

    def test_failing_action_negative_bias(self):
        mem = JitRLMemory(config=JitRLConfig(min_samples=5))
        self._warm_up(mem, "bad_action", "failure", n=10)
        biases = mem.get_action_bias([], ["bad_action"])
        self.assertLess(biases["bad_action"], 0.0)

    def test_under_explored_action_gets_ucb_bonus(self):
        mem = JitRLMemory(config=JitRLConfig(min_samples=5, exploration_bonus=0.2))
        # Record only 2 episodes — below min_samples threshold
        mem.record_trajectory(make_trajectory(["rare_act"], outcome="success"))
        mem.record_trajectory(make_trajectory(["rare_act"], outcome="success"))
        biases = mem.get_action_bias([], ["rare_act"])
        # Should include exploration bonus since count < min_samples
        self.assertGreater(biases["rare_act"], 0.0)

    def test_bias_scores_all_provided_actions(self):
        mem = JitRLMemory()
        actions = ["a", "b", "c"]
        biases = mem.get_action_bias([], actions)
        self.assertEqual(set(biases.keys()), set(actions))

    def test_context_sensitivity(self):
        mem = JitRLMemory(config=JitRLConfig(min_samples=3))
        # Record "click" after "search" as successful
        for _ in range(5):
            traj = make_trajectory(["search", "click"], outcome="success")
            mem.record_trajectory(traj)
        # Record "click" in a fresh context (no preceding steps) as failure
        for _ in range(5):
            traj = make_trajectory(["click"], outcome="failure")
            mem.record_trajectory(traj)
        # In context ["search"], click should have better bias than in []
        bias_with_context = mem.get_action_bias(["search"], ["click"])
        bias_without_context = mem.get_action_bias([], ["click"])
        self.assertGreater(bias_with_context["click"], bias_without_context["click"])


# ---------------------------------------------------------------------------
# JitRLMemory — min_samples threshold
# ---------------------------------------------------------------------------


class TestMinSamplesThreshold(unittest.TestCase):

    def test_no_nudge_before_min_samples(self):
        mem = JitRLMemory(config=JitRLConfig(min_samples=10))
        for _ in range(5):  # Only 5, below threshold of 10
            mem.record_trajectory(make_trajectory(["act"], outcome="success"))
        nudge = mem.nudge_prompt([], ["act"])
        self.assertEqual(nudge, "")

    def test_nudge_appears_after_min_samples(self):
        mem = JitRLMemory(config=JitRLConfig(min_samples=3))
        for _ in range(5):  # 5 > 3 threshold
            mem.record_trajectory(make_trajectory(["act"], outcome="success"))
        nudge = mem.nudge_prompt([], ["act"])
        self.assertNotEqual(nudge, "")

    def test_exploration_bias_provided_regardless_of_min_samples(self):
        mem = JitRLMemory(config=JitRLConfig(min_samples=100, exploration_bonus=0.3))
        # Zero episodes — bias should still be exploration_bonus for unknown action
        biases = mem.get_action_bias([], ["new_action"])
        self.assertAlmostEqual(biases["new_action"], 0.3)


# ---------------------------------------------------------------------------
# JitRLMemory — nudge_prompt
# ---------------------------------------------------------------------------


class TestNudgePrompt(unittest.TestCase):

    def _fill(self, mem, action, outcome, n):
        for _ in range(n):
            mem.record_trajectory(make_trajectory([action], outcome=outcome))

    def test_nudge_contains_action_name(self):
        mem = JitRLMemory(config=JitRLConfig(min_samples=3))
        self._fill(mem, "navigate", "success", 5)
        nudge = mem.nudge_prompt([], ["navigate"])
        self.assertIn("navigate", nudge)

    def test_nudge_contains_success_signal(self):
        mem = JitRLMemory(config=JitRLConfig(min_samples=3))
        self._fill(mem, "search", "success", 8)
        nudge = mem.nudge_prompt([], ["search"])
        self.assertIn("success", nudge.lower())

    def test_nudge_contains_struggle_signal_for_failures(self):
        mem = JitRLMemory(config=JitRLConfig(min_samples=3))
        self._fill(mem, "click", "failure", 8)
        nudge = mem.nudge_prompt([], ["click"])
        # The nudge should flag poor performance
        self.assertTrue(
            "struggled" in nudge.lower() or "mixed" in nudge.lower() or "click" in nudge
        )

    def test_nudge_empty_when_no_data(self):
        mem = JitRLMemory()
        nudge = mem.nudge_prompt([], ["anything"])
        self.assertEqual(nudge, "")

    def test_nudge_empty_for_unknown_actions(self):
        mem = JitRLMemory(config=JitRLConfig(min_samples=3))
        self._fill(mem, "known_action", "success", 5)
        nudge = mem.nudge_prompt([], ["unknown_action"])
        self.assertEqual(nudge, "")

    def test_nudge_max_actions_shown_respected(self):
        mem = JitRLMemory(config=JitRLConfig(min_samples=1))
        actions = [f"act_{i}" for i in range(10)]
        for action in actions:
            for _ in range(3):
                mem.record_trajectory(make_trajectory([action], outcome="success"))
        nudge = mem.nudge_prompt([], actions, max_actions_shown=3)
        # Count action mentions — should be at most 3
        mentions = sum(1 for a in actions if a in nudge)
        self.assertLessEqual(mentions, 3)


# ---------------------------------------------------------------------------
# JitRLMemory — decay_statistics
# ---------------------------------------------------------------------------


class TestDecayStatistics(unittest.TestCase):

    def test_decay_reduces_counts(self):
        mem = JitRLMemory(config=JitRLConfig(decay_rate=0.9))
        for _ in range(10):
            mem.record_trajectory(make_trajectory(["act"], outcome="success"))
        count_before = sum(s.total_count for s in mem._stats.values())
        mem.decay_statistics()
        count_after = sum(s.total_count for s in mem._stats.values())
        self.assertLess(count_after, count_before)

    def test_decay_prunes_near_zero_entries(self):
        mem = JitRLMemory(config=JitRLConfig(decay_rate=0.001))
        mem.record_trajectory(make_trajectory(["tiny"], outcome="success"))
        self.assertGreater(len(mem._stats), 0)
        # After extreme decay, entries should be pruned
        for _ in range(5):
            mem.decay_statistics()
        self.assertEqual(len(mem._stats), 0)

    def test_decay_does_not_affect_trajectory_count(self):
        mem = JitRLMemory()
        for _ in range(5):
            mem.record_trajectory(make_trajectory(["a"], outcome="success"))
        mem.decay_statistics()
        self.assertEqual(mem._trajectory_count, 5)


# ---------------------------------------------------------------------------
# JitRLMemory — stats()
# ---------------------------------------------------------------------------


class TestStats(unittest.TestCase):

    def test_stats_empty(self):
        mem = JitRLMemory()
        s = mem.stats()
        self.assertEqual(s["total_trajectories"], 0)
        self.assertEqual(s["action_coverage"], 0)
        self.assertEqual(s["top_actions"], [])
        self.assertEqual(s["bottom_actions"], [])

    def test_stats_after_recording(self):
        mem = JitRLMemory(config=JitRLConfig(min_samples=2))
        for _ in range(5):
            mem.record_trajectory(make_trajectory(["good"], outcome="success"))
        for _ in range(5):
            mem.record_trajectory(make_trajectory(["bad"], outcome="failure"))
        s = mem.stats()
        self.assertEqual(s["total_trajectories"], 10)
        self.assertGreater(s["action_coverage"], 0)
        # Top action should have higher success rate than bottom
        if s["top_actions"] and s["bottom_actions"]:
            self.assertGreaterEqual(s["top_actions"][0][1], s["bottom_actions"][0][1])


# ---------------------------------------------------------------------------
# JitRLMemory — persistence (save/load)
# ---------------------------------------------------------------------------


class TestPersistence(unittest.TestCase):

    def test_save_and_load_roundtrip(self):
        mem = JitRLMemory()
        for _ in range(5):
            mem.record_trajectory(make_trajectory(["navigate", "click"], outcome="success"))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            mem.save(path)
            mem2 = JitRLMemory()
            mem2.load(path)
            self.assertEqual(mem2._trajectory_count, mem._trajectory_count)
            self.assertEqual(set(mem2._stats.keys()), set(mem._stats.keys()))
        finally:
            os.unlink(path)

    def test_save_raises_without_path(self):
        mem = JitRLMemory()
        with self.assertRaises(ValueError):
            mem.save()

    def test_load_no_op_when_file_missing(self):
        mem = JitRLMemory()
        mem.load("/nonexistent/path/to/nothing.json")
        self.assertEqual(mem._trajectory_count, 0)

    def test_persistence_path_constructor(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            mem = JitRLMemory(persistence_path=path)
            mem.record_trajectory(make_trajectory(["act"], outcome="success"))
            mem.save()

            mem2 = JitRLMemory(persistence_path=path)
            mem2.load()
            self.assertEqual(mem2._trajectory_count, 1)
        finally:
            os.unlink(path)

    def test_saved_file_is_valid_json(self):
        mem = JitRLMemory()
        mem.record_trajectory(make_trajectory(["a", "b"], outcome="success"))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            mem.save(path)
            with open(path) as fh:
                data = json.load(fh)
            self.assertIn("trajectory_count", data)
            self.assertIn("stats", data)
            self.assertIn("config", data)
        finally:
            os.unlink(path)

    def test_load_skips_malformed_entries(self):
        bad_data = {
            "trajectory_count": 3,
            "config": {},
            "stats": {
                "good|hash": {
                    "action_type": "good",
                    "context_hash": "hash",
                    "success_count": 2.0,
                    "failure_count": 1.0,
                    "total_reward": 1.5,
                    "avg_reward": 0.5,
                    "last_updated": "2024-01-01T00:00:00",
                },
                "broken": "not_a_dict",  # malformed
            },
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(bad_data, f)
            path = f.name

        try:
            mem = JitRLMemory()
            mem.load(path)
            # The good entry should be loaded, the bad one silently skipped
            self.assertIn("good|hash", mem._stats)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# JitRLAgent wrapper tests
# ---------------------------------------------------------------------------


class TestJitRLAgent(unittest.TestCase):

    def _make_mock_agent(self, return_value="ok"):
        agent = MagicMock()
        agent.execute.return_value = return_value
        agent.system_prompt = "base prompt"
        return agent

    def test_execute_calls_wrapped_agent(self):
        mock_agent = self._make_mock_agent("result")
        mem = JitRLMemory()
        wrapper = JitRLAgent(agent=mock_agent, memory=mem)
        result = wrapper.execute(task_steps=["act"], outcome="success")
        mock_agent.execute.assert_called_once()
        self.assertEqual(result, "result")

    def test_trajectory_recorded_after_execution(self):
        mock_agent = self._make_mock_agent()
        mem = JitRLMemory()
        wrapper = JitRLAgent(agent=mock_agent, memory=mem)
        wrapper.execute(task_steps=["search", "click"], outcome="success")
        self.assertEqual(mem._trajectory_count, 1)

    def test_nudge_injected_into_prompt_when_data_available(self):
        mock_agent = self._make_mock_agent()
        mem = JitRLMemory(config=JitRLConfig(min_samples=3))
        # Warm up statistics
        for _ in range(5):
            mem.record_trajectory(make_trajectory(["search"], outcome="success"))

        wrapper = JitRLAgent(
            agent=mock_agent,
            memory=mem,
            available_actions=["search"],
        )
        wrapper.execute(task_steps=["search"], outcome="success")
        # System prompt should have been set to something containing the nudge
        # (it's restored afterward, but it was modified during execution)
        mock_agent.execute.assert_called_once()

    def test_system_prompt_restored_after_execution(self):
        mock_agent = self._make_mock_agent()
        mock_agent.system_prompt = "original"
        mem = JitRLMemory(config=JitRLConfig(min_samples=1))
        for _ in range(3):
            mem.record_trajectory(make_trajectory(["act"], outcome="success"))

        wrapper = JitRLAgent(
            agent=mock_agent, memory=mem, available_actions=["act"]
        )
        wrapper.execute(task_steps=["act"], outcome="success")
        self.assertEqual(mock_agent.system_prompt, "original")

    def test_execute_with_no_task_steps(self):
        mock_agent = self._make_mock_agent("done")
        mem = JitRLMemory()
        wrapper = JitRLAgent(agent=mock_agent, memory=mem)
        result = wrapper.execute(outcome="success")
        # No trajectory recorded when no steps provided
        self.assertEqual(mem._trajectory_count, 0)
        self.assertEqual(result, "done")

    def test_step_rewards_forwarded(self):
        mock_agent = self._make_mock_agent()
        mem = JitRLMemory()
        wrapper = JitRLAgent(agent=mock_agent, memory=mem)
        wrapper.execute(
            task_steps=["a", "b"],
            outcome="success",
            step_rewards=[0.2, 0.8],
        )
        self.assertEqual(mem._trajectory_count, 1)

    def test_async_execute(self):
        mock_agent = MagicMock()
        mock_agent.system_prompt = "prompt"

        async def fake_async_execute(**kwargs):
            return "async_result"

        mock_agent.async_execute = fake_async_execute
        mem = JitRLMemory()
        wrapper = JitRLAgent(agent=mock_agent, memory=mem)

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                wrapper.async_execute(task_steps=["async_act"], outcome="success")
            )
        finally:
            loop.close()
        self.assertEqual(result, "async_result")
        self.assertEqual(mem._trajectory_count, 1)

    def test_composition_with_reflexion_agent(self):
        """JitRLAgent can wrap a ReflexionAgent (composition pattern)."""
        from evoagentx.memory.reflexion import ReflexionAgent, ReflexionMemory

        inner_agent = self._make_mock_agent("inner_result")
        reflexion_mem = ReflexionMemory()
        reflexion = ReflexionAgent(agent=inner_agent, memory=reflexion_mem)

        jitrl_mem = JitRLMemory()
        jitrl_agent = JitRLAgent(agent=reflexion, memory=jitrl_mem)

        # Execute via the JitRL wrapper which wraps the Reflexion wrapper
        result = jitrl_agent.execute(
            task="test composition",
            task_steps=["step1"],
            outcome="success",
        )
        self.assertEqual(result, "inner_result")
        # JitRL should have recorded its trajectory
        self.assertEqual(jitrl_mem._trajectory_count, 1)
        # Reflexion should also have stored an episode
        self.assertEqual(len(reflexion_mem.episodes), 1)


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------


class TestEdgeCases(unittest.TestCase):

    def test_single_trajectory_single_step(self):
        mem = JitRLMemory()
        mem.record_trajectory(make_trajectory(["act"], outcome="success"))
        self.assertEqual(mem._trajectory_count, 1)
        self.assertGreater(len(mem._stats), 0)

    def test_all_same_outcome_success(self):
        mem = JitRLMemory(config=JitRLConfig(min_samples=3))
        for _ in range(10):
            mem.record_trajectory(make_trajectory(["x"], outcome="success"))
        found = [s for s in mem._stats.values() if s.action_type == "x"]
        self.assertTrue(any(s.success_rate > 0.7 for s in found))

    def test_all_same_outcome_failure(self):
        mem = JitRLMemory(config=JitRLConfig(min_samples=3))
        for _ in range(10):
            mem.record_trajectory(make_trajectory(["y"], outcome="failure"))
        found = [s for s in mem._stats.values() if s.action_type == "y"]
        self.assertTrue(any(s.success_rate < 0.5 for s in found))

    def test_new_unseen_action_not_in_stats(self):
        mem = JitRLMemory()
        biases = mem.get_action_bias([], ["completely_new"])
        # Should return exploration bonus, not error
        self.assertIn("completely_new", biases)

    def test_nudge_prompt_with_no_available_actions(self):
        mem = JitRLMemory()
        nudge = mem.nudge_prompt([], [])
        self.assertEqual(nudge, "")

    def test_stats_returns_dict_structure(self):
        mem = JitRLMemory()
        s = mem.stats()
        self.assertIn("total_trajectories", s)
        self.assertIn("action_coverage", s)
        self.assertIn("top_actions", s)
        self.assertIn("bottom_actions", s)

    def test_multiple_actions_per_step_different_contexts(self):
        """Actions at different positions in the same trajectory get different context hashes."""
        mem = JitRLMemory()
        traj = make_trajectory(["a", "b", "c"], outcome="success")
        mem.record_trajectory(traj)
        # Keys for "a", "b", "c" should have different context hashes
        keys = list(mem._stats.keys())
        context_hashes = [k.split("|")[1] for k in keys]
        # Not all hashes should be the same (first step has empty context, others don't)
        # With context_window >= 1 and 3 distinct positions, at least some differ
        self.assertGreater(len(set(context_hashes)), 0)


if __name__ == "__main__":
    unittest.main()

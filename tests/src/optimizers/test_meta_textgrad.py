"""Unit tests for evoagentx.optimizers.meta_textgrad."""

import json
import os
import tempfile
import unittest
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

from evoagentx.optimizers.meta_textgrad import (
    ContrastiveStrategy,
    ExemplarStrategy,
    MetaPolicy,
    MetaTextGradHistory,
    MetaTextGradOptimizer,
    SocraticStrategy,
    StrategyStats,
    StepwiseStrategy,
    STRATEGY_NAMES,
)
from evoagentx.optimizers.engine.registry import ParamRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeLLMResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    def __init__(self, response: str = "updated prompt") -> None:
        self._response = response
        self.call_count = 0

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs):
        self.call_count += 1
        return _FakeLLMResponse(self._response)


def _make_registry() -> ParamRegistry:
    registry = MagicMock(spec=ParamRegistry)
    registry.fields = {"instruction": None}
    registry.names.return_value = ["instruction"]
    registry.get.return_value = "Answer the question."
    registry.set.return_value = None
    return registry


def _make_optimizer(**kwargs) -> MetaTextGradOptimizer:
    registry = _make_registry()
    llm = _FakeLLM()
    program = MagicMock(return_value={"answer": "42"})
    evaluator = MagicMock(return_value=0.6)
    defaults = dict(
        registry=registry,
        program=program,
        evaluator=evaluator,
        llm=llm,
        task_type="qa",
        steps=3,
        training_examples=[{"q": "x"}],
        seed=0,
    )
    defaults.update(kwargs)
    return MetaTextGradOptimizer(**defaults)


# ---------------------------------------------------------------------------
# Strategy tests
# ---------------------------------------------------------------------------

class TestGradientStrategies(unittest.TestCase):

    def _check_strategy(self, strategy_cls):
        strategy = strategy_cls()
        sys_p, user_p = strategy.build_prompt(
            param_name="instruction",
            current_value="Do the task.",
            evaluation_feedback="score=0.5",
            task_context="Task type: qa",
        )
        self.assertIsInstance(sys_p, str)
        self.assertIsInstance(user_p, str)
        self.assertGreater(len(sys_p), 0)
        self.assertGreater(len(user_p), 0)

    def test_contrastive_strategy(self):
        self._check_strategy(ContrastiveStrategy)

    def test_stepwise_strategy(self):
        self._check_strategy(StepwiseStrategy)

    def test_socratic_strategy(self):
        self._check_strategy(SocraticStrategy)

    def test_exemplar_strategy(self):
        self._check_strategy(ExemplarStrategy)

    def test_strategy_names_covers_all(self):
        self.assertIn("contrastive", STRATEGY_NAMES)
        self.assertIn("stepwise", STRATEGY_NAMES)
        self.assertIn("socratic", STRATEGY_NAMES)
        self.assertIn("exemplar", STRATEGY_NAMES)


# ---------------------------------------------------------------------------
# StrategyStats
# ---------------------------------------------------------------------------

class TestStrategyStats(unittest.TestCase):

    def test_mean_reward_zero_pulls(self):
        s = StrategyStats()
        self.assertAlmostEqual(s.mean_reward, 0.0)

    def test_mean_reward_after_updates(self):
        s = StrategyStats()
        s.update(0.8)
        s.update(0.4)
        self.assertAlmostEqual(s.mean_reward, 0.6)

    def test_n_pulls_increments(self):
        s = StrategyStats()
        s.update(0.5)
        s.update(0.5)
        self.assertEqual(s.n_pulls, 2)


# ---------------------------------------------------------------------------
# MetaPolicy
# ---------------------------------------------------------------------------

class TestMetaPolicy(unittest.TestCase):

    def test_untried_strategies_selected_first(self):
        """Every strategy should be selected at least once before UCB kicks in."""
        policy = MetaPolicy(strategy_names=STRATEGY_NAMES)
        seen = set()
        for _ in range(len(STRATEGY_NAMES) * 2):
            name = policy.select(task_type="qa", step=len(seen))
            seen.add(name)
            policy.update(name, reward=0.5, task_type="qa")
        self.assertEqual(seen, set(STRATEGY_NAMES))

    def test_update_changes_stats(self):
        policy = MetaPolicy(strategy_names=["a", "b"])
        policy.update("a", reward=1.0, task_type="test")
        stats = policy._stats["test"]["a"]
        self.assertEqual(stats.n_pulls, 1)
        self.assertAlmostEqual(stats.cumulative_reward, 1.0)

    def test_to_dict_round_trip(self):
        policy = MetaPolicy(strategy_names=STRATEGY_NAMES, ucb_c=2.0)
        policy.update("stepwise", reward=0.7, task_type="coding")
        d = policy.to_dict()
        restored = MetaPolicy.from_dict(d)
        self.assertAlmostEqual(
            restored._stats["coding"]["stepwise"].mean_reward, 0.7
        )

    def test_separate_task_types(self):
        policy = MetaPolicy(strategy_names=["a", "b"])
        policy.update("a", reward=1.0, task_type="qa")
        policy.update("a", reward=0.0, task_type="coding")
        self.assertAlmostEqual(policy._stats["qa"]["a"].mean_reward, 1.0)
        self.assertAlmostEqual(policy._stats["coding"]["a"].mean_reward, 0.0)


# ---------------------------------------------------------------------------
# MetaTextGradHistory
# ---------------------------------------------------------------------------

class TestMetaTextGradHistory(unittest.TestCase):

    def test_defaults(self):
        h = MetaTextGradHistory()
        self.assertEqual(h.score_per_step, [])
        self.assertEqual(h.strategy_per_step, [])
        self.assertIsNone(h.best_config)
        self.assertEqual(h.strategy_usage, {})


# ---------------------------------------------------------------------------
# MetaTextGradOptimizer — unit behaviour
# ---------------------------------------------------------------------------

class TestMetaTextGradOptimizer(unittest.TestCase):

    def test_optimize_returns_config_and_history(self):
        opt = _make_optimizer()
        cfg, history = opt.optimize()
        self.assertIsInstance(cfg, dict)
        self.assertIsInstance(history, MetaTextGradHistory)

    def test_history_score_length(self):
        opt = _make_optimizer(steps=4)
        _, history = opt.optimize()
        # steps + 1 (initial evaluation)
        self.assertEqual(len(history.score_per_step), 5)

    def test_history_strategy_length(self):
        opt = _make_optimizer(steps=3)
        _, history = opt.optimize()
        self.assertEqual(len(history.strategy_per_step), 3)

    def test_strategy_usage_populated(self):
        opt = _make_optimizer(steps=5)
        _, history = opt.optimize()
        total_usage = sum(history.strategy_usage.values())
        self.assertEqual(total_usage, 5)

    def test_best_config_is_set(self):
        opt = _make_optimizer(steps=3)
        _, history = opt.optimize()
        self.assertIsNotNone(history.best_config)

    def test_meta_policy_updated_after_steps(self):
        opt = _make_optimizer(steps=4)
        opt.optimize()
        # At least one strategy should have been pulled
        total_pulls = sum(
            s.n_pulls
            for strats in opt.meta_policy._stats.values()
            for s in strats.values()
        )
        self.assertGreater(total_pulls, 0)

    def test_optimize_no_examples(self):
        opt = _make_optimizer(steps=2, training_examples=[])
        cfg, history = opt.optimize()
        self.assertIsNotNone(cfg)

    def test_warm_start_save_load(self):
        opt = _make_optimizer(steps=3)
        opt.optimize()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            opt.save_meta_policy(path)
            opt2 = _make_optimizer(steps=1)
            opt2.load_meta_policy(path)
            # Policy stats should be populated from saved state
            total_pulls = sum(
                s.n_pulls
                for strats in opt2.meta_policy._stats.values()
                for s in strats.values()
            )
            self.assertGreater(total_pulls, 0)
        finally:
            os.unlink(path)

    def test_pre_loaded_meta_policy_accepted(self):
        policy = MetaPolicy(strategy_names=STRATEGY_NAMES)
        opt = _make_optimizer(steps=2, meta_policy=policy)
        cfg, _ = opt.optimize()
        self.assertIsNotNone(cfg)


if __name__ == "__main__":
    unittest.main()

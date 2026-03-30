"""metaTextGrad — meta-learning layer on top of TextGrad-style optimization.

TextGrad treats the optimizer as fixed: it always computes gradients the same
way regardless of task type or what kind of feedback has worked in the past.
metaTextGrad adds a *meta-policy* that learns *which gradient strategy is most
effective* for a given task, then biases the gradient computation accordingly.

Design overview
---------------
1. **Gradient strategies** — A small catalogue of named strategies, each of
   which controls *how* the LLM feedback prompt is framed when asking for a
   gradient (e.g. ``"contrastive"``, ``"stepwise"``, ``"socratic"``).
2. **Meta-policy** — A bandit-style policy (ε-greedy with UCB fallback) that
   selects a strategy for each gradient step.  After the step, the observed
   score improvement is used to update the strategy's estimated value.
3. **Task-type conditioning** — The meta-policy keeps separate value estimates
   for each task type, enabling transfer within a session.
4. **Warm-start** — Policy state can be serialised and reloaded so that prior
   runs inform future optimization without retraining from scratch.
5. **Process reward integration** — When a ``StepwiseRewardEvaluator`` is
   provided, its per-step scores are used as a denser reward signal for the
   meta-policy (rather than only the final outcome delta).

Integration points
------------------
- Uses ``BaseLLM`` for gradient computation (no dependency on the ``textgrad``
  library — this is an independent re-implementation of the same concept).
- Extends ``BaseOptimizer`` so it fits into the existing optimizer stack.
- Optionally wraps ``StepwiseRewardEvaluator`` from
  ``evoagentx.evaluators.process_reward`` for richer rewards.
- Compatible with ``ConstrainedOptimizer`` from
  ``evoagentx.optimizers.constraint_layer``.

Usage
-----
    >>> from evoagentx.optimizers.meta_textgrad import MetaTextGradOptimizer
    >>> optimizer = MetaTextGradOptimizer(
    ...     registry=registry,
    ...     program=my_program,
    ...     evaluator=my_evaluator,
    ...     llm=my_llm,
    ...     task_type="qa",
    ...     steps=30,
    ... )
    >>> best_cfg, history = optimizer.optimize()
    >>> optimizer.save_meta_policy("meta_policy_qa.json")
"""

from __future__ import annotations

import json
import math
import random
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.logging import logger
from .engine.base import BaseOptimizer
from .engine.registry import ParamRegistry

# ---------------------------------------------------------------------------
# Optional process reward integration
# ---------------------------------------------------------------------------
try:
    from ..evaluators.process_reward import StepwiseRewardEvaluator  # type: ignore
    _PRM_AVAILABLE = True
except ImportError:
    _PRM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional CostTracker integration
# ---------------------------------------------------------------------------
try:
    from ..core.cost_tracker import CostTracker  # type: ignore
    _COST_TRACKER_AVAILABLE = True
except ImportError:
    _COST_TRACKER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Gradient strategies
# ---------------------------------------------------------------------------

class _GradientStrategy:
    """Defines how to frame the gradient computation prompt for one strategy.

    Each strategy corresponds to a different way of eliciting feedback from the
    LLM about how to improve a prompt.  The strategy is passed the current
    prompt and evaluation feedback, and returns a system + user prompt pair for
    the gradient LLM call.
    """

    name: str = "base"

    def build_prompt(
        self,
        param_name: str,
        current_value: str,
        evaluation_feedback: str,
        task_context: str,
    ) -> Tuple[str, str]:
        """Build the (system_prompt, user_prompt) pair for the gradient call.

        Returns:
            Tuple of (system_prompt, user_prompt) strings.
        """
        raise NotImplementedError


class ContrastiveStrategy(_GradientStrategy):
    """Ask the LLM to contrast a good vs. bad example to guide the edit."""

    name = "contrastive"

    def build_prompt(
        self, param_name: str, current_value: str, evaluation_feedback: str,
        task_context: str,
    ) -> Tuple[str, str]:
        system = (
            "You are an expert prompt engineer. "
            "Use contrastive reasoning: imagine a perfect version and a poor version "
            "of this prompt, then produce an improved version that moves toward perfect."
        )
        user = (
            f"## Parameter: {param_name}\n"
            f"## Current prompt\n{current_value}\n\n"
            f"## Evaluation feedback\n{evaluation_feedback}\n\n"
            f"## Task context\n{task_context}\n\n"
            "Describe what a near-perfect version would do differently from the current "
            "prompt, then output the improved prompt.\n"
            "Output ONLY the improved prompt — no commentary."
        )
        return system, user


class StepwiseStrategy(_GradientStrategy):
    """Ask the LLM to reason step-by-step about what to change."""

    name = "stepwise"

    def build_prompt(
        self, param_name: str, current_value: str, evaluation_feedback: str,
        task_context: str,
    ) -> Tuple[str, str]:
        system = (
            "You are an expert prompt engineer. "
            "Use step-by-step reasoning to identify the single most impactful "
            "change to this prompt, then apply it."
        )
        user = (
            f"## Parameter: {param_name}\n"
            f"## Current prompt\n{current_value}\n\n"
            f"## Evaluation feedback\n{evaluation_feedback}\n\n"
            f"## Task context\n{task_context}\n\n"
            "Step 1: Identify the biggest weakness in the current prompt.\n"
            "Step 2: Propose a targeted fix.\n"
            "Step 3: Output the improved prompt.\n"
            "Output ONLY the improved prompt after your reasoning — no commentary."
        )
        return system, user


class SocraticStrategy(_GradientStrategy):
    """Ask the LLM to question the prompt's assumptions before improving it."""

    name = "socratic"

    def build_prompt(
        self, param_name: str, current_value: str, evaluation_feedback: str,
        task_context: str,
    ) -> Tuple[str, str]:
        system = (
            "You are a Socratic prompt engineer. "
            "Challenge the assumptions embedded in the current prompt and use "
            "those insights to write a better version."
        )
        user = (
            f"## Parameter: {param_name}\n"
            f"## Current prompt\n{current_value}\n\n"
            f"## Evaluation feedback\n{evaluation_feedback}\n\n"
            f"## Task context\n{task_context}\n\n"
            "Ask two questions that expose hidden assumptions in the current prompt, "
            "then produce an improved version that addresses them.\n"
            "Output ONLY the improved prompt — no commentary."
        )
        return system, user


class ExemplarStrategy(_GradientStrategy):
    """Ask the LLM to provide an exemplar output then back-engineer the prompt."""

    name = "exemplar"

    def build_prompt(
        self, param_name: str, current_value: str, evaluation_feedback: str,
        task_context: str,
    ) -> Tuple[str, str]:
        system = (
            "You are an expert prompt engineer. "
            "Imagine the ideal model output for this task, then reverse-engineer "
            "the prompt that would most reliably produce it."
        )
        user = (
            f"## Parameter: {param_name}\n"
            f"## Current prompt\n{current_value}\n\n"
            f"## Evaluation feedback\n{evaluation_feedback}\n\n"
            f"## Task context\n{task_context}\n\n"
            "First, briefly describe the ideal model output. "
            "Then write the prompt that would produce it.\n"
            "Output ONLY the improved prompt — no commentary."
        )
        return system, user


# Registry of all available strategies, keyed by name
_STRATEGIES: Dict[str, _GradientStrategy] = {
    s.name: s
    for s in [
        ContrastiveStrategy(),
        StepwiseStrategy(),
        SocraticStrategy(),
        ExemplarStrategy(),
    ]
}
STRATEGY_NAMES = list(_STRATEGIES.keys())


# ---------------------------------------------------------------------------
# Meta-policy (UCB bandit)
# ---------------------------------------------------------------------------

@dataclass
class StrategyStats:
    """Running statistics for a single gradient strategy within a task type.

    Attributes:
        n_pulls: Number of times this strategy has been selected.
        cumulative_reward: Sum of reward signals observed after each pull.
        mean_reward: Current estimate of expected reward.
    """

    n_pulls: int = 0
    cumulative_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        """Average reward, or 0.0 if never pulled."""
        return self.cumulative_reward / self.n_pulls if self.n_pulls > 0 else 0.0

    def update(self, reward: float) -> None:
        """Record an observed reward for this strategy."""
        self.n_pulls += 1
        self.cumulative_reward += reward


class MetaPolicy:
    """UCB1 bandit meta-policy for gradient strategy selection.

    Maintains per-task-type statistics so that knowledge gained on one task
    type can be transferred to another through a shared prior.

    Args:
        strategy_names: Names of the strategies available for selection.
        ucb_c: Exploration constant for UCB1.  Higher values encourage more
            exploration; lower values exploit known good strategies sooner.
        default_task_type: Task type used when none is specified.
    """

    def __init__(
        self,
        strategy_names: List[str],
        ucb_c: float = 1.414,
        default_task_type: str = "general",
    ) -> None:
        self.strategy_names = strategy_names
        self.ucb_c = ucb_c
        self.default_task_type = default_task_type
        # stats[task_type][strategy_name] -> StrategyStats
        self._stats: Dict[str, Dict[str, StrategyStats]] = {}

    def _ensure_task_type(self, task_type: str) -> None:
        """Initialise statistics for a task type if not already present."""
        if task_type not in self._stats:
            self._stats[task_type] = {
                name: StrategyStats() for name in self.strategy_names
            }

    def select(self, task_type: Optional[str] = None, step: int = 0) -> str:
        """Select a strategy using UCB1 for the given task type.

        Any strategy that has never been tried is prioritised.  Once all
        strategies have been tried at least once, UCB1 selects based on
        estimated value + exploration bonus.

        Args:
            task_type: Task type identifier (e.g. ``"qa"``, ``"coding"``).
            step: Current optimization step (used as total pulls estimate).

        Returns:
            Name of the selected strategy.
        """
        tt = task_type or self.default_task_type
        self._ensure_task_type(tt)
        stats = self._stats[tt]

        # Phase 1: round-robin until every strategy tried once
        untried = [n for n in self.strategy_names if stats[n].n_pulls == 0]
        if untried:
            return random.choice(untried)

        # Phase 2: UCB1
        total_pulls = sum(s.n_pulls for s in stats.values())
        ucb_scores = {
            name: s.mean_reward + self.ucb_c * math.sqrt(
                math.log(total_pulls + 1) / (s.n_pulls + 1e-9)
            )
            for name, s in stats.items()
        }
        return max(ucb_scores, key=ucb_scores.__getitem__)

    def update(
        self,
        strategy_name: str,
        reward: float,
        task_type: Optional[str] = None,
    ) -> None:
        """Record an observed reward for a strategy + task type combination.

        Args:
            strategy_name: Name of the strategy that produced the reward.
            reward: Scalar reward signal (typically score delta, clipped to
                ``[-1, 1]``).
            task_type: Task type identifier.
        """
        tt = task_type or self.default_task_type
        self._ensure_task_type(tt)
        self._stats[tt][strategy_name].update(reward)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise policy state to a plain dict (for warm-start saving)."""
        return {
            "strategy_names": self.strategy_names,
            "ucb_c": self.ucb_c,
            "default_task_type": self.default_task_type,
            "stats": {
                tt: {name: asdict(s) for name, s in strats.items()}
                for tt, strats in self._stats.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetaPolicy":
        """Restore a policy from a serialised dict.

        Args:
            data: Dict produced by ``to_dict()``.

        Returns:
            Restored ``MetaPolicy`` instance.
        """
        policy = cls(
            strategy_names=data["strategy_names"],
            ucb_c=data.get("ucb_c", 1.414),
            default_task_type=data.get("default_task_type", "general"),
        )
        for tt, strats in data.get("stats", {}).items():
            policy._stats[tt] = {
                name: StrategyStats(**s_data) for name, s_data in strats.items()
            }
        return policy


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@dataclass
class MetaTextGradHistory:
    """Tracks the progress of a MetaTextGrad optimization run.

    Attributes:
        score_per_step: Score after each gradient step.
        strategy_per_step: Strategy selected at each step.
        best_config: Configuration that achieved the best score.
        strategy_usage: How many times each strategy was selected.
    """

    score_per_step: List[float] = field(default_factory=list)
    strategy_per_step: List[str] = field(default_factory=list)
    best_config: Optional[Dict[str, Any]] = None
    strategy_usage: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MetaTextGradOptimizer
# ---------------------------------------------------------------------------

class MetaTextGradOptimizer(BaseOptimizer):
    """Meta-learning gradient optimizer for prompt tuning.

    Applies a textual gradient descent loop over the registered parameters,
    selecting the gradient computation strategy at each step via a UCB1 bandit
    meta-policy.

    Args:
        registry: Parameter registry.
        program: Callable that runs the workflow; returns a result dict.
        evaluator: Callable ``(result_dict[, example]) -> float`` in [0, 1].
        llm: ``BaseLLM`` instance for gradient computation.
        task_type: Short identifier for the task domain (e.g. ``"qa"``,
            ``"coding"``, ``"summarisation"``).  Used to condition the
            meta-policy's strategy selection.
        steps: Total number of gradient steps.
        training_examples: Optional list of examples used for evaluation.
        ucb_c: UCB exploration constant.
        meta_policy: Pre-initialised ``MetaPolicy`` for warm-starting.
        process_reward_evaluator: Optional ``StepwiseRewardEvaluator`` for
            per-step reward signals.
        cost_tracker: Optional ``CostTracker`` for budget enforcement.
        seed: Random seed.
    """

    def __init__(
        self,
        registry: ParamRegistry,
        program: Callable[..., Dict[str, Any]],
        evaluator: Callable[..., float],
        llm: Any,
        task_type: str = "general",
        steps: int = 30,
        training_examples: Optional[List[Dict[str, Any]]] = None,
        ucb_c: float = 1.414,
        meta_policy: Optional[MetaPolicy] = None,
        process_reward_evaluator: Optional[Any] = None,
        cost_tracker: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(registry=registry, program=program, evaluator=evaluator)

        self.llm = llm
        self.task_type = task_type
        self.steps = steps
        self.training_examples = training_examples or []
        self.cost_tracker = cost_tracker
        self.history = MetaTextGradHistory()

        self.meta_policy = meta_policy or MetaPolicy(
            strategy_names=STRATEGY_NAMES,
            ucb_c=ucb_c,
            default_task_type=task_type,
        )

        if process_reward_evaluator is not None and not _PRM_AVAILABLE:
            logger.warning(
                "metaTextGrad: process_reward_evaluator supplied but "
                "process_reward module is not importable — ignoring."
            )
        self.prm = process_reward_evaluator if _PRM_AVAILABLE else None

        if seed is not None:
            random.seed(seed)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def optimize(self) -> Tuple[Dict[str, Any], MetaTextGradHistory]:
        """Run the metaTextGrad optimization loop.

        Returns:
            Tuple of ``(best_config, history)``.
        """
        logger.info(
            "metaTextGrad: starting — task_type=%s  steps=%d",
            self.task_type,
            self.steps,
        )

        best_cfg = deepcopy(self.get_current_cfg())
        best_score = self._evaluate(best_cfg)
        self.history.score_per_step.append(best_score)
        self.history.best_config = deepcopy(best_cfg)

        for step in range(self.steps):
            strategy_name = self.meta_policy.select(
                task_type=self.task_type, step=step
            )
            self.history.strategy_per_step.append(strategy_name)
            self.history.strategy_usage[strategy_name] = (
                self.history.strategy_usage.get(strategy_name, 0) + 1
            )

            new_cfg = self._gradient_step(
                cfg=deepcopy(best_cfg),
                strategy_name=strategy_name,
                step=step,
            )
            new_score = self._evaluate(new_cfg)

            # Compute reward as score delta (clipped to [-1, 1])
            reward = max(-1.0, min(1.0, new_score - best_score))
            # If process reward model available, blend with PRM reward
            if self.prm is not None:
                prm_reward = self._prm_reward(new_cfg)
                reward = 0.6 * reward + 0.4 * prm_reward

            self.meta_policy.update(strategy_name, reward, self.task_type)

            if new_score > best_score:
                best_score = new_score
                best_cfg = new_cfg
                self.history.best_config = deepcopy(best_cfg)
                logger.info(
                    "metaTextGrad: step %d — improved to %.4f via '%s'",
                    step + 1,
                    best_score,
                    strategy_name,
                )
            else:
                logger.debug(
                    "metaTextGrad: step %d — no improvement (%.4f) via '%s'",
                    step + 1,
                    new_score,
                    strategy_name,
                )

            self.history.score_per_step.append(best_score)

            if self.cost_tracker is not None and _COST_TRACKER_AVAILABLE:
                try:
                    self.cost_tracker.check_budget()
                except Exception as budget_err:  # noqa: BLE001
                    logger.warning("metaTextGrad: cost budget exceeded — %s", budget_err)
                    break

        self.apply_cfg(best_cfg)
        logger.info(
            "metaTextGrad: done — best_score=%.4f  steps=%d",
            best_score,
            len(self.history.score_per_step) - 1,
        )
        return best_cfg, self.history

    def save_meta_policy(self, path: str) -> None:
        """Serialise the meta-policy to a JSON file for warm-starting.

        Args:
            path: File path to write the JSON to.
        """
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.meta_policy.to_dict(), fh, indent=2)
        logger.info("metaTextGrad: meta-policy saved to %s", path)

    def load_meta_policy(self, path: str) -> None:
        """Load a previously saved meta-policy from a JSON file.

        Args:
            path: File path to read the JSON from.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.meta_policy = MetaPolicy.from_dict(data)
        logger.info("metaTextGrad: meta-policy loaded from %s", path)

    # ------------------------------------------------------------------
    # Gradient step
    # ------------------------------------------------------------------

    def _gradient_step(
        self,
        cfg: Dict[str, Any],
        strategy_name: str,
        step: int,
    ) -> Dict[str, Any]:
        """Apply one gradient step to all parameters using the chosen strategy.

        Args:
            cfg: Current parameter configuration to improve.
            strategy_name: Name of the strategy to use.
            step: Current step number (for logging).

        Returns:
            Updated configuration dict.
        """
        strategy = _STRATEGIES.get(strategy_name)
        if strategy is None:
            logger.warning(
                "metaTextGrad: unknown strategy '%s' — falling back to stepwise.",
                strategy_name,
            )
            strategy = _STRATEGIES["stepwise"]

        # Collect feedback by running the current config
        feedback = self._collect_feedback(cfg)

        for name in self.param_names():
            current_val = str(cfg.get(name, ""))
            sys_prompt, user_prompt = strategy.build_prompt(
                param_name=name,
                current_value=current_val,
                evaluation_feedback=feedback,
                task_context=f"Task type: {self.task_type}",
            )
            try:
                response = self.llm.generate(
                    prompt=user_prompt,
                    system_prompt=sys_prompt,
                )
                cfg[name] = (response.content or current_val).strip()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "metaTextGrad: gradient LLM call failed for '%s' — %s",
                    name,
                    exc,
                )

        return cfg

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _evaluate(self, cfg: Dict[str, Any]) -> float:
        """Apply config and return mean score over training examples.

        Args:
            cfg: Parameter configuration to evaluate.

        Returns:
            Mean score in [0, 1].
        """
        self.apply_cfg(cfg)
        if not self.training_examples:
            try:
                result = self.program()
                return max(0.0, min(1.0, float(self.evaluator(result))))
            except Exception as exc:  # noqa: BLE001
                logger.warning("metaTextGrad: evaluation error — %s", exc)
                return 0.0

        scores: List[float] = []
        for example in self.training_examples:
            try:
                result = self.program(example)
                scores.append(float(self.evaluator(result, example)))
            except Exception as exc:  # noqa: BLE001
                logger.warning("metaTextGrad: evaluation error — %s", exc)
                scores.append(0.0)
        return sum(scores) / len(scores) if scores else 0.0

    def _collect_feedback(self, cfg: Dict[str, Any]) -> str:
        """Run the program and collect a textual feedback summary.

        This is used as the ``evaluation_feedback`` passed to gradient
        strategies.  If training examples are available, a small sample is
        used.  The feedback is a plain-text description of observed failures.

        Args:
            cfg: Current configuration to run.

        Returns:
            Feedback string.
        """
        self.apply_cfg(cfg)
        sample = (
            self.training_examples[:3]
            if self.training_examples
            else [{}]
        )
        observations: List[str] = []
        for example in sample:
            try:
                result = self.program(example) if example else self.program()
                score = float(self.evaluator(result, example) if example else self.evaluator(result))
                observations.append(f"score={score:.3f}")
            except Exception as exc:  # noqa: BLE001
                observations.append(f"error: {exc}")

        if observations:
            return f"Recent evaluation results: {', '.join(observations)}"
        return "No evaluation data available."

    def _prm_reward(self, cfg: Dict[str, Any]) -> float:
        """Obtain a process reward signal using the step-wise evaluator.

        Args:
            cfg: Configuration to evaluate.

        Returns:
            Normalised reward in [-1, 1].
        """
        if self.prm is None:
            return 0.0
        try:
            self.apply_cfg(cfg)
            result = self.program()
            trajectory = result.get("trajectory", [])
            if not trajectory:
                return 0.0
            ts = self.prm.score_trajectory(steps=trajectory)
            return max(-1.0, min(1.0, ts.mean_score * 2.0 - 1.0))
        except Exception as exc:  # noqa: BLE001
            logger.warning("metaTextGrad: PRM evaluation error — %s", exc)
            return 0.0

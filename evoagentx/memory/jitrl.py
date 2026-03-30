"""Just-in-Time Reinforcement Learning (JitRL) for AgentAugi.

JitRL complements Reflexion's verbal self-improvement with statistical
behavioral nudges. While Reflexion stores natural-language lessons ("I failed
because X, next time I should Y"), JitRL tracks *which actions lead to
success or failure across many episodes* and biases future action selection
toward historically successful patterns.

Research context: JitRL-style statistical nudging has demonstrated +19 points
over Reflexion alone on WebArena by grounding action selection in empirical
success rates rather than relying solely on verbal reasoning about past mistakes.

Design
------
``JitRLMemory`` is the central class. It:

1. Accepts completed ``TrajectoryStatistics`` objects and updates per-action
   statistics using temporal-difference credit assignment with discounting.
2. Provides ``get_action_bias`` to score available actions based on historical
   success rates, including a UCB-style exploration bonus for under-sampled
   actions.
3. Provides ``nudge_prompt`` to generate a natural-language hint suitable for
   injection into an agent's system prompt.
4. Supports JSON-based persistence across sessions.

``JitRLAgent`` wraps any agent (including a ``ReflexionAgent``) to inject
nudge prompts before execution and record the resulting trajectory afterward.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ..core.logging import logger


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ActionStatistics:
    """Per-action statistics accumulated across episodes.

    Attributes:
        action_type: Identifier for the action (e.g. ``"search"``, ``"click"``).
        context_hash: Hash of the recent-step context in which the action was
            taken.  Two calls are considered "in the same context" only when
            their hashes match, giving JitRL context-sensitivity.
        success_count: Number of episodes in which this (action, context) pair
            contributed to a successful outcome (after credit assignment).
        failure_count: Number of episodes in which this pair contributed to a
            failed outcome.
        total_reward: Accumulated discounted reward across all episodes.
        avg_reward: Running average reward (updated incrementally).
        last_updated: Wall-clock time of the most recent update.
    """

    action_type: str
    context_hash: str
    success_count: float = 0.0
    failure_count: float = 0.0
    total_reward: float = 0.0
    avg_reward: float = 0.0
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def total_count(self) -> float:
        """Total weighted episode count (success + failure)."""
        return self.success_count + self.failure_count

    @property
    def success_rate(self) -> float:
        """Empirical success rate in [0, 1], or 0.5 when no data is available."""
        if self.total_count == 0:
            return 0.5
        return self.success_count / self.total_count

    def apply_decay(self, decay_rate: float) -> None:
        """Exponentially decay counts so old episodes fade out over time.

        Args:
            decay_rate: Multiplicative factor applied to all count and reward
                accumulators.  Typical value: 0.99.
        """
        self.success_count *= decay_rate
        self.failure_count *= decay_rate
        self.total_reward *= decay_rate
        # avg_reward is recomputed from total_reward / total_count so it stays
        # consistent even after decay; guard against zero-division.
        if self.total_count > 0:
            self.avg_reward = self.total_reward / self.total_count
        else:
            self.avg_reward = 0.0


@dataclass
class TrajectoryStep:
    """A single step within a trajectory.

    Attributes:
        action_type: Identifier for the action taken.
        reward: Immediate reward received after this step (can be 0 for most
            steps with reward reserved for the terminal step).
        metadata: Optional free-form metadata (e.g. tool arguments).
    """

    action_type: str
    reward: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryStatistics:
    """Statistics for a completed agent trajectory.

    Attributes:
        steps: Ordered list of (action_type, reward) pairs.
        outcome: Terminal outcome label — one of ``"success"``, ``"failure"``,
            ``"partial"``, or ``"unknown"``.
        total_reward: Scalar summary reward for the whole trajectory.
    """

    steps: List[TrajectoryStep] = field(default_factory=list)
    outcome: str = "unknown"
    total_reward: float = 0.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class JitRLConfig:
    """Hyper-parameters for JitRL behaviour.

    Attributes:
        learning_rate: Step size used when blending new rewards into the
            running average.  Higher values make the system more reactive to
            recent episodes; lower values give more stable estimates.
        discount_factor: Temporal discount applied to upstream (earlier) steps
            when performing credit assignment.  Actions close to the terminal
            step receive the most credit.
        exploration_bonus: UCB-style additive bonus applied to actions that
            have been tried fewer than ``min_samples`` times, encouraging
            exploration of under-sampled alternatives.
        min_samples: Minimum number of weighted episode samples required before
            any nudge is generated for a (action, context) pair.  Below this
            threshold the statistics are too noisy to be informative.
        context_window: Number of recent trajectory steps used when computing
            the context hash.  Larger windows create finer-grained context keys
            at the cost of more key diversity (sparser statistics).
        decay_rate: Exponential decay multiplier applied to all statistics on
            every call to ``decay_statistics()``.  Values close to 1.0 retain
            history for longer; values close to 0.0 forget quickly.
    """

    learning_rate: float = 0.1
    discount_factor: float = 0.95
    exploration_bonus: float = 0.1
    min_samples: int = 5
    context_window: int = 3
    decay_rate: float = 0.99


# ---------------------------------------------------------------------------
# JitRLMemory
# ---------------------------------------------------------------------------


class JitRLMemory:
    """Statistical action-selection memory based on historical trajectory data.

    JitRLMemory accumulates per-action success/failure statistics across many
    episodes and uses them to produce scoring biases and natural-language nudges
    that can be injected into an agent's prompt before each step.

    This is complementary to ``ReflexionMemory``: Reflexion provides *verbal*
    lessons from past failures; JitRLMemory provides *statistical* evidence
    about which actions tend to succeed in which contexts.

    Args:
        config: ``JitRLConfig`` with all hyper-parameters.
        persistence_path: Optional path for JSON-based cross-session storage.
    """

    def __init__(
        self,
        config: Optional[JitRLConfig] = None,
        persistence_path: Optional[str] = None,
    ) -> None:
        self.config = config or JitRLConfig()
        self.persistence_path = persistence_path
        # Key: "<action_type>|<context_hash>" → ActionStatistics
        self._stats: Dict[str, ActionStatistics] = {}
        self._trajectory_count: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_key(self, action_type: str, context_hash: str) -> str:
        return f"{action_type}|{context_hash}"

    def _get_or_create(self, action_type: str, context_hash: str) -> ActionStatistics:
        key = self._make_key(action_type, context_hash)
        if key not in self._stats:
            self._stats[key] = ActionStatistics(
                action_type=action_type,
                context_hash=context_hash,
            )
        return self._stats[key]

    def _compute_context_hash(self, recent_steps: List[str]) -> str:
        """Hash a sequence of recent action types into a fixed-length key.

        Only the last ``config.context_window`` entries are used so the hash
        stays bounded regardless of trajectory length.

        Args:
            recent_steps: Ordered list of action-type strings.

        Returns:
            8-character hex digest uniquely representing the context window.
        """
        window = recent_steps[-self.config.context_window :]
        combined = "|".join(window)
        return hashlib.sha256(combined.encode()).hexdigest()[:8]

    def _temporal_credit_assignment(
        self, trajectory: TrajectoryStatistics
    ) -> List[Tuple[str, float]]:
        """Distribute terminal reward backward through the trajectory.

        Actions near the end receive more credit; earlier actions receive
        discounted credit following ``γ^(T-t)`` where ``T`` is the last step
        index and ``t`` is the current step index.

        For *successful* outcomes the terminal reward is 1.0; for *failed*
        outcomes it is 0.0; for *partial* or *unknown* the base is 0.5.

        Any per-step reward recorded in the trajectory is additive to the
        discounted terminal credit.

        Args:
            trajectory: A completed trajectory with steps and outcome.

        Returns:
            List of ``(action_type, credit)`` pairs aligned with trajectory
            steps.
        """
        outcome_map = {
            "success": 1.0,
            "failure": 0.0,
            "partial": 0.5,
            "unknown": 0.5,
        }
        terminal_reward = outcome_map.get(trajectory.outcome.lower(), 0.5)

        n = len(trajectory.steps)
        if n == 0:
            return []

        credits: List[Tuple[str, float]] = []
        for t, step in enumerate(trajectory.steps):
            # Discount from the end: step at index t is n-1-t steps before the end.
            steps_before_end = (n - 1) - t
            discounted = terminal_reward * (self.config.discount_factor ** steps_before_end)
            total_credit = discounted + step.reward
            # Clip to [0, 1] range since we add per-step rewards that may push above 1.
            total_credit = max(0.0, min(1.0, total_credit))
            credits.append((step.action_type, total_credit))

        return credits

    def _ucb_exploration_bonus(self, stat: ActionStatistics) -> float:
        """Compute a UCB1-style exploration bonus for an under-sampled action.

        Returns the configured ``exploration_bonus`` value when the action has
        fewer than ``min_samples`` weighted observations, else 0.0.

        Args:
            stat: The action's current statistics.

        Returns:
            Non-negative exploration bonus.
        """
        if stat.total_count < self.config.min_samples:
            return self.config.exploration_bonus
        return 0.0

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------

    def record_trajectory(self, trajectory: TrajectoryStatistics) -> None:
        """Update action statistics from a completed trajectory.

        Applies temporal-difference credit assignment: actions closer to the
        terminal step receive more credit on success.  Updates are blended into
        the running average using the configured ``learning_rate``.

        Args:
            trajectory: A completed ``TrajectoryStatistics`` object.
        """
        if not trajectory.steps:
            logger.debug("JitRLMemory.record_trajectory: empty trajectory; skipping.")
            return

        credits = self._temporal_credit_assignment(trajectory)
        recent_actions: List[str] = []

        for action_type, credit in credits:
            context_hash = self._compute_context_hash(recent_actions)
            stat = self._get_or_create(action_type, context_hash)

            is_success = trajectory.outcome.lower() == "success"
            is_failure = trajectory.outcome.lower() == "failure"

            # Weight each step's contribution by the discounted credit value.
            if is_success:
                stat.success_count += credit
            elif is_failure:
                stat.failure_count += (1.0 - credit)
                # Still track partial failure signal.
                stat.failure_count = max(stat.failure_count, 0.0)
            else:
                # Partial / unknown: fractional contribution to both.
                stat.success_count += credit * 0.5
                stat.failure_count += (1.0 - credit) * 0.5

            # Blend reward into running average.
            old_avg = stat.avg_reward
            stat.total_reward += credit
            stat.avg_reward = old_avg + self.config.learning_rate * (credit - old_avg)
            stat.last_updated = datetime.now(timezone.utc).isoformat()

            recent_actions.append(action_type)

        self._trajectory_count += 1
        logger.debug(
            "JitRLMemory recorded trajectory #%d (outcome=%s, steps=%d)",
            self._trajectory_count,
            trajectory.outcome,
            len(trajectory.steps),
        )

    def get_action_bias(
        self,
        recent_steps: List[str],
        available_actions: List[str],
    ) -> Dict[str, float]:
        """Score available actions based on historical success rates.

        Actions with a high empirical success rate in the current context
        receive positive bias; actions that tend to fail receive negative bias;
        actions with insufficient data receive an exploration bonus.

        The bias is centered around 0.0: a perfectly neutral action (50%
        success rate with no exploration bonus) scores 0.0.

        Args:
            recent_steps: The action-type strings of the most recent steps
                (used to compute the context hash).
            available_actions: List of action-type identifiers to score.

        Returns:
            Dictionary mapping each action to a bias scalar.  Values are in
            approximately ``[-0.5, 0.5 + exploration_bonus]``.
        """
        context_hash = self._compute_context_hash(recent_steps)
        biases: Dict[str, float] = {}

        for action in available_actions:
            key = self._make_key(action, context_hash)
            if key in self._stats:
                stat = self._stats[key]
                # Center at 0: success_rate of 0.5 → bias of 0.0
                base_bias = stat.success_rate - 0.5
                exploration = self._ucb_exploration_bonus(stat)
                biases[action] = base_bias + exploration
            else:
                # Never seen this action in this context → pure exploration bonus.
                biases[action] = self.config.exploration_bonus

        return biases

    def nudge_prompt(
        self,
        recent_steps: List[str],
        available_actions: List[str],
        max_actions_shown: int = 5,
    ) -> str:
        """Generate a natural-language hint to inject into the agent's prompt.

        Only actions with at least ``min_samples`` observations are included in
        the nudge.  Actions without enough data are silently omitted (the
        exploration bonus in ``get_action_bias`` handles them implicitly).

        Args:
            recent_steps: Recent action-type strings (for context hashing).
            available_actions: Candidate actions to describe.
            max_actions_shown: Maximum number of actions mentioned in the hint.

        Returns:
            A formatted string ready for system-prompt injection, or an empty
            string if no action has accumulated enough statistics.
        """
        context_hash = self._compute_context_hash(recent_steps)

        eligible: List[Tuple[str, ActionStatistics]] = []
        for action in available_actions:
            key = self._make_key(action, context_hash)
            if key in self._stats:
                stat = self._stats[key]
                if stat.total_count >= self.config.min_samples:
                    eligible.append((action, stat))

        if not eligible:
            return ""

        # Sort descending by success rate so the most informative actions come first.
        eligible.sort(key=lambda x: x[1].success_rate, reverse=True)
        eligible = eligible[:max_actions_shown]

        lines = ["Based on past experience in similar contexts:"]
        for action, stat in eligible:
            pct = int(stat.success_rate * 100)
            n = int(stat.total_count)
            if stat.success_rate >= 0.6:
                verdict = "has been successful"
            elif stat.success_rate <= 0.4:
                verdict = "has struggled"
            else:
                verdict = "has produced mixed results"
            lines.append(f"  - Action '{action}' {verdict} ({pct}% success rate over {n} episodes).")

        return "\n".join(lines)

    def decay_statistics(self) -> None:
        """Apply exponential decay to all stored statistics.

        Prevents stale data from dominating by multiplying all count and reward
        accumulators by ``config.decay_rate`` on each call.  Old, low-count
        entries may eventually be pruned when their ``total_count`` drops below
        a negligible threshold.
        """
        to_prune: List[str] = []
        for key, stat in self._stats.items():
            stat.apply_decay(self.config.decay_rate)
            if stat.total_count < 1e-4:
                to_prune.append(key)

        for key in to_prune:
            del self._stats[key]

        if to_prune:
            logger.debug(
                "JitRLMemory pruned %d near-zero statistics entries after decay.",
                len(to_prune),
            )

    def stats(self) -> Dict[str, Any]:
        """Return a summary of current memory state.

        Returns:
            Dictionary with keys:
            - ``total_trajectories``: number of trajectories recorded.
            - ``action_coverage``: number of unique (action, context) keys.
            - ``top_actions``: up to 5 (action_type, success_rate) pairs with
              the highest success rates (among entries with enough data).
            - ``bottom_actions``: up to 5 pairs with the lowest success rates.
        """
        all_stats = list(self._stats.values())
        mature = [s for s in all_stats if s.total_count >= self.config.min_samples]
        mature_sorted = sorted(mature, key=lambda s: s.success_rate, reverse=True)

        top = [(s.action_type, round(s.success_rate, 3)) for s in mature_sorted[:5]]
        bottom = [(s.action_type, round(s.success_rate, 3)) for s in mature_sorted[-5:]]
        if mature_sorted:
            bottom = [(s.action_type, round(s.success_rate, 3)) for s in mature_sorted[-5:][::-1]]

        return {
            "total_trajectories": self._trajectory_count,
            "action_coverage": len(self._stats),
            "top_actions": top,
            "bottom_actions": bottom,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> None:
        """Persist all statistics to a JSON file.

        Args:
            path: Override path; falls back to ``self.persistence_path`` if
                ``None``.  Raises ``ValueError`` when neither is set.
        """
        target = path or self.persistence_path
        if not target:
            raise ValueError("No persistence path configured for JitRLMemory.save().")

        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)

        payload = {
            "trajectory_count": self._trajectory_count,
            "config": asdict(self.config),
            "stats": {key: asdict(stat) for key, stat in self._stats.items()},
        }
        with open(target, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        logger.debug(
            "JitRLMemory saved %d entries to %s", len(self._stats), target
        )

    def load(self, path: Optional[str] = None) -> None:
        """Load statistics from a JSON file, merging with any in-memory data.

        Existing keys are overwritten by the loaded data.

        Args:
            path: Override path; falls back to ``self.persistence_path`` if
                ``None``.  No-op if the file does not exist.
        """
        target = path or self.persistence_path
        if not target or not os.path.exists(target):
            return

        with open(target, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        self._trajectory_count = payload.get("trajectory_count", 0)

        for key, raw in payload.get("stats", {}).items():
            try:
                self._stats[key] = ActionStatistics(**raw)
            except (TypeError, KeyError) as exc:
                logger.warning("JitRLMemory: skipping malformed stats entry %s: %s", key, exc)

        logger.debug(
            "JitRLMemory loaded %d entries from %s", len(self._stats), target
        )


# ---------------------------------------------------------------------------
# JitRLAgent wrapper
# ---------------------------------------------------------------------------


class JitRLAgent:
    """Decorator that adds JitRL statistical nudging to any agent.

    Wraps an existing agent (or a ``ReflexionAgent`` for combined
    Reflexion+JitRL behaviour) without modifying its class.

    Before each ``execute()`` call the wrapper:

    1. Computes ``nudge_prompt`` based on recent trajectory context.
    2. Prepends the nudge to the agent's system prompt.

    After execution the wrapper:

    3. Records a ``TrajectoryStatistics`` object with the outcome and step
       rewards.

    Usage::

        base_agent = MyAgent(...)
        jitrl_memory = JitRLMemory(persistence_path="/tmp/jitrl.json")
        agent = JitRLAgent(agent=base_agent, memory=jitrl_memory)
        result = agent.execute(
            task_steps=["search", "click"],
            outcome="success",
        )

    To compose with Reflexion::

        reflexion = ReflexionAgent(agent=base_agent, memory=ReflexionMemory())
        agent = JitRLAgent(agent=reflexion, memory=jitrl_memory)

    Attributes:
        agent: The wrapped agent instance.
        memory: ``JitRLMemory`` used to store and retrieve statistics.
        available_actions: Optional default list of candidate action types
            (can be overridden per call).
    """

    def __init__(
        self,
        agent: Any,
        memory: JitRLMemory,
        available_actions: Optional[List[str]] = None,
    ) -> None:
        self.agent = agent
        self.memory = memory
        self.available_actions = available_actions or []
        self._recent_steps: List[str] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _inject_nudge(
        self,
        recent_steps: List[str],
        available_actions: List[str],
    ) -> Optional[str]:
        """Prepend nudge prompt to the agent's system_prompt.

        Returns the original system_prompt so it can be restored afterward.
        """
        original = getattr(self.agent, "system_prompt", None)
        nudge = self.memory.nudge_prompt(recent_steps, available_actions)
        if nudge:
            patched = f"{nudge}\n\n{original}" if original else nudge
            try:
                self.agent.system_prompt = patched
            except Exception:
                pass
        return original

    def _restore_prompt(self, original: Optional[str]) -> None:
        try:
            self.agent.system_prompt = original
        except Exception:
            pass

    def _record(
        self,
        steps: List[str],
        outcome: str,
        step_rewards: Optional[List[float]] = None,
    ) -> None:
        """Build and record a TrajectoryStatistics from the completed run."""
        rewards = step_rewards or [0.0] * len(steps)
        # Pad or trim rewards to match step count.
        if len(rewards) < len(steps):
            rewards = rewards + [0.0] * (len(steps) - len(rewards))
        traj_steps = [
            TrajectoryStep(action_type=a, reward=r)
            for a, r in zip(steps, rewards)
        ]
        total_reward = sum(rewards)
        trajectory = TrajectoryStatistics(
            steps=traj_steps,
            outcome=outcome,
            total_reward=total_reward,
        )
        self.memory.record_trajectory(trajectory)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def execute(
        self,
        task_steps: Optional[List[str]] = None,
        outcome: str = "unknown",
        step_rewards: Optional[List[float]] = None,
        available_actions: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute the wrapped agent with JitRL nudging (synchronous).

        Args:
            task_steps: Ordered list of action-type identifiers the agent will
                take.  Used as both the context for nudge generation *and* the
                steps recorded post-execution.  When ``None``, ``self._recent_steps``
                is used for context.
            outcome: Terminal outcome label (``"success"``, ``"failure"``,
                ``"partial"``, ``"unknown"``).
            step_rewards: Optional per-step reward signals aligned with
                ``task_steps``.
            available_actions: Override the default ``self.available_actions``
                for nudge generation.
            **kwargs: Forwarded to the wrapped agent's ``execute()`` method.

        Returns:
            The wrapped agent's return value.
        """
        steps = task_steps or self._recent_steps
        actions = available_actions or self.available_actions

        original_prompt = self._inject_nudge(steps, actions)
        result = None
        try:
            result = self.agent.execute(**kwargs)
        finally:
            self._restore_prompt(original_prompt)
            if steps:
                self._record(steps, outcome, step_rewards)
                self._recent_steps = list(steps)

        return result

    async def async_execute(
        self,
        task_steps: Optional[List[str]] = None,
        outcome: str = "unknown",
        step_rewards: Optional[List[float]] = None,
        available_actions: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute the wrapped agent with JitRL nudging (asynchronous).

        Args:
            task_steps: Action-type identifiers for this trajectory.
            outcome: Terminal outcome label.
            step_rewards: Per-step rewards aligned with ``task_steps``.
            available_actions: Override default available action list.
            **kwargs: Forwarded to the wrapped agent's ``async_execute()``.

        Returns:
            The wrapped agent's return value.
        """
        steps = task_steps or self._recent_steps
        actions = available_actions or self.available_actions

        original_prompt = self._inject_nudge(steps, actions)
        result = None
        try:
            result = await self.agent.async_execute(**kwargs)
        finally:
            self._restore_prompt(original_prompt)
            if steps:
                self._record(steps, outcome, step_rewards)
                self._recent_steps = list(steps)

        return result

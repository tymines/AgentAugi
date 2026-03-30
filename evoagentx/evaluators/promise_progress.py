"""Promise and Progress signals for AgentPRM-style process reward modeling.

These two signals are complementary and target different optimizer components:

``Promise``
    Estimated probability of eventual task success from the *current*
    trajectory state.  High promise means "the agent is on track and likely
    to succeed from here."  Feeds into MCTS value estimation (AFlow node
    values) and can bootstrap value functions without waiting for rollout
    completion.

``Progress``
    Delta improvement achieved by the *most recent action* relative to the
    state before it.  High progress means the last step materially advanced
    the goal.  Feeds into gradient computation (TextGrad step weights) and
    the fitness function (EvoPrompt: ``fitness = α*outcome + β*mean(progress)``).

Both signals are normalised to [0.0, 1.0].

Usage
-----
    >>> from evoagentx.evaluators.promise_progress import PromiseProgressEvaluator
    >>> evaluator = PromiseProgressEvaluator(llm=my_llm)
    >>> result = evaluator.evaluate(trajectory, context="Solve this math problem")
    >>> print(result.promise)   # 0.85 — agent is on track
    >>> print(result.progress)  # 0.40 — last step made modest headway
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.logging import logger
from ..models.base_model import BaseLLM


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PromiseProgressScore:
    """Promise and Progress scores for a trajectory state.

    Attributes:
        promise: Estimated probability (0–1) of eventual task success from
            the current trajectory state.
        progress: Normalised improvement (0–1) achieved by the most recent
            action compared to the previous state.
        promise_rationale: Explanation of the promise score.
        progress_rationale: Explanation of the progress score.
        step_index: Zero-based index of the step these scores describe.
    """

    promise: float
    progress: float
    promise_rationale: str = ""
    progress_rationale: str = ""
    step_index: int = 0

    def __post_init__(self) -> None:
        for name, val in (("promise", self.promise), ("progress", self.progress)):
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"PromiseProgressScore.{name} must be in [0, 1], got {val}"
                )

    @property
    def composite(self) -> float:
        """Equal-weight combination of promise and progress."""
        return 0.5 * self.promise + 0.5 * self.progress


@dataclass
class TrajectoryPromiseProgress:
    """Per-step promise+progress scores for an entire trajectory.

    Attributes:
        scores: Ordered list of ``PromiseProgressScore`` objects, one per step.
    """

    scores: List[PromiseProgressScore] = field(default_factory=list)

    @property
    def promises(self) -> List[float]:
        """All promise values as a plain list."""
        return [s.promise for s in self.scores]

    @property
    def progresses(self) -> List[float]:
        """All progress values as a plain list."""
        return [s.progress for s in self.scores]

    @property
    def mean_promise(self) -> float:
        """Average promise across the trajectory, or 0 if empty."""
        return sum(self.promises) / len(self.promises) if self.promises else 0.0

    @property
    def mean_progress(self) -> float:
        """Average progress across the trajectory, or 0 if empty."""
        return sum(self.progresses) / len(self.progresses) if self.progresses else 0.0

    @property
    def final_promise(self) -> float:
        """Promise score of the last step (most informative for AFlow)."""
        return self.scores[-1].promise if self.scores else 0.0

    def fitness_score(self, alpha: float = 0.7, beta: float = 0.3) -> float:
        """Compute a combined fitness useful as an EvoPrompt fitness component.

        The default weighting (α=0.7, β=0.3) prioritises promise (goal
        alignment) over average progress (step-level effort), following the
        tuning suggestion in the master implementation plan.

        Args:
            alpha: Weight for final promise.
            beta: Weight for mean progress.

        Returns:
            Scalar in [0, 1].
        """
        return alpha * self.final_promise + beta * self.mean_progress


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BasePromiseProgressEvaluator(ABC):
    """Abstract evaluator producing promise and progress signals."""

    @abstractmethod
    def evaluate(
        self,
        trajectory: List[Dict[str, Any]],
        context: str = "",
    ) -> TrajectoryPromiseProgress:
        """Compute promise and progress for every step of a trajectory.

        Args:
            trajectory: Ordered list of step dicts, each with at minimum:
                - ``"type"``: ``"tool_call"`` | ``"reasoning"`` | ``"unknown"``
                - ``"content"``: textual description of what happened.
            context: High-level task / goal description.

        Returns:
            ``TrajectoryPromiseProgress`` with one entry per step.
        """

    @abstractmethod
    def evaluate_step(
        self,
        step: Dict[str, Any],
        preceding: List[Dict[str, Any]],
        context: str = "",
    ) -> PromiseProgressScore:
        """Score a single step in isolation.

        Args:
            step: The step to evaluate.
            preceding: Steps that came before this one.
            context: High-level task / goal description.

        Returns:
            ``PromiseProgressScore`` for this step.
        """


# ---------------------------------------------------------------------------
# LLM-based implementation
# ---------------------------------------------------------------------------

_PROMISE_PROGRESS_PROMPT = """\
You are evaluating an agent's trajectory step by step.

## Task context
{context}

## Steps completed so far (most recent last)
{history}

## Current step
Type: {step_type}
Content: {step_content}

Answer the following two questions, each on a scale from 0.0 to 1.0:

1. **Promise** — Given everything the agent has done so far *including this step*,
   how likely is it that the agent will successfully complete the task?
   (0.0 = will almost certainly fail, 1.0 = will almost certainly succeed)

2. **Progress** — How much did *this specific step* advance the agent toward the
   goal compared to where it was before this step?
   (0.0 = no progress or harmful, 0.5 = modest improvement, 1.0 = major breakthrough)

Respond with a JSON object on a single line with exactly four keys:
  "promise": <float 0–1>
  "promise_rationale": "<one sentence>"
  "progress": <float 0–1>
  "progress_rationale": "<one sentence>"

Example:
{{"promise": 0.7, "promise_rationale": "Agent has the right approach but hasn't verified output yet.", "progress": 0.6, "progress_rationale": "Retrieved relevant context that narrows the search space."}}
"""


class LLMPromiseProgressEvaluator(BasePromiseProgressEvaluator):
    """Concrete evaluator that uses an LLM judge for promise and progress.

    Args:
        llm: Language model used for scoring.
        history_window: Maximum number of preceding steps to include in the
            prompt.  Larger windows give more context but cost more tokens.
        fallback_promise: Promise score to return on parse failure.
        fallback_progress: Progress score to return on parse failure.
    """

    def __init__(
        self,
        llm: BaseLLM,
        history_window: int = 5,
        fallback_promise: float = 0.5,
        fallback_progress: float = 0.5,
    ) -> None:
        self.llm = llm
        self.history_window = history_window
        self.fallback_promise = fallback_promise
        self.fallback_progress = fallback_progress

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """Issue a single LLM call and return the raw text."""
        response = self.llm.generate(prompt)
        if hasattr(response, "content"):
            return response.content
        return str(response)

    def _format_history(self, preceding: List[Dict[str, Any]]) -> str:
        """Serialise preceding steps for prompt inclusion."""
        window = preceding[-self.history_window :]
        if not window:
            return "(none)"
        lines = []
        for i, s in enumerate(window):
            lines.append(
                f"[{i + 1}] ({s.get('type', '?')}) {s.get('content', '')}"
            )
        return "\n".join(lines)

    def _parse_response(
        self, raw: str, step_index: int
    ) -> PromiseProgressScore:
        """Extract promise and progress from the LLM JSON response."""
        text = raw.strip()
        # Strip markdown fences
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(l for l in lines if not l.startswith("```")).strip()

        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            logger.warning(
                "PromiseProgressEvaluator: no JSON found in LLM response; "
                "using fallback. Raw: %s", raw[:200]
            )
            return self._fallback_score(step_index)

        try:
            data = json.loads(text[start:end])
            promise = max(0.0, min(1.0, float(data.get("promise", self.fallback_promise))))
            progress = max(0.0, min(1.0, float(data.get("progress", self.fallback_progress))))
            return PromiseProgressScore(
                promise=promise,
                progress=progress,
                promise_rationale=str(data.get("promise_rationale", "")),
                progress_rationale=str(data.get("progress_rationale", "")),
                step_index=step_index,
            )
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning(
                "PromiseProgressEvaluator: parse error (%s); using fallback. "
                "Raw: %s", exc, raw[:200]
            )
            return self._fallback_score(step_index)

    def _fallback_score(self, step_index: int) -> PromiseProgressScore:
        """Return a neutral fallback when scoring fails."""
        return PromiseProgressScore(
            promise=self.fallback_promise,
            progress=self.fallback_progress,
            promise_rationale="(parse error — fallback)",
            progress_rationale="(parse error — fallback)",
            step_index=step_index,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_step(
        self,
        step: Dict[str, Any],
        preceding: List[Dict[str, Any]],
        context: str = "",
    ) -> PromiseProgressScore:
        """Score a single step's promise and progress.

        Args:
            step: The step dict to evaluate.
            preceding: Steps executed before this one.
            context: High-level task description.

        Returns:
            ``PromiseProgressScore`` for this step.
        """
        step_index = len(preceding)
        prompt = _PROMISE_PROGRESS_PROMPT.format(
            context=context or "(no context provided)",
            history=self._format_history(preceding),
            step_type=step.get("type", "unknown"),
            step_content=step.get("content", ""),
        )
        raw = self._call_llm(prompt)
        result = self._parse_response(raw, step_index=step_index)
        return result

    def evaluate(
        self,
        trajectory: List[Dict[str, Any]],
        context: str = "",
    ) -> TrajectoryPromiseProgress:
        """Compute promise and progress for every step of a trajectory.

        Args:
            trajectory: Ordered list of step dicts.
            context: High-level task / goal description.

        Returns:
            ``TrajectoryPromiseProgress`` with one ``PromiseProgressScore``
            per step.
        """
        scores: List[PromiseProgressScore] = []
        for idx, step in enumerate(trajectory):
            preceding = trajectory[:idx]
            score = self.evaluate_step(
                step=step,
                preceding=preceding,
                context=context,
            )
            score.step_index = idx
            scores.append(score)
        return TrajectoryPromiseProgress(scores=scores)


# ---------------------------------------------------------------------------
# Aggregation utilities for optimizer integration
# ---------------------------------------------------------------------------

def trajectory_to_textgrad_weights(
    tpp: TrajectoryPromiseProgress,
) -> List[float]:
    """Convert trajectory promise+progress scores to TextGrad step weights.

    TextGrad's backward pass propagates through each step; steps with higher
    combined scores contribute more to the gradient signal.

    Args:
        tpp: Scored trajectory from ``LLMPromiseProgressEvaluator.evaluate``.

    Returns:
        List of normalised weights (sum to 1.0) aligned with trajectory steps.
        Returns an empty list when the trajectory is empty.
    """
    if not tpp.scores:
        return []
    raw = [s.composite for s in tpp.scores]
    total = sum(raw)
    if total == 0.0:
        n = len(raw)
        return [1.0 / n] * n
    return [w / total for w in raw]


def trajectory_to_evoprompt_fitness(
    tpp: TrajectoryPromiseProgress,
    outcome_score: float,
    alpha: float = 0.7,
    beta: float = 0.3,
) -> float:
    """Blend a process reward signal into an EvoPrompt candidate fitness.

    Follows the formula from the master plan:
        ``fitness = α × outcome_score + β × mean(step_rewards)``

    The step reward is taken as the mean composite (promise+progress) score
    across all trajectory steps.

    Args:
        tpp: Scored trajectory.
        outcome_score: Final-outcome evaluation score in [0, 1].
        alpha: Weight for the outcome component (default 0.7).
        beta: Weight for the process component (default 0.3).

    Returns:
        Combined fitness scalar in [0, 1].
    """
    process_score = sum(s.composite for s in tpp.scores) / max(1, len(tpp.scores))
    return alpha * outcome_score + beta * process_score


def trajectory_to_aflow_node_value(
    tpp: TrajectoryPromiseProgress,
) -> float:
    """Estimate an AFlow MCTS node value from trajectory promise+progress.

    AFlow expands MCTS nodes; this value estimate lets the tree search
    prioritise promising branches without waiting for full rollout completion.

    Returns the final promise score, which directly represents "probability
    of success from this node."

    Args:
        tpp: Scored trajectory ending at the candidate node.

    Returns:
        Node value estimate in [0, 1].
    """
    return tpp.final_promise

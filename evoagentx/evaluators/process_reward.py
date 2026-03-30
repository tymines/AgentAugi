"""Step-wise process reward evaluation for trajectory-based optimization.

Inspired by AgentPRM (arXiv:2511.08325) and ToolRM (arXiv:2510.26167), this
module provides fine-grained quality signals for each reasoning or tool-use step
in an agent trajectory, rather than scoring only the final outcome.

Empirical motivation: step-wise supervision achieves ~78% vs ~34% on hard
multi-step reasoning tasks compared to outcome-only evaluation. By exposing
per-step scores to TextGrad, EvoPrompt, and AFlow, each optimizer can receive
a richer gradient/fitness signal even when final-outcome evaluation is noisy.

Design
------
The core abstraction is ``StepwiseRewardEvaluator``, an abstract base class
that defines three scoring entry-points:

1. ``score_tool_call`` — was this the right tool to call at this point in the
   trajectory?
2. ``score_reasoning_step`` — is this reasoning intermediate sound and useful?
3. ``score_trajectory`` — produce a score for every step in a full trace.

``LLMStepwiseRewardEvaluator`` is the concrete implementation: it serialises
each step's context into a prompt and asks the LLM to rate quality on [0, 1].
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
class StepScore:
    """Quality score for a single trajectory step.

    Attributes:
        score: Normalised quality value in [0.0, 1.0]. Higher is better.
        rationale: Human-readable explanation produced alongside the score.
        step_index: Zero-based position of this step within its trajectory.
        step_type: One of ``"tool_call"``, ``"reasoning"``, or ``"unknown"``.
    """

    score: float
    rationale: str = ""
    step_index: int = 0
    step_type: str = "unknown"

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(
                f"StepScore.score must be in [0, 1], got {self.score}"
            )


@dataclass
class TrajectoryScores:
    """Collection of per-step scores for an entire execution trajectory.

    Attributes:
        step_scores: Ordered list of ``StepScore`` objects.
        mean_score: Average of all step scores (lazy-computed).
        min_score: Worst step score (identifies bottleneck steps).
        max_score: Best step score.
    """

    step_scores: List[StepScore] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        """Average of all step scores, or 0.0 when there are no steps."""
        if not self.step_scores:
            return 0.0
        return sum(s.score for s in self.step_scores) / len(self.step_scores)

    @property
    def min_score(self) -> float:
        """Score of the worst step, or 0.0 when there are no steps."""
        if not self.step_scores:
            return 0.0
        return min(s.score for s in self.step_scores)

    @property
    def max_score(self) -> float:
        """Score of the best step, or 0.0 when there are no steps."""
        if not self.step_scores:
            return 0.0
        return max(s.score for s in self.step_scores)

    def as_float_list(self) -> List[float]:
        """Return step scores as a plain list of floats (for optimizer APIs)."""
        return [s.score for s in self.step_scores]

    def weighted_score(self, alpha: float = 0.5, beta: float = 0.5) -> float:
        """Combine mean and min scores.

        Args:
            alpha: Weight for the mean score component.
            beta: Weight for the min score (bottleneck-aware) component.

        Returns:
            Scalar composite score in [0, 1].
        """
        return alpha * self.mean_score + beta * self.min_score


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class StepwiseRewardEvaluator(ABC):
    """Abstract evaluator that scores individual reasoning or tool-use steps.

    Subclasses implement the three abstract methods to provide scores that
    can be consumed by any of the five optimizers (TextGrad, AFlow, SEW,
    MIPRO, EvoPrompt) for richer feedback than outcome-only evaluation.

    All scores are normalised to [0.0, 1.0].
    """

    @abstractmethod
    def score_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        context: str,
        trajectory_so_far: List[Dict[str, Any]],
    ) -> StepScore:
        """Score whether this tool call was appropriate in context.

        Args:
            tool_name: Name of the tool that was invoked.
            args: Arguments passed to the tool at call time.
            context: High-level description of the current task / goal.
            trajectory_so_far: Preceding steps (dicts with at minimum
                ``"type"`` and ``"content"`` keys).

        Returns:
            ``StepScore`` with quality in [0, 1] and a short rationale.
        """

    @abstractmethod
    def score_reasoning_step(
        self,
        step_text: str,
        context: str,
    ) -> StepScore:
        """Score whether a single reasoning intermediate is sound and useful.

        Args:
            step_text: The text of the reasoning step (chain-of-thought chunk,
                intermediate conclusion, etc.).
            context: High-level description of the current task / goal.

        Returns:
            ``StepScore`` with quality in [0, 1] and a short rationale.
        """

    @abstractmethod
    def score_trajectory(
        self,
        trajectory: List[Dict[str, Any]],
        context: str = "",
    ) -> TrajectoryScores:
        """Score every step in a full execution trace.

        Args:
            trajectory: Ordered list of step dicts.  Each dict must contain:
                - ``"type"``: one of ``"tool_call"`` or ``"reasoning"``.
                - ``"content"``: textual description / reasoning text.
                - (for tool calls) ``"tool_name"`` and ``"args"``.
            context: Overall task / goal description.

        Returns:
            ``TrajectoryScores`` containing one ``StepScore`` per step.
        """


# ---------------------------------------------------------------------------
# LLM-based concrete implementation
# ---------------------------------------------------------------------------

_TOOL_CALL_PROMPT = """\
You are evaluating the quality of a single tool-call step in an agent trajectory.

## Task context
{context}

## Steps taken so far (most recent last)
{history}

## Current tool call
Tool: {tool_name}
Arguments: {args}

Rate the quality of this tool call on a scale from 0.0 (completely wrong /
harmful / irrelevant) to 1.0 (optimal choice given the context).

Respond with a JSON object on a single line with exactly two keys:
  "score": <float between 0.0 and 1.0>
  "rationale": "<one sentence explanation>"

Example: {{"score": 0.8, "rationale": "Good tool choice but argument slightly off."}}
"""

_REASONING_PROMPT = """\
You are evaluating the quality of a single reasoning step produced by an agent.

## Task context
{context}

## Reasoning step to evaluate
{step_text}

Rate the quality of this reasoning step on a scale from 0.0 (logically flawed,
irrelevant, or harmful) to 1.0 (correct, useful, and clearly advances the goal).

Respond with a JSON object on a single line with exactly two keys:
  "score": <float between 0.0 and 1.0>
  "rationale": "<one sentence explanation>"

Example: {{"score": 0.75, "rationale": "Sound logic but misses an edge case."}}
"""


class LLMStepwiseRewardEvaluator(StepwiseRewardEvaluator):
    """Concrete evaluator that uses an LLM judge to score trajectory steps.

    This implementation serialises each step into a structured prompt and
    parses the LLM's JSON response to extract a numeric score.

    Args:
        llm: The language model used to score steps.
        history_window: Number of preceding steps to include as context when
            scoring a tool call.  Larger windows give richer context but
            increase token cost.
        fallback_score: Score returned when the LLM response cannot be parsed.
            Defaults to 0.5 (neutral / unknown quality).
    """

    def __init__(
        self,
        llm: BaseLLM,
        history_window: int = 5,
        fallback_score: float = 0.5,
    ) -> None:
        self.llm = llm
        self.history_window = history_window
        self.fallback_score = fallback_score

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """Issue a single LLM call and return the raw text content."""
        response = self.llm.generate(prompt)
        # BaseLLM.generate returns a response object with a .content attribute.
        if hasattr(response, "content"):
            return response.content
        return str(response)

    def _parse_json_response(self, raw: str, step_index: int, step_type: str) -> StepScore:
        """Extract score + rationale from the LLM JSON response.

        Falls back to ``self.fallback_score`` on any parse error so that a
        single malformed response never breaks an entire optimization run.
        """
        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            inner = [l for l in lines if not l.startswith("```")]
            text = "\n".join(inner).strip()

        # Find the first JSON object in the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            logger.warning(
                "StepwiseRewardEvaluator: could not locate JSON in LLM response; "
                "using fallback score. Raw response: %s", raw[:200]
            )
            return StepScore(
                score=self.fallback_score,
                rationale="(parse error — fallback)",
                step_index=step_index,
                step_type=step_type,
            )

        try:
            data = json.loads(text[start:end])
            score = float(data.get("score", self.fallback_score))
            score = max(0.0, min(1.0, score))
            rationale = str(data.get("rationale", ""))
            return StepScore(
                score=score,
                rationale=rationale,
                step_index=step_index,
                step_type=step_type,
            )
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning(
                "StepwiseRewardEvaluator: JSON parse failed (%s); using fallback. "
                "Raw: %s", exc, raw[:200]
            )
            return StepScore(
                score=self.fallback_score,
                rationale=f"(parse error: {exc})",
                step_index=step_index,
                step_type=step_type,
            )

    def _format_history(self, trajectory_so_far: List[Dict[str, Any]]) -> str:
        """Format the preceding steps for prompt insertion."""
        window = trajectory_so_far[-self.history_window :]
        if not window:
            return "(none)"
        lines = []
        for i, step in enumerate(window):
            step_type = step.get("type", "unknown")
            content = step.get("content", "")
            lines.append(f"[{i + 1}] ({step_type}) {content}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        context: str,
        trajectory_so_far: List[Dict[str, Any]],
    ) -> StepScore:
        """Score whether this tool call was appropriate in context.

        Args:
            tool_name: Name of the tool that was invoked.
            args: Arguments passed to the tool.
            context: High-level task description.
            trajectory_so_far: Steps that preceded this call.

        Returns:
            ``StepScore`` with quality in [0, 1].
        """
        step_index = len(trajectory_so_far)
        prompt = _TOOL_CALL_PROMPT.format(
            context=context or "(no context provided)",
            history=self._format_history(trajectory_so_far),
            tool_name=tool_name,
            args=json.dumps(args, ensure_ascii=False, indent=None),
        )
        raw = self._call_llm(prompt)
        return self._parse_json_response(raw, step_index=step_index, step_type="tool_call")

    def score_reasoning_step(
        self,
        step_text: str,
        context: str,
    ) -> StepScore:
        """Score whether a reasoning intermediate is sound and useful.

        Args:
            step_text: The text of the reasoning step.
            context: High-level task description.

        Returns:
            ``StepScore`` with quality in [0, 1].
        """
        prompt = _REASONING_PROMPT.format(
            context=context or "(no context provided)",
            step_text=step_text,
        )
        raw = self._call_llm(prompt)
        return self._parse_json_response(raw, step_index=0, step_type="reasoning")

    def score_trajectory(
        self,
        trajectory: List[Dict[str, Any]],
        context: str = "",
    ) -> TrajectoryScores:
        """Score every step in a full execution trace.

        Args:
            trajectory: Ordered list of step dicts, each with ``"type"``
                (``"tool_call"`` or ``"reasoning"``) and ``"content"``.
                Tool-call steps additionally need ``"tool_name"`` and
                ``"args"``.
            context: Overall task / goal description.

        Returns:
            ``TrajectoryScores`` with one ``StepScore`` per step.
        """
        scores: List[StepScore] = []
        for idx, step in enumerate(trajectory):
            step_type = step.get("type", "unknown")
            preceding = trajectory[:idx]

            if step_type == "tool_call":
                step_score = self.score_tool_call(
                    tool_name=step.get("tool_name", "unknown_tool"),
                    args=step.get("args", {}),
                    context=context,
                    trajectory_so_far=preceding,
                )
            elif step_type == "reasoning":
                step_score = self.score_reasoning_step(
                    step_text=step.get("content", ""),
                    context=context,
                )
            else:
                logger.debug(
                    "score_trajectory: unknown step type '%s' at index %d; "
                    "using fallback score.",
                    step_type,
                    idx,
                )
                step_score = StepScore(
                    score=self.fallback_score,
                    rationale="(unknown step type)",
                    step_index=idx,
                    step_type=step_type,
                )

            step_score.step_index = idx
            scores.append(step_score)

        return TrajectoryScores(step_scores=scores)

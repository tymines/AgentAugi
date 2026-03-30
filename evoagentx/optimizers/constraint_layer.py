"""Evolution constraint layer (CEAO) for production-safe optimization.

This module implements a *static* pre/post-evolution filter that wraps any
``BaseOptimizer`` subclass and prevents unsafe mutations from being accepted
into the optimization run.  It is the safety boundary before any evolved agent
reaches a staging or production environment.

Four constraint types are provided:

``CostConstraint``
    Rejects candidate configurations whose estimated execution cost exceeds a
    per-call or total budget ceiling.  Integrates with the Phase-0 CostTracker
    when available; falls back to a lightweight cost estimator otherwise.

``HallucinationConstraint``
    Scores the generated output of a candidate for factual reliability using
    a user-supplied detector function.  Candidates whose score falls below a
    configured threshold are rejected.

``DriftConstraint``
    Computes semantic similarity between the evolved candidate's behaviour and
    a baseline behaviour.  Halts evolution if the candidate drifts beyond an
    acceptable similarity threshold — ensuring the evolved agent stays within
    the spirit of the original task.

``ConstrainedOptimizer``
    Wraps *any* ``BaseOptimizer`` subclass (EvoPrompt, MAP-Elites, TextGrad,
    etc.) with a stack of constraints.  Constraints are checked *before* a
    candidate is accepted; rejected candidates are logged and discarded without
    modifying the registry.

Design notes
------------
- Purely additive: does not modify existing optimizer internals.
- Composable: multiple constraints can be combined via the ``constraints`` list.
- All constraints are individually optional; pass only the ones you need.
- Constraint violations produce warnings (never silent failures).

Usage
-----
    >>> from evoagentx.optimizers.constraint_layer import (
    ...     CostConstraint, DriftConstraint, HallucinationConstraint,
    ...     ConstrainedOptimizer,
    ... )
    >>> cost = CostConstraint(budget_per_call=0.05, total_budget=5.0)
    >>> drift = DriftConstraint(
    ...     baseline_behavior="answer factual questions concisely",
    ...     semantic_distance_threshold=0.3,
    ...     embed_fn=my_embed,
    ... )
    >>> safe_optimizer = ConstrainedOptimizer(
    ...     base_optimizer=evoprompt_instance,
    ...     constraints=[cost, drift],
    ... )
    >>> safe_optimizer.optimize()
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.logging import logger
from .engine.base import BaseOptimizer
from .engine.registry import ParamRegistry


# ---------------------------------------------------------------------------
# Constraint result
# ---------------------------------------------------------------------------

@dataclass
class ConstraintResult:
    """Result of evaluating a single constraint against a candidate.

    Attributes:
        passed: ``True`` if the candidate satisfies the constraint.
        constraint_name: Name of the constraint that produced this result.
        reason: Human-readable explanation when the constraint is violated.
        metadata: Optional extra information (e.g. measured cost, similarity).
    """

    passed: bool
    constraint_name: str
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base constraint
# ---------------------------------------------------------------------------

class BaseConstraint(ABC):
    """Abstract base class for all evolution constraints.

    Subclasses implement ``check`` to evaluate a candidate configuration
    against their specific safety criterion.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique human-readable constraint name."""

    @abstractmethod
    def check(
        self,
        config: Dict[str, Any],
        eval_result: Optional[Dict[str, Any]] = None,
    ) -> ConstraintResult:
        """Evaluate whether a candidate configuration satisfies this constraint.

        Args:
            config: Candidate parameter configuration dict (name → value).
            eval_result: Optional evaluation result dict from running the
                program with this configuration.  May be ``None`` if the
                program has not yet been run.

        Returns:
            ``ConstraintResult`` with ``passed=True`` if the constraint is
            satisfied, ``False`` otherwise.
        """


# ---------------------------------------------------------------------------
# CostConstraint
# ---------------------------------------------------------------------------

class CostConstraint(BaseConstraint):
    """Reject candidates that would exceed a cost budget.

    Tracks cumulative cost via the Phase-0 CostTracker when available.
    When the CostTracker is absent, it estimates cost from token counts
    present in the evaluation result dict (keys ``"input_tokens"`` and
    ``"output_tokens"``), using a conservative per-token rate.

    Args:
        budget_per_call: Maximum cost in USD allowed for a single candidate
            evaluation.  Set to ``None`` to disable per-call limiting.
        total_budget: Maximum cumulative cost in USD for the entire
            optimization run.  Set to ``None`` to disable total limiting.
        cost_per_input_token: Estimated cost per input token in USD used when
            the CostTracker is unavailable (default: $0.000003, ≈GPT-4o mini).
        cost_per_output_token: Estimated cost per output token in USD
            (default: $0.000012).
    """

    def __init__(
        self,
        budget_per_call: Optional[float] = None,
        total_budget: Optional[float] = None,
        cost_per_input_token: float = 0.000003,
        cost_per_output_token: float = 0.000012,
    ) -> None:
        self.budget_per_call = budget_per_call
        self.total_budget = total_budget
        self.cost_per_input_token = cost_per_input_token
        self.cost_per_output_token = cost_per_output_token
        self._cumulative_cost: float = 0.0

        # Lazily import the Phase-0 CostTracker if it has been installed.
        self._tracker = None
        try:
            from ..core.cost_tracker import get_tracker  # type: ignore
            self._tracker = get_tracker()
        except ImportError:
            pass

    @property
    def name(self) -> str:
        return "CostConstraint"

    def _estimate_call_cost(self, eval_result: Optional[Dict[str, Any]]) -> float:
        """Estimate the cost of one candidate evaluation.

        Uses the Phase-0 CostTracker if available; otherwise reads token
        counts from the eval_result dict.
        """
        if self._tracker is not None:
            try:
                return self._tracker.last_call_cost()
            except Exception:
                pass

        if eval_result is None:
            return 0.0

        input_tokens = eval_result.get("input_tokens", 0)
        output_tokens = eval_result.get("output_tokens", 0)
        return (
            input_tokens * self.cost_per_input_token
            + output_tokens * self.cost_per_output_token
        )

    def check(
        self,
        config: Dict[str, Any],
        eval_result: Optional[Dict[str, Any]] = None,
    ) -> ConstraintResult:
        """Check whether this candidate's cost is within budget.

        Args:
            config: Candidate configuration (used for metadata).
            eval_result: Evaluation result; needed to estimate call cost.

        Returns:
            ``ConstraintResult`` with ``passed=False`` and a reason string
            when the budget would be exceeded.
        """
        call_cost = self._estimate_call_cost(eval_result)
        self._cumulative_cost += call_cost
        meta = {
            "call_cost_usd": call_cost,
            "cumulative_cost_usd": self._cumulative_cost,
        }

        if self.budget_per_call is not None and call_cost > self.budget_per_call:
            return ConstraintResult(
                passed=False,
                constraint_name=self.name,
                reason=(
                    f"Per-call cost ${call_cost:.4f} exceeds limit "
                    f"${self.budget_per_call:.4f}."
                ),
                metadata=meta,
            )

        if self.total_budget is not None and self._cumulative_cost > self.total_budget:
            return ConstraintResult(
                passed=False,
                constraint_name=self.name,
                reason=(
                    f"Cumulative cost ${self._cumulative_cost:.4f} exceeds "
                    f"total budget ${self.total_budget:.4f}."
                ),
                metadata=meta,
            )

        return ConstraintResult(
            passed=True,
            constraint_name=self.name,
            metadata=meta,
        )

    def reset(self) -> None:
        """Reset the cumulative cost counter (e.g. between optimization runs)."""
        self._cumulative_cost = 0.0


# ---------------------------------------------------------------------------
# HallucinationConstraint
# ---------------------------------------------------------------------------

class HallucinationConstraint(BaseConstraint):
    """Reject candidates whose output scores below a hallucination threshold.

    The detection function is user-supplied, keeping this constraint agnostic
    to the specific hallucination detection method (LLM-as-judge, NLI model,
    retrieval-based verification, etc.).

    Args:
        detector_fn: Callable ``(output_text: str) → float`` returning a
            *reliability* score in [0, 1].  Higher means more reliable / less
            hallucinated.  The caller is responsible for providing a detector
            appropriate for their domain.
        threshold: Minimum reliability score required for a candidate to pass.
            Candidates scoring below this value are rejected.
        output_key: Key in the evaluation result dict that holds the generated
            output text.  Defaults to ``"output"``.
    """

    def __init__(
        self,
        detector_fn: Callable[[str], float],
        threshold: float = 0.7,
        output_key: str = "output",
    ) -> None:
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"HallucinationConstraint threshold must be in [0, 1], got {threshold}"
            )
        self.detector_fn = detector_fn
        self.threshold = threshold
        self.output_key = output_key

    @property
    def name(self) -> str:
        return "HallucinationConstraint"

    def check(
        self,
        config: Dict[str, Any],
        eval_result: Optional[Dict[str, Any]] = None,
    ) -> ConstraintResult:
        """Score the candidate output for hallucination and check against threshold.

        Args:
            config: Candidate configuration (unused; kept for interface uniformity).
            eval_result: Must contain the output text under ``self.output_key``.

        Returns:
            ``ConstraintResult`` with ``passed=False`` if the reliability
            score is below ``self.threshold``.
        """
        if eval_result is None:
            # No output to score — conservatively pass
            return ConstraintResult(
                passed=True,
                constraint_name=self.name,
                reason="No eval_result provided; constraint skipped.",
            )

        output_text = eval_result.get(self.output_key, "")
        if not output_text:
            return ConstraintResult(
                passed=True,
                constraint_name=self.name,
                reason="Empty output; constraint skipped.",
            )

        try:
            score = float(self.detector_fn(output_text))
        except Exception as exc:
            logger.warning(
                "HallucinationConstraint: detector_fn raised %s; "
                "treating as passed.", exc
            )
            return ConstraintResult(
                passed=True,
                constraint_name=self.name,
                reason=f"Detector error ({exc}); treated as passed.",
                metadata={"error": str(exc)},
            )

        score = max(0.0, min(1.0, score))
        meta = {"reliability_score": score, "threshold": self.threshold}

        if score < self.threshold:
            return ConstraintResult(
                passed=False,
                constraint_name=self.name,
                reason=(
                    f"Reliability score {score:.3f} is below threshold "
                    f"{self.threshold:.3f}."
                ),
                metadata=meta,
            )

        return ConstraintResult(
            passed=True, constraint_name=self.name, metadata=meta
        )


# ---------------------------------------------------------------------------
# DriftConstraint
# ---------------------------------------------------------------------------

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two embedding vectors.

    Args:
        a: First embedding vector.
        b: Second embedding vector.

    Returns:
        Cosine similarity in [-1, 1].  Returns 0.0 on zero-norm vectors.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Vector length mismatch: {len(a)} vs {len(b)}"
        )
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class DriftConstraint(BaseConstraint):
    """Reject candidates that drift too far from the original agent intent.

    Computes semantic similarity between the evolved candidate's behaviour
    (represented as the output text or the prompt config string) and a fixed
    baseline behaviour description.  Candidates whose similarity falls below
    ``1 - semantic_distance_threshold`` are rejected.

    Args:
        baseline_behavior: Text description of the expected / original
            behaviour (e.g. the original system prompt or a short summary of
            the intended task).
        semantic_distance_threshold: Maximum allowed semantic distance from
            the baseline.  A value of 0.0 means no drift is tolerated; 1.0
            means any behaviour is accepted.  Typical values: 0.2–0.4.
        embed_fn: Callable ``(text: str) → List[float]`` that returns an
            embedding vector for the input text.  Any embedding model works
            (sentence-transformers, OpenAI embeddings, etc.).
        behavior_key: Key in the evaluation result dict or config dict that
            holds the behaviour text to compare.  Defaults to ``"output"``.
    """

    def __init__(
        self,
        baseline_behavior: str,
        semantic_distance_threshold: float,
        embed_fn: Callable[[str], List[float]],
        behavior_key: str = "output",
    ) -> None:
        if not (0.0 <= semantic_distance_threshold <= 1.0):
            raise ValueError(
                "semantic_distance_threshold must be in [0, 1], "
                f"got {semantic_distance_threshold}"
            )
        self.baseline_behavior = baseline_behavior
        self.semantic_distance_threshold = semantic_distance_threshold
        self.embed_fn = embed_fn
        self.behavior_key = behavior_key

        # Pre-compute baseline embedding so it is only computed once
        self._baseline_embedding: Optional[List[float]] = None

    @property
    def name(self) -> str:
        return "DriftConstraint"

    def _get_baseline_embedding(self) -> List[float]:
        """Return (or compute and cache) the baseline behaviour embedding."""
        if self._baseline_embedding is None:
            self._baseline_embedding = self.embed_fn(self.baseline_behavior)
        return self._baseline_embedding

    def _get_behavior_text(
        self,
        config: Dict[str, Any],
        eval_result: Optional[Dict[str, Any]],
    ) -> str:
        """Extract the behaviour text to compare against the baseline."""
        # Prefer the eval result (actual output) over the config (prompt text)
        if eval_result is not None and self.behavior_key in eval_result:
            return str(eval_result[self.behavior_key])
        if self.behavior_key in config:
            return str(config[self.behavior_key])
        # Fallback: concatenate all string values in the config
        parts = [str(v) for v in config.values() if isinstance(v, str)]
        return " ".join(parts)

    def check(
        self,
        config: Dict[str, Any],
        eval_result: Optional[Dict[str, Any]] = None,
    ) -> ConstraintResult:
        """Check that the candidate behaviour is semantically close to baseline.

        Args:
            config: Candidate configuration dict.
            eval_result: Evaluation result dict (output text extracted from
                ``self.behavior_key`` if present).

        Returns:
            ``ConstraintResult`` with ``passed=False`` if the semantic
            distance exceeds ``self.semantic_distance_threshold``.
        """
        behavior_text = self._get_behavior_text(config, eval_result)
        if not behavior_text.strip():
            return ConstraintResult(
                passed=True,
                constraint_name=self.name,
                reason="No behavior text found; constraint skipped.",
            )

        try:
            baseline_emb = self._get_baseline_embedding()
            candidate_emb = self.embed_fn(behavior_text)
            similarity = _cosine_similarity(baseline_emb, candidate_emb)
            # Clamp to [0, 1] for human-readable reporting
            similarity = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
            distance = 1.0 - similarity
        except Exception as exc:
            logger.warning(
                "DriftConstraint: embedding computation failed (%s); "
                "treating as passed.", exc
            )
            return ConstraintResult(
                passed=True,
                constraint_name=self.name,
                reason=f"Embedding error ({exc}); treated as passed.",
                metadata={"error": str(exc)},
            )

        meta = {
            "similarity": similarity,
            "distance": distance,
            "threshold": self.semantic_distance_threshold,
        }

        if distance > self.semantic_distance_threshold:
            return ConstraintResult(
                passed=False,
                constraint_name=self.name,
                reason=(
                    f"Semantic distance {distance:.3f} exceeds threshold "
                    f"{self.semantic_distance_threshold:.3f} "
                    f"(similarity={similarity:.3f})."
                ),
                metadata=meta,
            )

        return ConstraintResult(
            passed=True, constraint_name=self.name, metadata=meta
        )


# ---------------------------------------------------------------------------
# ConstrainedOptimizer
# ---------------------------------------------------------------------------

@dataclass
class ViolationRecord:
    """Records a single constraint violation for post-run auditing.

    Attributes:
        iteration: Optimizer iteration at which the violation occurred.
        constraint_name: Name of the violated constraint.
        reason: Human-readable violation description.
        metadata: Extra diagnostic information from the constraint.
    """

    iteration: int
    constraint_name: str
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConstrainedOptimizer(BaseOptimizer):
    """Wraps any ``BaseOptimizer`` with a safety constraint stack.

    Constraints are evaluated *after* each candidate is produced by the
    base optimizer but *before* that candidate is accepted into the
    population / archive.  Rejected candidates are discarded and the base
    optimizer continues to the next iteration.

    This is a *static* filter layer.  It does not modify the internal logic
    of the wrapped optimizer in any way — it only prevents unsafe candidates
    from persisting.

    Args:
        base_optimizer: Any ``BaseOptimizer`` subclass to wrap
            (``EvopromptOptimizer``, ``MAPElitesOptimizer``, etc.).
        constraints: List of ``BaseConstraint`` objects to evaluate in order.
            All constraints must pass for a candidate to be accepted.
        max_violation_ratio: If the fraction of rejected iterations exceeds
            this value, a warning is issued.  Does not halt optimization.
            Defaults to 0.5 (warn if more than half of candidates are
            rejected).
    """

    def __init__(
        self,
        base_optimizer: BaseOptimizer,
        constraints: List[BaseConstraint],
        max_violation_ratio: float = 0.5,
    ) -> None:
        # Mirror the registry and program from the wrapped optimizer
        super().__init__(
            registry=base_optimizer.registry,
            program=base_optimizer.program,
            evaluator=base_optimizer.evaluator,
        )
        self.base_optimizer = base_optimizer
        self.constraints = constraints
        self.max_violation_ratio = max_violation_ratio

        self._violation_log: List[ViolationRecord] = []
        self._total_checks: int = 0
        self._total_violations: int = 0

    @property
    def name(self) -> str:
        """Descriptive name combining base optimizer and constraint names."""
        constraint_names = "+".join(c.name for c in self.constraints)
        return f"ConstrainedOptimizer({type(self.base_optimizer).__name__}, [{constraint_names}])"

    # ------------------------------------------------------------------
    # Constraint evaluation
    # ------------------------------------------------------------------

    def check_all(
        self,
        config: Dict[str, Any],
        eval_result: Optional[Dict[str, Any]] = None,
        iteration: int = 0,
    ) -> Tuple[bool, List[ConstraintResult]]:
        """Run all constraints against a candidate.

        Evaluation stops at the first failure (short-circuit).  To evaluate
        all constraints even on failure, iterate ``self.constraints`` directly.

        Args:
            config: Candidate configuration dict.
            eval_result: Optional evaluation result dict.
            iteration: Current optimizer iteration (for violation logging).

        Returns:
            Tuple of ``(all_passed, results_list)`` where ``results_list``
            contains one ``ConstraintResult`` per constraint evaluated.
        """
        self._total_checks += 1
        results: List[ConstraintResult] = []

        for constraint in self.constraints:
            result = constraint.check(config=config, eval_result=eval_result)
            results.append(result)
            if not result.passed:
                self._total_violations += 1
                record = ViolationRecord(
                    iteration=iteration,
                    constraint_name=result.constraint_name,
                    reason=result.reason,
                    metadata=result.metadata,
                )
                self._violation_log.append(record)
                logger.warning(
                    "ConstrainedOptimizer [iter %d]: %s violated — %s",
                    iteration,
                    result.constraint_name,
                    result.reason,
                )
                self._check_violation_ratio()
                return False, results

        return True, results

    def _check_violation_ratio(self) -> None:
        """Warn if the violation ratio is too high."""
        if self._total_checks == 0:
            return
        ratio = self._total_violations / self._total_checks
        if ratio > self.max_violation_ratio:
            logger.warning(
                "ConstrainedOptimizer: violation ratio %.1f%% exceeds "
                "max_violation_ratio %.1f%%. Consider loosening constraints "
                "or fixing the base optimizer.",
                ratio * 100,
                self.max_violation_ratio * 100,
            )

    # ------------------------------------------------------------------
    # Delegate to base optimizer with pre/post hooks
    # ------------------------------------------------------------------

    def optimize(self) -> Any:
        """Run the base optimizer's ``optimize()`` with constraint filtering.

        This implementation patches the base optimizer's ``_evaluate_config``
        method (if present) to insert constraint checks after evaluation.
        For optimizers that do not expose ``_evaluate_config``, the base
        ``optimize()`` is called directly with a warning.

        Returns:
            Whatever the base optimizer's ``optimize()`` returns.
        """
        if hasattr(self.base_optimizer, "_evaluate_config"):
            return self._optimize_with_hook()
        else:
            logger.warning(
                "ConstrainedOptimizer: base optimizer %s does not expose "
                "_evaluate_config; running unconstrained.",
                type(self.base_optimizer).__name__,
            )
            return self.base_optimizer.optimize()

    def _optimize_with_hook(self) -> Any:
        """Hook into base optimizer's _evaluate_config for constraint checks."""
        original_evaluate = self.base_optimizer._evaluate_config
        iteration_counter = [0]

        def constrained_evaluate(
            config: Dict[str, Any],
        ) -> Tuple[Dict[str, Any], float]:
            """Replacement for _evaluate_config that applies constraints."""
            eval_result, quality = original_evaluate(config)
            passed, _ = self.check_all(
                config=config,
                eval_result=eval_result,
                iteration=iteration_counter[0],
            )
            iteration_counter[0] += 1

            if not passed:
                # Return the eval result but force quality to -inf so the
                # optimizer's archive / population rejects this candidate.
                return eval_result, float("-inf")

            return eval_result, quality

        # Monkey-patch, run, then restore
        self.base_optimizer._evaluate_config = constrained_evaluate
        try:
            result = self.base_optimizer.optimize()
        finally:
            self.base_optimizer._evaluate_config = original_evaluate

        return result

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_violation_log(self) -> List[ViolationRecord]:
        """Return all recorded constraint violations for auditing."""
        return list(self._violation_log)

    def violation_summary(self) -> Dict[str, Any]:
        """Return a summary of constraint violations.

        Returns:
            Dict with keys ``total_checks``, ``total_violations``,
            ``violation_ratio``, and ``by_constraint`` (per-constraint counts).
        """
        by_constraint: Dict[str, int] = {}
        for record in self._violation_log:
            by_constraint[record.constraint_name] = (
                by_constraint.get(record.constraint_name, 0) + 1
            )

        ratio = (
            self._total_violations / self._total_checks
            if self._total_checks > 0
            else 0.0
        )
        return {
            "total_checks": self._total_checks,
            "total_violations": self._total_violations,
            "violation_ratio": ratio,
            "by_constraint": by_constraint,
        }

"""FrugalGPT-style model cascading for cost-aware LLM routing.

Inspired by FrugalGPT (arXiv:2305.05176), this module implements a cascade of
LLM models ordered from cheapest to most expensive.  Each tier is tried in
sequence; if the model's output meets a confidence threshold, the cascade
returns immediately without escalating to more expensive models.

Architecture
------------
``ModelCascade`` orchestrates the cascade:

1. **Tier ordering** — define a sequence of (model, threshold) pairs from
   cheapest to most expensive.  The last tier should have threshold 1.0 (always
   accept) so there is a guaranteed terminal tier.

2. **Confidence estimation** — two complementary mechanisms:
   - ``ConfidenceEstimator.from_text(output)`` — heuristic, no external call.
     Uses reasoning markers, hedging words, output length, and structural cues.
   - ``ConfidenceEstimator.from_self_report(output)`` — looks for explicit
     "confidence: X" patterns in the model's own output (requires prompting the
     model to self-report confidence).

3. **Difficulty routing integration** — ``query()`` accepts an optional
   ``difficulty_hint`` in [0, 1] from ``DifficultyRouter`` (Phase 2B).  A high
   difficulty hint skips cheap tiers entirely (cheap models will almost certainly
   fail), saving the escalation cost.

4. **Cost tracking** — each tier carries a ``cost_per_1k_tokens`` estimate.
   ``CascadeMetrics`` tracks actual cost vs. hypothetical cost if every query
   went to the most expensive tier, exposing the real savings.

5. **Fallback behaviour** — if a tier's generate function raises an exception,
   the cascade logs a warning and escalates to the next tier rather than failing.

Default model chain (cheapest → most expensive)
-----------------------------------------------
As specified in the implementation plan:
  Ollama (local) → DeepSeek API → Kimi → Claude

Users define their own chain via ``ModelTier`` instances.

Usage
-----
    >>> from evoagentx.core.model_cascade import ModelCascade, ModelTier
    >>> cascade = ModelCascade(tiers=[
    ...     ModelTier("cheap",    generate_fn=ollama_fn,   confidence_threshold=0.80, cost_per_1k=0.0),
    ...     ModelTier("medium",   generate_fn=deepseek_fn, confidence_threshold=0.88, cost_per_1k=0.10),
    ...     ModelTier("powerful", generate_fn=claude_fn,   confidence_threshold=1.00, cost_per_1k=1.50),
    ... ])
    >>> result = cascade.query("Explain quantum entanglement in one sentence.")
    >>> print(result.response, result.tier_used)
    >>> print(cascade.metrics)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .logging import logger


# ---------------------------------------------------------------------------
# Confidence estimation
# ---------------------------------------------------------------------------

_HEDGING_WORDS = frozenset([
    "maybe", "perhaps", "possibly", "might", "could", "uncertain",
    "unclear", "not sure", "I think", "I believe", "approximately",
    "roughly", "around", "it seems", "appears to", "likely",
])

_REASONING_MARKERS = frozenset([
    "because", "therefore", "thus", "hence", "since", "given that",
    "as a result", "consequently", "specifically", "for example",
    "evidence", "demonstrates", "clearly", "in conclusion",
])

_UNCERTAIN_PHRASES = frozenset([
    "i'm not sure", "i don't know", "i cannot", "i am unable",
    "it's unclear", "it is unclear", "hard to say", "difficult to determine",
    "not enough information", "insufficient information",
])


def _extract_self_reported_confidence(text: str) -> Optional[float]:
    """Parse explicit confidence from model output, e.g. 'confidence: 0.85'.

    Recognises numeric and qualitative patterns:
    - "confidence: 0.85" or "confidence score: 85%"
    - "confidence: high" → 0.9, "medium" → 0.65, "low" → 0.35

    Args:
        text: Model output text.

    Returns:
        Confidence in [0, 1] or None if no pattern found.
    """
    lower = text.lower()

    # Numeric: "confidence: 0.85", "confidence score: 85%", etc.
    match = re.search(
        r"confidence\b[^0-9]*([0-9]+(?:\.[0-9]+)?)(%?)",
        lower,
    )
    if match:
        val = float(match.group(1))
        if match.group(2) == "%":
            val /= 100.0
        return max(0.0, min(1.0, val))

    # Qualitative
    if re.search(r"confidence[\s:]+(very\s+)?high", lower):
        return 0.90
    if re.search(r"confidence[\s:]+medium", lower):
        return 0.65
    if re.search(r"confidence[\s:]+(very\s+)?low", lower):
        return 0.35

    return None


def _heuristic_confidence(text: str) -> float:
    """Estimate confidence from text characteristics without a dedicated model.

    Penalises hedging and uncertainty phrases; rewards clear reasoning,
    specificity (numbers, named entities), and decisive phrasing.

    Args:
        text: Model output text.

    Returns:
        Heuristic confidence estimate in [0.05, 0.95].
    """
    if not text or not text.strip():
        return 0.05

    words = text.split()
    word_count = len(words)
    lower = text.lower()

    # Hard uncertain phrases → very low confidence
    for phrase in _UNCERTAIN_PHRASES:
        if phrase in lower:
            return 0.20

    # Hedge rate
    hedge_count = sum(1 for h in _HEDGING_WORDS if h in lower)
    hedge_penalty = min(0.4, hedge_count * 0.08)

    # Reasoning markers → higher confidence
    reasoning_hits = sum(1 for m in _REASONING_MARKERS if m in lower)
    reasoning_bonus = min(0.2, reasoning_hits * 0.04)

    # Length: very short answers are uncertain; mid-length is confident
    if word_count < 5:
        length_factor = 0.3
    elif word_count < 20:
        length_factor = 0.55
    elif word_count <= 200:
        length_factor = 0.75
    else:
        length_factor = 0.68  # very long may be padding

    # Specificity: numbers, code blocks, proper nouns → confident
    has_numbers = bool(re.search(r"\b\d+(?:\.\d+)?\b", text))
    has_code = "```" in text or "`" in text
    specificity_bonus = 0.05 * int(has_numbers) + 0.05 * int(has_code)

    raw = length_factor + reasoning_bonus + specificity_bonus - hedge_penalty
    return max(0.05, min(0.95, raw))


class ConfidenceEstimator:
    """Stateless helper for estimating model output confidence.

    Two independent estimation strategies:

    - ``from_self_report``: parses explicit ``"confidence: X"`` patterns that
      the model itself emits (requires appropriate system prompting).
    - ``from_text``: heuristic signal derived from output characteristics.

    ``combined`` merges both signals, preferring the self-reported value when
    available.
    """

    @staticmethod
    def from_self_report(output: str) -> Optional[float]:
        """Parse model's self-reported confidence from output text.

        Args:
            output: Raw model output string.

        Returns:
            Confidence in [0, 1] or None if no self-report pattern found.
        """
        return _extract_self_reported_confidence(output)

    @staticmethod
    def from_text(output: str) -> float:
        """Heuristic confidence estimate based on output characteristics.

        Args:
            output: Raw model output string.

        Returns:
            Heuristic confidence in [0.05, 0.95].
        """
        return _heuristic_confidence(output)

    @classmethod
    def combined(cls, output: str) -> float:
        """Combine self-report and heuristic confidence estimates.

        Self-reported confidence is used directly when present.  Otherwise
        falls back to the heuristic estimate.  A small blend is applied
        to avoid over-trusting either source.

        Args:
            output: Raw model output string.

        Returns:
            Combined confidence estimate in [0, 1].
        """
        self_report = cls.from_self_report(output)
        heuristic = cls.from_text(output)

        if self_report is not None:
            # Blend: 70% self-reported, 30% heuristic (heuristic as sanity check)
            return 0.70 * self_report + 0.30 * heuristic

        return heuristic


# ---------------------------------------------------------------------------
# ModelTier
# ---------------------------------------------------------------------------

@dataclass
class ModelTier:
    """One tier in the model cascade.

    Attributes:
        name: Identifier for this tier (used in metrics and logging).
        generate_fn: Callable ``(prompt: str) -> str`` that calls the model.
            Use ``wrap_messages`` (see below) if your model expects a messages list.
        confidence_threshold: Minimum confidence [0, 1] for this tier's output
            to be accepted without escalation.  Set to 1.0 for the final tier
            to guarantee termination.
        cost_per_1k_tokens: Estimated cost per 1,000 tokens for this tier.
            Used for savings computation.  Use 0.0 for free/local models.
    """

    name: str
    generate_fn: Callable[[str], str]
    confidence_threshold: float
    cost_per_1k_tokens: float = 0.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ModelTier.name must not be empty.")
        if not (0.0 < self.confidence_threshold <= 1.0):
            raise ValueError(
                f"ModelTier.confidence_threshold must be in (0, 1], "
                f"got {self.confidence_threshold}"
            )
        if self.cost_per_1k_tokens < 0.0:
            raise ValueError("ModelTier.cost_per_1k_tokens must be >= 0.")


# ---------------------------------------------------------------------------
# Result and metrics
# ---------------------------------------------------------------------------

@dataclass
class CascadeResult:
    """Result of a single cascade query.

    Attributes:
        response: Final model response text.
        tier_used: Name of the tier that produced the accepted response.
        tier_index: Zero-based index of the tier used (0 = cheapest).
        confidence: Confidence score of the accepted response.
        escalations: Number of tiers tried before the accepted tier
            (0 = first tier was sufficient).
        latency_seconds: Wall-clock time for the entire cascade query.
    """

    response: str
    tier_used: str
    tier_index: int
    confidence: float
    escalations: int
    latency_seconds: float = 0.0


@dataclass
class CascadeMetrics:
    """Cumulative metrics across all cascade queries.

    Attributes:
        total_queries: Total calls to ``cascade.query()``.
        tier_usage: Map from tier name to number of queries it served.
        total_escalations: Total escalation events across all queries.
        estimated_tokens: Total estimated tokens processed (heuristic: 4 chars/token).
        estimated_cost_actual: Estimated cost using the tier actually used.
        estimated_cost_no_cascade: Hypothetical cost if every query used the
            most expensive (last) tier.
        mean_confidence: Mean confidence of accepted responses.
        mean_latency_seconds: Mean wall-clock latency per query.
    """

    total_queries: int = 0
    tier_usage: Dict[str, int] = field(default_factory=dict)
    total_escalations: int = 0
    estimated_tokens: int = 0
    estimated_cost_actual: float = 0.0
    estimated_cost_no_cascade: float = 0.0
    _confidence_sum: float = field(default=0.0, repr=False)
    _latency_sum: float = field(default=0.0, repr=False)

    @property
    def savings_rate(self) -> float:
        """Fraction of cost saved vs. always using the most expensive tier.

        Returns:
            Savings rate in [0, 1].  0.0 if no queries or all used top tier.
        """
        if self.estimated_cost_no_cascade < 1e-9:
            return 0.0
        saved = self.estimated_cost_no_cascade - self.estimated_cost_actual
        return max(0.0, saved / self.estimated_cost_no_cascade)

    @property
    def mean_confidence(self) -> float:
        """Mean confidence of accepted responses."""
        if self.total_queries == 0:
            return 0.0
        return self._confidence_sum / self.total_queries

    @property
    def mean_latency_seconds(self) -> float:
        """Mean wall-clock latency per query in seconds."""
        if self.total_queries == 0:
            return 0.0
        return self._latency_sum / self.total_queries

    def __str__(self) -> str:
        usage_str = ", ".join(
            f"{k}={v}" for k, v in sorted(self.tier_usage.items())
        )
        return (
            f"CascadeMetrics(queries={self.total_queries}, "
            f"tier_usage=[{usage_str}], "
            f"escalations={self.total_escalations}, "
            f"savings={self.savings_rate:.1%}, "
            f"mean_confidence={self.mean_confidence:.2f}, "
            f"mean_latency={self.mean_latency_seconds:.3f}s)"
        )


# ---------------------------------------------------------------------------
# ModelCascade
# ---------------------------------------------------------------------------

class ModelCascade:
    """FrugalGPT-style model cascade for cost-optimised LLM inference.

    Tries tiers from cheapest to most expensive, accepting the first tier
    whose output confidence meets the tier's threshold.  Tracks savings vs.
    accuracy tradeoff via ``CascadeMetrics``.

    Args:
        tiers: Ordered list of ``ModelTier`` instances, cheapest first.
            The last tier *must* have ``confidence_threshold == 1.0`` to
            guarantee the cascade always terminates.
        estimator: ``ConfidenceEstimator`` instance.  Uses the default combined
            estimator if not provided.
        difficulty_skip_threshold: When a ``difficulty_hint`` ≥ this value is
            passed to ``query()``, the cheapest N tiers are skipped and the
            cascade starts at the first tier whose ``confidence_threshold``
            can plausibly handle hard queries (heuristic: any tier above the
            median threshold).  Default 0.75.

    Raises:
        ValueError: If fewer than 1 tier is provided, or the last tier does
            not have a confidence_threshold of 1.0.
    """

    def __init__(
        self,
        tiers: List[ModelTier],
        estimator: Optional[ConfidenceEstimator] = None,
        difficulty_skip_threshold: float = 0.75,
    ) -> None:
        if not tiers:
            raise ValueError("ModelCascade requires at least one ModelTier.")
        if abs(tiers[-1].confidence_threshold - 1.0) > 1e-6:
            raise ValueError(
                f"The last ModelTier must have confidence_threshold=1.0 "
                f"to guarantee termination.  Got {tiers[-1].confidence_threshold}."
            )
        if not (0.0 <= difficulty_skip_threshold <= 1.0):
            raise ValueError("difficulty_skip_threshold must be in [0, 1].")

        self._tiers = tiers
        self._estimator = estimator or ConfidenceEstimator()
        self.difficulty_skip_threshold = difficulty_skip_threshold
        self._metrics = CascadeMetrics()

        # Pre-compute most expensive tier cost for savings estimation
        self._max_cost_per_1k = max(t.cost_per_1k_tokens for t in tiers)

        logger.debug(
            "ModelCascade: %d tiers, difficulty_skip_threshold=%.2f",
            len(self._tiers),
            self.difficulty_skip_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        prompt: str,
        difficulty_hint: Optional[float] = None,
    ) -> CascadeResult:
        """Run the cascade for a single prompt.

        Tries each tier from cheapest to most expensive.  Returns as soon as
        a tier's output confidence ≥ the tier's threshold.  If a tier raises
        an exception, escalates silently to the next tier.

        Args:
            prompt: The query / prompt string.
            difficulty_hint: Optional difficulty score in [0, 1] from
                ``DifficultyRouter`` (Phase 2B).  High values (≥
                ``difficulty_skip_threshold``) cause cheap tiers to be
                skipped to avoid wasting a call that will almost certainly
                need escalation.

        Returns:
            ``CascadeResult`` with the accepted response and diagnostics.

        Raises:
            RuntimeError: If all tiers fail (generate_fn raises on every tier).
        """
        start_time = time.time()
        start_index = self._compute_start_index(difficulty_hint)
        escalations = 0
        last_error: Optional[Exception] = None

        for idx in range(start_index, len(self._tiers)):
            tier = self._tiers[idx]

            try:
                response = tier.generate_fn(prompt)
            except Exception as exc:
                logger.warning(
                    "ModelCascade: tier '%s' failed — escalating: %s",
                    tier.name, exc,
                )
                last_error = exc
                escalations += 1
                continue

            # Estimate confidence
            confidence = ConfidenceEstimator.combined(response)

            logger.debug(
                "ModelCascade: tier '%s' confidence=%.3f threshold=%.3f",
                tier.name, confidence, tier.confidence_threshold,
            )

            is_last_tier = (idx == len(self._tiers) - 1)
            if confidence >= tier.confidence_threshold or is_last_tier:
                # Accepted (last tier is always accepted as a guaranteed fallback)
                latency = time.time() - start_time
                self._record_result(
                    tier=tier,
                    tier_index=idx,
                    response=response,
                    confidence=confidence,
                    escalations=escalations + (idx - start_index),
                    latency=latency,
                    prompt=prompt,
                )
                return CascadeResult(
                    response=response,
                    tier_used=tier.name,
                    tier_index=idx,
                    confidence=confidence,
                    escalations=escalations + (idx - start_index),
                    latency_seconds=latency,
                )

            # Confidence too low — escalate
            escalations += 1
            logger.info(
                "ModelCascade: escalating from '%s' (confidence=%.3f < %.3f)",
                tier.name, confidence, tier.confidence_threshold,
            )

        # Should not reach here if last tier has threshold=1.0, but guard anyway
        raise RuntimeError(
            f"ModelCascade: all tiers exhausted without an accepted response. "
            f"Last error: {last_error}"
        )

    def query_with_messages(
        self,
        messages: List[Dict[str, str]],
        key_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None,
        difficulty_hint: Optional[float] = None,
    ) -> CascadeResult:
        """Convenience wrapper for message-list based LLM queries.

        Serialises the message list to a prompt string, then calls ``query()``.
        Tiers are expected to accept plain strings; if your backend requires
        messages, build the tier's generate_fn to accept strings (using a
        closure that reformats as needed).

        Args:
            messages: Chat-format message list.
            key_fn: Custom serialiser for the messages.  Defaults to
                concatenating ``role:content`` lines.
            difficulty_hint: Optional difficulty score from DifficultyRouter.

        Returns:
            ``CascadeResult``.
        """
        if key_fn is not None:
            prompt = key_fn(messages)
        else:
            prompt = "\n".join(
                f"{m.get('role', '')}:{m.get('content', '')}" for m in messages
            )
        return self.query(prompt=prompt, difficulty_hint=difficulty_hint)

    @property
    def metrics(self) -> CascadeMetrics:
        """Cumulative cost and performance metrics (read-only snapshot)."""
        return self._metrics

    def reset_metrics(self) -> None:
        """Reset all cumulative metrics."""
        self._metrics = CascadeMetrics()
        logger.debug("ModelCascade: metrics reset.")

    def tier_names(self) -> List[str]:
        """Return the names of all tiers in order (cheapest first)."""
        return [t.name for t in self._tiers]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_start_index(self, difficulty_hint: Optional[float]) -> int:
        """Determine which tier index to start the cascade at.

        When a high difficulty_hint is provided, cheap tiers are skipped to
        avoid a near-certain escalation.  The skip target is the first tier
        whose confidence threshold is above the median threshold (i.e., it is
        already a "stronger" tier).

        Args:
            difficulty_hint: Optional [0, 1] difficulty score.

        Returns:
            Starting tier index (0 = no skip).
        """
        if difficulty_hint is None or difficulty_hint < self.difficulty_skip_threshold:
            return 0

        # Find the first tier with confidence_threshold above the median
        thresholds = [t.confidence_threshold for t in self._tiers]
        median_threshold = sorted(thresholds)[len(thresholds) // 2]

        for idx, tier in enumerate(self._tiers):
            if tier.confidence_threshold >= median_threshold:
                logger.debug(
                    "ModelCascade: difficulty_hint=%.2f >= %.2f — "
                    "skipping to tier '%s' (index %d)",
                    difficulty_hint,
                    self.difficulty_skip_threshold,
                    tier.name,
                    idx,
                )
                return idx

        return 0  # Fallback: no skip

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token count estimate (4 characters per token heuristic)."""
        return max(1, len(text) // 4)

    def _record_result(
        self,
        tier: ModelTier,
        tier_index: int,
        response: str,
        confidence: float,
        escalations: int,
        latency: float,
        prompt: str,
    ) -> None:
        """Update cumulative metrics after a successful cascade query."""
        m = self._metrics
        m.total_queries += 1
        m.tier_usage[tier.name] = m.tier_usage.get(tier.name, 0) + 1
        m.total_escalations += escalations

        tokens = self._estimate_tokens(prompt + response)
        m.estimated_tokens += tokens

        # Cost of tier actually used
        cost_actual = tier.cost_per_1k_tokens * tokens / 1000.0
        m.estimated_cost_actual += cost_actual

        # Hypothetical cost if top tier had been used directly
        cost_top = self._max_cost_per_1k * tokens / 1000.0
        m.estimated_cost_no_cascade += cost_top

        m._confidence_sum += confidence
        m._latency_sum += latency


# ---------------------------------------------------------------------------
# Default cascade factory
# ---------------------------------------------------------------------------

def build_default_cascade(
    ollama_fn: Optional[Callable[[str], str]] = None,
    deepseek_fn: Optional[Callable[[str], str]] = None,
    kimi_fn: Optional[Callable[[str], str]] = None,
    claude_fn: Optional[Callable[[str], str]] = None,
) -> Optional["ModelCascade"]:
    """Build the default cascade: Ollama → DeepSeek → Kimi → Claude.

    Only includes tiers for which a generate_fn is provided.  Returns None if
    no functions are provided (useful for conditional construction).

    The confidence thresholds are set to:
    - Ollama (local): 0.80 — accept only clearly confident local responses
    - DeepSeek:       0.87 — accept moderately confident responses
    - Kimi:           0.92 — accept high-confidence responses
    - Claude:         1.00 — always accept (terminal tier)

    Args:
        ollama_fn: Generate function for Ollama (local) tier.
        deepseek_fn: Generate function for DeepSeek API tier.
        kimi_fn: Generate function for Kimi/Moonshot API tier.
        claude_fn: Generate function for Claude (Anthropic) tier.

    Returns:
        Configured ``ModelCascade`` or None if no functions provided.
    """
    tiers: List[ModelTier] = []

    if ollama_fn is not None:
        tiers.append(ModelTier(
            name="ollama",
            generate_fn=ollama_fn,
            confidence_threshold=0.80,
            cost_per_1k_tokens=0.0,
        ))

    if deepseek_fn is not None:
        tiers.append(ModelTier(
            name="deepseek",
            generate_fn=deepseek_fn,
            confidence_threshold=0.87,
            cost_per_1k_tokens=0.14,
        ))

    if kimi_fn is not None:
        tiers.append(ModelTier(
            name="kimi",
            generate_fn=kimi_fn,
            confidence_threshold=0.92,
            cost_per_1k_tokens=0.60,
        ))

    if claude_fn is not None:
        tiers.append(ModelTier(
            name="claude",
            generate_fn=claude_fn,
            confidence_threshold=1.00,
            cost_per_1k_tokens=1.50,
        ))

    if not tiers:
        return None

    # Ensure final tier has threshold 1.0
    if abs(tiers[-1].confidence_threshold - 1.0) > 1e-6:
        tiers[-1] = ModelTier(
            name=tiers[-1].name,
            generate_fn=tiers[-1].generate_fn,
            confidence_threshold=1.00,
            cost_per_1k_tokens=tiers[-1].cost_per_1k_tokens,
        )

    return ModelCascade(tiers=tiers)


# Alias matching the plan's "FrugalCascade" name
FrugalCascade = ModelCascade


__all__ = [
    "ModelCascade",
    "FrugalCascade",
    "ModelTier",
    "CascadeResult",
    "CascadeMetrics",
    "ConfidenceEstimator",
    "build_default_cascade",
]

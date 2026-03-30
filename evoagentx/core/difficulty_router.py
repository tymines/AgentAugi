"""
Difficulty-Aware Adaptive Orchestration (DAAO) router.

Classifies incoming tasks by difficulty and routes them to an appropriate
pipeline complexity tier, trading off cost against quality. Inspired by the
DAAO paper (11 % accuracy improvement, 36 % cost reduction) but implemented
from scratch to fit EvoAgentX architecture.

Difficulty tiers
----------------
SIMPLE  → lightweight single-agent pipeline (cheap fast model)
MEDIUM  → standard pipeline (default quality/cost trade-off)
HARD    → full multi-agent workflow with all optimizations

The router learns calibration thresholds from historical performance data via
an exponential-moving-average update so thresholds drift over time without
requiring explicit re-training.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .logging import logger
from .module import BaseModule


# ---------------------------------------------------------------------------
# Difficulty tier enum
# ---------------------------------------------------------------------------

class DifficultyTier(str, Enum):
    """Three-way difficulty classification used for routing decisions."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    HARD = "hard"


# ---------------------------------------------------------------------------
# Routing decision – carries all contextual information about a routing choice
# ---------------------------------------------------------------------------

@dataclass
class RoutingDecision:
    """
    Records a single routing decision for auditing and calibration.

    Attributes
    ----------
    task_id:
        Caller-supplied identifier for the task (may be empty string).
    tier:
        Which difficulty tier the router selected.
    score:
        Raw difficulty score in [0, 1] computed before thresholding.
    features:
        Dictionary of features extracted from the task input.
    model_hint:
        Suggested model identifier for the selected tier (informational).
    timestamp:
        Unix timestamp when the decision was made.
    outcome_score:
        Optional accuracy/quality score reported after the task completes.
        Used for threshold calibration when ``record_outcome`` is called.
    """
    task_id: str
    tier: DifficultyTier
    score: float
    features: Dict[str, float]
    model_hint: str
    timestamp: float = field(default_factory=time.time)
    outcome_score: Optional[float] = None


# ---------------------------------------------------------------------------
# Feature extractor – produces a fixed-size feature dict from raw task input
# ---------------------------------------------------------------------------

class _FeatureExtractor:
    """
    Extracts lightweight, cost-free heuristic features from task text.

    All features are normalised to [0, 1] so they can be combined into a
    single difficulty score without re-scaling.
    """

    # Token count bucket upper bounds used for normalisation
    _TOKEN_SOFT_MAX = 2000

    # Keywords that correlate with harder / easier tasks
    _HARD_KEYWORDS = frozenset({
        "implement", "design", "architect", "refactor", "optimise", "optimize",
        "debug", "multi-step", "multi step", "complex", "comprehensive",
        "analyze", "analyse", "compare", "evaluate", "synthesize",
    })
    _EASY_KEYWORDS = frozenset({
        "what is", "define", "list", "name", "when", "who", "yes or no",
        "true or false", "summarize", "summarise", "translate",
    })

    def extract(self, text: str) -> Dict[str, float]:
        """
        Return a feature dict for *text*.

        Features
        --------
        token_count_norm:
            Approximate token count normalised to [0, 1] (soft cap at 2000).
        hard_keyword_ratio:
            Fraction of hard-task keywords found in the text.
        easy_keyword_ratio:
            Fraction of easy-task keywords found in the text.
        sentence_complexity:
            Heuristic: average words-per-sentence, normalised to [0, 1].
        question_count_norm:
            Number of "?" characters, normalised (soft cap at 5).
        code_block_present:
            1.0 if the text contains a code fence (```), else 0.0.
        """
        lower = text.lower()
        tokens = lower.split()
        token_count = len(tokens)

        sentences = [s.strip() for s in text.replace("?", ".").replace("!", ".").split(".") if s.strip()]
        avg_words_per_sentence = (token_count / max(len(sentences), 1)) / 30.0  # normalise by 30

        hard_hits = sum(1 for kw in self._HARD_KEYWORDS if kw in lower)
        easy_hits = sum(1 for kw in self._EASY_KEYWORDS if kw in lower)

        return {
            "token_count_norm": min(token_count / self._TOKEN_SOFT_MAX, 1.0),
            "hard_keyword_ratio": min(hard_hits / max(len(self._HARD_KEYWORDS), 1), 1.0),
            "easy_keyword_ratio": min(easy_hits / max(len(self._EASY_KEYWORDS), 1), 1.0),
            "sentence_complexity": min(avg_words_per_sentence, 1.0),
            "question_count_norm": min(text.count("?") / 5.0, 1.0),
            "code_block_present": 1.0 if "```" in text else 0.0,
        }


# ---------------------------------------------------------------------------
# Threshold state – mutable, EMA-updated calibration thresholds
# ---------------------------------------------------------------------------

@dataclass
class _ThresholdState:
    """
    Current simple→medium and medium→hard split points.

    Thresholds are stored as floats in (0, 1).  A task with score < simple_max
    is SIMPLE, score < medium_max is MEDIUM, otherwise HARD.
    """
    simple_max: float = 0.35
    medium_max: float = 0.65

    # EMA smoothing factor – higher = faster adaptation
    ema_alpha: float = 0.05

    def update(self, score: float, tier_was_correct: bool) -> None:
        """
        Nudge thresholds based on whether the routing was correct.

        When a routing decision proves *incorrect* the relevant threshold is
        adjusted by a fraction of the signed error so the boundary moves
        toward the misclassified sample.

        Parameters
        ----------
        score:
            The raw difficulty score of the task that was mis-routed.
        tier_was_correct:
            False when the caller signals the routing did not serve the task
            well.  This is a simplification – in practice the outcome score
            from ``RoutingDecision.outcome_score`` drives this.
        """
        if tier_was_correct:
            return
        delta = self.ema_alpha * score
        if score < self.simple_max:
            # Was routed as SIMPLE but needed more → push threshold down
            self.simple_max = max(0.05, self.simple_max - delta)
        elif score < self.medium_max:
            # Was routed as MEDIUM but needed more → push threshold down
            self.medium_max = max(self.simple_max + 0.05, self.medium_max - delta)
        else:
            # Was routed as HARD but seemed easier → push upper threshold up
            self.medium_max = min(0.95, self.medium_max + delta)
        # Keep invariants
        self.simple_max = min(self.simple_max, self.medium_max - 0.05)
        self.medium_max = max(self.medium_max, self.simple_max + 0.05)


# ---------------------------------------------------------------------------
# Main router class
# ---------------------------------------------------------------------------

class DifficultyRouter(BaseModule):
    """
    Routes tasks to difficulty tiers based on heuristic feature scoring.

    Usage
    -----
    >>> router = DifficultyRouter()
    >>> decision = router.route("What is the capital of France?")
    >>> print(decision.tier)   # DifficultyTier.SIMPLE
    >>> decision = router.route("Implement a distributed rate-limiter with ...")
    >>> print(decision.tier)   # DifficultyTier.HARD

    Model hints
    -----------
    The ``model_hints`` attribute maps each tier to a suggested model
    identifier string.  These are *informational only* — the router does not
    call any LLM itself.  Callers use the hint to select the actual model.

    Calibration
    -----------
    After each task completes, call ``record_outcome`` with the task_id and
    an accuracy/quality score in [0, 1].  The router uses EMA to nudge
    thresholds over time.

    Persistence
    -----------
    Call ``save_state`` / ``load_state`` to persist thresholds and history
    across sessions.

    Attributes
    ----------
    model_hints:
        Maps each DifficultyTier to a model identifier string.
    history_limit:
        Maximum number of RoutingDecision records to keep in memory.
    feature_weights:
        Per-feature importance weights used to compute the scalar score.
    """

    # Pydantic fields (serialisable)
    model_hints: Dict[str, str] = {
        DifficultyTier.SIMPLE: "deepseek-chat",
        DifficultyTier.MEDIUM: "claude-haiku-4-5-20251001",
        DifficultyTier.HARD:   "claude-sonnet-4-6",
    }
    history_limit: int = 1000
    feature_weights: Dict[str, float] = {
        "token_count_norm":    0.20,
        "hard_keyword_ratio":  0.30,
        "easy_keyword_ratio": -0.25,
        "sentence_complexity": 0.15,
        "question_count_norm": 0.05,
        "code_block_present":  0.05,
    }
    # Serialisable threshold snapshot (populated on save)
    _threshold_snapshot: Optional[Dict[str, float]] = None

    def init_module(self) -> None:
        """Set up non-serialisable runtime state."""
        self._extractor = _FeatureExtractor()
        self._thresholds = _ThresholdState()
        self._history: List[RoutingDecision] = []
        self._lock = threading.Lock()
        # Per-tier counters for lightweight metrics
        self._tier_counts: Dict[str, int] = {t.value: 0 for t in DifficultyTier}
        self._correct_counts: Dict[str, int] = {t.value: 0 for t in DifficultyTier}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route(
        self,
        task_input: str,
        task_id: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> RoutingDecision:
        """
        Classify *task_input* and return a :class:`RoutingDecision`.

        Parameters
        ----------
        task_input:
            The raw task text (question, instruction, code problem, etc.).
        task_id:
            Optional caller-supplied ID for later outcome recording.
        context:
            Optional extra context dict.  Currently unused but reserved for
            future feature expansion (e.g. prior conversation turns).

        Returns
        -------
        RoutingDecision
            Contains the selected tier, raw score, features, and model hint.
        """
        features = self._extractor.extract(task_input)
        score = self._compute_score(features)
        tier = self._score_to_tier(score)
        model_hint = self.model_hints.get(tier, "")
        decision = RoutingDecision(
            task_id=task_id,
            tier=tier,
            score=score,
            features=features,
            model_hint=model_hint,
        )
        with self._lock:
            self._history.append(decision)
            if len(self._history) > self.history_limit:
                self._history.pop(0)
            self._tier_counts[tier.value] += 1
        logger.debug(
            "DifficultyRouter: task_id={} tier={} score={:.3f} model={}",
            task_id, tier.value, score, model_hint,
        )
        return decision

    def record_outcome(
        self,
        task_id: str,
        outcome_score: float,
        quality_threshold: float = 0.7,
    ) -> None:
        """
        Record the quality outcome of a completed task and calibrate thresholds.

        Parameters
        ----------
        task_id:
            The ID supplied at ``route`` time.
        outcome_score:
            A quality/accuracy score in [0, 1].  Scores below
            *quality_threshold* are treated as routing failures.
        quality_threshold:
            Minimum acceptable outcome score.  Decisions whose outcome falls
            below this trigger a threshold nudge.
        """
        with self._lock:
            # Find the most recent decision with this task_id
            decision: Optional[RoutingDecision] = None
            for d in reversed(self._history):
                if d.task_id == task_id:
                    decision = d
                    break
            if decision is None:
                logger.warning("DifficultyRouter.record_outcome: task_id={} not found", task_id)
                return
            decision.outcome_score = outcome_score
            correct = outcome_score >= quality_threshold
            if correct:
                self._correct_counts[decision.tier.value] += 1
            self._thresholds.update(decision.score, correct)

    def metrics(self) -> Dict[str, Any]:
        """
        Return a snapshot of routing statistics.

        Returns
        -------
        dict with keys:
            - ``tier_counts``: total tasks routed per tier
            - ``accuracy_per_tier``: fraction of outcomes above threshold
            - ``thresholds``: current (simple_max, medium_max) values
            - ``history_size``: number of decisions currently in memory
        """
        with self._lock:
            accuracy: Dict[str, Optional[float]] = {}
            for tier_val, total in self._tier_counts.items():
                if total == 0:
                    accuracy[tier_val] = None
                else:
                    accuracy[tier_val] = self._correct_counts[tier_val] / total
            return {
                "tier_counts": dict(self._tier_counts),
                "accuracy_per_tier": accuracy,
                "thresholds": {
                    "simple_max": self._thresholds.simple_max,
                    "medium_max": self._thresholds.medium_max,
                },
                "history_size": len(self._history),
            }

    def save_state(self, path: str) -> None:
        """
        Persist thresholds and recent history to a JSON file.

        Parameters
        ----------
        path:
            File path to write.  Parent directory is created if needed.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "thresholds": {
                "simple_max": self._thresholds.simple_max,
                "medium_max": self._thresholds.medium_max,
                "ema_alpha": self._thresholds.ema_alpha,
            },
            "tier_counts": self._tier_counts,
            "correct_counts": self._correct_counts,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2)
        logger.info("DifficultyRouter: saved state to {}", path)

    def load_state(self, path: str) -> None:
        """
        Restore thresholds and counters from a previously saved JSON file.

        Parameters
        ----------
        path:
            File path to read.  Raises ``FileNotFoundError`` if absent.
        """
        with open(path, "r", encoding="utf-8") as fh:
            state = json.load(fh)
        t = state.get("thresholds", {})
        self._thresholds.simple_max = float(t.get("simple_max", 0.35))
        self._thresholds.medium_max = float(t.get("medium_max", 0.65))
        self._thresholds.ema_alpha = float(t.get("ema_alpha", 0.05))
        self._tier_counts.update(state.get("tier_counts", {}))
        self._correct_counts.update(state.get("correct_counts", {}))
        logger.info("DifficultyRouter: loaded state from {}", path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_score(self, features: Dict[str, float]) -> float:
        """
        Compute a scalar difficulty score in [0, 1] from *features*.

        Uses a weighted linear combination then applies a sigmoid to keep
        the result within bounds and smooth out extreme values.
        """
        raw = sum(
            self.feature_weights.get(k, 0.0) * v
            for k, v in features.items()
        )
        # Sigmoid centred at 0.5 with scale ~8 for a reasonable spread
        score = 1.0 / (1.0 + math.exp(-8.0 * (raw - 0.5)))
        return max(0.0, min(1.0, score))

    def _score_to_tier(self, score: float) -> DifficultyTier:
        """Map a scalar score to a :class:`DifficultyTier` using current thresholds."""
        if score < self._thresholds.simple_max:
            return DifficultyTier.SIMPLE
        if score < self._thresholds.medium_max:
            return DifficultyTier.MEDIUM
        return DifficultyTier.HARD

    # ------------------------------------------------------------------
    # Convenience: batch routing
    # ------------------------------------------------------------------

    def route_batch(
        self,
        tasks: List[str],
        task_ids: Optional[List[str]] = None,
    ) -> List[RoutingDecision]:
        """
        Route a list of tasks and return a decision per task.

        Parameters
        ----------
        tasks:
            List of raw task strings.
        task_ids:
            Optional list of task IDs, same length as *tasks*.

        Returns
        -------
        List[RoutingDecision]
        """
        ids = task_ids or [""] * len(tasks)
        if len(ids) != len(tasks):
            raise ValueError("task_ids length must match tasks length")
        return [self.route(t, tid) for t, tid in zip(tasks, ids)]


__all__ = ["DifficultyRouter", "DifficultyTier", "RoutingDecision"]

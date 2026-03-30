"""Cost tracking for LLM calls across optimization runs.

Intercepts every LLM call, records input/output tokens, maps them to a cost
per provider, and maintains per-session totals.  Raises ``CostBudgetExceeded``
when a configured budget ceiling is breached.

Usage
-----
Basic (global tracker, no session boundaries):

    >>> from evoagentx.core.cost_tracker import get_tracker
    >>> tracker = get_tracker()
    >>> tracker.record("openai", "gpt-4o", input_tokens=500, output_tokens=100)
    >>> print(tracker.total_cost())

Session-scoped (recommended for per-run isolation):

    >>> with tracker.session("textgrad-run-1"):
    ...     result = optimizer.optimize(dataset)
    ...     cost = tracker.session_cost()   # cost for this session only
    ...     print(f"Optimization cost: ${cost:.4f}")

Budget enforcement:

    >>> tracker.set_budget(max_usd=5.00)
    >>> tracker.record("openai", "gpt-4o", ...)   # raises if cumulative > $5
"""

from __future__ import annotations

import threading
import contextvars
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .logging import logger


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class CostBudgetExceeded(Exception):
    """Raised when cumulative cost surpasses the configured budget ceiling."""

    def __init__(self, current_cost: float, budget: float, context: str = ""):
        msg = (
            f"Cost budget exceeded: accumulated ${current_cost:.4f} "
            f"surpasses limit of ${budget:.4f}"
        )
        if context:
            msg += f" (context: {context})"
        super().__init__(msg)
        self.current_cost = current_cost
        self.budget = budget


# ---------------------------------------------------------------------------
# Pricing table
# ---------------------------------------------------------------------------

@dataclass
class ModelPricing:
    """Cost in USD per 1 000 tokens for a specific model."""

    input_cost_per_1k: float   # USD per 1K input tokens
    output_cost_per_1k: float  # USD per 1K output tokens

    def compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Return the total cost in USD for the given token counts."""
        return (
            input_tokens * self.input_cost_per_1k / 1000.0
            + output_tokens * self.output_cost_per_1k / 1000.0
        )


# Pricing as of early 2026 — update as providers change their rates.
# Key format: "<provider>/<model>" (lower-cased).
_DEFAULT_PRICING: Dict[str, ModelPricing] = {
    # OpenAI
    "openai/gpt-4o":                  ModelPricing(input_cost_per_1k=0.0025, output_cost_per_1k=0.010),
    "openai/gpt-4o-mini":             ModelPricing(input_cost_per_1k=0.00015, output_cost_per_1k=0.0006),
    "openai/gpt-4-turbo":             ModelPricing(input_cost_per_1k=0.010, output_cost_per_1k=0.030),
    "openai/gpt-4":                   ModelPricing(input_cost_per_1k=0.030, output_cost_per_1k=0.060),
    "openai/gpt-3.5-turbo":           ModelPricing(input_cost_per_1k=0.0005, output_cost_per_1k=0.0015),
    "openai/o1":                      ModelPricing(input_cost_per_1k=0.015, output_cost_per_1k=0.060),
    "openai/o3-mini":                 ModelPricing(input_cost_per_1k=0.0011, output_cost_per_1k=0.0044),
    # Anthropic / Claude
    "anthropic/claude-opus-4-6":      ModelPricing(input_cost_per_1k=0.015, output_cost_per_1k=0.075),
    "anthropic/claude-sonnet-4-6":    ModelPricing(input_cost_per_1k=0.003, output_cost_per_1k=0.015),
    "anthropic/claude-haiku-4-5":     ModelPricing(input_cost_per_1k=0.0008, output_cost_per_1k=0.004),
    # DeepSeek
    "deepseek/deepseek-chat":         ModelPricing(input_cost_per_1k=0.00014, output_cost_per_1k=0.00028),
    "deepseek/deepseek-reasoner":     ModelPricing(input_cost_per_1k=0.00055, output_cost_per_1k=0.00219),
    # Kimi / Moonshot
    "moonshot/moonshot-v1-8k":        ModelPricing(input_cost_per_1k=0.00120, output_cost_per_1k=0.00120),
    "moonshot/moonshot-v1-32k":       ModelPricing(input_cost_per_1k=0.00240, output_cost_per_1k=0.00240),
    # Google Gemini
    "google/gemini-2.0-flash":        ModelPricing(input_cost_per_1k=0.00010, output_cost_per_1k=0.00040),
    "google/gemini-1.5-pro":          ModelPricing(input_cost_per_1k=0.00125, output_cost_per_1k=0.005),
    # Ollama / local (treat as free)
    "ollama/llama3":                  ModelPricing(input_cost_per_1k=0.0, output_cost_per_1k=0.0),
    "ollama/mistral":                 ModelPricing(input_cost_per_1k=0.0, output_cost_per_1k=0.0),
}

# Fallback pricing for unknown models (conservative estimate to avoid surprises)
_FALLBACK_PRICING = ModelPricing(input_cost_per_1k=0.002, output_cost_per_1k=0.008)


def _resolve_pricing(provider: str, model: str) -> ModelPricing:
    """Return the best matching ModelPricing for the given provider/model pair."""
    key = f"{provider.lower()}/{model.lower()}"
    if key in _DEFAULT_PRICING:
        return _DEFAULT_PRICING[key]
    # Try prefix match (e.g. "gpt-4o-2024-11-20" matches "openai/gpt-4o")
    for registered_key, pricing in _DEFAULT_PRICING.items():
        reg_provider, reg_model = registered_key.split("/", 1)
        if provider.lower() == reg_provider and model.lower().startswith(reg_model):
            return pricing
    return _FALLBACK_PRICING


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class UsageRecord:
    """Single LLM call usage record."""

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: float = field(default_factory=time.time)
    session_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class CostSummary:
    """Aggregated cost summary over a set of records."""

    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int
    total_calls: int
    per_model: Dict[str, Dict] = field(default_factory=dict)
    per_session: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            f"Total cost: ${self.total_cost_usd:.4f}",
            f"Total calls: {self.total_calls}",
            f"Input tokens: {self.total_input_tokens:,}",
            f"Output tokens: {self.total_output_tokens:,}",
        ]
        if self.per_model:
            lines.append("Per-model breakdown:")
            for key, info in sorted(self.per_model.items()):
                lines.append(
                    f"  {key}: ${info['cost_usd']:.4f} "
                    f"({info['calls']} calls, "
                    f"{info['input_tokens']:,} in / {info['output_tokens']:,} out)"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# ContextVar: active session ID
# ---------------------------------------------------------------------------

_active_session: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_active_session", default=None
)


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------

class CostTracker:
    """Thread-safe, session-aware cost tracker for LLM calls.

    A single global instance is provided via :func:`get_tracker`.  Multiple
    independent instances can be created for testing.

    Parameters
    ----------
    max_budget_usd:
        When set, any ``record()`` call that pushes cumulative cost above this
        value raises ``CostBudgetExceeded``.  Set per-session with
        ``set_budget()`` inside a ``session()`` context.
    """

    def __init__(self, max_budget_usd: Optional[float] = None) -> None:
        self._lock = threading.Lock()
        self._records: List[UsageRecord] = []
        self._custom_pricing: Dict[str, ModelPricing] = {}
        self._global_budget: Optional[float] = max_budget_usd
        # session_id -> budget override
        self._session_budgets: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Pricing registry
    # ------------------------------------------------------------------

    def register_pricing(self, provider: str, model: str, pricing: ModelPricing) -> None:
        """Register or override pricing for a specific provider/model pair."""
        key = f"{provider.lower()}/{model.lower()}"
        self._custom_pricing[key] = pricing

    def _get_pricing(self, provider: str, model: str) -> ModelPricing:
        key = f"{provider.lower()}/{model.lower()}"
        if key in self._custom_pricing:
            return self._custom_pricing[key]
        return _resolve_pricing(provider, model)

    # ------------------------------------------------------------------
    # Recording usage
    # ------------------------------------------------------------------

    def record(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        metadata: Optional[Dict] = None,
    ) -> UsageRecord:
        """Record a single LLM call and check budget constraints.

        Parameters
        ----------
        provider:
            LLM provider name, e.g. ``"openai"``, ``"anthropic"``, ``"deepseek"``.
        model:
            Exact model identifier, e.g. ``"gpt-4o"``, ``"claude-opus-4-6"``.
        input_tokens:
            Number of prompt (input) tokens consumed.
        output_tokens:
            Number of completion (output) tokens generated.
        metadata:
            Optional dict of arbitrary metadata attached to the record.

        Returns
        -------
        UsageRecord
            The recorded entry.

        Raises
        ------
        CostBudgetExceeded
            When cumulative cost (global or session) exceeds the configured budget.
        """
        pricing = self._get_pricing(provider, model)
        cost = pricing.compute_cost(input_tokens, output_tokens)
        session_id = _active_session.get()

        record = UsageRecord(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            session_id=session_id,
            metadata=metadata or {},
        )

        with self._lock:
            self._records.append(record)
            self._check_budgets(session_id, cost)

        logger.debug(
            f"CostTracker | {provider}/{model} | "
            f"in={input_tokens} out={output_tokens} | "
            f"cost=${cost:.5f} | session={session_id}"
        )
        return record

    def _check_budgets(self, session_id: Optional[str], new_cost: float) -> None:
        """Raise CostBudgetExceeded if any budget ceiling is breached."""
        # Global budget check
        if self._global_budget is not None:
            total = sum(r.cost_usd for r in self._records)
            if total > self._global_budget:
                raise CostBudgetExceeded(total, self._global_budget)

        # Session-specific budget check
        if session_id and session_id in self._session_budgets:
            session_total = sum(
                r.cost_usd for r in self._records if r.session_id == session_id
            )
            budget = self._session_budgets[session_id]
            if session_total > budget:
                raise CostBudgetExceeded(session_total, budget, context=session_id)

    # ------------------------------------------------------------------
    # Budget management
    # ------------------------------------------------------------------

    def set_budget(self, max_usd: float, session_id: Optional[str] = None) -> None:
        """Set a budget ceiling.

        If ``session_id`` is provided (or a session is currently active), the
        budget applies only to that session.  Otherwise it applies globally.
        """
        sid = session_id or _active_session.get()
        if sid:
            self._session_budgets[sid] = max_usd
        else:
            self._global_budget = max_usd

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    @contextmanager
    def session(self, session_id: str):
        """Context manager that scopes all records to ``session_id``.

        Example
        -------
        >>> with tracker.session("run-001"):
        ...     optimizer.optimize(dataset)
        ...     print(tracker.session_cost())
        """
        token = _active_session.set(session_id)
        try:
            yield self
        finally:
            _active_session.reset(token)

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def total_cost(self, session_id: Optional[str] = None) -> float:
        """Return total cost in USD, optionally filtered to a session."""
        with self._lock:
            records = self._filter_records(session_id)
            return sum(r.cost_usd for r in records)

    def session_cost(self) -> float:
        """Return cost for the currently active session (0.0 if none)."""
        sid = _active_session.get()
        if sid is None:
            return 0.0
        return self.total_cost(session_id=sid)

    def total_tokens(
        self, session_id: Optional[str] = None
    ) -> tuple[int, int]:
        """Return (input_tokens, output_tokens), optionally filtered to a session."""
        with self._lock:
            records = self._filter_records(session_id)
            inp = sum(r.input_tokens for r in records)
            out = sum(r.output_tokens for r in records)
        return inp, out

    def summary(self, session_id: Optional[str] = None) -> CostSummary:
        """Return a :class:`CostSummary` aggregating all (or session-filtered) records."""
        with self._lock:
            records = self._filter_records(session_id)

        per_model: Dict[str, Dict] = defaultdict(
            lambda: {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "calls": 0}
        )
        per_session: Dict[str, float] = defaultdict(float)

        for r in records:
            key = f"{r.provider}/{r.model}"
            per_model[key]["cost_usd"] += r.cost_usd
            per_model[key]["input_tokens"] += r.input_tokens
            per_model[key]["output_tokens"] += r.output_tokens
            per_model[key]["calls"] += 1
            if r.session_id:
                per_session[r.session_id] += r.cost_usd

        return CostSummary(
            total_cost_usd=sum(r.cost_usd for r in records),
            total_input_tokens=sum(r.input_tokens for r in records),
            total_output_tokens=sum(r.output_tokens for r in records),
            total_calls=len(records),
            per_model=dict(per_model),
            per_session=dict(per_session),
        )

    def records(self, session_id: Optional[str] = None) -> List[UsageRecord]:
        """Return a copy of all (or session-filtered) usage records."""
        with self._lock:
            return list(self._filter_records(session_id))

    def _filter_records(self, session_id: Optional[str]) -> List[UsageRecord]:
        if session_id is None:
            return self._records
        return [r for r in self._records if r.session_id == session_id]

    def reset(self, session_id: Optional[str] = None) -> None:
        """Clear all records (or only those belonging to ``session_id``)."""
        with self._lock:
            if session_id is None:
                self._records.clear()
            else:
                self._records = [r for r in self._records if r.session_id != session_id]


# ---------------------------------------------------------------------------
# Token estimation (used when actual counts are unavailable)
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token count estimate from raw text.

    Uses a 4-chars-per-token heuristic, which is reasonably accurate for
    English prose and code without requiring a tokenizer dependency.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: List[Dict]) -> int:
    """Estimate token count for a list of chat messages.

    Each message dict is expected to have at least a ``"content"`` key.
    """
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            # multimodal content list
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    total += estimate_tokens(item.get("text", ""))
        # add ~4 tokens per message for role/format overhead
        total += 4
    return total


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_global_tracker: Optional[CostTracker] = None
_tracker_lock = threading.Lock()


def get_tracker() -> CostTracker:
    """Return the process-wide :class:`CostTracker` singleton.

    Creates the instance on first call (thread-safe).
    """
    global _global_tracker
    if _global_tracker is None:
        with _tracker_lock:
            if _global_tracker is None:
                _global_tracker = CostTracker()
    return _global_tracker

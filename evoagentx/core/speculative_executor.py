"""
PASTE — Predictive Async Speculative Tool Execution.

While the LLM is still generating a response, PASTE predicts and pre-executes
the *next* tool call in the background.  When the LLM finally emits its decision,
the pre-computed result is returned immediately (cache hit) or discarded (miss).
This achieves significant latency reduction for sequential tool chains where the
next step is predictable from history (e.g. search → parse → summarise).

PASTE is *complementary* to :class:`~evoagentx.core.parallel_executor.ParallelExecutor`:
- ``ParallelExecutor`` handles *independent* calls (same LLM turn, topological sort).
- ``SpeculativeExecutor`` handles *sequential* calls across turns (predict ahead).

Key classes
-----------
SpeculativeConfig
    Tuneable parameters: depth, confidence threshold, parallelism cap, timeouts.
ToolCallRecord
    Lightweight record of one completed tool call kept as prediction history.
ToolPrediction
    A single prediction: tool name, expected args, confidence, reasoning.
SpeculationResult
    Outcome of a resolve() call: whether the speculation was used and how much
    latency was saved.
SpeculativeExecutor
    Orchestrates prediction, background execution, and resolution.

Usage
-----
::

    from evoagentx.core.speculative_executor import (
        SpeculativeExecutor, SpeculativeConfig, ToolCallRecord
    )

    config = SpeculativeConfig(confidence_threshold=0.7)
    executor = SpeculativeExecutor(config=config)

    # After each tool call:
    history = [ToolCallRecord("search", {"q": "foo"}, "result...")]
    prediction = executor.predict_next_tool(history, available_tools=["parse", "summarise"])

    if prediction:
        await executor.speculatively_execute(prediction, tool_registry={
            "parse": parse_fn,
            "summarise": summarise_fn,
        })

    # ... LLM decides which tool to call next ...

    result = await executor.resolve("parse", {"text": "result..."})
    if result and result.was_used:
        parsed = result.result          # pre-computed — no extra latency
    else:
        parsed = await parse_fn(...)    # fallback: execute normally
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .logging import logger
from .cost_tracker import CostTracker


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SpeculativeConfig:
    """
    Configuration for :class:`SpeculativeExecutor`.

    Attributes
    ----------
    max_speculative_depth:
        How many sequential steps ahead to predict.  Currently only depth=1
        speculations are launched; larger values allow deeper pattern context
        windows (i.e. longer n-gram patterns).  Default: 1.
    confidence_threshold:
        Minimum confidence score [0, 1] required to launch a speculation.
        Predictions below this threshold are discarded.  Default: 0.7.
    max_parallel_speculations:
        Maximum number of speculative tasks that may run simultaneously.
        When the cap is reached, new speculations are skipped.  Default: 2.
    discard_on_mismatch:
        When True (default), all active speculations that don't match the
        LLM's actual decision are cancelled immediately during ``resolve()``.
    timeout_per_speculation:
        Per-task timeout in seconds.  ``None`` means no timeout.
    """
    max_speculative_depth: int = 1
    confidence_threshold: float = 0.7
    max_parallel_speculations: int = 2
    discard_on_mismatch: bool = True
    timeout_per_speculation: Optional[float] = None


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ToolCallRecord:
    """
    A completed tool-call entry in the execution history.

    Attributes
    ----------
    tool:
        Name of the tool that was called.
    args:
        Keyword arguments passed to the tool.
    result:
        Return value of the tool call.
    duration_ms:
        Wall-clock time the call took in milliseconds.
    """
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    duration_ms: float = 0.0


@dataclass
class ToolPrediction:
    """
    A single prediction for the next tool call.

    Attributes
    ----------
    predicted_tool:
        Name of the tool predicted to be called next.
    predicted_args:
        Expected keyword arguments.  Empty dict when prediction is based on
        pattern matching alone (tool-name-level prediction only).
    confidence:
        Score in [0, 1] reflecting prediction reliability.
    basis:
        Human-readable explanation of why this prediction was made.
    """
    predicted_tool: str
    predicted_args: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    basis: str = ""


@dataclass
class SpeculationResult:
    """
    Outcome of a :meth:`SpeculativeExecutor.resolve` call.

    Attributes
    ----------
    prediction:
        The :class:`ToolPrediction` that was (or wasn't) used.
    result:
        Pre-computed tool output if *was_used* is True; ``None`` otherwise.
    was_used:
        True when the speculation matched the LLM's actual decision.
    latency_saved_ms:
        Estimated wall-clock milliseconds saved by the speculation.
    """
    prediction: ToolPrediction
    result: Any
    was_used: bool
    latency_saved_ms: float = 0.0


# ---------------------------------------------------------------------------
# Internal stats accumulator
# ---------------------------------------------------------------------------

@dataclass
class _SpecStats:
    hits: int = 0
    misses: int = 0
    total_latency_saved_ms: float = 0.0
    wasted_ms: float = 0.0


# ---------------------------------------------------------------------------
# Speculative executor
# ---------------------------------------------------------------------------

class SpeculativeExecutor:
    """
    Predict-ahead speculative tool executor (PASTE).

    The executor maintains a rolling prediction model built from the agent's
    tool-call history.  At each step it can:

    1. Predict which tool is likely to be called next.
    2. Launch that tool in an asyncio background task.
    3. When the LLM makes its actual decision, check for a match:
       - Hit  → return the pre-computed result, saving the tool's latency.
       - Miss → cancel the speculation, execute normally.

    Parameters
    ----------
    config:
        :class:`SpeculativeConfig` instance.  Defaults to standard settings.
    cost_tracker:
        Optional :class:`~evoagentx.core.cost_tracker.CostTracker` for
        accounting speculative calls separately from real calls.
    """

    def __init__(
        self,
        config: Optional[SpeculativeConfig] = None,
        cost_tracker: Optional[CostTracker] = None,
    ) -> None:
        self.config = config or SpeculativeConfig()
        self._cost_tracker = cost_tracker
        # tool_name -> (task, prediction, start_time_ms)
        self._active: Dict[str, Tuple[asyncio.Task, ToolPrediction, float]] = {}
        self._stats = _SpecStats()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_next_tool(
        self,
        history: List[ToolCallRecord],
        available_tools: List[str],
        llm_predictor: Optional[Callable[[List[ToolCallRecord], List[str]], Optional[ToolPrediction]]] = None,
    ) -> Optional[ToolPrediction]:
        """
        Predict which tool will be called next given the execution history.

        First tries *pattern matching* on tool-call sequences.  If an optional
        *llm_predictor* callable is supplied and pattern matching yields no
        high-confidence result, the LLM predictor is tried as a fallback.

        Parameters
        ----------
        history:
            Ordered list of completed :class:`ToolCallRecord` objects.
        available_tools:
            Names of tools the agent can legally call.  Predictions not in
            this list are silently discarded.
        llm_predictor:
            Optional callable ``(history, available_tools) → ToolPrediction | None``
            that uses an LLM to produce a prediction when pattern matching fails.

        Returns
        -------
        ToolPrediction or None
            A prediction above ``confidence_threshold``, or ``None``.
        """
        if not history:
            return None

        # Try pattern-based prediction first
        match = self._pattern_match(history)
        if match is not None:
            tool, args, confidence, basis = match
            if tool in available_tools and confidence >= self.config.confidence_threshold:
                return ToolPrediction(
                    predicted_tool=tool,
                    predicted_args=args,
                    confidence=confidence,
                    basis=basis,
                )

        # Fall back to LLM predictor if supplied
        if llm_predictor is not None:
            prediction = llm_predictor(history, available_tools)
            if prediction is not None and prediction.predicted_tool in available_tools:
                if prediction.confidence >= self.config.confidence_threshold:
                    return prediction

        return None

    # ------------------------------------------------------------------
    # Speculative execution
    # ------------------------------------------------------------------

    async def speculatively_execute(
        self,
        prediction: ToolPrediction,
        tool_registry: Dict[str, Callable],
    ) -> bool:
        """
        Launch a background asyncio task for the predicted tool call.

        The task is stored internally and later examined by :meth:`resolve`.

        Parameters
        ----------
        prediction:
            The prediction to speculatively execute.
        tool_registry:
            Mapping of tool name → callable (sync or async).

        Returns
        -------
        bool
            True if the speculation was started; False if skipped (confidence
            too low, cap reached, tool unknown, or already speculating on it).
        """
        if prediction.confidence < self.config.confidence_threshold:
            logger.debug(
                "SpeculativeExecutor: skipping '{}' — confidence {:.3f} < threshold {:.3f}",
                prediction.predicted_tool, prediction.confidence, self.config.confidence_threshold,
            )
            return False

        if len(self._active) >= self.config.max_parallel_speculations:
            logger.debug(
                "SpeculativeExecutor: speculation cap ({}) reached, skipping '{}'",
                self.config.max_parallel_speculations, prediction.predicted_tool,
            )
            return False

        if prediction.predicted_tool in self._active:
            logger.debug(
                "SpeculativeExecutor: already speculating on '{}'",
                prediction.predicted_tool,
            )
            return False

        tool_fn = tool_registry.get(prediction.predicted_tool)
        if tool_fn is None:
            logger.debug(
                "SpeculativeExecutor: tool '{}' not in registry",
                prediction.predicted_tool,
            )
            return False

        start_ms = time.perf_counter() * 1000.0

        async def _run() -> Any:
            timeout = self.config.timeout_per_speculation

            async def _invoke() -> Any:
                if asyncio.iscoroutinefunction(tool_fn):
                    return await tool_fn(**prediction.predicted_args)
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: tool_fn(**prediction.predicted_args),
                )

            if timeout is not None:
                return await asyncio.wait_for(_invoke(), timeout=timeout)
            return await _invoke()

        task = asyncio.create_task(_run())
        self._active[prediction.predicted_tool] = (task, prediction, start_ms)

        logger.debug(
            "SpeculativeExecutor: launched speculation for '{}' (confidence={:.3f})",
            prediction.predicted_tool, prediction.confidence,
        )

        # Record in cost tracker as speculative (zero tokens — tool call, not LLM call)
        if self._cost_tracker is not None:
            try:
                self._cost_tracker.record(
                    provider="speculative",
                    model=prediction.predicted_tool,
                    input_tokens=0,
                    output_tokens=0,
                    metadata={
                        "type": "speculative_tool",
                        "confidence": prediction.confidence,
                        "basis": prediction.basis,
                    },
                )
            except Exception:
                pass  # Never let cost tracking break execution

        return True

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    async def resolve(
        self,
        actual_tool: str,
        actual_args: Optional[Dict[str, Any]] = None,
    ) -> Optional[SpeculationResult]:
        """
        Check whether the LLM's actual decision matches any active speculation.

        Parameters
        ----------
        actual_tool:
            The tool name the LLM decided to call.
        actual_args:
            The actual keyword arguments.  When the prediction had non-empty
            ``predicted_args``, they must match exactly for a cache hit.

        Returns
        -------
        SpeculationResult or None
            A result with ``was_used=True`` on a cache hit; ``None`` when no
            matching speculation existed.
        """
        if actual_args is None:
            actual_args = {}

        # Pop the matching speculation (if any) before cancelling others
        matched = self._active.pop(actual_tool, None)

        # Cancel all remaining non-matching speculations
        if self.config.discard_on_mismatch and self._active:
            await self._cancel_all_active()

        if matched is None:
            self._stats.misses += 1
            logger.debug(
                "SpeculativeExecutor: miss — no speculation for '{}'", actual_tool
            )
            return None

        task, prediction, start_ms = matched

        # Args match check: only validate when prediction provided specific args
        if prediction.predicted_args and prediction.predicted_args != actual_args:
            elapsed_ms = time.perf_counter() * 1000.0 - start_ms
            self._stats.wasted_ms += elapsed_ms
            await self._cancel_task(task)
            self._stats.misses += 1
            logger.debug(
                "SpeculativeExecutor: miss — args mismatch for '{}'", actual_tool
            )
            return None

        # Wait for the speculation to finish (may already be done)
        resolve_start_ms = time.perf_counter() * 1000.0
        try:
            result = await task
        except asyncio.TimeoutError:
            self._stats.misses += 1
            elapsed_ms = time.perf_counter() * 1000.0 - start_ms
            self._stats.wasted_ms += elapsed_ms
            logger.warning(
                "SpeculativeExecutor: speculation '{}' timed out at resolve", actual_tool
            )
            return None
        except asyncio.CancelledError:
            self._stats.misses += 1
            return None
        except Exception as exc:
            self._stats.misses += 1
            logger.warning(
                "SpeculativeExecutor: speculation '{}' raised {}: {}",
                actual_tool, type(exc).__name__, exc,
            )
            return None

        # Latency saved = wall-clock ms the speculation was already running
        # before resolve() was called.
        latency_saved_ms = max(0.0, resolve_start_ms - start_ms)
        self._stats.hits += 1
        self._stats.total_latency_saved_ms += latency_saved_ms

        logger.debug(
            "SpeculativeExecutor: hit for '{}' — saved {:.1f} ms",
            actual_tool, latency_saved_ms,
        )

        return SpeculationResult(
            prediction=prediction,
            result=result,
            was_used=True,
            latency_saved_ms=latency_saved_ms,
        )

    # ------------------------------------------------------------------
    # Pattern matching
    # ------------------------------------------------------------------

    def _pattern_match(
        self,
        history: List[ToolCallRecord],
    ) -> Optional[Tuple[str, Dict[str, Any], float, str]]:
        """
        Find repeated n-gram sequences in *history* to predict the next tool.

        Scans the tool-name sequence for the longest suffix that also appears
        earlier in the history (i.e. has a known "continuation").  Among all
        continuations of that suffix, picks the most frequent one.

        Returns
        -------
        (tool_name, args, confidence, basis) or None
            ``args`` is always an empty dict for pattern-based predictions.
        """
        tool_names = [r.tool for r in history]
        n = len(tool_names)

        if n < 2:
            return None

        best_tool: Optional[str] = None
        best_confidence = 0.0
        best_basis = ""

        # Try context lengths from max_speculative_depth+1 down to 1.
        # +1 because the context window itself is 1 longer than the depth
        # (we need at least depth=1 items before what we're predicting).
        max_ctx_len = min(n - 1, self.config.max_speculative_depth + 1)

        for ctx_len in range(max_ctx_len, 0, -1):
            # The "context" is the last ctx_len tool names in history
            context = tool_names[-ctx_len:]

            # Find every occurrence of this context that has a successor
            next_tools: List[str] = []
            for i in range(n - ctx_len):
                if tool_names[i : i + ctx_len] == context:
                    # tool_names[i + ctx_len] is what followed this context
                    if i + ctx_len < n:
                        next_tools.append(tool_names[i + ctx_len])

            if not next_tools:
                continue

            counts: Counter = Counter(next_tools)
            predicted_next, freq = counts.most_common(1)[0]
            confidence = self._confidence_from_pattern(ctx_len, freq, len(next_tools))

            if confidence > best_confidence:
                best_confidence = confidence
                best_tool = predicted_next
                best_basis = (
                    f"pattern(ctx_len={ctx_len}, "
                    f"freq={freq}/{len(next_tools)}, "
                    f"conf={confidence:.3f})"
                )

        if best_tool is None:
            return None

        return (best_tool, {}, best_confidence, best_basis)

    def _confidence_from_pattern(
        self,
        pattern_length: int,
        freq: int,
        total: int,
    ) -> float:
        """
        Compute prediction confidence from pattern statistics.

        Combines two signals:

        * *Frequency ratio* ``freq / total``: how consistently the pattern
          predicts this next tool (1.0 = always, 0.5 = half the time).
        * *Length bonus*: longer patterns are more specific and therefore more
          reliable.  Bonus grows as ``1 - 1/(L+1)``:
          L=1 → 0.50, L=2 → 0.67, L=3 → 0.75, L=∞ → 1.00.

        Combined score: ``freq_ratio * (0.5 + 0.5 * length_bonus)``.

        Parameters
        ----------
        pattern_length:
            Number of tool names in the matched context window.
        freq:
            How many times this specific next tool was observed after the context.
        total:
            Total observations after the context (including other next-tools).
        """
        if total <= 0:
            return 0.0
        freq_ratio = freq / total
        length_bonus = 1.0 - 1.0 / (pattern_length + 1)
        return freq_ratio * (0.5 + 0.5 * length_bonus)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """
        Return a snapshot of speculation statistics.

        Keys
        ----
        hits, misses, total
            Raw counts.
        hit_rate, miss_rate
            Fractions in [0, 1].
        total_latency_saved_ms
            Cumulative latency saved across all cache hits.
        avg_latency_saved_ms
            Average per hit (0.0 if no hits).
        total_wasted_ms
            Cumulative compute wasted on cancelled / mismatched speculations.
        active_speculations
            Number of currently running background tasks.
        """
        total = self._stats.hits + self._stats.misses
        hit_rate = self._stats.hits / total if total > 0 else 0.0
        avg_saved = (
            self._stats.total_latency_saved_ms / self._stats.hits
            if self._stats.hits > 0
            else 0.0
        )
        return {
            "hits": self._stats.hits,
            "misses": self._stats.misses,
            "total": total,
            "hit_rate": hit_rate,
            "miss_rate": 1.0 - hit_rate,
            "total_latency_saved_ms": self._stats.total_latency_saved_ms,
            "avg_latency_saved_ms": avg_saved,
            "total_wasted_ms": self._stats.wasted_ms,
            "active_speculations": len(self._active),
        }

    def reset_stats(self) -> None:
        """Reset all statistics counters."""
        self._stats = _SpecStats()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _cancel_all_active(self) -> None:
        """Cancel and await all active speculative tasks."""
        for tool_name, (task, _pred, start_ms) in list(self._active.items()):
            elapsed_ms = time.perf_counter() * 1000.0 - start_ms
            self._stats.wasted_ms += max(0.0, elapsed_ms)
            await self._cancel_task(task)
        self._active.clear()

    @staticmethod
    async def _cancel_task(task: asyncio.Task) -> None:
        """Cancel a task and swallow the resulting CancelledError."""
        if not task.done():
            task.cancel()
        try:
            await task
        except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
            pass

    # ------------------------------------------------------------------
    # Sync convenience wrapper
    # ------------------------------------------------------------------

    def resolve_sync(
        self,
        actual_tool: str,
        actual_args: Optional[Dict[str, Any]] = None,
    ) -> Optional[SpeculationResult]:
        """Synchronous wrapper around :meth:`resolve` for non-async contexts."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(
                        asyncio.run, self.resolve(actual_tool, actual_args)
                    )
                    return future.result()
            return loop.run_until_complete(self.resolve(actual_tool, actual_args))
        except RuntimeError:
            return asyncio.run(self.resolve(actual_tool, actual_args))


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    "SpeculativeExecutor",
    "SpeculativeConfig",
    "ToolCallRecord",
    "ToolPrediction",
    "SpeculationResult",
]

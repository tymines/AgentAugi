"""
Cycle support layer for WorkFlowGraph.

The base :class:`~evoagentx.workflow.workflow_graph.WorkFlowGraph` already
can *represent* cycles (it detects them via DFS), but it does not manage
iteration bookkeeping, break conditions, or per-cycle strategy escalation.

This module provides:

``CycleConfig``
    Declarative per-cycle configuration (max iterations, break conditions,
    strategy escalation).

``CycleState``
    Runtime state for an active cycle: iteration counter, convergence
    history, break reason.

``CyclicWorkFlowGraph``
    WorkFlowGraph subclass that wraps the existing graph and adds managed
    cycle execution semantics.

Design notes
------------
* Every cycle in the graph is identified by its start node name (matching
  the key used in ``WorkFlowGraph._loops``).
* Callers advance cycles by calling ``step_cycle(start_node)`` after each
  loop-end node completes.
* The graph signals when a cycle should *stop* via ``should_break_cycle``.
* Strategy escalation maps iteration numbers to strategy labels so the
  caller can select heavier approaches on later iterations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import Field

from ..core.logging import logger
from .workflow_graph import WorkFlowGraph, WorkFlowNode, WorkFlowEdge, WorkFlowNodeState


# ---------------------------------------------------------------------------
# Break reason enum
# ---------------------------------------------------------------------------

class CycleBreakReason(str, Enum):
    """Why a cycle stopped iterating."""
    MAX_ITERATIONS = "max_iterations"
    CONVERGENCE    = "convergence"
    QUALITY_MET    = "quality_met"
    EXTERNAL       = "external"
    NOT_BROKEN     = "not_broken"


# ---------------------------------------------------------------------------
# Per-cycle configuration
# ---------------------------------------------------------------------------

@dataclass
class CycleConfig:
    """
    Configuration for a single named cycle.

    Attributes
    ----------
    max_iterations:
        Hard upper bound on loop repetitions.  Prevents infinite loops.
        Defaults to 5.
    quality_threshold:
        If a quality score supplied to ``step_cycle`` meets or exceeds this
        value the loop is considered done.  Set to ``None`` to disable
        quality-based stopping.
    convergence_window:
        Number of consecutive iterations where quality improvement must be
        below ``convergence_delta`` before declaring convergence.
    convergence_delta:
        Minimum quality improvement required to continue iterating.
    strategy_escalation:
        Maps iteration index (0-based) to a strategy label string.
        Callers inspect this to pick increasingly expensive execution paths.
        Example: ``{0: "simple", 2: "medium", 4: "hard"}``.
    """
    max_iterations: int = 5
    quality_threshold: Optional[float] = None
    convergence_window: int = 2
    convergence_delta: float = 0.01
    strategy_escalation: Dict[int, str] = field(default_factory=dict)

    def strategy_for_iteration(self, iteration: int) -> str:
        """
        Return the strategy label for *iteration*.

        Picks the highest configured key that does not exceed *iteration*.
        Falls back to ``"default"`` when no escalation is configured.
        """
        if not self.strategy_escalation:
            return "default"
        applicable = [k for k in self.strategy_escalation if k <= iteration]
        if not applicable:
            return "default"
        return self.strategy_escalation[max(applicable)]


# ---------------------------------------------------------------------------
# Runtime cycle state
# ---------------------------------------------------------------------------

@dataclass
class CycleState:
    """
    Mutable runtime state for one active cycle.

    Attributes
    ----------
    start_node:
        Name of the loop-start node.
    iteration:
        Current 0-based iteration counter.
    quality_history:
        List of quality scores supplied via ``step_cycle``, one per past
        iteration.
    break_reason:
        Why the cycle stopped (or NOT_BROKEN while still running).
    started_at:
        Unix timestamp when the cycle began.
    """
    start_node: str
    iteration: int = 0
    quality_history: List[float] = field(default_factory=list)
    break_reason: CycleBreakReason = CycleBreakReason.NOT_BROKEN
    started_at: float = field(default_factory=time.time)

    @property
    def is_broken(self) -> bool:
        """True once the cycle has been stopped for any reason."""
        return self.break_reason != CycleBreakReason.NOT_BROKEN

    def advance(self) -> None:
        """Increment the iteration counter."""
        self.iteration += 1

    def record_quality(self, score: float) -> None:
        """Append a quality score for the current iteration."""
        self.quality_history.append(score)


# ---------------------------------------------------------------------------
# CyclicWorkFlowGraph
# ---------------------------------------------------------------------------

class CyclicWorkFlowGraph(WorkFlowGraph):
    """
    WorkFlowGraph subclass with explicit managed-cycle semantics.

    All base graph functionality is inherited unchanged.  This class adds:

    * ``register_cycle`` – declare a named cycle with its :class:`CycleConfig`.
    * ``begin_cycle`` / ``step_cycle`` / ``should_break_cycle`` – runtime API
      for callers driving the execution loop.
    * ``cycle_status`` – snapshot of all active cycles.

    Example
    -------
    >>> graph = CyclicWorkFlowGraph(goal="refine answer", nodes=[...], edges=[...])
    >>> graph.register_cycle("Refine", CycleConfig(max_iterations=3, quality_threshold=0.9))
    >>> graph.begin_cycle("Refine")
    >>> while not graph.should_break_cycle("Refine"):
    ...     result = run_node(graph.get_node("Refine"))
    ...     quality = evaluate(result)
    ...     graph.step_cycle("Refine", quality_score=quality)
    """

    # Pydantic field: serialisable cycle configs keyed by start-node name
    cycle_configs: Dict[str, Dict] = Field(default_factory=dict)

    def init_module(self) -> None:
        """Initialise runtime cycle state after base graph setup."""
        super().init_module()
        # Runtime state – not serialised
        self._cycle_states: Dict[str, CycleState] = {}

    # ------------------------------------------------------------------
    # Configuration API
    # ------------------------------------------------------------------

    def register_cycle(self, start_node: str, config: CycleConfig) -> None:
        """
        Declare cycle configuration for the loop starting at *start_node*.

        Parameters
        ----------
        start_node:
            Name of the loop-start node.  Must exist in the graph.
        config:
            :class:`CycleConfig` instance describing termination policy.

        Raises
        ------
        KeyError
            If *start_node* does not exist in the graph.
        ValueError
            If the node is not actually a loop start.
        """
        if not self.node_exists(start_node):
            raise KeyError(f"Node '{start_node}' does not exist in the graph.")
        if not self.is_loop_start(start_node):
            raise ValueError(
                f"Node '{start_node}' is not a loop-start node.  "
                f"Detected loop-start nodes: {list(self._loops.keys())}"
            )
        self.cycle_configs[start_node] = {
            "max_iterations": config.max_iterations,
            "quality_threshold": config.quality_threshold,
            "convergence_window": config.convergence_window,
            "convergence_delta": config.convergence_delta,
            "strategy_escalation": config.strategy_escalation,
        }
        logger.debug(
            "CyclicWorkFlowGraph: registered cycle for node '{}' max_iter={}",
            start_node, config.max_iterations,
        )

    def _get_config(self, start_node: str) -> CycleConfig:
        """Return the :class:`CycleConfig` for *start_node*, creating a default if absent."""
        raw = self.cycle_configs.get(start_node, {})
        return CycleConfig(
            max_iterations=raw.get("max_iterations", 5),
            quality_threshold=raw.get("quality_threshold"),
            convergence_window=raw.get("convergence_window", 2),
            convergence_delta=raw.get("convergence_delta", 0.01),
            strategy_escalation=raw.get("strategy_escalation", {}),
        )

    # ------------------------------------------------------------------
    # Runtime API
    # ------------------------------------------------------------------

    def begin_cycle(self, start_node: str) -> CycleState:
        """
        Initialise runtime state for a cycle at *start_node*.

        Resets any previous state for the same node.  Call this once before
        entering the loop.

        Parameters
        ----------
        start_node:
            Name of the loop-start node.

        Returns
        -------
        CycleState
            The freshly initialised state object.
        """
        state = CycleState(start_node=start_node)
        self._cycle_states[start_node] = state
        logger.debug("CyclicWorkFlowGraph: begin_cycle '{}'", start_node)
        return state

    def step_cycle(
        self,
        start_node: str,
        quality_score: Optional[float] = None,
    ) -> CycleState:
        """
        Advance the iteration counter for *start_node*'s cycle and evaluate
        whether the cycle should stop.

        Call this *after* each loop body completes (i.e. after the loop-end
        node finishes).

        Parameters
        ----------
        start_node:
            Name of the loop-start node.
        quality_score:
            Optional quality/accuracy score in [0, 1] from this iteration.
            Used for quality-threshold and convergence checks.

        Returns
        -------
        CycleState
            Updated state.  Check ``state.is_broken`` to know if the cycle
            should stop, and ``state.break_reason`` for why.
        """
        state = self._ensure_state(start_node)
        cfg   = self._get_config(start_node)

        if quality_score is not None:
            state.record_quality(quality_score)

        state.advance()

        # --- check termination conditions in priority order ---

        # 1. Hard iteration limit
        if state.iteration >= cfg.max_iterations:
            state.break_reason = CycleBreakReason.MAX_ITERATIONS
            logger.debug(
                "CyclicWorkFlowGraph: cycle '{}' stopped — max_iterations ({}) reached",
                start_node, cfg.max_iterations,
            )
            return state

        # 2. Quality threshold
        if cfg.quality_threshold is not None and state.quality_history:
            if state.quality_history[-1] >= cfg.quality_threshold:
                state.break_reason = CycleBreakReason.QUALITY_MET
                logger.debug(
                    "CyclicWorkFlowGraph: cycle '{}' stopped — quality {:.3f} >= threshold {:.3f}",
                    start_node, state.quality_history[-1], cfg.quality_threshold,
                )
                return state

        # 3. Convergence (improvement < delta for window iterations)
        if self._is_converged(state, cfg):
            state.break_reason = CycleBreakReason.CONVERGENCE
            logger.debug(
                "CyclicWorkFlowGraph: cycle '{}' stopped — convergence detected",
                start_node,
            )
            return state

        return state

    def should_break_cycle(self, start_node: str) -> bool:
        """
        Return ``True`` if the cycle at *start_node* should stop.

        This is a convenience method — equivalent to checking
        ``cycle_state(start_node).is_broken``.
        """
        state = self._cycle_states.get(start_node)
        if state is None:
            return False
        return state.is_broken

    def break_cycle(self, start_node: str, reason: CycleBreakReason = CycleBreakReason.EXTERNAL) -> None:
        """
        Externally force a cycle to stop.

        Parameters
        ----------
        start_node:
            Name of the loop-start node.
        reason:
            The reason to record.  Defaults to ``EXTERNAL``.
        """
        state = self._ensure_state(start_node)
        state.break_reason = reason
        logger.debug("CyclicWorkFlowGraph: cycle '{}' externally broken ({})", start_node, reason)

    def current_strategy(self, start_node: str) -> str:
        """
        Return the escalation strategy label for the *current* iteration of
        the cycle at *start_node*.

        Returns ``"default"`` if no escalation is configured.
        """
        state = self._cycle_states.get(start_node)
        cfg   = self._get_config(start_node)
        iteration = state.iteration if state else 0
        return cfg.strategy_for_iteration(iteration)

    def cycle_state(self, start_node: str) -> Optional[CycleState]:
        """Return the :class:`CycleState` for *start_node*, or ``None`` if not begun."""
        return self._cycle_states.get(start_node)

    def cycle_status(self) -> Dict[str, Dict]:
        """
        Return a snapshot of all active cycle states as plain dicts.

        Useful for logging and monitoring.
        """
        result = {}
        for node_name, state in self._cycle_states.items():
            result[node_name] = {
                "iteration": state.iteration,
                "is_broken": state.is_broken,
                "break_reason": state.break_reason.value,
                "quality_history": list(state.quality_history),
                "elapsed": time.time() - state.started_at,
            }
        return result

    def reset_cycle(self, start_node: str) -> None:
        """Remove runtime state for *start_node*, allowing ``begin_cycle`` to restart it."""
        self._cycle_states.pop(start_node, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_state(self, start_node: str) -> CycleState:
        """Return existing state or auto-create one for *start_node*."""
        if start_node not in self._cycle_states:
            return self.begin_cycle(start_node)
        return self._cycle_states[start_node]

    @staticmethod
    def _is_converged(state: CycleState, cfg: CycleConfig) -> bool:
        """
        Return True if quality improvements have flattened out.

        Requires at least ``convergence_window + 1`` quality data points.
        """
        history = state.quality_history
        if len(history) < cfg.convergence_window + 1:
            return False
        recent = history[-(cfg.convergence_window + 1):]
        improvements = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
        return all(abs(imp) < cfg.convergence_delta for imp in improvements)


__all__ = [
    "CyclicWorkFlowGraph",
    "CycleConfig",
    "CycleState",
    "CycleBreakReason",
]

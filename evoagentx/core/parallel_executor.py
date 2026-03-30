"""
Parallel tool execution engine (GAP-framework inspired).

Identifies independent tool calls in a batch, executes them concurrently
via ``asyncio``, and aggregates results — including partial-failure handling
and per-call timeout enforcement.

Key concepts
------------
ToolCall
    A lightweight descriptor for a single callable invocation.  Has a name,
    the callable, positional/keyword args, an optional list of *depends_on*
    names, and a per-call timeout.

DependencyGraph
    Analyses a list of ToolCall instances and produces topologically sorted
    execution waves: calls that can run concurrently within a wave, waves
    that must run sequentially because later waves depend on earlier ones.

ParallelExecutor
    Drives execution wave by wave, running all calls within a wave
    concurrently.  Tracks cost (token counts or arbitrary numeric cost),
    accumulates results, and handles partial failures gracefully.

Usage
-----
::

    from evoagentx.core.parallel_executor import ParallelExecutor, ToolCall

    async def main():
        executor = ParallelExecutor(max_concurrency=4, default_timeout=10.0)

        calls = [
            ToolCall("search_web",  search_fn,   args=["python asyncio"]),
            ToolCall("read_doc",    read_fn,      args=["docs/index.md"]),
            ToolCall("summarise",   summarise_fn, depends_on=["search_web", "read_doc"]),
        ]
        result = await executor.execute(calls)
        print(result.outputs["summarise"])
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from .logging import logger


# ---------------------------------------------------------------------------
# ToolCall descriptor
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    """
    Describes a single callable to be executed by :class:`ParallelExecutor`.

    Attributes
    ----------
    name:
        Unique identifier within a batch.
    fn:
        The callable to invoke.  May be a regular function or a coroutine
        function.
    args:
        Positional arguments to pass to *fn*.
    kwargs:
        Keyword arguments to pass to *fn*.
    depends_on:
        Names of other :class:`ToolCall` objects in the same batch that
        must complete before this call is eligible to run.  Dependency
        outputs are NOT automatically injected; callers must handle that
        in their callable via closure or shared state if needed.
    timeout:
        Per-call timeout in seconds.  ``None`` means use the executor
        default.
    cost_estimate:
        Optional numeric cost estimate (e.g. expected token usage).  Used
        by the executor to track budgets.
    """
    name: str
    fn: Callable
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    cost_estimate: float = 0.0


# ---------------------------------------------------------------------------
# Per-call result
# ---------------------------------------------------------------------------

@dataclass
class CallResult:
    """
    Outcome of a single :class:`ToolCall` execution.

    Attributes
    ----------
    name:
        The ToolCall name.
    output:
        Return value of the callable, or ``None`` on failure.
    error:
        Exception instance if the call failed, else ``None``.
    duration:
        Wall-clock seconds taken.
    cost_actual:
        Actual cost incurred (defaults to the call's ``cost_estimate``).
    timed_out:
        True if the call exceeded its timeout.
    """
    name: str
    output: Any
    error: Optional[Exception]
    duration: float
    cost_actual: float = 0.0
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        """True when the call succeeded without error."""
        return self.error is None and not self.timed_out


# ---------------------------------------------------------------------------
# Batch execution result
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """
    Aggregated result of executing a batch of :class:`ToolCall` objects.

    Attributes
    ----------
    outputs:
        Mapping from call name to its output value (only for successful calls).
    results:
        All :class:`CallResult` instances, including failures.
    total_duration:
        Wall-clock time for the entire batch.
    total_cost:
        Sum of ``cost_actual`` across all calls.
    failed_calls:
        Names of calls that errored or timed out.
    """
    outputs: Dict[str, Any]
    results: List[CallResult]
    total_duration: float
    total_cost: float
    failed_calls: List[str]

    @property
    def success(self) -> bool:
        """True when all calls in the batch succeeded."""
        return len(self.failed_calls) == 0

    def get(self, name: str, default: Any = None) -> Any:
        """Return the output for *name*, or *default* if missing or failed."""
        return self.outputs.get(name, default)


# ---------------------------------------------------------------------------
# Dependency graph analyser
# ---------------------------------------------------------------------------

class _DependencyGraph:
    """
    Analyses :class:`ToolCall` dependencies and produces execution waves.

    A wave is a list of call names that have no unresolved dependencies and
    can therefore run concurrently.  Waves are returned in topological order.

    Raises
    ------
    ValueError
        If *calls* reference unknown dependency names or contain cycles.
    """

    def __init__(self, calls: List[ToolCall]) -> None:
        self._calls: Dict[str, ToolCall] = {}
        for call in calls:
            if call.name in self._calls:
                raise ValueError(f"Duplicate ToolCall name: '{call.name}'")
            self._calls[call.name] = call

        self._validate_dependencies()
        self._waves: List[List[str]] = self._compute_waves()

    def _validate_dependencies(self) -> None:
        known = set(self._calls.keys())
        for call in self._calls.values():
            for dep in call.depends_on:
                if dep not in known:
                    raise ValueError(
                        f"ToolCall '{call.name}' depends on unknown call '{dep}'"
                    )

    def _compute_waves(self) -> List[List[str]]:
        """
        Kahn's algorithm for topological sort, collecting nodes per level.
        """
        in_degree: Dict[str, int] = {name: 0 for name in self._calls}
        dependents: Dict[str, List[str]] = {name: [] for name in self._calls}

        for call in self._calls.values():
            for dep in call.depends_on:
                in_degree[call.name] += 1
                dependents[dep].append(call.name)

        queue = [name for name, deg in in_degree.items() if deg == 0]
        waves: List[List[str]] = []

        while queue:
            waves.append(list(queue))
            next_queue: List[str] = []
            for name in queue:
                for child in dependents[name]:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        next_queue.append(child)
            queue = next_queue

        processed = sum(len(w) for w in waves)
        if processed != len(self._calls):
            raise ValueError(
                "Dependency cycle detected in ToolCall batch. "
                f"Processed {processed}/{len(self._calls)} calls."
            )
        return waves

    @property
    def waves(self) -> List[List[str]]:
        """Ordered list of execution waves."""
        return self._waves

    def get_call(self, name: str) -> ToolCall:
        return self._calls[name]


# ---------------------------------------------------------------------------
# Parallel executor
# ---------------------------------------------------------------------------

class ParallelExecutor:
    """
    Executes a batch of :class:`ToolCall` objects with dependency-aware
    parallelism.

    Parameters
    ----------
    max_concurrency:
        Maximum number of calls running simultaneously within a wave.
        A value of 0 means unlimited.
    default_timeout:
        Default per-call timeout in seconds.  Individual calls may override.
    cost_budget:
        Optional upper bound on cumulative ``cost_estimate`` across the batch.
        Raises :class:`CostBudgetExceeded` if exceeded before execution starts.
    """

    def __init__(
        self,
        max_concurrency: int = 8,
        default_timeout: float = 30.0,
        cost_budget: Optional[float] = None,
    ) -> None:
        if max_concurrency < 0:
            raise ValueError("max_concurrency must be >= 0")
        self.max_concurrency = max_concurrency
        self.default_timeout = default_timeout
        self.cost_budget = cost_budget

    async def execute(self, calls: List[ToolCall]) -> ExecutionResult:
        """
        Execute *calls* concurrently where possible, sequentially where
        dependencies require.

        Parameters
        ----------
        calls:
            List of :class:`ToolCall` objects to run.

        Returns
        -------
        ExecutionResult
            Aggregated results including outputs, errors, timing, and cost.

        Raises
        ------
        CostBudgetExceeded
            When the sum of ``cost_estimate`` values exceeds ``cost_budget``.
        ValueError
            On duplicate call names or dependency cycles.
        """
        if not calls:
            return ExecutionResult(
                outputs={},
                results=[],
                total_duration=0.0,
                total_cost=0.0,
                failed_calls=[],
            )

        # Pre-flight cost check
        estimated_total = sum(c.cost_estimate for c in calls)
        if self.cost_budget is not None and estimated_total > self.cost_budget:
            raise CostBudgetExceeded(
                f"Estimated cost {estimated_total:.4f} exceeds budget {self.cost_budget:.4f}"
            )

        dep_graph = _DependencyGraph(calls)
        semaphore = asyncio.Semaphore(self.max_concurrency) if self.max_concurrency > 0 else None

        all_results: List[CallResult] = []
        outputs: Dict[str, Any] = {}
        batch_start = time.perf_counter()

        for wave in dep_graph.waves:
            wave_calls = [dep_graph.get_call(name) for name in wave]
            wave_results = await self._run_wave(wave_calls, semaphore)
            for res in wave_results:
                all_results.append(res)
                if res.ok:
                    outputs[res.name] = res.output

        total_duration = time.perf_counter() - batch_start
        total_cost = sum(r.cost_actual for r in all_results)
        failed = [r.name for r in all_results if not r.ok]

        if failed:
            logger.warning(
                "ParallelExecutor: {} call(s) failed: {}",
                len(failed), failed,
            )

        return ExecutionResult(
            outputs=outputs,
            results=all_results,
            total_duration=total_duration,
            total_cost=total_cost,
            failed_calls=failed,
        )

    # ------------------------------------------------------------------
    # Wave execution
    # ------------------------------------------------------------------

    async def _run_wave(
        self,
        wave_calls: List[ToolCall],
        semaphore: Optional[asyncio.Semaphore],
    ) -> List[CallResult]:
        """Run all calls in *wave_calls* concurrently."""
        tasks = [
            asyncio.create_task(self._run_one(call, semaphore))
            for call in wave_calls
        ]
        return list(await asyncio.gather(*tasks))

    async def _run_one(
        self,
        call: ToolCall,
        semaphore: Optional[asyncio.Semaphore],
    ) -> CallResult:
        """
        Execute a single :class:`ToolCall`, enforcing timeout and semaphore.
        """
        timeout = call.timeout if call.timeout is not None else self.default_timeout
        start = time.perf_counter()

        async def _invoke() -> Any:
            if asyncio.iscoroutinefunction(call.fn):
                return await call.fn(*call.args, **call.kwargs)
            # Run synchronous callables in a thread pool to avoid blocking the
            # event loop.
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                lambda: call.fn(*call.args, **call.kwargs),
            )

        async def _guarded() -> CallResult:
            try:
                if semaphore is not None:
                    async with semaphore:
                        output = await asyncio.wait_for(_invoke(), timeout=timeout)
                else:
                    output = await asyncio.wait_for(_invoke(), timeout=timeout)
                duration = time.perf_counter() - start
                logger.debug(
                    "ParallelExecutor: '{}' completed in {:.3f}s",
                    call.name, duration,
                )
                return CallResult(
                    name=call.name,
                    output=output,
                    error=None,
                    duration=duration,
                    cost_actual=call.cost_estimate,
                )
            except asyncio.TimeoutError:
                duration = time.perf_counter() - start
                logger.warning(
                    "ParallelExecutor: '{}' timed out after {:.1f}s",
                    call.name, timeout,
                )
                return CallResult(
                    name=call.name,
                    output=None,
                    error=asyncio.TimeoutError(f"'{call.name}' timed out after {timeout}s"),
                    duration=duration,
                    cost_actual=call.cost_estimate,
                    timed_out=True,
                )
            except Exception as exc:
                duration = time.perf_counter() - start
                logger.warning(
                    "ParallelExecutor: '{}' raised {}: {}",
                    call.name, type(exc).__name__, exc,
                )
                return CallResult(
                    name=call.name,
                    output=None,
                    error=exc,
                    duration=duration,
                    cost_actual=call.cost_estimate,
                )

        return await _guarded()

    # ------------------------------------------------------------------
    # Sync convenience wrapper
    # ------------------------------------------------------------------

    def execute_sync(self, calls: List[ToolCall]) -> ExecutionResult:
        """
        Synchronous wrapper around :meth:`execute`.

        Creates (or reuses) an event loop and blocks until complete.  Useful
        in non-async contexts such as unit tests or REPL usage.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an existing event loop (e.g. Jupyter).
                # Create a new loop in a thread — simpler than nest_asyncio.
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, self.execute(calls))
                    return future.result()
            return loop.run_until_complete(self.execute(calls))
        except RuntimeError:
            return asyncio.run(self.execute(calls))


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class CostBudgetExceeded(Exception):
    """Raised when estimated batch cost exceeds the configured budget."""


__all__ = [
    "ParallelExecutor",
    "ToolCall",
    "CallResult",
    "ExecutionResult",
    "CostBudgetExceeded",
]

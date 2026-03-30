"""Baseline runner — freeze benchmark results before any optimisation work.

This module runs all five existing AgentAugi optimisers (TextGrad, AFlow, SEW,
MIPRO, EvoPrompt) against all four Phase-0 benchmarks (TAU-bench, SWE-Bench,
AgentBench, GAIA) and saves the results as an immutable JSON baseline.

**Purpose:** Establish the measurement foundation.  Every claimed improvement
in later phases must be validated by comparing against these frozen numbers.

Usage
-----
CLI-style run::

    from evoagentx.benchmark.baseline_runner import BaselineRunner, RunConfig

    runner = BaselineRunner(
        output_dir="~/.evoagentx/baselines",
        llm_config=my_llm_config,
    )
    baseline = runner.run(
        optimizers=["textgrad", "evoprompt"],
        benchmarks=["tau_bench", "gaia"],
        k=4,                    # pass^k trials per task
        sample_tasks=50,        # limit tasks per benchmark for speed
    )
    print(baseline.summary())
    baseline.save()

Programmatic diff::

    from evoagentx.benchmark.baseline_runner import BaselineReport

    old = BaselineReport.load("~/.evoagentx/baselines/baseline_2026-03-29.json")
    new = BaselineReport.load("~/.evoagentx/baselines/baseline_2026-04-15.json")
    diff = old.compare(new)
    print(diff)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from ..core.logging import logger
from ..core.cost_tracker import get_tracker


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUPPORTED_OPTIMIZERS = ["textgrad", "aflow", "sew", "mipro", "evoprompt"]
SUPPORTED_BENCHMARKS = ["tau_bench", "swe_bench", "agent_bench", "gaia"]


@dataclass
class RunConfig:
    """Configuration for a single optimizer × benchmark evaluation.

    Parameters
    ----------
    optimizer_name:
        One of ``SUPPORTED_OPTIMIZERS``.
    benchmark_name:
        One of ``SUPPORTED_BENCHMARKS``.
    k:
        Number of pass^k trials per task (for TAU-bench wrapper).
    sample_tasks:
        If > 0, only evaluate a random subset of tasks (faster).
    seed:
        Random seed for reproducible sampling.
    eval_mode:
        Data split: ``"test"`` (default), ``"dev"``, or ``"train"``.
    extra:
        Optimizer-specific keyword arguments forwarded to the run.
    """
    optimizer_name: str
    benchmark_name: str
    k: int = 4
    sample_tasks: int = 0
    seed: int = 42
    eval_mode: str = "test"
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-run result
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Evaluation result for one optimizer × benchmark combination."""

    optimizer: str
    benchmark: str
    metrics: Dict[str, Any]         # primary metrics dict
    tau_metrics: Optional[Dict] = None  # pass^k breakdown if available
    cost_usd: float = 0.0
    wall_seconds: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    error: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "optimizer": self.optimizer,
            "benchmark": self.benchmark,
            "metrics": self.metrics,
            "tau_metrics": self.tau_metrics,
            "cost_usd": self.cost_usd,
            "wall_seconds": self.wall_seconds,
            "timestamp": self.timestamp,
            "error": self.error,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "RunResult":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Baseline report
# ---------------------------------------------------------------------------

@dataclass
class BaselineReport:
    """Immutable snapshot of all optimizer × benchmark results.

    Parameters
    ----------
    results:
        All :class:`RunResult` objects from this baseline run.
    created_at:
        ISO-8601 timestamp of when the baseline was frozen.
    metadata:
        Arbitrary notes (e.g., git commit, environment info).
    """

    results: List[RunResult]
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def get(self, optimizer: str, benchmark: str) -> Optional[RunResult]:
        """Return the result for a specific optimizer × benchmark pair."""
        for r in self.results:
            if r.optimizer == optimizer and r.benchmark == benchmark:
                return r
        return None

    def get_metric(
        self,
        optimizer: str,
        benchmark: str,
        metric: str,
        default: Any = None,
    ) -> Any:
        """Return a specific metric value, or default if not found."""
        result = self.get(optimizer, benchmark)
        if result is None:
            return default
        return result.metrics.get(metric, default)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------

    def summary(self, metric: str = "score") -> str:
        """Print a human-readable summary table.

        Parameters
        ----------
        metric:
            The key to extract from each ``metrics`` dict for the table.
        """
        optimizers = sorted({r.optimizer for r in self.results})
        benchmarks = sorted({r.benchmark for r in self.results})

        col_width = max(16, max(len(b) for b in benchmarks) + 2)
        opt_width = max(12, max(len(o) for o in optimizers) + 2)

        header = f"{'Optimizer':{opt_width}}" + "".join(f"{b:>{col_width}}" for b in benchmarks)
        separator = "-" * len(header)
        lines = [
            f"=== Baseline Report (metric={metric}) — {self.created_at[:10]} ===",
            header,
            separator,
        ]
        for opt in optimizers:
            row = f"{opt:{opt_width}}"
            for bench in benchmarks:
                result = self.get(opt, bench)
                if result is None or result.error:
                    val = "  —"
                else:
                    v = result.metrics.get(metric, result.metrics.get("pass_at_1", None))
                    val = f"{v:.3f}" if isinstance(v, float) else str(v) if v is not None else "  —"
                row += f"{val:>{col_width}}"
            lines.append(row)

        total_cost = sum(r.cost_usd for r in self.results)
        lines += [separator, f"Total evaluation cost: ${total_cost:.4f}"]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare(
        self,
        candidate: "BaselineReport",
        metric: str = "score",
    ) -> Dict[str, Any]:
        """Return a structured diff (candidate − self) for every shared run.

        Positive deltas indicate improvement.
        """
        deltas: List[Dict] = []
        for r_base in self.results:
            r_cand = candidate.get(r_base.optimizer, r_base.benchmark)
            if r_cand is None:
                continue
            base_val = r_base.metrics.get(metric, 0.0)
            cand_val = r_cand.metrics.get(metric, 0.0)
            deltas.append({
                "optimizer": r_base.optimizer,
                "benchmark": r_base.benchmark,
                "baseline": base_val,
                "candidate": cand_val,
                "delta": round(float(cand_val) - float(base_val), 4),
                "improved": float(cand_val) > float(base_val),
            })

        improved = sum(1 for d in deltas if d["improved"])
        mean_delta = (
            sum(d["delta"] for d in deltas) / len(deltas) if deltas else 0.0
        )
        return {
            "num_comparisons": len(deltas),
            "num_improved": improved,
            "num_regressed": sum(1 for d in deltas if not d["improved"] and d["delta"] != 0),
            "mean_delta": round(mean_delta, 4),
            "baseline_date": self.created_at[:10],
            "candidate_date": candidate.created_at[:10],
            "details": deltas,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        return {
            "created_at": self.created_at,
            "metadata": self.metadata,
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path: str) -> None:
        """Write the report to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        logger.info(f"BaselineReport saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BaselineReport":
        """Reload a previously saved baseline report."""
        with open(os.path.expanduser(path), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        results = [RunResult.from_dict(r) for r in data.get("results", [])]
        return cls(
            results=results,
            created_at=data.get("created_at", ""),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# BaselineRunner
# ---------------------------------------------------------------------------

class BaselineRunner:
    """Orchestrates evaluation of all optimizers across all benchmarks.

    This class is the entry point for generating a frozen baseline.  It
    wraps each optimizer in a TAU-bench reliability harness and records
    cost via :class:`~evoagentx.core.cost_tracker.CostTracker`.

    Parameters
    ----------
    output_dir:
        Directory where baseline JSON files are written.
    llm_config:
        LLM configuration passed to optimizers that need it.
    agent_manager:
        Agent manager for building workflow graphs.
    verbose:
        Print progress to stdout.
    """

    def __init__(
        self,
        output_dir: str = "~/.evoagentx/baselines",
        llm_config: Any = None,
        agent_manager: Any = None,
        verbose: bool = True,
    ) -> None:
        self.output_dir = os.path.expanduser(output_dir)
        self.llm_config = llm_config
        self.agent_manager = agent_manager
        self.verbose = verbose
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Main run method
    # ------------------------------------------------------------------

    def run(
        self,
        optimizers: Optional[List[str]] = None,
        benchmarks: Optional[List[str]] = None,
        k: int = 4,
        sample_tasks: int = 0,
        seed: int = 42,
        save: bool = True,
    ) -> BaselineReport:
        """Run all specified optimizer × benchmark combinations.

        Parameters
        ----------
        optimizers:
            Subset of ``SUPPORTED_OPTIMIZERS`` to run.  Defaults to all five.
        benchmarks:
            Subset of ``SUPPORTED_BENCHMARKS`` to run.  Defaults to all four.
        k:
            Number of pass^k reliability trials per task.
        sample_tasks:
            If > 0, evaluate only a random sample of ``sample_tasks`` per benchmark
            (useful for smoke-testing).
        seed:
            Random seed for reproducible task sampling.
        save:
            Whether to persist the report automatically.
        """
        opt_list = optimizers or SUPPORTED_OPTIMIZERS
        bench_list = benchmarks or SUPPORTED_BENCHMARKS

        invalid_opts = [o for o in opt_list if o not in SUPPORTED_OPTIMIZERS]
        invalid_bench = [b for b in bench_list if b not in SUPPORTED_BENCHMARKS]
        if invalid_opts:
            raise ValueError(f"Unknown optimizers: {invalid_opts}. Valid: {SUPPORTED_OPTIMIZERS}")
        if invalid_bench:
            raise ValueError(f"Unknown benchmarks: {invalid_bench}. Valid: {SUPPORTED_BENCHMARKS}")

        logger.info(
            f"BaselineRunner: {len(opt_list)} optimizers × {len(bench_list)} benchmarks "
            f"(k={k}, sample_tasks={sample_tasks or 'all'})"
        )

        results: List[RunResult] = []
        tracker = get_tracker()

        for opt_name in opt_list:
            for bench_name in bench_list:
                config = RunConfig(
                    optimizer_name=opt_name,
                    benchmark_name=bench_name,
                    k=k,
                    sample_tasks=sample_tasks,
                    seed=seed,
                )
                result = self._run_one(config, tracker)
                results.append(result)
                if self.verbose:
                    status = f"${result.cost_usd:.4f} | {result.wall_seconds:.1f}s"
                    if result.error:
                        status = f"ERROR: {result.error[:60]}"
                    logger.info(
                        f"  [{opt_name} / {bench_name}] "
                        f"score={result.metrics.get('score', result.metrics.get('pass_at_1', '?'))} "
                        f"| {status}"
                    )

        report = BaselineReport(
            results=results,
            metadata={
                "k": k,
                "sample_tasks": sample_tasks,
                "seed": seed,
                "total_cost_usd": sum(r.cost_usd for r in results),
            },
        )

        if save:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
            path = os.path.join(self.output_dir, f"baseline_{date_str}.json")
            report.save(path)
            # Always keep a "latest" symlink / copy
            latest_path = os.path.join(self.output_dir, "baseline_latest.json")
            report.save(latest_path)

        if self.verbose:
            print(report.summary())
        return report

    # ------------------------------------------------------------------
    # Single run
    # ------------------------------------------------------------------

    def _run_one(self, config: RunConfig, tracker: Any) -> RunResult:
        """Execute one optimizer × benchmark pair and return a :class:`RunResult`."""
        session_id = f"{config.optimizer_name}/{config.benchmark_name}"
        t_start = time.time()

        with tracker.session(session_id):
            try:
                metrics, tau_metrics = self._evaluate(config)
                cost = tracker.session_cost()
                return RunResult(
                    optimizer=config.optimizer_name,
                    benchmark=config.benchmark_name,
                    metrics=metrics,
                    tau_metrics=tau_metrics,
                    cost_usd=cost,
                    wall_seconds=time.time() - t_start,
                    config=config.__dict__,
                )
            except Exception as exc:
                logger.warning(
                    f"BaselineRunner: {config.optimizer_name}/{config.benchmark_name} "
                    f"failed: {exc}"
                )
                return RunResult(
                    optimizer=config.optimizer_name,
                    benchmark=config.benchmark_name,
                    metrics={},
                    cost_usd=tracker.session_cost(),
                    wall_seconds=time.time() - t_start,
                    error=str(exc),
                    config=config.__dict__,
                )

    # ------------------------------------------------------------------
    # Evaluation dispatch
    # ------------------------------------------------------------------

    def _evaluate(self, config: RunConfig) -> tuple[Dict, Optional[Dict]]:
        """Build the optimizer + benchmark, run evaluation, return metrics.

        This is the integration point.  Since Phase 0 is infrastructure-only
        and the optimizers need to be initialised with an LLM config, this
        method gracefully degrades when the LLM config is not provided:

        * If ``llm_config`` is set, it runs a real evaluation.
        * Otherwise, it returns a stub result tagged ``"not_configured"`` so
          the harness still produces a valid (empty) baseline JSON.
        """
        bench = self._build_benchmark(config)
        if bench is None:
            return {"score": None, "note": "benchmark_not_available"}, None

        if self.llm_config is None:
            logger.warning(
                f"BaselineRunner: no llm_config provided, skipping real evaluation "
                f"for {config.optimizer_name}/{config.benchmark_name}"
            )
            return {"score": None, "note": "llm_not_configured"}, None

        graph = self._build_optimizer_graph(config)
        if graph is None:
            return {"score": None, "note": "optimizer_not_configured"}, None

        # Run TAU-bench pass^k harness around the graph
        from .tau_bench import TAUBenchRunner
        from ..evaluators.evaluator import Evaluator

        evaluator = Evaluator(
            llm=self._build_llm(),
            num_workers=1,
        )

        runner = TAUBenchRunner.from_benchmark(
            benchmark=bench,
            graph=graph,
            evaluator=evaluator,
            k=config.k,
            seed=config.seed,
        )
        report = runner.run(sample_k=config.sample_tasks or None)
        metrics = {
            "pass_at_1": round(report.pass_at_1, 4),
            f"pass_k": round(report.pass_k, 4),
            "consistency_rate": round(report.consistency_rate, 4),
            "score": round(report.pass_at_1, 4),
            "num_tasks": len(report.task_results),
        }
        return metrics, report.to_dict()

    # ------------------------------------------------------------------
    # Builder helpers
    # ------------------------------------------------------------------

    def _build_benchmark(self, config: RunConfig) -> Optional[Any]:
        """Instantiate the appropriate Benchmark for the config."""
        name = config.benchmark_name
        try:
            if name == "tau_bench":
                from .tau_bench import TAUBench
                return TAUBench()
            if name == "swe_bench":
                from .swe_bench import SWEBench
                return SWEBench()
            if name == "agent_bench":
                from .agent_bench import AgentBench
                return AgentBench()
            if name == "gaia":
                from .gaia_bench import GAIA
                return GAIA()
        except FileNotFoundError:
            logger.warning(f"BaselineRunner: dataset for {name} not found at default path")
            return None
        except Exception as exc:
            logger.warning(f"BaselineRunner: could not load {name}: {exc}")
            return None
        return None

    def _build_llm(self) -> Any:
        """Build an LLM instance from self.llm_config."""
        if self.llm_config is None:
            return None
        from ..models import LiteLLM
        return LiteLLM(config=self.llm_config)

    def _build_optimizer_graph(self, config: RunConfig) -> Optional[Any]:
        """Return a ready-to-run workflow graph for the specified optimizer.

        In the baseline run, we evaluate the *current* (pre-optimisation)
        graph.  Optimisers are not invoked here — we measure the baseline
        graph performance before any training.
        """
        if self.agent_manager is None:
            return None
        try:
            # Return the default graph from the agent manager without optimising
            return self.agent_manager.build_graph()
        except Exception as exc:
            logger.warning(f"BaselineRunner: could not build graph: {exc}")
            return None


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def run_phase0_baseline(
    llm_config: Any = None,
    agent_manager: Any = None,
    output_dir: str = "~/.evoagentx/baselines",
    k: int = 4,
    sample_tasks: int = 50,
    optimizers: Optional[List[str]] = None,
    benchmarks: Optional[List[str]] = None,
) -> BaselineReport:
    """One-shot function to generate and save the Phase 0 baseline.

    This is the function to call at the very start of Phase 0.  The resulting
    JSON file is the immutable ground truth all future improvements are
    compared against.

    Parameters
    ----------
    llm_config:
        LLM configuration for the executor.
    agent_manager:
        Agent manager used to build workflow graphs.
    output_dir:
        Where to write the baseline JSON files.
    k:
        Number of pass^k reliability trials.
    sample_tasks:
        Limit each benchmark to this many tasks for the baseline run.
    optimizers:
        Subset of optimizers.  Defaults to all five.
    benchmarks:
        Subset of benchmarks.  Defaults to all four.

    Returns
    -------
    BaselineReport
        The frozen baseline.
    """
    runner = BaselineRunner(
        output_dir=output_dir,
        llm_config=llm_config,
        agent_manager=agent_manager,
        verbose=True,
    )
    return runner.run(
        optimizers=optimizers,
        benchmarks=benchmarks,
        k=k,
        sample_tasks=sample_tasks,
        save=True,
    )

"""TAU-bench-style pass^k reliability evaluation harness.

TAU-bench (Task Automation Under Uncertainty) introduced the *pass^k* metric
to distinguish one-off luck from reliable performance.  A system that passes
60 % of tasks on attempt-1 but only 25 % consistently is worse than one at
50 % that passes every time.

Metric definitions
------------------
Let ``c_i`` = number of correct completions for task ``i`` across ``k`` trials.

* **pass@1** = mean(c_i / k)   — probability of success on a single attempt.
* **pass^k** = mean(c_i == k)  — fraction of tasks where *all* k attempts pass
  (complete consistency).
* **consistency_rate** = mean of c_i / k for tasks where at least one attempt
  succeeded — measures reliability among partially-solved tasks.

Usage
-----
Standalone task set::

    from evoagentx.benchmark.tau_bench import TAUBench, TAUBenchTask

    bench = TAUBench(tasks=[
        TAUBenchTask(
            task_id="q001",
            question="What is 2+2?",
            expected_answer="4",
            check_fn=lambda pred, label: pred.strip() == label,
        ),
    ])

    # run against a callable agent: fn(question: str) -> str
    runner = TAUBenchRunner(bench, agent_fn=my_agent, k=8, num_workers=4)
    report = runner.run()
    print(report.summary())

Wrapping an existing Benchmark::

    from evoagentx.benchmark.tau_bench import TAUBenchRunner
    from evoagentx.benchmark.gsm8k import GSM8K

    gsm = GSM8K(mode="test")
    runner = TAUBenchRunner.from_benchmark(
        benchmark=gsm,
        graph=my_workflow_graph,
        evaluator=my_evaluator,
        k=8,
    )
    report = runner.run()
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..core.logging import logger
from .benchmark import Benchmark


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------

@dataclass
class TAUBenchTask:
    """A single evaluatable task for the reliability harness.

    Parameters
    ----------
    task_id:
        Unique identifier for this task.
    question:
        The natural-language prompt / question given to the agent.
    expected_answer:
        Ground-truth answer used by ``check_fn``.
    check_fn:
        ``(prediction: str, expected: Any) -> bool`` — returns ``True`` if the
        prediction is correct.  Defaults to exact-match after stripping whitespace.
    metadata:
        Arbitrary key-value pairs (difficulty level, domain, source, etc.).
    """

    task_id: str
    question: str
    expected_answer: Any
    check_fn: Optional[Callable[[str, Any], bool]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def evaluate(self, prediction: str) -> bool:
        """Return True if prediction is correct according to check_fn."""
        if self.check_fn is not None:
            return bool(self.check_fn(prediction, self.expected_answer))
        # Default: case-insensitive exact match after stripping
        return str(prediction).strip().lower() == str(self.expected_answer).strip().lower()


# ---------------------------------------------------------------------------
# Per-task trial result
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    """Result of a single trial (one attempt at one task)."""

    task_id: str
    trial_index: int
    prediction: str
    is_correct: bool
    latency_seconds: float
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Aggregate per-task result
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    """Aggregated result across all k trials for one task."""

    task_id: str
    k: int
    trials: List[TrialResult]

    @property
    def correct_count(self) -> int:
        return sum(1 for t in self.trials if t.is_correct)

    @property
    def pass_at_1(self) -> float:
        """Empirical probability of success on a single random attempt."""
        return self.correct_count / self.k if self.k > 0 else 0.0

    @property
    def pass_k(self) -> bool:
        """True iff ALL k attempts are correct (perfect consistency)."""
        return self.correct_count == self.k

    @property
    def mean_latency(self) -> float:
        latencies = [t.latency_seconds for t in self.trials if t.error is None]
        return sum(latencies) / len(latencies) if latencies else 0.0


# ---------------------------------------------------------------------------
# Evaluation report
# ---------------------------------------------------------------------------

@dataclass
class TAUBenchReport:
    """Full evaluation report for a pass^k run."""

    benchmark_name: str
    k: int
    task_results: List[TaskResult]
    total_wall_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    @property
    def pass_at_1(self) -> float:
        """Mean pass@1 across all tasks."""
        if not self.task_results:
            return 0.0
        return sum(r.pass_at_1 for r in self.task_results) / len(self.task_results)

    @property
    def pass_k(self) -> float:
        """Fraction of tasks where *all* k attempts are correct (pass^k)."""
        if not self.task_results:
            return 0.0
        return sum(1 for r in self.task_results if r.pass_k) / len(self.task_results)

    @property
    def consistency_rate(self) -> float:
        """Mean pass@1 among tasks with at least one correct attempt."""
        solved = [r for r in self.task_results if r.correct_count > 0]
        if not solved:
            return 0.0
        return sum(r.pass_at_1 for r in solved) / len(solved)

    @property
    def mean_latency(self) -> float:
        all_latencies = [r.mean_latency for r in self.task_results]
        return sum(all_latencies) / len(all_latencies) if all_latencies else 0.0

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"=== TAU-bench Report: {self.benchmark_name} (k={self.k}) ===",
            f"Tasks evaluated : {len(self.task_results)}",
            f"pass@1          : {self.pass_at_1:.3f}  ({self.pass_at_1 * 100:.1f}%)",
            f"pass^{self.k:<2}         : {self.pass_k:.3f}  ({self.pass_k * 100:.1f}%)",
            f"consistency rate: {self.consistency_rate:.3f}",
            f"mean latency    : {self.mean_latency:.2f}s / task",
            f"total wall time : {self.total_wall_seconds:.1f}s",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "benchmark_name": self.benchmark_name,
            "k": self.k,
            "pass_at_1": self.pass_at_1,
            f"pass_k": self.pass_k,
            "consistency_rate": self.consistency_rate,
            "mean_latency": self.mean_latency,
            "total_wall_seconds": self.total_wall_seconds,
            "num_tasks": len(self.task_results),
            "metadata": self.metadata,
            "task_results": [
                {
                    "task_id": r.task_id,
                    "correct_count": r.correct_count,
                    "k": r.k,
                    "pass_at_1": r.pass_at_1,
                    "pass_k": r.pass_k,
                    "mean_latency": r.mean_latency,
                }
                for r in self.task_results
            ],
        }

    def save(self, path: str) -> None:
        """Persist the report to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        logger.info(f"TAUBenchReport saved to {path}")

    @classmethod
    def load(cls, path: str) -> "TAUBenchReport":
        """Reload a previously saved report (metadata only, no trial details)."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        dummy_tasks = [
            TaskResult(
                task_id=t["task_id"],
                k=t["k"],
                trials=[
                    TrialResult(
                        task_id=t["task_id"],
                        trial_index=i,
                        prediction="",
                        is_correct=(i < t["correct_count"]),
                        latency_seconds=t.get("mean_latency", 0.0),
                    )
                    for i in range(t["k"])
                ],
            )
            for t in data.get("task_results", [])
        ]
        return cls(
            benchmark_name=data["benchmark_name"],
            k=data["k"],
            task_results=dummy_tasks,
            total_wall_seconds=data.get("total_wall_seconds", 0.0),
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# TAUBench — a Benchmark subclass for defining task sets
# ---------------------------------------------------------------------------

class TAUBench(Benchmark):
    """Benchmark class for reliability evaluation using pass^k.

    Can be initialised either from an explicit list of :class:`TAUBenchTask`
    objects (useful for custom task sets) or by loading from a JSONL file.

    JSONL format (one task per line)::

        {"task_id": "q1", "question": "What is 2+2?", "expected_answer": "4"}

    Parameters
    ----------
    tasks:
        Explicit list of TAUBenchTask objects.  If provided, ``path`` is
        ignored.
    path:
        Directory containing ``train.jsonl``, ``dev.jsonl``, or ``test.jsonl``.
    name:
        Display name for the benchmark.
    mode:
        Which splits to load: ``"all"``, ``"train"``, ``"dev"``, or ``"test"``.
    """

    def __init__(
        self,
        tasks: Optional[List[TAUBenchTask]] = None,
        path: Optional[str] = None,
        name: str = "TAUBench",
        mode: str = "all",
        **kwargs,
    ) -> None:
        self._explicit_tasks = tasks
        path = os.path.expanduser(path or "~/.evoagentx/data/tau_bench")
        super().__init__(name=name, path=path, mode=mode, **kwargs)

    # ------------------------------------------------------------------
    # Benchmark abstract method implementations
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        if self._explicit_tasks is not None:
            # Use the provided task list directly as test data
            self._test_data = [self._task_to_dict(t) for t in self._explicit_tasks]
            return

        file_map = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
        for split, fname in file_map.items():
            if self.mode not in ("all", split):
                continue
            fpath = os.path.join(self.path, fname)
            if os.path.exists(fpath):
                data = self._load_jsonl(fpath)
                setattr(self, f"_{split}_data", data)
            else:
                logger.debug(f"TAUBench: {fpath} not found, skipping {split} split.")

    @staticmethod
    def _load_jsonl(path: str) -> List[Dict]:
        records = []
        with open(path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning(f"TAUBench: skipping malformed line {i} in {path}: {exc}")
        return records

    @staticmethod
    def _task_to_dict(task: TAUBenchTask) -> Dict:
        return {
            "id": task.task_id,
            "question": task.question,
            "expected_answer": task.expected_answer,
            "metadata": task.metadata,
        }

    def _get_id(self, example: Dict) -> Any:
        return example.get("id") or example.get("task_id")

    def _get_label(self, example: Dict) -> Any:
        return example.get("expected_answer") or example.get("label")

    def evaluate(self, prediction: Any, label: Any) -> Dict:
        """Exact-match evaluation (used by the base Evaluator)."""
        correct = str(prediction).strip().lower() == str(label).strip().lower()
        return {"correct": int(correct), "exact_match": float(correct)}


# ---------------------------------------------------------------------------
# TAUBenchRunner — executes k trials and produces a TAUBenchReport
# ---------------------------------------------------------------------------

class TAUBenchRunner:
    """Runs pass^k evaluation for a set of :class:`TAUBenchTask` objects.

    Parameters
    ----------
    benchmark:
        The :class:`TAUBench` instance to evaluate.
    agent_fn:
        Callable ``(question: str) -> str`` that produces the agent's answer.
        This is the entry point for running whatever workflow / graph is being
        evaluated.
    k:
        Number of trials per task (the ``k`` in pass^k).  Typical values: 4, 8.
    num_workers:
        Parallel workers for trial execution.  Set to 1 to disable parallelism.
    eval_mode:
        Which data split to use: ``"test"`` (default), ``"dev"``, or ``"train"``.
    seed:
        Optional random seed for reproducibility.
    """

    def __init__(
        self,
        benchmark: TAUBench,
        agent_fn: Callable[[str], str],
        k: int = 8,
        num_workers: int = 1,
        eval_mode: str = "test",
        seed: Optional[int] = None,
    ) -> None:
        self._bench = benchmark
        self._agent_fn = agent_fn
        self._k = k
        self._num_workers = num_workers
        self._eval_mode = eval_mode
        self._seed = seed

    # ------------------------------------------------------------------
    # Factory — build from an existing Benchmark + workflow graph
    # ------------------------------------------------------------------

    @classmethod
    def from_benchmark(
        cls,
        benchmark: Benchmark,
        graph: Any,               # WorkFlowGraph or ActionGraph
        evaluator: Any,           # evoagentx Evaluator
        k: int = 8,
        num_workers: int = 1,
        eval_mode: str = "test",
        seed: Optional[int] = None,
        collate_fn: Optional[Callable] = None,
        output_fn: Optional[Callable] = None,
    ) -> "TAUBenchRunner":
        """Wrap any existing Benchmark with the pass^k harness.

        Parameters
        ----------
        benchmark:
            Any :class:`Benchmark` subclass (GSM8K, HotPotQA, MBPP, …).
        graph:
            The workflow graph to evaluate.
        evaluator:
            An :class:`~evoagentx.evaluators.evaluator.Evaluator` instance.
        collate_fn:
            Maps a raw benchmark example to the graph's input dict.
        output_fn:
            Maps the graph's output to the string fed to ``benchmark.evaluate``.
        """
        def _agent_fn(question: str) -> str:
            # collate_fn converts the question string back to a graph-compatible input
            if collate_fn is not None:
                inputs = collate_fn(question)
            else:
                inputs = {"question": question}

            output = graph.execute(**inputs) if hasattr(graph, "execute") else graph(inputs)
            if output_fn is not None:
                return output_fn(output)
            return str(output)

        # Build TAUBench tasks from the existing benchmark's test data
        raw_data = benchmark.get_test_data() if eval_mode == "test" else (
            benchmark.get_dev_data() if eval_mode == "dev"
            else benchmark.get_train_data()
        )
        tasks = [
            TAUBenchTask(
                task_id=str(benchmark._get_id(ex)),
                question=str(ex.get("question") or ex.get("prompt") or str(ex)),
                expected_answer=benchmark._get_label(ex),
                check_fn=lambda pred, label: bool(
                    benchmark.evaluate(pred, label).get("correct", 0)
                ),
            )
            for ex in raw_data
        ]
        tau = TAUBench(tasks=tasks, name=f"TAUBench({benchmark.name})")
        return cls(
            benchmark=tau,
            agent_fn=_agent_fn,
            k=k,
            num_workers=num_workers,
            eval_mode="test",   # tasks already loaded into test split
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(
        self,
        sample_k: Optional[int] = None,
        indices: Optional[List[int]] = None,
    ) -> TAUBenchReport:
        """Execute all trials and return a :class:`TAUBenchReport`.

        Parameters
        ----------
        sample_k:
            Run only a random sample of ``sample_k`` tasks (useful for quick
            checks).
        indices:
            Run only the tasks at the specified indices.
        """
        data = self._bench._get_eval_data_raw(
            eval_mode=self._eval_mode,
            sample_k=sample_k,
            indices=indices,
            seed=self._seed,
        )
        if not data:
            logger.warning("TAUBenchRunner: no data found for the selected mode/indices.")
            return TAUBenchReport(
                benchmark_name=self._bench.name,
                k=self._k,
                task_results=[],
            )

        logger.info(
            f"TAUBenchRunner: evaluating {len(data)} tasks × {self._k} trials "
            f"({len(data) * self._k} total calls) | workers={self._num_workers}"
        )
        wall_start = time.time()
        task_results = self._run_all_tasks(data)
        wall_end = time.time()

        report = TAUBenchReport(
            benchmark_name=self._bench.name,
            k=self._k,
            task_results=task_results,
            total_wall_seconds=wall_end - wall_start,
        )
        logger.info(f"\n{report.summary()}")
        return report

    def _run_all_tasks(self, data: List[Dict]) -> List[TaskResult]:
        """Execute all tasks, respecting num_workers."""
        if self._num_workers <= 1:
            return [self._run_single_task(ex) for ex in data]

        results: List[TaskResult] = []
        with ThreadPoolExecutor(max_workers=self._num_workers) as pool:
            futures = {pool.submit(self._run_single_task, ex): ex for ex in data}
            for fut in as_completed(futures):
                results.append(fut.result())
        return results

    def _run_single_task(self, example: Dict) -> TaskResult:
        """Run k trials for one task example."""
        task_id = str(self._bench._get_id(example))
        label = self._bench._get_label(example)
        question = example.get("question") or example.get("prompt") or str(example)

        trials: List[TrialResult] = []
        for i in range(self._k):
            t_start = time.time()
            error = None
            is_correct = False
            prediction = ""
            try:
                prediction = self._agent_fn(str(question))
                metrics = self._bench.evaluate(prediction, label)
                is_correct = bool(metrics.get("correct", metrics.get("exact_match", 0)))
            except Exception as exc:
                error = str(exc)
                logger.debug(f"TAUBench trial error task={task_id} trial={i}: {exc}")

            trials.append(TrialResult(
                task_id=task_id,
                trial_index=i,
                prediction=prediction,
                is_correct=is_correct,
                latency_seconds=time.time() - t_start,
                error=error,
            ))

        return TaskResult(task_id=task_id, k=self._k, trials=trials)


# ---------------------------------------------------------------------------
# Monkey-patch helper: _get_eval_data_raw on Benchmark
# ---------------------------------------------------------------------------

def _get_eval_data_raw(
    self: Benchmark,
    eval_mode: str = "test",
    indices: Optional[List[int]] = None,
    sample_k: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Dict]:
    """Extension of Benchmark._get_eval_data that does not lose raw records."""
    if eval_mode == "test":
        data = self._test_data or []
    elif eval_mode == "dev":
        data = self._dev_data or []
    else:
        data = self._train_data or []
    return self._get_data(data, indices=indices, sample_k=sample_k, seed=seed)


# Attach as a method to the Benchmark class without modifying source
Benchmark._get_eval_data_raw = _get_eval_data_raw  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Convenience: compare two reports
# ---------------------------------------------------------------------------

def compare_reports(
    baseline: TAUBenchReport,
    candidate: TAUBenchReport,
) -> Dict[str, float]:
    """Return a dict of metric deltas (candidate − baseline).

    Positive values mean the candidate improved.
    """
    return {
        "pass_at_1_delta": candidate.pass_at_1 - baseline.pass_at_1,
        "pass_k_delta": candidate.pass_k - baseline.pass_k,
        "consistency_rate_delta": candidate.consistency_rate - baseline.consistency_rate,
        "mean_latency_delta": candidate.mean_latency - baseline.mean_latency,
        "baseline_pass_at_1": baseline.pass_at_1,
        "candidate_pass_at_1": candidate.pass_at_1,
        "baseline_pass_k": baseline.pass_k,
        "candidate_pass_k": candidate.pass_k,
    }

"""AgentBench multi-domain evaluation harness.

AgentBench evaluates agents across 8 heterogeneous environments — spanning
web browsing, operating system operations, database queries, knowledge retrieval,
code execution, and more — to produce a multi-dimensional performance profile.

This module provides:

* :class:`AgentBenchDomain` — enum of the 8 canonical evaluation domains.
* :class:`AgentBenchTask` — one task in a specific domain.
* :class:`AgentBench` — :class:`~evoagentx.benchmark.benchmark.Benchmark`
  subclass that loads, evaluates, and aggregates results.
* Per-domain :class:`DomainScorer` implementations.

Dataset format (JSONL per domain, one task per line)::

    {
        "task_id": "os_001",
        "domain": "os",
        "instruction": "List all Python files modified in the last 24 hours",
        "expected": ["file1.py", "file2.py"],
        "evaluation_type": "set_match",
        "metadata": {}
    }

Usage
-----
    >>> from evoagentx.benchmark.agent_bench import AgentBench, AgentBenchDomain
    >>> bench = AgentBench(path="~/.evoagentx/data/agent_bench")
    >>> example = bench.get_example_by_index(0)
    >>> metrics = bench.evaluate(prediction="file1.py\\nfile2.py", label=["file1.py", "file2.py"])
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from ..core.logging import logger
from .benchmark import Benchmark


# ---------------------------------------------------------------------------
# Domain enum
# ---------------------------------------------------------------------------

class AgentBenchDomain(str, Enum):
    """The 8 evaluation domains from AgentBench.

    Values correspond to the ``"domain"`` field in JSONL task files.
    """
    OS = "os"                           # shell / file-system operations
    DB = "db"                           # SQL database queries
    KNOWLEDGE_GRAPH = "kg"              # graph database traversal
    DIGITAL_CARD_GAME = "card"          # strategic card game play
    LATERAL_THINKING = "lateral"        # creative / divergent reasoning
    WEB_SHOPPING = "shop"               # e-commerce navigation & purchase
    WEB_BROWSING = "web"                # general web navigation
    HOUSE3D = "house"                   # embodied agent in 3D environment


# Canonical ordering for aggregate reporting
DOMAIN_ORDER = [d for d in AgentBenchDomain]


# ---------------------------------------------------------------------------
# Evaluation types
# ---------------------------------------------------------------------------

class EvaluationType(str, Enum):
    """How the agent's output is compared to the expected answer."""
    EXACT_MATCH = "exact_match"
    SET_MATCH = "set_match"          # order-insensitive list comparison
    SUBSTRING = "substring"          # expected is substring of prediction
    REGEX = "regex"                  # expected is a regex pattern
    NUMERIC_TOLERANCE = "numeric"    # within ±tolerance of a number
    LLM_JUDGE = "llm_judge"          # requires an LLM evaluator (not auto)


# ---------------------------------------------------------------------------
# Task container
# ---------------------------------------------------------------------------

@dataclass
class AgentBenchTask:
    """One task in a specific AgentBench domain.

    Parameters
    ----------
    task_id:
        Unique identifier within its domain.
    domain:
        The :class:`AgentBenchDomain` this task belongs to.
    instruction:
        The natural-language directive given to the agent.
    expected:
        The expected output (format depends on evaluation_type).
    evaluation_type:
        How to compare the prediction to ``expected``.
    numeric_tolerance:
        Tolerance for :attr:`EvaluationType.NUMERIC_TOLERANCE` tasks.
    metadata:
        Arbitrary additional metadata.
    """
    task_id: str
    domain: AgentBenchDomain
    instruction: str
    expected: Any
    evaluation_type: EvaluationType = EvaluationType.EXACT_MATCH
    numeric_tolerance: float = 0.01
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "domain": self.domain.value,
            "instruction": self.instruction,
            "expected": self.expected,
            "evaluation_type": self.evaluation_type.value,
            "numeric_tolerance": self.numeric_tolerance,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "AgentBenchTask":
        domain_val = d.get("domain", "web")
        try:
            domain = AgentBenchDomain(domain_val)
        except ValueError:
            logger.warning(f"AgentBench: unknown domain '{domain_val}', defaulting to WEB_BROWSING")
            domain = AgentBenchDomain.WEB_BROWSING

        eval_type_val = d.get("evaluation_type", "exact_match")
        try:
            eval_type = EvaluationType(eval_type_val)
        except ValueError:
            eval_type = EvaluationType.EXACT_MATCH

        return cls(
            task_id=d.get("task_id", ""),
            domain=domain,
            instruction=d.get("instruction", d.get("question", "")),
            expected=d.get("expected", d.get("answer", "")),
            evaluation_type=eval_type,
            numeric_tolerance=float(d.get("numeric_tolerance", 0.01)),
            metadata={k: v for k, v in d.items() if k not in (
                "task_id", "domain", "instruction", "question",
                "expected", "answer", "evaluation_type", "numeric_tolerance"
            )},
        )


# ---------------------------------------------------------------------------
# Per-task scoring
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase and strip whitespace."""
    return text.strip().lower()


def _parse_list_output(text: str) -> List[str]:
    """Parse newline- or comma-separated agent output into a list of items."""
    if "\n" in text:
        items = [_normalise(l) for l in text.split("\n") if l.strip()]
    else:
        items = [_normalise(p) for p in text.split(",") if p.strip()]
    return items


def score_task(prediction: str, task: AgentBenchTask) -> Dict[str, Any]:
    """Evaluate a single prediction against an AgentBenchTask.

    Returns a metrics dict with at least ``"correct"`` (0 or 1) and
    ``"score"`` (float 0.0–1.0).
    """
    pred = str(prediction).strip()

    if task.evaluation_type == EvaluationType.EXACT_MATCH:
        correct = _normalise(pred) == _normalise(str(task.expected))
        return {"correct": int(correct), "score": float(correct)}

    if task.evaluation_type == EvaluationType.SET_MATCH:
        expected_items: List[str]
        if isinstance(task.expected, list):
            expected_items = [_normalise(str(e)) for e in task.expected]
        else:
            expected_items = _parse_list_output(str(task.expected))
        pred_items = _parse_list_output(pred)
        expected_set = set(expected_items)
        pred_set = set(pred_items)
        if not expected_set:
            return {"correct": 1, "score": 1.0}
        precision = len(pred_set & expected_set) / len(pred_set) if pred_set else 0.0
        recall = len(pred_set & expected_set) / len(expected_set)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "correct": int(expected_set == pred_set),
            "score": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        }

    if task.evaluation_type == EvaluationType.SUBSTRING:
        expected_lower = _normalise(str(task.expected))
        correct = expected_lower in _normalise(pred)
        return {"correct": int(correct), "score": float(correct)}

    if task.evaluation_type == EvaluationType.REGEX:
        pattern = str(task.expected)
        try:
            match = bool(re.search(pattern, pred, re.IGNORECASE | re.DOTALL))
        except re.error:
            match = False
        return {"correct": int(match), "score": float(match)}

    if task.evaluation_type == EvaluationType.NUMERIC_TOLERANCE:
        try:
            pred_num = float(pred.replace(",", ""))
            exp_num = float(str(task.expected).replace(",", ""))
            correct = abs(pred_num - exp_num) <= task.numeric_tolerance * abs(exp_num + 1e-9)
        except ValueError:
            correct = False
        return {"correct": int(correct), "score": float(correct)}

    if task.evaluation_type == EvaluationType.LLM_JUDGE:
        # Placeholder — returns 0 when no LLM judge is provided.
        # Phase 1A process-reward work will wire up a judge here.
        return {"correct": 0, "score": 0.0, "note": "llm_judge not configured"}

    return {"correct": 0, "score": 0.0}


# ---------------------------------------------------------------------------
# AgentBench Benchmark class
# ---------------------------------------------------------------------------

class AgentBench(Benchmark):
    """Multi-domain agent evaluation benchmark.

    Aggregates task scores across all :class:`AgentBenchDomain` values.

    Expected directory layout::

        ~/.evoagentx/data/agent_bench/
            os/
                test.jsonl
            db/
                test.jsonl
            ...   (one sub-directory per domain)

    Alternatively, a flat layout with all tasks in one file::

        ~/.evoagentx/data/agent_bench/
            test.jsonl    (all tasks, each record has a "domain" field)

    Parameters
    ----------
    path:
        Root directory of the dataset.
    mode:
        Which splits to load.
    domains:
        Limit loading to specific domains.  Defaults to all 8.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        mode: str = "test",
        domains: Optional[List[AgentBenchDomain]] = None,
        **kwargs,
    ) -> None:
        self._domains = domains or list(AgentBenchDomain)
        path = os.path.expanduser(path or "~/.evoagentx/data/agent_bench")
        super().__init__(name="AgentBench", path=path, mode=mode, **kwargs)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        all_records: Dict[str, List[Dict]] = {"train": [], "dev": [], "test": []}

        # Try flat layout first
        for split in ("train", "dev", "test"):
            flat_path = os.path.join(self.path, f"{split}.jsonl")
            if os.path.exists(flat_path):
                records = self._load_jsonl(flat_path)
                # filter to selected domains
                filtered = [
                    r for r in records
                    if AgentBenchDomain(r.get("domain", "web")) in self._domains
                ]
                all_records[split].extend(filtered)
                logger.info(f"AgentBench: loaded {len(filtered)} {split} tasks from {flat_path}")

        # Try per-domain directories
        for domain in self._domains:
            for split in ("train", "dev", "test"):
                fpath = os.path.join(self.path, domain.value, f"{split}.jsonl")
                if os.path.exists(fpath):
                    records = self._load_jsonl(fpath)
                    # inject domain field if missing
                    for r in records:
                        r.setdefault("domain", domain.value)
                    all_records[split].extend(records)
                    logger.info(f"AgentBench: loaded {len(records)} {split}/{domain.value} tasks")

        for split in ("train", "dev", "test"):
            if self.mode in ("all", split) and all_records[split]:
                setattr(self, f"_{split}_data", all_records[split])

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
                    logger.warning(f"AgentBench: skipping malformed line {i} in {path}: {exc}")
        return records

    # ------------------------------------------------------------------
    # Benchmark interface
    # ------------------------------------------------------------------

    def _get_id(self, example: Dict) -> Any:
        return example.get("task_id") or example.get("id")

    def _get_label(self, example: Dict) -> Any:
        return example.get("expected") or example.get("answer")

    def get_task(self, example: Dict) -> AgentBenchTask:
        return AgentBenchTask.from_dict(example)

    def evaluate(self, prediction: Any, label: Any) -> Dict:
        """Evaluate a prediction.

        If the task object is available (via :meth:`get_task`), use
        :func:`score_task` for domain-aware scoring.  Otherwise, fall back
        to exact match.
        """
        pred_str = str(prediction)
        if isinstance(label, dict) and "expected" in label:
            # label is the full example dict
            task = AgentBenchTask.from_dict(label)
            return score_task(pred_str, task)
        # Simple exact-match fallback
        correct = _normalise(pred_str) == _normalise(str(label))
        return {"correct": int(correct), "score": float(correct)}

    def evaluate_with_example(self, prediction: Any, example: Dict) -> Dict:
        """Evaluate a prediction using the full example dict for domain context."""
        task = self.get_task(example)
        metrics = score_task(str(prediction), task)
        metrics["domain"] = task.domain.value
        return metrics

    # ------------------------------------------------------------------
    # Aggregate metrics
    # ------------------------------------------------------------------

    def aggregate_domain_scores(self, per_task_metrics: List[Dict]) -> Dict[str, Any]:
        """Group per-task metrics by domain and compute per-domain averages.

        Parameters
        ----------
        per_task_metrics:
            List of dicts, each with at least ``"domain"`` and ``"score"`` keys.

        Returns
        -------
        dict
            Keys: domain names + ``"overall"``.  Values: mean score.
        """
        domain_scores: Dict[str, List[float]] = {d.value: [] for d in DOMAIN_ORDER}
        for m in per_task_metrics:
            d = m.get("domain", "unknown")
            if d in domain_scores:
                domain_scores[d].append(float(m.get("score", 0.0)))

        result = {}
        all_scores: List[float] = []
        for domain in DOMAIN_ORDER:
            scores = domain_scores.get(domain.value, [])
            avg = sum(scores) / len(scores) if scores else 0.0
            result[domain.value] = round(avg, 4)
            all_scores.extend(scores)

        result["overall"] = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0
        return result

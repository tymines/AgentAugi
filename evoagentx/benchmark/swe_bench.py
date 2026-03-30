"""SWE-Bench task format adapter.

SWE-Bench (Software Engineering Benchmark) evaluates an agent's ability to
resolve real-world GitHub issues by producing a code patch.  Each task
provides a problem statement (issue description), the relevant code context,
and a set of tests that the patch must pass.

This module provides:

* :class:`SWEBenchTask` — a structured container for one SWE-Bench instance.
* :class:`SWEBench` — a :class:`~evoagentx.benchmark.benchmark.Benchmark`
  subclass that can load the standard JSONL dataset format.
* :func:`evaluate_patch_heuristic` — a heuristic evaluator for when the actual
  test suite cannot be executed (e.g. CI is unavailable).  Scores are
  estimates, not ground truth.

Dataset format (JSONL, one task per line)::

    {
        "instance_id": "django__django-11099",
        "repo": "django/django",
        "base_commit": "abc123",
        "problem_statement": "Fix crash when ...",
        "hints_text": "Consider modifying ...",
        "test_patch": "diff --git a/tests/...",
        "patch": "diff --git a/django/...",    # ground truth patch (for eval)
        "FAIL_TO_PASS": ["test_foo", "test_bar"],
        "PASS_TO_PASS": ["test_baz"]
    }

Usage
-----
    >>> from evoagentx.benchmark.swe_bench import SWEBench
    >>> bench = SWEBench(path="~/.evoagentx/data/swe_bench", mode="test")
    >>> example = bench.get_example_by_index(0, mode="test")
    >>> pred_patch = my_agent(example["problem_statement"])
    >>> metrics = bench.evaluate(pred_patch, bench._get_label(example))
    >>> print(metrics)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..core.logging import logger
from .benchmark import Benchmark


# ---------------------------------------------------------------------------
# Task container
# ---------------------------------------------------------------------------

@dataclass
class SWEBenchTask:
    """One SWE-Bench problem instance.

    Parameters
    ----------
    instance_id:
        Unique identifier, typically ``"<org>__<repo>-<issue_number>"``.
    repo:
        Repository in ``"org/repo"`` format.
    base_commit:
        Git commit hash of the codebase at evaluation time.
    problem_statement:
        The verbatim GitHub issue text shown to the agent.
    hints_text:
        Additional hints from issue comments (optional).
    test_patch:
        The test diff that is applied before running tests.
    reference_patch:
        The ground-truth minimal patch (used for reference-based eval).
    fail_to_pass:
        Tests that currently fail and should pass after applying the patch.
    pass_to_pass:
        Tests that currently pass and must still pass after applying the patch
        (regression tests).
    metadata:
        Arbitrary additional fields from the dataset.
    """

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: str = ""
    test_patch: str = ""
    reference_patch: str = ""
    fail_to_pass: List[str] = field(default_factory=list)
    pass_to_pass: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "base_commit": self.base_commit,
            "problem_statement": self.problem_statement,
            "hints_text": self.hints_text,
            "test_patch": self.test_patch,
            "reference_patch": self.reference_patch,
            "fail_to_pass": self.fail_to_pass,
            "pass_to_pass": self.pass_to_pass,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "SWEBenchTask":
        return cls(
            instance_id=d.get("instance_id", ""),
            repo=d.get("repo", ""),
            base_commit=d.get("base_commit", ""),
            problem_statement=d.get("problem_statement", ""),
            hints_text=d.get("hints_text", ""),
            test_patch=d.get("test_patch", ""),
            reference_patch=d.get("patch", ""),
            fail_to_pass=d.get("FAIL_TO_PASS", []),
            pass_to_pass=d.get("PASS_TO_PASS", []),
            metadata={k: v for k, v in d.items() if k not in (
                "instance_id", "repo", "base_commit", "problem_statement",
                "hints_text", "test_patch", "patch", "FAIL_TO_PASS", "PASS_TO_PASS"
            )},
        )


# ---------------------------------------------------------------------------
# Patch quality heuristics
# ---------------------------------------------------------------------------

def _extract_changed_files(patch: str) -> Set[str]:
    """Extract the set of files touched in a unified diff patch."""
    files: Set[str] = set()
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            files.add(line[6:].strip())
        elif line.startswith("--- a/"):
            files.add(line[6:].strip())
    return files


def _count_diff_lines(patch: str) -> tuple[int, int]:
    """Return (added_lines, removed_lines) from a unified diff."""
    added = sum(1 for l in patch.splitlines() if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in patch.splitlines() if l.startswith("-") and not l.startswith("---"))
    return added, removed


def evaluate_patch_heuristic(
    prediction: str,
    reference: str,
    fail_to_pass: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Heuristic evaluation of a predicted patch against a reference patch.

    This is used when the actual test environment is unavailable.  It does NOT
    execute tests — it estimates correctness from structural similarity.

    Scoring:
    * ``has_patch`` (1/0) — predicted text contains a non-empty diff.
    * ``files_overlap`` (0–1) — Jaccard overlap of files touched.
    * ``line_similarity`` (0–1) — ratio of matching diff lines.
    * ``addresses_keywords`` (0–1) — fraction of keywords from fail_to_pass test
      names that appear in the predicted patch.
    * ``heuristic_score`` — weighted composite of the above.

    Returns a dict of all metrics.
    """
    if not prediction or not prediction.strip():
        return {
            "has_patch": 0,
            "files_overlap": 0.0,
            "line_similarity": 0.0,
            "addresses_keywords": 0.0,
            "heuristic_score": 0.0,
            "resolved": False,
        }

    has_patch = int(
        "diff --git" in prediction or "--- a/" in prediction or "+++ b/" in prediction
    )

    # File overlap
    pred_files = _extract_changed_files(prediction)
    ref_files = _extract_changed_files(reference)
    if pred_files or ref_files:
        union = pred_files | ref_files
        intersection = pred_files & ref_files
        files_overlap = len(intersection) / len(union)
    else:
        files_overlap = 0.0

    # Line-level similarity (set-based Jaccard on non-hunk-header diff lines)
    def _diff_lines(patch: str) -> Set[str]:
        return {
            l.strip() for l in patch.splitlines()
            if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))
            and len(l.strip()) > 2
        }

    pred_lines = _diff_lines(prediction)
    ref_lines = _diff_lines(reference)
    if pred_lines or ref_lines:
        union_l = pred_lines | ref_lines
        inter_l = pred_lines & ref_lines
        line_similarity = len(inter_l) / len(union_l)
    else:
        line_similarity = 0.0

    # Keyword presence
    addresses_keywords = 0.0
    if fail_to_pass:
        found = sum(
            1 for kw in fail_to_pass
            if re.search(re.escape(kw.split("::")[-1]), prediction, re.IGNORECASE)
        )
        addresses_keywords = found / len(fail_to_pass)

    heuristic_score = (
        0.25 * has_patch
        + 0.35 * files_overlap
        + 0.25 * line_similarity
        + 0.15 * addresses_keywords
    )

    return {
        "has_patch": has_patch,
        "files_overlap": round(files_overlap, 4),
        "line_similarity": round(line_similarity, 4),
        "addresses_keywords": round(addresses_keywords, 4),
        "heuristic_score": round(heuristic_score, 4),
        "resolved": heuristic_score >= 0.5,
    }


# ---------------------------------------------------------------------------
# SWEBench Benchmark class
# ---------------------------------------------------------------------------

class SWEBench(Benchmark):
    """Benchmark adapter for the SWE-Bench dataset.

    Loads the standard SWE-Bench JSONL files and evaluates predicted patches
    using :func:`evaluate_patch_heuristic` (structural similarity) or, when
    integrated with a live test environment, exact test-pass evaluation.

    Expected directory structure::

        ~/.evoagentx/data/swe_bench/
            test.jsonl
            dev.jsonl    (optional)
            train.jsonl  (optional)

    Parameters
    ----------
    path:
        Directory containing the dataset JSONL files.
    mode:
        Which splits to load: ``"all"``, ``"test"``, ``"dev"``, ``"train"``.
    use_heuristic:
        If ``True`` (default), ``evaluate()`` uses the heuristic patch scorer.
        Set to ``False`` if plugging in an external test runner.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        mode: str = "test",
        use_heuristic: bool = True,
        **kwargs,
    ) -> None:
        self.use_heuristic = use_heuristic
        path = os.path.expanduser(path or "~/.evoagentx/data/swe_bench")
        super().__init__(name="SWEBench", path=path, mode=mode, **kwargs)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        file_map = {
            "train": "train.jsonl",
            "dev": "dev.jsonl",
            "test": "test.jsonl",
        }
        for split, fname in file_map.items():
            if self.mode not in ("all", split):
                continue
            fpath = os.path.join(self.path, fname)
            if os.path.exists(fpath):
                data = self._load_jsonl(fpath)
                setattr(self, f"_{split}_data", data)
                logger.info(f"SWEBench: loaded {len(data)} {split} instances from {fpath}")
            else:
                logger.debug(f"SWEBench: {fpath} not found, skipping {split} split.")

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
                    logger.warning(f"SWEBench: skipping malformed line {i}: {exc}")
        return records

    # ------------------------------------------------------------------
    # Benchmark interface
    # ------------------------------------------------------------------

    def _get_id(self, example: Dict) -> Any:
        return example.get("instance_id")

    def _get_label(self, example: Dict) -> Any:
        """Return the ground-truth patch for an example."""
        return example.get("patch", "")

    def get_task(self, example: Dict) -> SWEBenchTask:
        """Convert a raw dict to a :class:`SWEBenchTask`."""
        return SWEBenchTask.from_dict(example)

    def evaluate(self, prediction: Any, label: Any) -> Dict:
        """Evaluate a predicted patch against the reference patch.

        Parameters
        ----------
        prediction:
            The agent-generated patch (unified diff string).
        label:
            The ground-truth patch (unified diff string).

        Returns
        -------
        dict
            Metrics from :func:`evaluate_patch_heuristic`, including
            ``"resolved"`` (bool) and ``"heuristic_score"`` (float 0–1).
        """
        if self.use_heuristic:
            return evaluate_patch_heuristic(
                prediction=str(prediction),
                reference=str(label),
            )
        # Placeholder for integration with external test-runner
        return {"resolved": False, "heuristic_score": 0.0, "has_patch": 0}

    def get_problem_prompt(self, example: Dict) -> str:
        """Return the prompt to send to the agent for a given example.

        Combines the problem statement with any hints.
        """
        task = self.get_task(example)
        parts = [task.problem_statement]
        if task.hints_text:
            parts.append(f"\n\nHints:\n{task.hints_text}")
        return "\n".join(parts)

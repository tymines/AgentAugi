"""GAIA benchmark adapter — real-world reasoning and tool use.

GAIA (General AI Assistants) tests an agent's ability to answer real-world
questions that require planning, web search, file reading, and multi-step
reasoning.  Questions are divided into three difficulty levels:

* **Level 1** — straightforward, minimal tool use.
* **Level 2** — moderate; typically requires one or two web searches + synthesis.
* **Level 3** — hard; requires complex multi-hop reasoning, multiple tools, and
  long-horizon planning.

Dataset format (JSONL, one task per line)::

    {
        "task_id": "abc123",
        "level": 1,
        "question": "What is the population of Liechtenstein?",
        "final_answer": "38,000",
        "annotator_metadata": {
            "steps": 2,
            "number_of_tools": 1,
            "tools": ["web_search"]
        },
        "file_name": ""   # optional attached file
    }

Usage
-----
    >>> from evoagentx.benchmark.gaia_bench import GAIA
    >>> bench = GAIA(path="~/.evoagentx/data/gaia", mode="test")
    >>> example = bench.get_example_by_index(0, mode="test")
    >>> metrics = bench.evaluate(prediction="38000", label="38,000")
    >>> print(metrics)
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

from ..core.logging import logger
from .benchmark import Benchmark


# ---------------------------------------------------------------------------
# Difficulty level
# ---------------------------------------------------------------------------

class GAIALevel(IntEnum):
    """GAIA difficulty level (1 = easiest, 3 = hardest)."""
    LEVEL_1 = 1
    LEVEL_2 = 2
    LEVEL_3 = 3


# ---------------------------------------------------------------------------
# Task container
# ---------------------------------------------------------------------------

@dataclass
class GAIATask:
    """One GAIA question-answer pair.

    Parameters
    ----------
    task_id:
        Unique identifier for the task.
    level:
        Difficulty level (1–3).
    question:
        The question text shown to the agent.
    final_answer:
        Ground-truth answer string.
    file_name:
        Optional file attached to the task (e.g., a PDF, spreadsheet).
    annotator_metadata:
        Metadata about the reference solution (steps, tools used, etc.).
    """
    task_id: str
    level: GAIALevel
    question: str
    final_answer: str
    file_name: str = ""
    annotator_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "task_id": self.task_id,
            "level": int(self.level),
            "question": self.question,
            "final_answer": self.final_answer,
            "file_name": self.file_name,
            "annotator_metadata": self.annotator_metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "GAIATask":
        lvl = d.get("level", 1)
        try:
            level = GAIALevel(int(lvl))
        except (ValueError, TypeError):
            level = GAIALevel.LEVEL_1

        return cls(
            task_id=d.get("task_id", d.get("id", "")),
            level=level,
            question=d.get("question", d.get("Question", "")),
            final_answer=d.get("final_answer", d.get("answer", d.get("Answer", ""))),
            file_name=d.get("file_name", d.get("file", "")),
            annotator_metadata=d.get("annotator_metadata", {}),
        )


# ---------------------------------------------------------------------------
# Answer normalisation + scoring
# ---------------------------------------------------------------------------

def _normalise_number(text: str) -> Optional[float]:
    """Try to parse a numeric value from a string, handling commas and units."""
    # Strip currency symbols, percent signs, and commas
    cleaned = re.sub(r"[,$%€£¥]", "", text.strip())
    cleaned = cleaned.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _normalise_text(text: str) -> str:
    """Canonical normalisation: lowercase, remove punctuation, collapse spaces."""
    text = unicodedata.normalize("NFKD", text)
    text = text.lower()
    # Remove punctuation except for hyphens and decimals in numbers
    text = re.sub(r"[^\w\s\-\.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_final_answer(text: str) -> str:
    """Extract the final answer from agent output.

    Looks for common answer patterns like:
    - "The answer is X"
    - "Final answer: X"
    - "Answer: X"
    - Last line of text (fallback)
    """
    patterns = [
        r"(?:final answer|the answer is|answer is|answer:)\s*[:\-]?\s*(.+?)(?:\.|$)",
        r"(?:therefore|thus|so),?\s+(?:the answer is\s+)?(.+?)(?:\.|$)",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    # Fallback: return the last non-empty line
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else text.strip()


def evaluate_gaia_answer(prediction: str, expected: str) -> Dict[str, Any]:
    """Compare a predicted GAIA answer to the expected answer.

    Applies the GAIA evaluation protocol:

    1. Try numeric comparison (within 1% tolerance).
    2. Exact normalised text match.
    3. Partial match (expected tokens ⊆ prediction tokens).

    Returns a dict with ``"correct"`` (0/1) and ``"match_type"`` explaining
    which comparison succeeded.
    """
    pred_clean = _extract_final_answer(prediction)
    exp_clean = expected.strip()

    # --- 1. Numeric comparison ---
    pred_num = _normalise_number(pred_clean)
    exp_num = _normalise_number(exp_clean)
    if pred_num is not None and exp_num is not None:
        tolerance = max(abs(exp_num) * 0.01, 0.5)  # 1% or 0.5 units
        if abs(pred_num - exp_num) <= tolerance:
            return {"correct": 1, "score": 1.0, "match_type": "numeric"}
        return {"correct": 0, "score": 0.0, "match_type": "numeric_mismatch"}

    # --- 2. Exact normalised match ---
    pred_norm = _normalise_text(pred_clean)
    exp_norm = _normalise_text(exp_clean)
    if pred_norm == exp_norm:
        return {"correct": 1, "score": 1.0, "match_type": "exact"}

    # --- 3. Substring / token containment ---
    exp_tokens = set(exp_norm.split())
    pred_tokens = set(pred_norm.split())
    if exp_tokens and exp_tokens.issubset(pred_tokens):
        return {"correct": 1, "score": 0.9, "match_type": "subset"}

    # --- 4. Partial overlap F1 ---
    if exp_tokens and pred_tokens:
        overlap = len(exp_tokens & pred_tokens)
        f1 = 2 * overlap / (len(exp_tokens) + len(pred_tokens))
    else:
        f1 = 0.0

    return {"correct": 0, "score": round(f1, 4), "match_type": "partial"}


# ---------------------------------------------------------------------------
# GAIA Benchmark class
# ---------------------------------------------------------------------------

class GAIA(Benchmark):
    """Benchmark adapter for the GAIA real-world reasoning dataset.

    Loads GAIA JSONL files and evaluates agent answers using
    :func:`evaluate_gaia_answer`.

    Expected directory layout::

        ~/.evoagentx/data/gaia/
            test.jsonl
            dev.jsonl

    Parameters
    ----------
    path:
        Directory containing the JSONL files.
    mode:
        Which splits to load.
    levels:
        Restrict evaluation to specific difficulty levels.  Defaults to all.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        mode: str = "test",
        levels: Optional[List[GAIALevel]] = None,
        **kwargs,
    ) -> None:
        self._levels = levels or list(GAIALevel)
        path = os.path.expanduser(path or "~/.evoagentx/data/gaia")
        super().__init__(name="GAIA", path=path, mode=mode, **kwargs)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_data(self) -> None:
        file_map = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}
        for split, fname in file_map.items():
            if self.mode not in ("all", split):
                continue
            fpath = os.path.join(self.path, fname)
            if os.path.exists(fpath):
                data = self._load_jsonl(fpath)
                # Filter by level if specified
                level_ints = {int(l) for l in self._levels}
                filtered = [r for r in data if int(r.get("level", 1)) in level_ints]
                setattr(self, f"_{split}_data", filtered)
                logger.info(
                    f"GAIA: loaded {len(filtered)}/{len(data)} {split} tasks "
                    f"(levels={[int(l) for l in self._levels]})"
                )
            else:
                logger.debug(f"GAIA: {fpath} not found, skipping {split} split.")

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
                    logger.warning(f"GAIA: skipping malformed line {i}: {exc}")
        return records

    # ------------------------------------------------------------------
    # Benchmark interface
    # ------------------------------------------------------------------

    def _get_id(self, example: Dict) -> Any:
        return example.get("task_id") or example.get("id")

    def _get_label(self, example: Dict) -> Any:
        return example.get("final_answer") or example.get("answer")

    def get_task(self, example: Dict) -> GAIATask:
        """Convert a raw dict to a :class:`GAIATask`."""
        return GAIATask.from_dict(example)

    def evaluate(self, prediction: Any, label: Any) -> Dict:
        """Evaluate a prediction using :func:`evaluate_gaia_answer`.

        Parameters
        ----------
        prediction:
            The agent's answer string.
        label:
            The ground-truth answer string.
        """
        return evaluate_gaia_answer(str(prediction), str(label))

    # ------------------------------------------------------------------
    # Level-stratified reporting
    # ------------------------------------------------------------------

    def aggregate_by_level(
        self,
        per_task_metrics: List[Tuple[int, Dict]],
    ) -> Dict[str, Any]:
        """Aggregate scores broken down by difficulty level.

        Parameters
        ----------
        per_task_metrics:
            List of ``(level: int, metrics_dict)`` pairs.

        Returns
        -------
        dict
            Per-level accuracy + overall accuracy.
        """
        level_correct: Dict[int, List[int]] = {1: [], 2: [], 3: []}
        for level, metrics in per_task_metrics:
            level_correct.setdefault(level, []).append(metrics.get("correct", 0))

        result = {}
        all_correct: List[int] = []
        for lvl in [1, 2, 3]:
            vals = level_correct.get(lvl, [])
            acc = sum(vals) / len(vals) if vals else 0.0
            result[f"level_{lvl}_accuracy"] = round(acc, 4)
            result[f"level_{lvl}_count"] = len(vals)
            all_correct.extend(vals)

        result["overall_accuracy"] = (
            round(sum(all_correct) / len(all_correct), 4) if all_correct else 0.0
        )
        result["total_count"] = len(all_correct)
        return result

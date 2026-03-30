"""ReasoningBank — shared, persistent store of successful reasoning patterns.

Agents consult the bank before tackling a new task, retrieving the most
relevant past reasoning chains to include as in-context examples.  Unlike
Reflexion memory (which stores failures and corrections), the ReasoningBank
stores *successful* trajectories so agents can repeat winning strategies.

Inspired by the ReasoningBank pattern (arXiv:2509.25140) but written from
scratch for the EvoAgentX architecture.

Design
------
- Each ``ReasoningEntry`` captures the task description, ordered reasoning
  steps, final outcome, and a quality score.
- Similarity search is keyword/token-overlap-based by default (zero
  dependencies).  If the caller supplies an ``embed_fn`` the search uses
  cosine similarity over dense vectors instead.
- JSON persistence keeps the bank available across sessions.
- Periodic pruning removes entries whose quality score falls below a
  configurable threshold.

Usage
-----
    >>> from evoagentx.memory.reasoning_bank import ReasoningBank
    >>> bank = ReasoningBank(min_quality=0.6, max_entries=500)
    >>> bank.store(
    ...     task="Sort a list of integers",
    ...     steps=["Identify the list", "Apply sorted()", "Return result"],
    ...     outcome="Correct sorted list returned",
    ...     quality_score=0.95,
    ... )
    >>> similar = bank.retrieve_similar("Sort numbers in ascending order", k=3)
    >>> for entry in similar:
    ...     print(entry.task, entry.quality_score)
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class ReasoningEntry:
    """One entry in the ReasoningBank.

    Attributes
    ----------
    entry_id:
        Unique identifier (auto-assigned sequential integer as string).
    task:
        Natural-language description of the task this reasoning solved.
    steps:
        Ordered list of reasoning steps or actions taken.
    outcome:
        Human-readable description of the final outcome.
    quality_score:
        Quality rating in [0, 1].  Higher is better.  Entries below the
        bank's ``min_quality`` threshold are pruned.
    metadata:
        Arbitrary extra information (e.g. model used, duration, tags).
    created_at:
        Unix timestamp when the entry was stored.
    embedding:
        Optional dense vector (list of floats) for the task description,
        populated when an ``embed_fn`` is supplied to the bank.
    """

    entry_id: str
    task: str
    steps: List[str]
    outcome: str
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    embedding: Optional[List[float]] = field(default=None, repr=False)

    def step_summary(self, max_steps: int = 5) -> str:
        """Return a compact multi-line summary of the first ``max_steps`` steps."""
        shown = self.steps[:max_steps]
        lines = [f"  {i + 1}. {s}" for i, s in enumerate(shown)]
        if len(self.steps) > max_steps:
            lines.append(f"  ... ({len(self.steps) - max_steps} more steps)")
        return "\n".join(lines)

    def to_prompt_excerpt(self, max_steps: int = 5) -> str:
        """Format the entry as an in-context example for an LLM prompt."""
        return (
            f"Past example (quality={self.quality_score:.2f}):\n"
            f"Task: {self.task}\n"
            f"Reasoning steps:\n{self.step_summary(max_steps)}\n"
            f"Outcome: {self.outcome}"
        )


# ---------------------------------------------------------------------------
# Similarity utilities
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> List[str]:
    """Split ``text`` into lowercase word tokens, stripping punctuation."""
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in text.lower())
    return [w for w in cleaned.split() if w]


def _jaccard_similarity(a: str, b: str) -> float:
    """Jaccard similarity between the token sets of strings ``a`` and ``b``."""
    tokens_a = set(_tokenise(a))
    tokens_b = set(_tokenise(b))
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError(
            f"Embedding dimension mismatch: {len(a)} vs {len(b)}"
        )
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# ReasoningBank
# ---------------------------------------------------------------------------

class ReasoningBank:
    """Persistent repository of successful reasoning chains.

    Parameters
    ----------
    min_quality:
        Entries with ``quality_score`` below this threshold are rejected on
        store and removed during pruning.  Default 0.5.
    max_entries:
        Maximum number of entries kept in memory.  When the bank is full the
        lowest-quality entry is evicted before a new one is added.  Default 1000.
    embed_fn:
        Optional callable ``(text: str) -> List[float]`` that produces a
        dense embedding for similarity search.  When supplied, cosine
        similarity is used; otherwise Jaccard similarity over tokens.
    persist_path:
        If set, the bank is automatically saved to this JSON file after
        every ``store()`` call and loaded on construction.

    Examples
    --------
    >>> bank = ReasoningBank(min_quality=0.7, max_entries=200)
    >>> bank.store("Multiply two numbers", ["parse inputs", "compute a*b"], "42", 0.9)
    >>> results = bank.retrieve_similar("Compute a product", k=2)
    """

    def __init__(
        self,
        min_quality: float = 0.5,
        max_entries: int = 1000,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        persist_path: Optional[str] = None,
    ) -> None:
        self._min_quality = min_quality
        self._max_entries = max_entries
        self._embed_fn = embed_fn
        self._persist_path = persist_path
        self._entries: List[ReasoningEntry] = []
        self._next_id: int = 1

        if persist_path and os.path.exists(persist_path):
            self.load(persist_path)

    # ------------------------------------------------------------------
    # Storing entries
    # ------------------------------------------------------------------

    def store(
        self,
        task: str,
        steps: List[str],
        outcome: str,
        quality_score: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ReasoningEntry]:
        """Add a new reasoning chain to the bank if quality is sufficient.

        Parameters
        ----------
        task:
            Natural-language description of the task.
        steps:
            Ordered list of reasoning steps.
        outcome:
            Description of the final outcome.
        quality_score:
            Quality rating in [0, 1].  Entries below ``min_quality`` are
            silently discarded.
        metadata:
            Optional extra metadata dict.

        Returns
        -------
        ReasoningEntry or None
            The stored entry, or None if the quality threshold was not met.
        """
        quality_score = max(0.0, min(1.0, quality_score))

        if quality_score < self._min_quality:
            return None

        embedding: Optional[List[float]] = None
        if self._embed_fn is not None:
            try:
                embedding = self._embed_fn(task)
            except Exception:
                embedding = None

        entry = ReasoningEntry(
            entry_id=str(self._next_id),
            task=task,
            steps=list(steps),
            outcome=outcome,
            quality_score=quality_score,
            metadata=metadata or {},
            embedding=embedding,
        )
        self._next_id += 1

        # Evict lowest-quality entry if at capacity
        if len(self._entries) >= self._max_entries:
            self._evict_lowest()

        self._entries.append(entry)

        if self._persist_path:
            self._save_quietly()

        return entry

    def _evict_lowest(self) -> None:
        """Remove the entry with the lowest quality score."""
        if not self._entries:
            return
        worst_idx = min(
            range(len(self._entries)),
            key=lambda i: self._entries[i].quality_score,
        )
        self._entries.pop(worst_idx)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_similar(
        self,
        task: str,
        k: int = 5,
        min_similarity: float = 0.0,
    ) -> List[ReasoningEntry]:
        """Return the ``k`` most similar entries to ``task``.

        Parameters
        ----------
        task:
            The query task description.
        k:
            Maximum number of entries to return.
        min_similarity:
            Only include entries with similarity ≥ this value.

        Returns
        -------
        List[ReasoningEntry]
            Sorted by descending similarity (most similar first).
        """
        if not self._entries:
            return []

        scored: List[Tuple[float, ReasoningEntry]] = []

        query_embedding: Optional[List[float]] = None
        if self._embed_fn is not None:
            try:
                query_embedding = self._embed_fn(task)
            except Exception:
                query_embedding = None

        for entry in self._entries:
            sim = self._compute_similarity(
                task, entry.task, query_embedding, entry.embedding
            )
            if sim >= min_similarity:
                scored.append((sim, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:k]]

    def _compute_similarity(
        self,
        query: str,
        candidate: str,
        query_emb: Optional[List[float]],
        candidate_emb: Optional[List[float]],
    ) -> float:
        """Compute similarity between query and candidate strings.

        Uses cosine similarity when both embeddings are available, otherwise
        falls back to Jaccard token overlap.
        """
        if query_emb is not None and candidate_emb is not None:
            try:
                return _cosine_similarity(query_emb, candidate_emb)
            except ValueError:
                pass
        return _jaccard_similarity(query, candidate)

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune(self, quality_threshold: Optional[float] = None) -> int:
        """Remove entries below ``quality_threshold``.

        Parameters
        ----------
        quality_threshold:
            Minimum quality to retain.  Defaults to ``self._min_quality``.

        Returns
        -------
        int
            Number of entries removed.
        """
        threshold = quality_threshold if quality_threshold is not None else self._min_quality
        before = len(self._entries)
        self._entries = [e for e in self._entries if e.quality_score >= threshold]
        removed = before - len(self._entries)
        if removed and self._persist_path:
            self._save_quietly()
        return removed

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def size(self) -> int:
        """Return the number of entries currently in the bank."""
        return len(self._entries)

    def stats(self) -> Dict[str, Any]:
        """Return summary statistics for the bank.

        Returns
        -------
        Dict
            Keys: ``count``, ``mean_quality``, ``min_quality``, ``max_quality``,
            ``min_quality_threshold``, ``max_capacity``.
        """
        if not self._entries:
            return {
                "count": 0,
                "mean_quality": 0.0,
                "min_quality": 0.0,
                "max_quality": 0.0,
                "min_quality_threshold": self._min_quality,
                "max_capacity": self._max_entries,
            }
        scores = [e.quality_score for e in self._entries]
        return {
            "count": len(scores),
            "mean_quality": sum(scores) / len(scores),
            "min_quality": min(scores),
            "max_quality": max(scores),
            "min_quality_threshold": self._min_quality,
            "max_capacity": self._max_entries,
        }

    def get_by_id(self, entry_id: str) -> Optional[ReasoningEntry]:
        """Return the entry with the given ``entry_id``, or None."""
        for entry in self._entries:
            if entry.entry_id == entry_id:
                return entry
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist all entries to a JSON file at ``path``.

        Parameters
        ----------
        path:
            File path (parent directory is created if needed).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "version": 1,
            "min_quality": self._min_quality,
            "max_entries": self._max_entries,
            "next_id": self._next_id,
            "entries": [self._entry_to_dict(e) for e in self._entries],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def load(self, path: str) -> int:
        """Load entries from a JSON file at ``path``.

        Merges with any existing in-memory entries (deduplicating by
        ``entry_id``).  Returns the number of entries loaded.

        Parameters
        ----------
        path:
            File path to read.

        Returns
        -------
        int
            Number of new entries loaded.
        """
        if not os.path.exists(path):
            return 0

        with open(path, "r", encoding="utf-8") as fh:
            raw_text = fh.read().strip()

        if not raw_text:
            return 0

        try:
            data = json.loads(raw_text)
        except (json.JSONDecodeError, ValueError):
            return 0

        existing_ids = {e.entry_id for e in self._entries}
        loaded = 0

        for raw in data.get("entries", []):
            eid = raw.get("entry_id", "")
            if eid in existing_ids:
                continue
            try:
                entry = self._entry_from_dict(raw)
                self._entries.append(entry)
                existing_ids.add(eid)
                loaded += 1
            except (KeyError, TypeError, ValueError):
                continue

        # Sync next_id to avoid collisions
        file_next_id = data.get("next_id", 1)
        max_numeric = 0
        for e in self._entries:
            try:
                max_numeric = max(max_numeric, int(e.entry_id))
            except ValueError:
                pass
        self._next_id = max(self._next_id, file_next_id, max_numeric + 1)

        return loaded

    def _save_quietly(self) -> None:
        """Save to ``_persist_path`` without raising on failure."""
        try:
            self.save(self._persist_path)  # type: ignore[arg-type]
        except Exception:
            pass

    @staticmethod
    def _entry_to_dict(entry: ReasoningEntry) -> Dict[str, Any]:
        d = asdict(entry)
        return d

    @staticmethod
    def _entry_from_dict(raw: Dict[str, Any]) -> ReasoningEntry:
        return ReasoningEntry(
            entry_id=str(raw["entry_id"]),
            task=str(raw["task"]),
            steps=[str(s) for s in raw.get("steps", [])],
            outcome=str(raw.get("outcome", "")),
            quality_score=float(raw["quality_score"]),
            metadata=dict(raw.get("metadata", {})),
            created_at=float(raw.get("created_at", 0.0)),
            embedding=raw.get("embedding"),
        )

    # ------------------------------------------------------------------
    # Context manager (for scoped use with automatic save)
    # ------------------------------------------------------------------

    def __enter__(self) -> "ReasoningBank":
        return self

    def __exit__(self, *args: Any) -> None:
        if self._persist_path:
            self._save_quietly()

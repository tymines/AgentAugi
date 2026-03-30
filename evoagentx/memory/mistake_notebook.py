"""
Mistake Notebook — lightweight persistent learning from failure patterns.

Agents record failure incidents as structured :class:`MistakeEntry` objects.
Before attempting a task the agent consults the notebook to surface relevant
past mistakes and their fixes, avoiding the same failure twice.

Key design properties
---------------------
- **Cross-agent aggregation**: a single notebook is typically shared so one
  agent's mistake immediately helps all others.
- **Lightweight**: no vector index or LLM required — similarity is computed
  via keyword overlap so the notebook works in any environment.
- **Automatic stale-entry cleanup**: entries older than a configurable number
  of days can be purged in batch.
- **Resolution tracking**: once a failure pattern is resolved it can be marked
  accordingly and excluded from future ``consult()`` calls.

Typical usage::

    nb = MistakeNotebook(persistence_path="/tmp/mistakes.json")
    nb.load()

    # Record a new mistake.
    entry = MistakeEntry(
        attempted="call tool X with parameter Y",
        went_wrong="tool X requires parameter Z, not Y",
        fix="always pass Z when calling tool X",
        category=MistakeCategory.TOOL_MISUSE,
    )
    nb.record(entry)

    # Consult before a task.
    hints = nb.format_for_prompt("call tool X")
    print(hints)  # injected into the agent's system prompt
"""

import os
import re
import json
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field

from ..core.module import BaseModule
from ..core.module_utils import generate_id, get_timestamp
from ..core.logging import logger


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------


class MistakeCategory(str, Enum):
    """Coarse category for classifying mistakes."""

    REASONING_ERROR = "reasoning_error"
    TOOL_MISUSE = "tool_misuse"
    CONTEXT_LOSS = "context_loss"
    FORMAT_ERROR = "format_error"
    HALLUCINATION = "hallucination"
    PLANNING_ERROR = "planning_error"
    OTHER = "other"


class MistakeEntry(BaseModule):
    """A single recorded failure incident.

    Attributes:
        entry_id: Unique identifier for this entry.
        attempted: What the agent tried to do.
        went_wrong: A description of what failed and why.
        fix: The corrective action or insight that resolved the issue.
        category: Coarse failure category for fast filtering.
        task_type: Optional task category the mistake occurred in.
        agent_name: Name of the agent that recorded this mistake.
        timestamp: ISO-formatted creation time.
        resolved: Whether this failure pattern has been permanently fixed.
        tags: Free-form tags for additional filtering (e.g. tool name, domain).
    """

    entry_id: str = Field(default_factory=generate_id)
    attempted: str = Field(..., description="What was attempted")
    went_wrong: str = Field(..., description="What went wrong and why")
    fix: str = Field(..., description="The corrective action or key insight")
    category: MistakeCategory = Field(default=MistakeCategory.OTHER)
    task_type: Optional[str] = Field(default=None, description="Task category")
    agent_name: Optional[str] = Field(default=None, description="Agent that recorded this")
    timestamp: str = Field(default_factory=get_timestamp)
    resolved: bool = Field(
        default=False, description="True once this pattern is permanently fixed"
    )
    tags: List[str] = Field(default_factory=list, description="Free-form tags")

    def combined_text(self) -> str:
        """Return all searchable text fields concatenated.

        Returns:
            String combining attempted, went_wrong, fix, and tags.
        """
        tag_str = " ".join(self.tags)
        return f"{self.attempted} {self.went_wrong} {self.fix} {tag_str}"

    def keywords(self) -> List[str]:
        """Extract alphabetic tokens (≥3 chars) from combined text.

        Returns:
            Deduplicated list of lowercase keyword strings.
        """
        return list(set(re.findall(r"\b[a-zA-Z]{3,}\b", self.combined_text().lower())))


# ---------------------------------------------------------------------------
# MistakeNotebook
# ---------------------------------------------------------------------------


class MistakeNotebook(BaseModule):
    """Persistent collection of failure entries with keyword-based retrieval.

    Entries are stored in insertion order.  Similarity scoring during
    :meth:`consult` uses Jaccard overlap between query tokens and entry
    keyword sets.

    Attributes:
        entries: All stored :class:`MistakeEntry` objects.
        max_entries: Maximum number of entries before oldest are pruned.
        persistence_path: Path to the JSON file for cross-session persistence.
            ``None`` disables disk I/O.
    """

    entries: List[MistakeEntry] = Field(default_factory=list)
    max_entries: int = Field(
        default=1000, ge=1, description="Maximum entries before pruning oldest"
    )
    persistence_path: Optional[str] = Field(
        default=None, description="Path to JSON file for persistence"
    )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def record(
        self, entry: MistakeEntry, auto_save: bool = True
    ) -> str:
        """Add a new mistake entry to the notebook.

        Silently deduplicates by entry_id (re-recording the same entry ID is
        a no-op).  Prunes oldest unresolved entries when capacity is exceeded.

        Args:
            entry: The :class:`MistakeEntry` to record.
            auto_save: If ``True`` and :attr:`persistence_path` is set, save
                to disk immediately after recording.

        Returns:
            The ``entry_id`` of the stored entry.
        """
        if any(e.entry_id == entry.entry_id for e in self.entries):
            logger.debug("MistakeNotebook: entry %s already recorded; skipping.", entry.entry_id)
            return entry.entry_id

        self.entries.append(entry)

        if len(self.entries) > self.max_entries:
            # Prune oldest unresolved entry to make room.
            for i, e in enumerate(self.entries):
                if not e.resolved:
                    removed = self.entries.pop(i)
                    logger.debug(
                        "MistakeNotebook pruned entry %s (oldest unresolved) to stay within max_entries=%d",
                        removed.entry_id,
                        self.max_entries,
                    )
                    break
            else:
                # All entries are resolved; prune the absolute oldest.
                self.entries.pop(0)

        if auto_save:
            self.save()

        return entry.entry_id

    def resolve(self, entry_id: str, auto_save: bool = True) -> bool:
        """Mark an entry as resolved so it is excluded from future consultations.

        Args:
            entry_id: The entry to resolve.
            auto_save: Persist immediately if ``True``.

        Returns:
            ``True`` if the entry was found and marked resolved, else ``False``.
        """
        for entry in self.entries:
            if entry.entry_id == entry_id:
                entry.resolved = True
                if auto_save:
                    self.save()
                return True
        logger.warning("MistakeNotebook.resolve: entry_id %s not found.", entry_id)
        return False

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def consult(
        self,
        task: str,
        top_k: int = 5,
        category_filter: Optional[MistakeCategory] = None,
        task_type_filter: Optional[str] = None,
        include_resolved: bool = False,
    ) -> List[MistakeEntry]:
        """Retrieve the most relevant mistake entries for a given task.

        Only unresolved entries are returned by default (set
        ``include_resolved=True`` for historical analysis).

        Similarity is Jaccard overlap of length-≥3 alphabetic tokens.

        Args:
            task: Free-text description of the task about to be attempted.
            top_k: Maximum number of entries to return.
            category_filter: Restrict results to this :class:`MistakeCategory`.
            task_type_filter: Restrict results to this task type string.
            include_resolved: If ``False`` (default), skip resolved entries.

        Returns:
            List of :class:`MistakeEntry` objects, most relevant first.
        """
        candidates = [
            e for e in self.entries
            if (include_resolved or not e.resolved)
            and (category_filter is None or e.category == category_filter)
            and (task_type_filter is None or e.task_type == task_type_filter)
        ]

        if not candidates:
            return []

        query_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", task.lower()))
        if not query_tokens:
            return candidates[-top_k:]

        scored: List[tuple] = []
        for entry in candidates:
            entry_tokens = set(entry.keywords())
            union = query_tokens | entry_tokens
            intersection = query_tokens & entry_tokens
            score = len(intersection) / len(union) if union else 0.0
            scored.append((score, entry))

        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:top_k] if _ > 0]

    def format_for_prompt(
        self,
        task: str,
        top_k: int = 3,
        category_filter: Optional[MistakeCategory] = None,
        task_type_filter: Optional[str] = None,
    ) -> str:
        """Build a formatted string of relevant mistakes for system prompt injection.

        Returns an empty string when no relevant entries are found, so
        callers can safely check truthiness before injecting.

        Args:
            task: Free-text description of the upcoming task.
            top_k: Maximum number of entries to include.
            category_filter: Restrict to this category.
            task_type_filter: Restrict to this task type.

        Returns:
            A formatted multi-line string, or ``""`` if no relevant entries.
        """
        entries = self.consult(
            task,
            top_k=top_k,
            category_filter=category_filter,
            task_type_filter=task_type_filter,
        )
        if not entries:
            return ""

        lines = ["Avoid repeating these past mistakes:"]
        for i, entry in enumerate(entries, 1):
            cat = entry.category.value.replace("_", " ")
            lines.append(
                f"{i}. [{cat}] Attempted: {entry.attempted[:80]}\n"
                f"   Went wrong: {entry.went_wrong[:120]}\n"
                f"   Fix: {entry.fix[:120]}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup_stale(
        self,
        max_age_days: int = 30,
        resolved_only: bool = False,
        auto_save: bool = True,
    ) -> int:
        """Remove entries older than ``max_age_days``.

        Args:
            max_age_days: Entries with ``timestamp`` older than this are removed.
            resolved_only: If ``True``, only remove *resolved* stale entries.
            auto_save: Persist after cleanup if ``True``.

        Returns:
            Number of entries removed.
        """
        cutoff = datetime.now() - timedelta(days=max_age_days)
        before = len(self.entries)
        remaining: List[MistakeEntry] = []
        for entry in self.entries:
            try:
                ts = datetime.strptime(entry.timestamp, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                remaining.append(entry)
                continue
            is_stale = ts < cutoff
            if is_stale and (not resolved_only or entry.resolved):
                continue  # drop
            remaining.append(entry)

        n_removed = before - len(remaining)
        self.entries = remaining

        if n_removed > 0:
            logger.info(
                "MistakeNotebook cleaned up %d stale entries (max_age_days=%d).",
                n_removed,
                max_age_days,
            )
            if auto_save:
                self.save()

        return n_removed

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return entry counts by category and resolution status.

        Returns:
            Dictionary with ``total``, ``resolved``, ``unresolved``, and
            per-category counts.
        """
        by_cat: Dict[str, int] = {c.value: 0 for c in MistakeCategory}
        resolved_count = 0
        for entry in self.entries:
            by_cat[entry.category.value] += 1
            if entry.resolved:
                resolved_count += 1

        return {
            "total": len(self.entries),
            "resolved": resolved_count,
            "unresolved": len(self.entries) - resolved_count,
            "by_category": by_cat,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write all entries to :attr:`persistence_path` as JSON.

        Creates parent directories as needed.  No-op if
        :attr:`persistence_path` is ``None``.
        """
        if not self.persistence_path:
            return
        try:
            parent = os.path.dirname(self.persistence_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            records = [e.model_dump() for e in self.entries]
            with open(self.persistence_path, "w", encoding="utf-8") as fh:
                json.dump(records, fh, indent=2, default=str)
            logger.debug(
                "MistakeNotebook saved %d entries to %s",
                len(self.entries),
                self.persistence_path,
            )
        except Exception as exc:
            logger.error("Failed to save MistakeNotebook: %s", exc)

    def load(self) -> None:
        """Load entries from :attr:`persistence_path`, skipping duplicates.

        No-op if path is ``None`` or the file does not exist.  Malformed
        records are logged and skipped.
        """
        if not self.persistence_path or not os.path.exists(self.persistence_path):
            return
        try:
            with open(self.persistence_path, "r", encoding="utf-8") as fh:
                records: List[Dict[str, Any]] = json.load(fh)
            existing_ids = {e.entry_id for e in self.entries}
            loaded = 0
            for r in records:
                try:
                    entry = MistakeEntry(**r)
                    if entry.entry_id not in existing_ids:
                        self.entries.append(entry)
                        existing_ids.add(entry.entry_id)
                        loaded += 1
                except Exception as exc:
                    logger.warning("Skipping malformed mistake entry: %s", exc)
            logger.debug(
                "MistakeNotebook loaded %d new entries from %s",
                loaded,
                self.persistence_path,
            )
        except Exception as exc:
            logger.error("Failed to load MistakeNotebook: %s", exc)

"""
Paged Memory System for AgentAugi.

Implements a tiered memory model inspired by the MemGPT/Letta "LLM as OS" paradigm,
where memory is managed across three tiers to prevent unbounded context growth:

- Hot  (in-context): Bounded deque of the most recent, relevant messages.
  Always available without I/O. Fits within an LLM's context window.
- Warm (summarized): Compact summaries of evicted message blocks.
  Kept in memory but not injected into context unless explicitly recalled.
- Cold (persisted): Full message history written to a JSON file on disk.
  Survives process restarts; searchable via keyword matching.

Automatic page-in/out: when the hot tier exceeds its capacity, the oldest
``eviction_block_size`` messages are evicted — unconditionally written to cold
storage, and optionally summarised into a warm page if a summarizer callable
has been registered with :meth:`PagedMemory.set_summarizer`.

Cross-session persistence: the cold JSON file can be reloaded at startup so
agents remember context from previous runs.

This module addresses the unbounded memory growth FIXME documented in
``evoagentx/rag/rag.py`` by keeping the active message list bounded.
"""

import os
import re
import json
from enum import Enum
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import Field

from ..core.module import BaseModule
from ..core.module_utils import generate_id, get_timestamp
from ..core.message import Message
from ..core.logging import logger
from .memory import BaseMemory


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------


class PageTier(str, Enum):
    """Tier classification for a memory page."""

    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


class WarmPage(BaseModule):
    """A compressed summary covering a block of evicted hot messages.

    When the hot tier overflows, the oldest ``eviction_block_size`` messages
    are compressed into a WarmPage and removed from the hot buffer. The
    original messages are still written to cold storage; the WarmPage summary
    exists so the agent can quickly understand what was evicted without
    reloading the full cold file.

    Attributes:
        page_id: Unique identifier for this warm page.
        summary: Human-readable text summarising the evicted messages.
        message_ids: Ordered list of message_id values for the covered messages.
        created_at: ISO timestamp when this page was created.
        last_accessed: ISO timestamp of the most recent recall.
        access_count: How many times this page has been recalled.
        keywords: Top keywords extracted from the evicted messages, used for
            relevance scoring during :meth:`PagedMemory.recall_warm`.
    """

    page_id: str = Field(default_factory=generate_id)
    summary: str = Field(..., description="Summary of the messages in this page")
    message_ids: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=get_timestamp)
    last_accessed: str = Field(default_factory=get_timestamp)
    access_count: int = Field(default=0)
    keywords: List[str] = Field(
        default_factory=list,
        description="Key terms extracted for relevance scoring",
    )

    def touch(self) -> None:
        """Update last_accessed timestamp and increment access_count."""
        self.last_accessed = get_timestamp()
        self.access_count += 1


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class PagedMemory(BaseMemory):
    """Tiered memory with automatic hot→warm→cold eviction.

    The :attr:`messages` list (inherited from :class:`BaseMemory`) is the full
    append-only history. The *hot buffer* is a bounded view over the most
    recent messages; it is rebuilt from ``messages`` during
    :meth:`init_module`.

    Tier summary:

    +---------+----------------------------------+-----------------------------+
    | Tier    | Storage                          | Capacity                    |
    +=========+==================================+=============================+
    | Hot     | In-memory deque                  | ``hot_capacity`` messages   |
    +---------+----------------------------------+-----------------------------+
    | Warm    | In-memory list of WarmPage dicts | ``warm_capacity`` summaries |
    +---------+----------------------------------+-----------------------------+
    | Cold    | JSON file on disk                | Unlimited                   |
    +---------+----------------------------------+-----------------------------+

    Typical usage::

        memory = PagedMemory(hot_capacity=30, cold_storage_path="/tmp/agent.json")
        memory.add_message(msg)
        hot = memory.get(n=10)
        warm_hits = memory.recall_warm("user authentication")
        cold_hits = memory.recall_cold("database error")

    Attributes:
        hot_capacity: Maximum number of messages in the hot tier.
        warm_capacity: Maximum number of warm pages to retain in memory.
        eviction_block_size: Number of messages moved out of hot per overflow.
        cold_storage_path: Path to the JSON file used for cold storage.
            If ``None``, cold persistence is disabled.
        warm_pages: In-memory list of WarmPage summaries.
    """

    hot_capacity: int = Field(
        default=50, ge=1, description="Max messages in hot tier"
    )
    warm_capacity: int = Field(
        default=20, ge=1, description="Max warm page summaries to retain in memory"
    )
    eviction_block_size: int = Field(
        default=10, ge=1, description="Number of messages moved out of hot per overflow"
    )
    cold_storage_path: Optional[str] = Field(
        default=None, description="Path to cold storage JSON file; None disables persistence"
    )
    warm_pages: List[WarmPage] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_module(self) -> None:
        """Rebuild the hot buffer from stored messages and initialise indices.

        ``_hot_buffer`` and ``_summarizer`` are runtime-only instance
        attributes (not Pydantic fields), set here following the same
        pattern used by BaseMemory for ``_by_action`` / ``_by_wf_goal``.
        """
        super().init_module()
        # Rebuild hot buffer from persisted message history.
        self._hot_buffer: deque = deque(self.messages[-self.hot_capacity:])
        self._summarizer: Optional[Callable[[List[Message]], str]] = None

    # ------------------------------------------------------------------
    # Summarizer registration
    # ------------------------------------------------------------------

    def set_summarizer(self, fn: Callable[[List[Message]], str]) -> None:
        """Register a callable that summarises a list of Messages into a string.

        When set, the summarizer is called during eviction to produce warm page
        summaries. If not set, a compact keyword-based digest is used instead.

        Args:
            fn: Callable with signature ``(messages: List[Message]) -> str``.
        """
        self._summarizer = fn

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def add_message(self, message: Message) -> None:
        """Add a message to the hot tier, evicting if capacity is exceeded.

        The message is appended to :attr:`messages` (full history) and to the
        hot buffer. When ``len(hot_buffer) > hot_capacity``, the oldest
        ``eviction_block_size`` messages are evicted to warm and cold tiers.

        Args:
            message: The :class:`~evoagentx.core.message.Message` to store.
        """
        if not message:
            return
        if message in self.messages:
            return

        # Append to full history.
        self.messages.append(message)

        # Update base-class indices.
        if self._by_action is not None and message.action:
            self._by_action[message.action].append(message)
        if self._by_wf_goal is not None and message.wf_goal:
            self._by_wf_goal[message.wf_goal].append(message)

        # Append to hot buffer.
        if self._hot_buffer is None:
            self._hot_buffer = deque()
        self._hot_buffer.append(message)

        # Evict if over capacity.
        if len(self._hot_buffer) > self.hot_capacity:
            self._evict()

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def _evict(self) -> None:
        """Evict the oldest ``eviction_block_size`` messages from the hot tier.

        Each eviction:
        1. Pops ``eviction_block_size`` messages from the left of the hot buffer.
        2. Appends them to the cold storage JSON file (if path is configured).
        3. Creates a WarmPage summary (using summarizer if available).
        4. Prunes warm pages beyond ``warm_capacity`` by dropping the
           least-accessed oldest entry.

        Note: evicted messages remain in :attr:`messages` (full history).
        """
        n_evict = min(self.eviction_block_size, len(self._hot_buffer))
        evicted: List[Message] = [self._hot_buffer.popleft() for _ in range(n_evict)]

        # Write to cold storage unconditionally.
        if self.cold_storage_path:
            self._write_cold(evicted)

        # Create warm page summary.
        summary = self._make_summary(evicted)
        keywords = self._extract_keywords(evicted)
        warm_page = WarmPage(
            summary=summary,
            message_ids=[m.message_id for m in evicted if m.message_id],
            keywords=keywords,
        )
        self.warm_pages.append(warm_page)

        # Enforce warm capacity: drop least-accessed oldest page.
        if len(self.warm_pages) > self.warm_capacity:
            self.warm_pages.sort(key=lambda p: (p.access_count, p.last_accessed))
            self.warm_pages.pop(0)

        logger.debug(
            "PagedMemory evicted %d messages. hot=%d warm=%d",
            n_evict,
            len(self._hot_buffer),
            len(self.warm_pages),
        )

    def _make_summary(self, messages: List[Message]) -> str:
        """Produce a text summary for a list of messages.

        Calls the registered summarizer if available, otherwise falls back to
        a compact digest showing the first few content snippets.

        Args:
            messages: The messages to summarise.

        Returns:
            A string summary.
        """
        if self._summarizer is not None:
            try:
                return self._summarizer(messages)
            except Exception as exc:
                logger.warning("Summarizer raised an exception; using digest fallback: %s", exc)

        # Fallback: compact first-N digest.
        parts: List[str] = []
        for msg in messages[:3]:
            snippet = str(msg.content)[:120].replace("\n", " ")
            role = msg.agent or "unknown"
            parts.append(f"[{role}] {snippet}")
        if len(messages) > 3:
            parts.append(f"... +{len(messages) - 3} more")
        return " | ".join(parts) if parts else "(empty)"

    def _extract_keywords(
        self, messages: List[Message], max_keywords: int = 20
    ) -> List[str]:
        """Extract the most frequent long words from messages for relevance scoring.

        Args:
            messages: Source messages.
            max_keywords: Maximum number of keywords to return.

        Returns:
            List of keyword strings, most frequent first.
        """
        combined = " ".join(str(m.content) for m in messages)
        words = re.findall(r"\b[a-zA-Z]{4,}\b", combined.lower())
        freq: Dict[str, int] = {}
        for word in words:
            freq[word] = freq.get(word, 0) + 1
        sorted_words = sorted(freq.items(), key=lambda kv: -kv[1])
        return [w for w, _ in sorted_words[:max_keywords]]

    # ------------------------------------------------------------------
    # Cold storage I/O
    # ------------------------------------------------------------------

    def _write_cold(self, messages: List[Message]) -> None:
        """Append messages to the cold storage JSON file, skipping duplicates.

        Args:
            messages: Messages to persist.
        """
        try:
            path = self.cold_storage_path
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)

            existing: List[Dict[str, Any]] = []
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    existing = json.load(fh)

            existing_ids = {r.get("message_id") for r in existing if r.get("message_id")}
            new_records = [
                msg.model_dump()
                for msg in messages
                if not msg.message_id or msg.message_id not in existing_ids
            ]
            existing.extend(new_records)

            with open(path, "w", encoding="utf-8") as fh:
                json.dump(existing, fh, indent=2, default=str)
        except Exception as exc:
            logger.error("Failed to write cold storage at %s: %s", self.cold_storage_path, exc)

    def load_cold(self) -> List[Message]:
        """Load all messages from the cold storage JSON file.

        Returns an empty list if the file does not exist or cold storage is
        disabled. Deserialization errors are logged and the affected record
        is skipped.

        Returns:
            Ordered list of :class:`Message` objects from cold storage.
        """
        if not self.cold_storage_path or not os.path.exists(self.cold_storage_path):
            return []
        try:
            with open(self.cold_storage_path, "r", encoding="utf-8") as fh:
                records: List[Dict[str, Any]] = json.load(fh)
            messages: List[Message] = []
            for r in records:
                try:
                    messages.append(Message(**r))
                except Exception as exc:
                    logger.warning("Skipping malformed cold storage record: %s", exc)
            return messages
        except Exception as exc:
            logger.error("Failed to load cold storage: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get(self, n: Optional[int] = None, **kwargs) -> List[Message]:
        """Return the most recent messages from the hot tier.

        Args:
            n: Maximum number of messages to return. ``None`` returns all hot
               messages.

        Returns:
            List of :class:`Message` objects, oldest first.
        """
        buf = list(self._hot_buffer) if self._hot_buffer is not None else []
        if n is None:
            return buf
        assert n >= 0, "n must be a non-negative integer"
        return buf[-n:]

    def recall_warm(self, query: str, top_k: int = 3) -> List[WarmPage]:
        """Retrieve warm pages most relevant to a query using keyword overlap.

        Relevance is defined as the Jaccard coefficient between the query's
        long words and each warm page's keyword set.  Recalled pages are
        touched (access count incremented).

        Args:
            query: The query string.
            top_k: Maximum number of warm pages to return.

        Returns:
            List of :class:`WarmPage` objects, most relevant first.
        """
        if not self.warm_pages:
            return []

        query_words = set(re.findall(r"\b[a-zA-Z]{4,}\b", query.lower()))
        if not query_words:
            return self.warm_pages[:top_k]

        scored: List[Tuple[float, WarmPage]] = []
        for page in self.warm_pages:
            kw_set = set(page.keywords)
            union = query_words | kw_set
            intersection = query_words & kw_set
            # Jaccard similarity; avoid division by zero.
            score = len(intersection) / len(union) if union else 0.0
            scored.append((score, page))

        scored.sort(key=lambda x: -x[0])
        results = [page for score, page in scored[:top_k] if score > 0]
        for page in results:
            page.touch()
        return results

    def recall_cold(self, query: str, top_k: int = 5) -> List[Message]:
        """Search cold storage for messages relevant to a query.

        Uses simple word-overlap scoring on message content strings.

        Args:
            query: The search query.
            top_k: Maximum number of messages to return.

        Returns:
            List of matching :class:`Message` objects, most relevant first.
        """
        cold_messages = self.load_cold()
        if not cold_messages:
            return []

        query_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", query.lower()))
        if not query_words:
            return cold_messages[:top_k]

        scored: List[Tuple[int, Message]] = []
        for msg in cold_messages:
            content_words = set(
                re.findall(r"\b[a-zA-Z]{3,}\b", str(msg.content).lower())
            )
            overlap = len(query_words & content_words)
            if overlap > 0:
                scored.append((overlap, msg))

        scored.sort(key=lambda x: -x[0])
        return [msg for _, msg in scored[:top_k]]

    def page_in(self, messages: List[Message]) -> None:
        """Promote messages from warm/cold back into the hot tier.

        Messages already present in the hot buffer are skipped.  If adding the
        recalled messages causes the hot tier to overflow, eviction runs again.

        Args:
            messages: Messages to promote to the hot tier.
        """
        if self._hot_buffer is None:
            self._hot_buffer = deque()

        hot_ids = {m.message_id for m in self._hot_buffer if m.message_id}
        for msg in messages:
            if msg.message_id not in hot_ids:
                self._hot_buffer.append(msg)
                if msg.message_id:
                    hot_ids.add(msg.message_id)

        while len(self._hot_buffer) > self.hot_capacity:
            self._evict()

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear all three memory tiers (does NOT delete the cold storage file)."""
        super().clear()
        if self._hot_buffer is not None:
            self._hot_buffer.clear()
        self.warm_pages.clear()

    # ------------------------------------------------------------------
    # Properties & introspection
    # ------------------------------------------------------------------

    @property
    def hot_size(self) -> int:
        """Number of messages currently in the hot tier."""
        return len(self._hot_buffer) if self._hot_buffer is not None else 0

    @property
    def warm_size(self) -> int:
        """Number of warm page summaries currently in memory."""
        return len(self.warm_pages)

    def stats(self) -> Dict[str, Any]:
        """Return a snapshot of memory tier usage.

        Returns:
            Dictionary with keys: ``hot_size``, ``warm_size``,
            ``total_messages``, ``hot_capacity``, ``warm_capacity``,
            ``cold_storage_path``.
        """
        return {
            "hot_size": self.hot_size,
            "warm_size": self.warm_size,
            "total_messages": self.size,
            "hot_capacity": self.hot_capacity,
            "warm_capacity": self.warm_capacity,
            "cold_storage_path": self.cold_storage_path,
        }

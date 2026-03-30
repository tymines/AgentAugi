"""
Reflexion-based Episodic Memory for AgentAugi.

Implements an episodic memory system inspired by the Reflexion concept:
after each task attempt, an agent generates a verbal self-reflection on
what went well or wrong.  These reflections are stored as episodes and
retrieved before similar tasks so the agent can avoid repeating mistakes
and reinforce successful strategies.

Design decisions
----------------
- Reflections are stored as :class:`Episode` objects, each capturing the
  task description, the outcome, a human-readable reflection, and metadata.
- Retrieval is keyword-based by default (no external embeddings required),
  making the module self-contained.
- :class:`ReflexionMemory` handles persistence via a JSON file so episodes
  survive process restarts.
- :class:`ReflexionAgent` is a lightweight wrapper that adds reflection loops
  to *any* existing :class:`~evoagentx.agents.agent.Agent` without requiring
  subclassing. The wrapper intercepts ``execute()``/``async_execute()`` calls,
  injects prior reflections into the system prompt, and stores a new episode
  after each attempt.

Complements TextGrad (which optimises prompts via gradient feedback) by
adding behaviour-level learning from trial-and-error experience.
"""

import os
import re
import json
import asyncio
from enum import Enum
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from pydantic import Field

from ..core.module import BaseModule
from ..core.module_utils import generate_id, get_timestamp
from ..core.logging import logger


# ---------------------------------------------------------------------------
# Supporting types
# ---------------------------------------------------------------------------


class TaskOutcome(str, Enum):
    """Outcome classification for a task episode."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class Episode(BaseModule):
    """A single task episode with its verbal self-reflection.

    Attributes:
        episode_id: Unique identifier for this episode.
        task_description: Free-text description of the task that was attempted.
        task_type: Optional coarse category (e.g. "code_generation", "qa").
        outcome: Whether the attempt succeeded, failed, or partially succeeded.
        attempt_summary: Brief description of what the agent did.
        reflection: The verbal self-reflection generated after the attempt.
            This is the most important field — it should explain *what went
            wrong* and *what to do differently next time* (on failure) or
            *why this approach worked* (on success).
        timestamp: ISO-formatted creation time.
        metadata: Arbitrary key-value pairs for additional context.
    """

    episode_id: str = Field(default_factory=generate_id)
    task_description: str = Field(..., description="What was attempted")
    task_type: Optional[str] = Field(
        default=None, description="Coarse task category for faster filtering"
    )
    outcome: TaskOutcome = Field(default=TaskOutcome.UNKNOWN)
    attempt_summary: str = Field(
        default="", description="Short description of the approach taken"
    )
    reflection: str = Field(
        ..., description="Verbal self-reflection on the outcome — the learning signal"
    )
    timestamp: str = Field(default_factory=get_timestamp)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def keywords(self) -> List[str]:
        """Extract keyword tokens from the task description and reflection.

        Returns:
            List of lowercase alphabetic tokens of length ≥ 3.
        """
        combined = f"{self.task_description} {self.reflection}"
        return list(set(re.findall(r"\b[a-zA-Z]{3,}\b", combined.lower())))


# ---------------------------------------------------------------------------
# ReflexionMemory
# ---------------------------------------------------------------------------


class ReflexionMemory(BaseModule):
    """Persistent store for task episodes, with similarity-based retrieval.

    Episodes are scored against a query using keyword overlap (Jaccard
    similarity).  The store can optionally filter by :class:`TaskOutcome` and
    task type to narrow recall.

    Attributes:
        episodes: All stored :class:`Episode` objects (ordered by insertion).
        max_episodes: Maximum number of episodes to retain; oldest are pruned
            once this limit is reached.
        persistence_path: Optional path to a JSON file for cross-session
            persistence.  ``None`` disables disk persistence.
    """

    episodes: List[Episode] = Field(default_factory=list)
    max_episodes: int = Field(
        default=500, ge=1, description="Maximum episodes to keep before pruning oldest"
    )
    persistence_path: Optional[str] = Field(
        default=None, description="Path to JSON file for cross-session persistence"
    )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_episode(self, episode: Episode) -> None:
        """Store a new episode, pruning oldest entries if capacity is exceeded.

        Args:
            episode: The :class:`Episode` to store.
        """
        if any(e.episode_id == episode.episode_id for e in self.episodes):
            logger.debug("Episode %s already stored; skipping.", episode.episode_id)
            return

        self.episodes.append(episode)

        if len(self.episodes) > self.max_episodes:
            n_remove = len(self.episodes) - self.max_episodes
            removed = self.episodes[:n_remove]
            self.episodes = self.episodes[n_remove:]
            logger.debug(
                "ReflexionMemory pruned %d oldest episode(s) to stay within max_episodes=%d",
                n_remove,
                self.max_episodes,
            )
            _ = removed  # intentionally discarded

        if self.persistence_path:
            self.save()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def find_similar(
        self,
        task: str,
        outcome_filter: Optional[TaskOutcome] = None,
        task_type_filter: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Episode]:
        """Retrieve episodes most similar to a task description.

        Similarity is measured as the Jaccard coefficient between the query
        tokens and the episode's combined keyword set.

        Args:
            task: Free-text description of the upcoming task.
            outcome_filter: If provided, only return episodes with this outcome.
            task_type_filter: If provided, only return episodes with this task type.
            top_k: Maximum number of episodes to return.

        Returns:
            List of :class:`Episode` objects, most similar first.
        """
        if not self.episodes:
            return []

        candidates = self.episodes
        if outcome_filter is not None:
            candidates = [e for e in candidates if e.outcome == outcome_filter]
        if task_type_filter is not None:
            candidates = [e for e in candidates if e.task_type == task_type_filter]

        if not candidates:
            return []

        query_tokens = set(re.findall(r"\b[a-zA-Z]{3,}\b", task.lower()))
        if not query_tokens:
            return candidates[-top_k:]

        scored: List[tuple] = []
        for episode in candidates:
            ep_tokens = set(episode.keywords())
            union = query_tokens | ep_tokens
            intersection = query_tokens & ep_tokens
            score = len(intersection) / len(union) if union else 0.0
            scored.append((score, episode))

        scored.sort(key=lambda x: -x[0])
        return [ep for _, ep in scored[:top_k] if _ > 0]

    def get_reflections_for_task(
        self,
        task: str,
        outcome_filter: Optional[TaskOutcome] = None,
        task_type_filter: Optional[str] = None,
        top_k: int = 3,
    ) -> str:
        """Build a formatted string of past reflections for prompt injection.

        Retrieves the most similar episodes and formats their reflections as a
        numbered list suitable for prepending to an LLM system prompt.

        Args:
            task: Free-text description of the upcoming task.
            outcome_filter: Restrict retrieval to episodes with this outcome.
            task_type_filter: Restrict retrieval to episodes with this task type.
            top_k: Maximum number of past reflections to include.

        Returns:
            A formatted string, or an empty string if no relevant episodes exist.
        """
        episodes = self.find_similar(
            task,
            outcome_filter=outcome_filter,
            task_type_filter=task_type_filter,
            top_k=top_k,
        )
        if not episodes:
            return ""

        lines = ["Past reflections on similar tasks:"]
        for i, ep in enumerate(episodes, 1):
            outcome_label = ep.outcome.value.upper()
            lines.append(
                f"{i}. [{outcome_label}] Task: {ep.task_description[:80]}\n"
                f"   Reflection: {ep.reflection}"
            )
        return "\n".join(lines)

    def recent(self, n: int = 10) -> List[Episode]:
        """Return the n most recent episodes.

        Args:
            n: Number of episodes to return.

        Returns:
            List of the most recent :class:`Episode` objects, oldest first.
        """
        return self.episodes[-n:]

    def stats(self) -> Dict[str, Any]:
        """Return a summary of episode counts by outcome.

        Returns:
            Dictionary mapping outcome labels to counts, plus ``total``.
        """
        counts: Dict[str, int] = {o.value: 0 for o in TaskOutcome}
        for ep in self.episodes:
            counts[ep.outcome.value] += 1
        return {"total": len(self.episodes), **counts}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Persist all episodes to :attr:`persistence_path` as JSON.

        Creates parent directories as needed. No-op if
        :attr:`persistence_path` is ``None``.
        """
        if not self.persistence_path:
            return
        try:
            parent = os.path.dirname(self.persistence_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            records = [ep.model_dump() for ep in self.episodes]
            with open(self.persistence_path, "w", encoding="utf-8") as fh:
                json.dump(records, fh, indent=2, default=str)
            logger.debug("ReflexionMemory saved %d episodes to %s", len(self.episodes), self.persistence_path)
        except Exception as exc:
            logger.error("Failed to save ReflexionMemory: %s", exc)

    def load(self) -> None:
        """Load episodes from :attr:`persistence_path`, merging with any in-memory data.

        Episodes already present (by episode_id) are not duplicated.
        No-op if the path is ``None`` or the file does not exist.
        """
        if not self.persistence_path or not os.path.exists(self.persistence_path):
            return
        try:
            with open(self.persistence_path, "r", encoding="utf-8") as fh:
                records: List[Dict[str, Any]] = json.load(fh)
            existing_ids = {e.episode_id for e in self.episodes}
            loaded = 0
            for r in records:
                try:
                    ep = Episode(**r)
                    if ep.episode_id not in existing_ids:
                        self.episodes.append(ep)
                        existing_ids.add(ep.episode_id)
                        loaded += 1
                except Exception as exc:
                    logger.warning("Skipping malformed episode record: %s", exc)
            logger.debug("ReflexionMemory loaded %d new episodes from %s", loaded, self.persistence_path)
        except Exception as exc:
            logger.error("Failed to load ReflexionMemory: %s", exc)


# ---------------------------------------------------------------------------
# ReflexionAgent wrapper
# ---------------------------------------------------------------------------


class ReflexionAgent:
    """Decorator that adds Reflexion-style learning loops to any Agent.

    Wraps an existing :class:`~evoagentx.agents.agent.Agent` without modifying
    its class.  Before each execution the wrapper:

    1. Retrieves relevant past reflections from :attr:`memory`.
    2. Prepends them to the agent's system prompt (temporarily).

    After each execution the wrapper:

    3. Calls the registered ``reflect_fn`` to generate a verbal reflection.
    4. Stores a new :class:`Episode` in :attr:`memory`.

    Usage::

        base_agent = MyAgent(...)
        reflexion = ReflexionAgent(
            agent=base_agent,
            memory=ReflexionMemory(persistence_path="/tmp/reflexion.json"),
            reflect_fn=my_llm_reflect,
            top_k_reflections=3,
        )
        result = reflexion.execute(inputs={"task": "..."})

    Attributes:
        agent: The wrapped agent instance.
        memory: The :class:`ReflexionMemory` used to store and retrieve episodes.
        reflect_fn: Callable ``(task, attempt_summary, outcome, result) -> str``
            that produces a verbal reflection string.  If ``None``, a simple
            rule-based reflection is generated automatically.
        top_k_reflections: Number of past reflections to inject before each task.
        task_type: Optional task type label applied to all stored episodes.
    """

    def __init__(
        self,
        agent: Any,
        memory: ReflexionMemory,
        reflect_fn: Optional[Callable] = None,
        top_k_reflections: int = 3,
        task_type: Optional[str] = None,
    ) -> None:
        self.agent = agent
        self.memory = memory
        self.reflect_fn = reflect_fn
        self.top_k_reflections = top_k_reflections
        self.task_type = task_type

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_reflection_prefix(self, task: str) -> str:
        """Retrieve past reflections and format them for system prompt injection."""
        return self.memory.get_reflections_for_task(
            task,
            top_k=self.top_k_reflections,
        )

    def _inject_reflections(self, task: str) -> Optional[str]:
        """Temporarily patch the agent's system prompt with past reflections.

        Returns the original system prompt so it can be restored.

        Args:
            task: Task description used to query the memory.

        Returns:
            The agent's original system_prompt (may be ``None``).
        """
        original = getattr(self.agent, "system_prompt", None)
        prefix = self._build_reflection_prefix(task)
        if prefix:
            patched = f"{prefix}\n\n{original}" if original else prefix
            try:
                self.agent.system_prompt = patched
            except Exception:
                pass  # read-only attribute; skip injection silently
        return original

    def _restore_system_prompt(self, original: Optional[str]) -> None:
        """Restore the agent's system_prompt to its original value."""
        try:
            self.agent.system_prompt = original
        except Exception:
            pass

    def _generate_reflection(
        self,
        task: str,
        attempt_summary: str,
        outcome: TaskOutcome,
        result: Any,
    ) -> str:
        """Generate a verbal reflection, using reflect_fn if available.

        Falls back to a rule-based template when no reflect_fn is registered.

        Args:
            task: Task description.
            attempt_summary: What was tried.
            outcome: Whether it succeeded or failed.
            result: The raw result from the agent (for context).

        Returns:
            A verbal reflection string.
        """
        if self.reflect_fn is not None:
            try:
                return self.reflect_fn(task, attempt_summary, outcome, result)
            except Exception as exc:
                logger.warning("reflect_fn raised an exception; using fallback: %s", exc)

        # Rule-based fallback.
        if outcome == TaskOutcome.SUCCESS:
            return (
                f"The approach for '{task[:60]}' succeeded. "
                "The strategy was effective and should be reused for similar tasks."
            )
        elif outcome == TaskOutcome.FAILURE:
            return (
                f"The attempt on '{task[:60]}' failed. "
                "Review the approach: ensure inputs are correct, tools are used properly, "
                "and all required context is available before retrying."
            )
        else:
            return (
                f"The attempt on '{task[:60]}' produced a partial result. "
                "Consider refining the approach or breaking the task into smaller steps."
            )

    def _store_episode(
        self,
        task: str,
        attempt_summary: str,
        outcome: TaskOutcome,
        reflection: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Episode:
        """Create and store an Episode in memory.

        Args:
            task: Task description.
            attempt_summary: Short description of what was done.
            outcome: Task outcome classification.
            reflection: Verbal self-reflection.
            metadata: Optional extra metadata.

        Returns:
            The stored :class:`Episode`.
        """
        episode = Episode(
            task_description=task,
            task_type=self.task_type,
            outcome=outcome,
            attempt_summary=attempt_summary,
            reflection=reflection,
            metadata=metadata or {},
        )
        self.memory.add_episode(episode)
        return episode

    # ------------------------------------------------------------------
    # Public execute interface
    # ------------------------------------------------------------------

    def execute(
        self,
        task: str = "",
        outcome: TaskOutcome = TaskOutcome.UNKNOWN,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute the wrapped agent with reflexion support (synchronous).

        Args:
            task: Free-text description of the task being attempted.
            outcome: The expected or known outcome (can be updated post-hoc
                via :meth:`update_last_episode`).
            metadata: Optional metadata to attach to the episode.
            **kwargs: Forwarded to the wrapped agent's ``execute()`` method.

        Returns:
            The wrapped agent's return value.
        """
        original_prompt = self._inject_reflections(task)
        result = None
        try:
            result = self.agent.execute(**kwargs)
            if outcome == TaskOutcome.UNKNOWN:
                outcome = TaskOutcome.SUCCESS
        except Exception as exc:
            if outcome == TaskOutcome.UNKNOWN:
                outcome = TaskOutcome.FAILURE
            logger.warning("ReflexionAgent.execute caught exception: %s", exc)
            raise
        finally:
            self._restore_system_prompt(original_prompt)
            attempt_summary = str(result)[:200] if result is not None else "no result"
            reflection = self._generate_reflection(task, attempt_summary, outcome, result)
            self._store_episode(task, attempt_summary, outcome, reflection, metadata)

        return result

    async def async_execute(
        self,
        task: str = "",
        outcome: TaskOutcome = TaskOutcome.UNKNOWN,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Execute the wrapped agent with reflexion support (asynchronous).

        Args:
            task: Free-text description of the task being attempted.
            outcome: The expected or known outcome.
            metadata: Optional metadata.
            **kwargs: Forwarded to the wrapped agent's ``async_execute()`` method.

        Returns:
            The wrapped agent's return value.
        """
        original_prompt = self._inject_reflections(task)
        result = None
        try:
            result = await self.agent.async_execute(**kwargs)
            if outcome == TaskOutcome.UNKNOWN:
                outcome = TaskOutcome.SUCCESS
        except Exception as exc:
            if outcome == TaskOutcome.UNKNOWN:
                outcome = TaskOutcome.FAILURE
            logger.warning("ReflexionAgent.async_execute caught exception: %s", exc)
            raise
        finally:
            self._restore_system_prompt(original_prompt)
            attempt_summary = str(result)[:200] if result is not None else "no result"
            reflection = self._generate_reflection(task, attempt_summary, outcome, result)
            self._store_episode(task, attempt_summary, outcome, reflection, metadata)

        return result

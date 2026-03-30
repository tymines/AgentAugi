"""ADAS Design Archive — Automated Design of Agentic Systems.

Inspired by the ICLR 2025 paper "Automated Design of Agentic Systems" (ADAS),
this module maintains a persistent, searchable archive of *agent designs* that
have been tested — including both successful and failed designs.  Learning from
failures is a key design goal: a design that is known to fail on a task type is
just as valuable as one that succeeds, because it prevents re-testing similar
designs in the future.

Components
----------

``DesignEntry``
    A single archived design.  Stores the agent configuration, a performance
    profile across multiple task types, a human-readable rationale, and whether
    the design was considered successful.

``DesignArchive``
    The central archive.  Supports:

    - ``add(entry)`` — Insert a new design, deduplicating against similar ones.
    - ``search(query, k)`` — Retrieve the k most similar designs to a query
      configuration using a lightweight structural similarity metric.
    - ``top_k(k, task_type)`` — Return the k best-performing designs for a
      specific task type.
    - ``failed_designs(task_type)`` — Return known failures to guide the meta
      agent away from re-testing similar designs.
    - ``meta_agent_search(seed_cfg, llm, n_iterations)`` — Iteratively generate
      new agent designs by combining archive knowledge with LLM creativity.

``MetaAgentSearch``
    Orchestrates the iterative new-design generation loop.  At each iteration
    it selects a few high-performing designs and a few failures from the archive,
    passes them to the LLM with the instruction to produce a novel design that
    avoids known failure modes, evaluates the design, then archives the result.

Integration
-----------
- Depends on: ``BaseLLM`` for design generation.
- Integrates with: ``MAPElitesOptimizer`` (archive cells can be imported as
  design entries for multi-objective Pareto tracking).
- Compatible with: ``AFlowOptimizer`` outputs (workflow configs map naturally
  to design entries).

Usage
-----
    >>> from evoagentx.core.design_archive import DesignArchive, DesignEntry
    >>> archive = DesignArchive()
    >>> entry = DesignEntry(
    ...     config={"agent_type": "ReAct", "max_steps": 5},
    ...     performance={"qa": 0.72, "coding": 0.45},
    ...     rationale="ReAct with short horizon works well for factual QA.",
    ...     success=True,
    ...     task_types=["qa"],
    ... )
    >>> archive.add(entry)
    >>> results = archive.top_k(k=3, task_type="qa")
"""

from __future__ import annotations

import json
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .logging import logger


# ---------------------------------------------------------------------------
# LLM prompt templates
# ---------------------------------------------------------------------------

_META_SYSTEM = (
    "You are an expert AI system designer. "
    "Your task is to propose novel agent system designs by learning from the "
    "successes and failures in the provided archive."
)

_META_USER = """## Archive of successful designs
{successes}

## Archive of known failures (avoid similar designs)
{failures}

## Task type to optimise for
{task_type}

## Design constraints
{constraints}

## Instruction
Propose ONE novel agent system design that:
1. Builds on patterns from successful designs.
2. Explicitly avoids the failure modes observed in failed designs.
3. Is meaningfully different from all archived designs.

Output a JSON object with these keys:
- "config": dict of design parameters (agent_type, tools, prompting_strategy, etc.)
- "rationale": 2-3 sentence explanation of why this design might work
- "expected_strengths": list of 2-3 expected strengths
- "expected_weaknesses": list of 1-2 anticipated weaknesses

Output ONLY valid JSON — no markdown, no preamble."""

_SIMILARITY_SYSTEM = (
    "You are a structured data analyst. "
    "Estimate the structural similarity between two agent configuration dicts."
)

_SIMILARITY_USER = """## Config A
{config_a}

## Config B
{config_b}

On a scale from 0.0 (completely different) to 1.0 (identical), how similar are
these two agent configurations? Consider: component types, parameter values, and
overall design philosophy.

Output ONLY a single float value, e.g. 0.73"""


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PerformanceProfile:
    """Performance metrics for a design across one or more task types.

    Attributes:
        scores: Mapping from task type to evaluation score in [0, 1].
        n_trials: Number of evaluation trials per task type.
        std_devs: Standard deviation of scores (if multiple trials run).
    """

    scores: Dict[str, float] = field(default_factory=dict)
    n_trials: Dict[str, int] = field(default_factory=dict)
    std_devs: Dict[str, float] = field(default_factory=dict)

    def mean_score(self) -> float:
        """Average score across all task types."""
        if not self.scores:
            return 0.0
        return sum(self.scores.values()) / len(self.scores)

    def score_for(self, task_type: str) -> float:
        """Return the score for a specific task type, or 0.0 if not tested."""
        return self.scores.get(task_type, 0.0)

    def update(self, task_type: str, score: float, n: int = 1) -> None:
        """Record or update a score for a task type.

        Uses a running mean update: existing scores are weighted by their
        trial count, making multiple-trial scores more reliable.

        Args:
            task_type: Identifier of the task type.
            score: Evaluation score in [0, 1].
            n: Number of trials that produced this score.
        """
        prev_n = self.n_trials.get(task_type, 0)
        prev_score = self.scores.get(task_type, score)
        total_n = prev_n + n
        self.scores[task_type] = (prev_score * prev_n + score * n) / total_n
        self.n_trials[task_type] = total_n


@dataclass
class DesignEntry:
    """A single archived agent design.

    Attributes:
        config: Agent configuration dict (design parameters).
        performance: Performance profile across task types.
        rationale: Human-readable explanation of the design.
        success: Whether this design is considered successful.
        task_types: Task types this design was evaluated on.
        design_id: Unique identifier assigned at insertion.
        source: How this design was created (``"manual"``, ``"meta_agent"``,
            ``"aflow"``, ``"user"``, etc.).
        tags: Arbitrary metadata tags.
    """

    config: Dict[str, Any]
    performance: PerformanceProfile = field(default_factory=PerformanceProfile)
    rationale: str = ""
    success: bool = True
    task_types: List[str] = field(default_factory=list)
    design_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source: str = "manual"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise this entry to a plain dict for JSON persistence."""
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DesignEntry":
        """Restore a ``DesignEntry`` from a serialised dict."""
        perf_data = data.pop("performance", {})
        entry = cls(**data)
        entry.performance = PerformanceProfile(**perf_data)
        return entry


# ---------------------------------------------------------------------------
# Similarity metric (structural, no LLM required for basic use)
# ---------------------------------------------------------------------------

def _structural_similarity(cfg_a: Dict[str, Any], cfg_b: Dict[str, Any]) -> float:
    """Compute a fast structural similarity score between two config dicts.

    Uses a key–value overlap heuristic:
    - Full match (same key, same value) contributes 1.0 / max_keys.
    - Key-only match contributes 0.3 / max_keys.
    - Missing keys contribute 0.0.

    This is intentionally lightweight — use LLM-assisted similarity only when
    the structural score is ambiguous (0.4–0.6 range).

    Args:
        cfg_a: First configuration dict.
        cfg_b: Second configuration dict.

    Returns:
        Similarity score in [0.0, 1.0].
    """
    all_keys = set(cfg_a) | set(cfg_b)
    if not all_keys:
        return 1.0
    score = 0.0
    for key in all_keys:
        if key in cfg_a and key in cfg_b:
            if cfg_a[key] == cfg_b[key]:
                score += 1.0
            else:
                score += 0.3  # partial credit for same key, different value
    return score / len(all_keys)


# ---------------------------------------------------------------------------
# Design Archive
# ---------------------------------------------------------------------------

class DesignArchive:
    """Persistent archive of tested agent designs with similarity-based search.

    The archive stores both successful and failed designs, enabling the meta
    agent search to learn from the full history of experimentation.

    Args:
        similarity_threshold: Structural similarity above which a new design
            is considered a *duplicate* and rejected on insertion.
        llm: Optional ``BaseLLM`` instance used for LLM-assisted similarity
            scoring and meta-agent design generation.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        llm: Optional[Any] = None,
    ) -> None:
        if not (0.0 < similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be in (0, 1].")
        self.similarity_threshold = similarity_threshold
        self.llm = llm
        self._entries: List[DesignEntry] = []

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add(self, entry: DesignEntry) -> bool:
        """Insert a design into the archive if it is not a duplicate.

        A design is rejected if its structural similarity to any existing entry
        exceeds ``similarity_threshold``.

        Args:
            entry: Design to insert.

        Returns:
            ``True`` if inserted; ``False`` if rejected as a duplicate.
        """
        for existing in self._entries:
            sim = _structural_similarity(entry.config, existing.config)
            if sim >= self.similarity_threshold:
                # Update performance of existing entry rather than duplicating
                for tt, sc in entry.performance.scores.items():
                    n = entry.performance.n_trials.get(tt, 1)
                    existing.performance.update(tt, sc, n)
                # Update success flag if new trial was more successful
                if entry.success and not existing.success:
                    existing.success = True
                    existing.rationale += f" [updated: {entry.rationale}]"
                logger.debug(
                    "DesignArchive: duplicate rejected (sim=%.3f) — updated existing %s",
                    sim,
                    existing.design_id,
                )
                return False

        self._entries.append(entry)
        logger.info(
            "DesignArchive: added design %s (success=%s  source=%s)",
            entry.design_id,
            entry.success,
            entry.source,
        )
        return True

    def search(
        self,
        query_config: Dict[str, Any],
        k: int = 5,
        include_failures: bool = True,
    ) -> List[Tuple[DesignEntry, float]]:
        """Return the k most structurally similar designs to a query config.

        Args:
            query_config: Configuration dict to compare against.
            k: Maximum number of results to return.
            include_failures: Whether to include failed designs in results.

        Returns:
            List of ``(entry, similarity_score)`` tuples, sorted descending.
        """
        pool = self._entries if include_failures else [e for e in self._entries if e.success]
        scored = [
            (entry, _structural_similarity(query_config, entry.config))
            for entry in pool
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

    def top_k(
        self,
        k: int = 5,
        task_type: Optional[str] = None,
    ) -> List[DesignEntry]:
        """Return the k best-performing successful designs.

        Args:
            k: Number of designs to return.
            task_type: If provided, rank by performance on this specific task
                type.  Otherwise rank by mean performance across all tasks.

        Returns:
            List of ``DesignEntry`` objects sorted by performance descending.
        """
        successful = [e for e in self._entries if e.success]
        if task_type:
            successful.sort(
                key=lambda e: e.performance.score_for(task_type),
                reverse=True,
            )
        else:
            successful.sort(
                key=lambda e: e.performance.mean_score(),
                reverse=True,
            )
        return successful[:k]

    def failed_designs(
        self,
        task_type: Optional[str] = None,
    ) -> List[DesignEntry]:
        """Return known failed designs, optionally filtered by task type.

        Args:
            task_type: If provided, only return designs that were tested on
                this task type.

        Returns:
            List of failed ``DesignEntry`` objects.
        """
        failures = [e for e in self._entries if not e.success]
        if task_type:
            failures = [
                e for e in failures
                if task_type in e.task_types or task_type in e.performance.scores
            ]
        return failures

    def __len__(self) -> int:
        """Return the total number of entries in the archive."""
        return len(self._entries)

    def __iter__(self):
        """Iterate over all archive entries."""
        return iter(self._entries)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the archive to a JSON file.

        Args:
            path: Output file path.
        """
        data = [e.to_dict() for e in self._entries]
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        logger.info("DesignArchive: saved %d entries to %s", len(self._entries), path)

    @classmethod
    def load(
        cls,
        path: str,
        similarity_threshold: float = 0.85,
        llm: Optional[Any] = None,
    ) -> "DesignArchive":
        """Restore an archive from a JSON file.

        Args:
            path: Path to the JSON file produced by ``save()``.
            similarity_threshold: Threshold for duplicate detection.
            llm: Optional LLM for meta-agent search.

        Returns:
            Populated ``DesignArchive`` instance.
        """
        archive = cls(similarity_threshold=similarity_threshold, llm=llm)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for entry_data in data:
            entry = DesignEntry.from_dict(entry_data)
            archive._entries.append(entry)
        logger.info("DesignArchive: loaded %d entries from %s", len(archive._entries), path)
        return archive

    # ------------------------------------------------------------------
    # Meta-agent search
    # ------------------------------------------------------------------

    def meta_agent_search(
        self,
        task_type: str,
        evaluator: Any,
        n_iterations: int = 10,
        n_successes_in_context: int = 3,
        n_failures_in_context: int = 2,
        design_constraints: str = "No specific constraints.",
    ) -> List[DesignEntry]:
        """Iteratively generate and evaluate new agent designs.

        At each iteration:
        1. Select top-k successes and recent failures from the archive.
        2. Ask the LLM to propose a novel design.
        3. Evaluate the design using ``evaluator``.
        4. Archive the result (success or failure).

        Args:
            task_type: Task type to optimise designs for.
            evaluator: Callable ``(config) -> float`` that evaluates a design
                configuration and returns a score in [0, 1].
            n_iterations: Number of meta-agent search iterations.
            n_successes_in_context: How many successful designs to show the LLM.
            n_failures_in_context: How many failed designs to show the LLM.
            design_constraints: Free-text constraints passed to the LLM.

        Returns:
            List of newly created ``DesignEntry`` objects (one per iteration).
        """
        if self.llm is None:
            raise RuntimeError(
                "meta_agent_search requires an LLM — pass llm= to DesignArchive()."
            )

        new_entries: List[DesignEntry] = []
        for iteration in range(n_iterations):
            logger.info(
                "DesignArchive: meta-agent search iteration %d/%d",
                iteration + 1,
                n_iterations,
            )

            successes = self.top_k(k=n_successes_in_context, task_type=task_type)
            failures = self.failed_designs(task_type=task_type)[:n_failures_in_context]

            design_data = self._propose_design(
                successes=successes,
                failures=failures,
                task_type=task_type,
                constraints=design_constraints,
            )
            if design_data is None:
                logger.warning(
                    "DesignArchive: LLM failed to produce a valid design on iteration %d",
                    iteration + 1,
                )
                continue

            config = design_data.get("config", {})
            rationale = design_data.get("rationale", "")

            try:
                score = float(evaluator(config))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "DesignArchive: evaluator error on iteration %d — %s",
                    iteration + 1,
                    exc,
                )
                score = 0.0

            entry = DesignEntry(
                config=config,
                performance=PerformanceProfile(
                    scores={task_type: score},
                    n_trials={task_type: 1},
                ),
                rationale=rationale,
                success=score >= 0.5,
                task_types=[task_type],
                source="meta_agent",
                tags=design_data.get("expected_strengths", []),
            )
            inserted = self.add(entry)
            if inserted:
                new_entries.append(entry)

        logger.info(
            "DesignArchive: meta-agent search complete — %d new entries added",
            len(new_entries),
        )
        return new_entries

    def _propose_design(
        self,
        successes: List[DesignEntry],
        failures: List[DesignEntry],
        task_type: str,
        constraints: str,
    ) -> Optional[Dict[str, Any]]:
        """Ask the LLM to propose a new agent design.

        Args:
            successes: Successful designs to show as positive examples.
            failures: Failed designs to show as negative examples.
            task_type: Target task type.
            constraints: Free-text design constraints.

        Returns:
            Parsed dict with keys ``config``, ``rationale``, etc., or ``None``
            on parse failure.
        """
        def _fmt(entries: List[DesignEntry]) -> str:
            if not entries:
                return "None available."
            parts = []
            for e in entries:
                score = e.performance.score_for(task_type)
                parts.append(
                    f"- config={json.dumps(e.config)}  score={score:.3f}  "
                    f"rationale={e.rationale!r}"
                )
            return "\n".join(parts)

        user_msg = _META_USER.format(
            successes=_fmt(successes),
            failures=_fmt(failures),
            task_type=task_type,
            constraints=constraints,
        )

        try:
            response = self.llm.generate(
                prompt=user_msg,
                system_prompt=_META_SYSTEM,
            )
            raw = (response.content or "").strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("DesignArchive: JSON parse error in LLM response — %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("DesignArchive: LLM call failed — %s", exc)
            return None

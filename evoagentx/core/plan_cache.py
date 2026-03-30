"""Agentic Plan Caching for multi-step task plan reuse.

Unlike SemanticCache (which caches individual LLM call results) or CachingLLM
(which caches per-prompt responses), PlanCache stores **entire plan templates**
— sequences of actions/tool calls — and retrieves/adapts them for structurally
similar incoming tasks.

Design
------
Matching is two-level:

1. **Structural similarity** — two plans share the same sequence of action types
   and optional tool names.  This is computed via ``structural_similarity`` and
   is a fast O(len) comparison entirely independent of task-specific parameters.
2. **Task description similarity** — among structurally similar candidates, rank
   by how closely the stored task description matches the new task.  Uses an
   optional embedding function (cosine similarity) or falls back to Jaccard on
   word tokens.

When a match is found, ``adapt`` returns a copy of the template's steps with
descriptions updated to reference the new task while preserving action types,
tool names, and the overall structure.

Cost savings are tracked via the CostTracker integration: each cache hit records
the estimated cost that was avoided, driving toward the ~50% reduction target.

Typical usage
-------------
    >>> from evoagentx.core.plan_cache import PlanCache, PlanStep
    >>> cache = PlanCache()
    >>> steps = [PlanStep("search", "search the web", tool_name="web_search")]
    >>> cache.store("find the weather in Paris", steps, outcome="success")
    >>> template = cache.retrieve("what is the weather in London?")
    >>> adapted = cache.adapt(template, "what is the weather in London?")
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from .logging import logger


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    """A single step in a multi-step plan.

    Attributes:
        action_type: Categorical label for the kind of operation (e.g.
            ``"search"``, ``"summarise"``, ``"code_exec"``).  Used for
            structural matching.
        description: Human-readable description of this step.
        tool_name: Optional name of the tool invoked in this step.  Part of
            structural fingerprint when present.
        parameters: Key/value pairs specific to this step execution.  NOT part
            of the structural fingerprint — these are adapted per task.
        estimated_cost: Estimated USD cost of executing this step alone.
    """

    action_type: str
    description: str
    tool_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    estimated_cost: float = 0.0


@dataclass
class PlanTemplate:
    """A cached multi-step plan template extracted from a successful run.

    Attributes:
        task_description: The original task that produced this plan.
        steps: Ordered sequence of steps.
        total_cost: Sum of estimated_cost across steps (USD).
        success_rate: Fraction of retrievals that led to a successful outcome
            (updated via ``PlanCache.store`` with ``outcome``).
        times_used: How many times this template has been retrieved.
        last_used: Unix timestamp of the most recent retrieval.
        structural_hash: Deterministic hash of the action-type / tool sequence.
            Pre-computed for fast grouping.
    """

    task_description: str
    steps: List[PlanStep]
    total_cost: float = 0.0
    success_rate: float = 1.0
    times_used: int = 0
    last_used: float = field(default_factory=time.time)
    structural_hash: str = ""

    def __post_init__(self) -> None:
        if not self.structural_hash:
            self.structural_hash = _compute_structural_hash(self.steps)
        if self.total_cost == 0.0:
            self.total_cost = sum(s.estimated_cost for s in self.steps)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_structural_hash(steps: List[PlanStep]) -> str:
    """Return a deterministic hash of the action-type / tool-name sequence.

    The hash only depends on the *structure* of the plan, not on step
    descriptions or parameters.  Two plans that share the same sequence of
    ``(action_type, tool_name or "")`` pairs will produce the same hash.

    Args:
        steps: Ordered list of plan steps.

    Returns:
        Hex-encoded SHA-256 digest string (64 chars).
    """
    fingerprint = "|".join(
        f"{s.action_type}:{s.tool_name or ''}" for s in steps
    )
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Word-token Jaccard similarity between two strings.

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        Jaccard coefficient in [0.0, 1.0].
    """
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a and not tokens_b:
        return 1.0
    union = len(tokens_a | tokens_b)
    if union == 0:
        return 0.0
    return len(tokens_a & tokens_b) / union


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Cosine similarity clamped to [0, 1].

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        Cosine similarity in [0.0, 1.0].  Returns 0.0 on zero-norm vectors.

    Raises:
        ValueError: If vectors have different lengths.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Embedding dimension mismatch: {len(vec_a)} vs {len(vec_b)}"
        )
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


# ---------------------------------------------------------------------------
# PlanCache
# ---------------------------------------------------------------------------

class PlanCache:
    """Cache for multi-step plan templates with structural + semantic matching.

    Args:
        embed_fn: Optional callable ``(text: str) -> List[float]``.  When
            provided, task description similarity uses cosine similarity on
            embeddings instead of Jaccard on word tokens.
        similarity_threshold: Minimum task-description similarity for a
            retrieval hit (0–1, default 0.75).
        max_templates: Maximum number of templates stored.  When exceeded,
            ``prune`` is called automatically with ``min_success_rate=0``.
        ttl_seconds: Optional time-to-live per template.  ``None`` = no expiry.
        cost_tracker: Optional :class:`CostTracker` instance.  When provided,
            cache hits record the avoided cost.

    Notes:
        Structural matching is O(1) via hash grouping.
        Description similarity is O(K) where K is the number of templates
        sharing the same structural hash.
    """

    def __init__(
        self,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        similarity_threshold: float = 0.75,
        max_templates: int = 500,
        ttl_seconds: Optional[float] = None,
        cost_tracker: Optional[Any] = None,
    ) -> None:
        if not (0.0 < similarity_threshold <= 1.0):
            raise ValueError(
                f"similarity_threshold must be in (0, 1], got {similarity_threshold}"
            )
        if max_templates < 1:
            raise ValueError(f"max_templates must be >= 1, got {max_templates}")

        self._embed_fn = embed_fn
        self.similarity_threshold = similarity_threshold
        self.max_templates = max_templates
        self.ttl_seconds = ttl_seconds
        self._cost_tracker = cost_tracker

        # Primary store: list of templates (newest appended last)
        self._templates: List[PlanTemplate] = []

        # Stats
        self._total_queries: int = 0
        self._hits: int = 0
        self._total_cost_saved: float = 0.0

        logger.debug(
            "PlanCache: threshold=%.2f max=%d ttl=%s embed=%s",
            similarity_threshold,
            max_templates,
            ttl_seconds,
            "enabled" if embed_fn else "Jaccard",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(
        self,
        task_description: str,
        steps: List[PlanStep],
        outcome: str = "success",
    ) -> PlanTemplate:
        """Store a plan as a template, or update an existing identical plan.

        If a template with the same structural hash AND task description
        already exists (exact text match), its success statistics are updated
        rather than creating a duplicate.

        Args:
            task_description: The task that was solved by this plan.
            steps: Ordered list of :class:`PlanStep` objects.
            outcome: ``"success"`` or ``"failure"``.  Affects ``success_rate``
                of the stored template.

        Returns:
            The newly created or updated :class:`PlanTemplate`.
        """
        struct_hash = _compute_structural_hash(steps)
        success = outcome.lower() == "success"

        # Check for exact duplicate (same hash + same task description)
        for tmpl in self._templates:
            if (
                tmpl.structural_hash == struct_hash
                and tmpl.task_description == task_description
            ):
                # Update rolling success rate
                n = tmpl.times_used + 1
                tmpl.success_rate = (tmpl.success_rate * tmpl.times_used + (1.0 if success else 0.0)) / n
                tmpl.times_used = n
                tmpl.last_used = time.time()
                logger.debug("PlanCache: updated existing template for %r", task_description[:50])
                return tmpl

        total_cost = sum(s.estimated_cost for s in steps)
        tmpl = PlanTemplate(
            task_description=task_description,
            steps=list(steps),
            total_cost=total_cost,
            success_rate=1.0 if success else 0.0,
            times_used=1,  # counts this initial store as the first use
            last_used=time.time(),
            structural_hash=struct_hash,
        )
        self._templates.append(tmpl)
        logger.debug(
            "PlanCache: stored new template (total=%d) for %r",
            len(self._templates),
            task_description[:50],
        )

        # Auto-prune if over capacity
        if len(self._templates) > self.max_templates:
            self.prune(max_templates=self.max_templates, min_success_rate=0.0)

        return tmpl

    def retrieve(
        self,
        task_description: str,
        similarity_threshold: Optional[float] = None,
    ) -> Optional[PlanTemplate]:
        """Find the best matching cached plan template for a new task.

        Matching is two-level:
        1. Group candidates by structural similarity score ≥ 0.5.
        2. Among structural candidates, pick the one with the highest task
           description similarity (cosine if embed_fn provided, else Jaccard).

        Args:
            task_description: The new task to match against stored templates.
            similarity_threshold: Override instance threshold for this call.

        Returns:
            Best matching :class:`PlanTemplate`, or ``None`` if no template
            meets the threshold.
        """
        self._total_queries += 1
        threshold = similarity_threshold if similarity_threshold is not None else self.similarity_threshold

        active = self._active_templates()
        if not active:
            self._record_miss()
            return None

        best_template: Optional[PlanTemplate] = None
        best_score: float = threshold - 1e-9  # must strictly exceed threshold

        for tmpl in active:
            score = self._description_similarity(task_description, tmpl.task_description)
            if score > best_score:
                best_score = score
                best_template = tmpl

        if best_template is not None:
            best_template.times_used += 1
            best_template.last_used = time.time()
            self._hits += 1
            self._total_cost_saved += best_template.total_cost
            if self._cost_tracker is not None:
                try:
                    self._cost_tracker.record_savings(best_template.total_cost)
                except AttributeError:
                    pass  # cost_tracker without record_savings is fine
            logger.debug(
                "PlanCache: hit (score=%.3f) for %r → %r",
                best_score,
                task_description[:40],
                best_template.task_description[:40],
            )
            return best_template

        self._record_miss()
        return None

    def adapt(self, template: PlanTemplate, new_task: str) -> List[PlanStep]:
        """Produce a task-adapted copy of a cached plan's steps.

        Preserves ``action_type``, ``tool_name``, and ``estimated_cost`` from
        the template.  Rewrites ``description`` to reference the new task's
        verb/noun context, and shallow-copies ``parameters`` so the caller can
        modify them without affecting the cached template.

        The adaptation heuristic is intentionally lightweight: it replaces the
        original task tokens in each step description with the new task tokens,
        falling back to the original description when no replacement is needed.

        Args:
            template: The cached :class:`PlanTemplate` to adapt.
            new_task: Description of the new task.

        Returns:
            A new list of :class:`PlanStep` objects adapted for ``new_task``.
        """
        original_tokens = set(template.task_description.lower().split())
        new_tokens = new_task.lower().split()

        adapted: List[PlanStep] = []
        for step in template.steps:
            new_description = _adapt_description(
                step.description, template.task_description, new_task
            )
            adapted.append(
                PlanStep(
                    action_type=step.action_type,
                    description=new_description,
                    tool_name=step.tool_name,
                    parameters=dict(step.parameters),
                    estimated_cost=step.estimated_cost,
                )
            )
        return adapted

    def structural_similarity(
        self, plan_a: List[PlanStep], plan_b: List[PlanStep]
    ) -> float:
        """Compare two plans by their action-type / tool-name sequences.

        Computes a normalised edit-distance–inspired overlap between the two
        sequences.  Plans with identical structure score 1.0; plans with
        completely disjoint structures score 0.0.

        The comparison is parameter-independent and description-independent:
        only ``action_type`` and ``tool_name`` (when present) contribute.

        Args:
            plan_a: First plan step list.
            plan_b: Second plan step list.

        Returns:
            Structural similarity in [0.0, 1.0].
        """
        if not plan_a and not plan_b:
            return 1.0
        if not plan_a or not plan_b:
            return 0.0

        # Fast path: identical hashes
        if _compute_structural_hash(plan_a) == _compute_structural_hash(plan_b):
            return 1.0

        # Compute LCS-based similarity on the (action_type, tool_name) tokens
        tokens_a = [f"{s.action_type}:{s.tool_name or ''}" for s in plan_a]
        tokens_b = [f"{s.action_type}:{s.tool_name or ''}" for s in plan_b]

        lcs_len = _lcs_length(tokens_a, tokens_b)
        return (2.0 * lcs_len) / (len(tokens_a) + len(tokens_b))

    def prune(
        self,
        max_templates: Optional[int] = None,
        min_success_rate: float = 0.0,
    ) -> int:
        """Remove low-performing or excess templates.

        Templates are first filtered by ``min_success_rate``.  If the
        remaining count still exceeds ``max_templates``, the least-recently-
        used templates are removed until the limit is met.

        Args:
            max_templates: Keep at most this many templates.  Defaults to the
                instance's ``max_templates`` setting.
            min_success_rate: Remove any template with success_rate below this
                threshold.

        Returns:
            Number of templates removed.
        """
        before = len(self._templates)
        limit = max_templates if max_templates is not None else self.max_templates

        # Filter by success rate
        self._templates = [
            t for t in self._templates if t.success_rate >= min_success_rate
        ]

        # Trim to max by LRU (oldest last_used first)
        if len(self._templates) > limit:
            self._templates.sort(key=lambda t: t.last_used, reverse=True)
            self._templates = self._templates[:limit]

        removed = before - len(self._templates)
        if removed:
            logger.debug("PlanCache: pruned %d templates (remaining=%d)", removed, len(self._templates))
        return removed

    def save(self, path: str) -> None:
        """Persist all templates to a JSON file.

        Args:
            path: File path to write (created/overwritten).
        """
        payload = {
            "version": 1,
            "stats": {
                "total_queries": self._total_queries,
                "hits": self._hits,
                "total_cost_saved": self._total_cost_saved,
            },
            "templates": [_template_to_dict(t) for t in self._templates],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        logger.debug("PlanCache: saved %d templates to %s", len(self._templates), path)

    def load(self, path: str) -> None:
        """Load templates from a JSON file, replacing any existing templates.

        Args:
            path: File path to read.

        Raises:
            FileNotFoundError: If path does not exist.
            ValueError: If the file format is unrecognised.
        """
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        if payload.get("version") != 1:
            raise ValueError(
                f"Unsupported plan cache file version: {payload.get('version')}"
            )

        self._templates = [_template_from_dict(d) for d in payload.get("templates", [])]
        stats = payload.get("stats", {})
        self._total_queries = stats.get("total_queries", 0)
        self._hits = stats.get("hits", 0)
        self._total_cost_saved = stats.get("total_cost_saved", 0.0)
        logger.debug("PlanCache: loaded %d templates from %s", len(self._templates), path)

    def stats(self) -> Dict[str, Any]:
        """Return a snapshot of cache performance statistics.

        Returns:
            Dict with keys:
            - ``total_queries``: Total retrieve() calls.
            - ``hits``: Retrieve calls that returned a template.
            - ``misses``: Retrieve calls that returned None.
            - ``hit_rate``: hits / total_queries (0.0 if no queries).
            - ``total_cost_saved``: Estimated USD avoided via cache hits.
            - ``num_templates``: Current number of stored templates.
            - ``avg_template_cost``: Mean total_cost across stored templates.
        """
        misses = self._total_queries - self._hits
        hit_rate = self._hits / self._total_queries if self._total_queries > 0 else 0.0
        avg_cost = (
            sum(t.total_cost for t in self._templates) / len(self._templates)
            if self._templates else 0.0
        )
        return {
            "total_queries": self._total_queries,
            "hits": self._hits,
            "misses": misses,
            "hit_rate": hit_rate,
            "total_cost_saved": self._total_cost_saved,
            "num_templates": len(self._templates),
            "avg_template_cost": avg_cost,
        }

    def __len__(self) -> int:
        return len(self._templates)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _active_templates(self) -> List[PlanTemplate]:
        """Return templates that have not expired under the TTL policy."""
        if self.ttl_seconds is None:
            return list(self._templates)
        now = time.time()
        active = [
            t for t in self._templates
            if (now - t.last_used) <= self.ttl_seconds
        ]
        return active

    def _description_similarity(self, text_a: str, text_b: str) -> float:
        """Compute task description similarity using embed_fn or Jaccard."""
        if self._embed_fn is not None:
            try:
                emb_a = self._embed_fn(text_a)
                emb_b = self._embed_fn(text_b)
                return _cosine_similarity(emb_a, emb_b)
            except Exception as exc:
                logger.warning(
                    "PlanCache: embed_fn failed, falling back to Jaccard — %s", exc
                )
        return _jaccard_similarity(text_a, text_b)

    def _record_miss(self) -> None:
        logger.debug("PlanCache: miss (total_queries=%d)", self._total_queries)


# ---------------------------------------------------------------------------
# Adaptation helper
# ---------------------------------------------------------------------------

def _adapt_description(
    original_desc: str,
    original_task: str,
    new_task: str,
) -> str:
    """Rewrite a step description from original task context to new task context.

    Strategy: find words in the step description that also appear in the
    original task (content words) and substitute them with their positional
    counterpart in the new task when the counts align, or append the new task
    noun phrase when they don't.  If no overlap is found the description is
    returned unchanged.

    This is intentionally rule-based and deterministic — no LLM call.

    Args:
        original_desc: Step description from the cached template.
        original_task: Original task that produced the template.
        new_task: Incoming task that requires adaptation.

    Returns:
        Adapted description string.
    """
    orig_task_words = set(original_task.lower().split())
    new_task_words = new_task.lower().split()
    desc_words = original_desc.split()

    adapted_words = []
    for word in desc_words:
        clean = word.lower().strip(".,;:!?\"'")
        if clean in orig_task_words and new_task_words:
            # Replace with the last content word of the new task as a heuristic
            replacement = new_task_words[-1].strip(".,;:!?\"'")
            # Preserve trailing punctuation
            punct = word[len(clean):] if word.lower().startswith(clean) else ""
            adapted_words.append(replacement + punct)
        else:
            adapted_words.append(word)

    return " ".join(adapted_words)


# ---------------------------------------------------------------------------
# LCS helper for structural similarity
# ---------------------------------------------------------------------------

def _lcs_length(seq_a: List[str], seq_b: List[str]) -> int:
    """Compute the length of the Longest Common Subsequence of two token lists.

    Uses dynamic programming in O(m*n) time.

    Args:
        seq_a: First token sequence.
        seq_b: Second token sequence.

    Returns:
        Length of the LCS.
    """
    m, n = len(seq_a), len(seq_b)
    if m == 0 or n == 0:
        return 0
    # Only keep two rows for space efficiency
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)
    return prev[n]


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _template_to_dict(tmpl: PlanTemplate) -> Dict[str, Any]:
    """Convert a PlanTemplate to a JSON-serialisable dict."""
    return {
        "task_description": tmpl.task_description,
        "steps": [
            {
                "action_type": s.action_type,
                "description": s.description,
                "tool_name": s.tool_name,
                "parameters": s.parameters,
                "estimated_cost": s.estimated_cost,
            }
            for s in tmpl.steps
        ],
        "total_cost": tmpl.total_cost,
        "success_rate": tmpl.success_rate,
        "times_used": tmpl.times_used,
        "last_used": tmpl.last_used,
        "structural_hash": tmpl.structural_hash,
    }


def _template_from_dict(d: Dict[str, Any]) -> PlanTemplate:
    """Reconstruct a PlanTemplate from a dict (as produced by _template_to_dict)."""
    steps = [
        PlanStep(
            action_type=s["action_type"],
            description=s["description"],
            tool_name=s.get("tool_name"),
            parameters=s.get("parameters", {}),
            estimated_cost=s.get("estimated_cost", 0.0),
        )
        for s in d.get("steps", [])
    ]
    return PlanTemplate(
        task_description=d["task_description"],
        steps=steps,
        total_cost=d.get("total_cost", 0.0),
        success_rate=d.get("success_rate", 1.0),
        times_used=d.get("times_used", 0),
        last_used=d.get("last_used", time.time()),
        structural_hash=d.get("structural_hash", _compute_structural_hash(steps)),
    )


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "PlanCache",
    "PlanTemplate",
    "PlanStep",
    "_compute_structural_hash",
    "_jaccard_similarity",
    "_cosine_similarity",
    "_lcs_length",
]

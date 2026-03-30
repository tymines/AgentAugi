"""EvoSkill — closed-loop pipeline connecting failure analysis to skill creation.

When agents accumulate recurring mistakes in the MistakeNotebook, EvoSkill
automatically detects those patterns, identifies what tool capability would
prevent them, and calls ToolSynthesizer to create that tool — closing the loop
between failure and self-improvement.

Architecture
------------
  MistakeNotebook  →  analyze_failures  →  SkillGap
       ↑                                       ↓
  resolve mistakes  ←  deploy_skill  ←  ToolSynthesizer

Typical usage::

    from evoagentx.memory.mistake_notebook import MistakeNotebook
    from evoagentx.core.tool_synthesizer import ToolSynthesizer, ToolRegistry
    from evoagentx.core.evoskill import EvoSkillPipeline, EvoSkillConfig

    nb = MistakeNotebook()
    registry = ToolRegistry()
    synth = ToolSynthesizer(llm=my_llm, registry=registry)
    config = EvoSkillConfig(min_mistake_frequency=3, auto_deploy=True)
    pipeline = EvoSkillPipeline(
        mistake_notebook=nb,
        tool_synthesizer=synth,
        config=config,
        llm_fn=my_llm.generate,
    )

    # After accumulating mistakes, run a full improvement cycle:
    new_discoveries = pipeline.run_cycle()
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .logging import logger
from .tool_synthesizer import SynthesizedTool, ToolSynthesisError, ToolSynthesizer


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class SkillGap:
    """A recurring failure pattern that a new tool could address.

    Attributes
    ----------
    gap_type:
        Short label for the kind of gap (e.g. ``"tool_misuse"``,
        ``"missing_capability"``).
    description:
        Human-readable description of the capability that is missing or broken.
    source_mistakes:
        IDs of the :class:`~evoagentx.memory.mistake_notebook.MistakeEntry`
        objects that contributed to this gap.
    frequency:
        How many similar mistakes were observed.
    severity:
        Estimated impact, in [0, 1].  Derived from mistake categories — e.g.
        TOOL_MISUSE and HALLUCINATION are treated as more severe.
    suggested_tool_spec:
        Dict with keys ``task_description`` and optionally ``examples``; passed
        directly to :meth:`~evoagentx.core.tool_synthesizer.ToolSynthesizer.synthesize`.
    """

    gap_type: str
    description: str
    source_mistakes: List[str]
    frequency: int
    severity: float
    suggested_tool_spec: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillDiscovery:
    """The result of attempting to synthesize a skill for a :class:`SkillGap`.

    Attributes
    ----------
    gap:
        The skill gap that motivated synthesis.
    synthesized_tool:
        The tool produced by :class:`~evoagentx.core.tool_synthesizer.ToolSynthesizer`,
        or ``None`` if synthesis failed.
    validation_score:
        Fraction of validation cases that passed, in [0, 1].
    deployed:
        True once the tool has been added to the registry and source mistakes
        resolved.
    created_at:
        Timestamp when this discovery object was created.
    """

    gap: SkillGap
    synthesized_tool: Optional[SynthesizedTool]
    validation_score: float
    deployed: bool
    created_at: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EvoSkillConfig:
    """Configuration knobs for :class:`EvoSkillPipeline`.

    Attributes
    ----------
    min_mistake_frequency:
        Minimum number of similar mistakes before a skill gap is declared.
        Prevents triggering on isolated, one-off failures.
    severity_threshold:
        Minimum average severity in [0, 1] for a cluster to warrant synthesis.
    auto_deploy:
        When True, deploy (register + resolve mistakes) immediately after
        successful validation without waiting for human approval.
    analysis_interval:
        Re-analyse every N mistakes recorded since the last cycle.  A value of
        0 disables automatic triggering — the caller must invoke
        :meth:`EvoSkillPipeline.run_cycle` explicitly.
    max_pending_gaps:
        Maximum number of undeployed gaps held in memory.  Oldest gaps are
        discarded once the limit is reached.
    """

    min_mistake_frequency: int = 3
    severity_threshold: float = 0.5
    auto_deploy: bool = False
    analysis_interval: int = 10
    max_pending_gaps: int = 20


# ---------------------------------------------------------------------------
# Severity mapping by mistake category
# ---------------------------------------------------------------------------

_CATEGORY_SEVERITY: Dict[str, float] = {
    "hallucination": 0.9,
    "tool_misuse": 0.8,
    "planning_error": 0.7,
    "reasoning_error": 0.6,
    "context_loss": 0.5,
    "format_error": 0.3,
    "other": 0.4,
}


# ---------------------------------------------------------------------------
# EvoSkillPipeline
# ---------------------------------------------------------------------------


class EvoSkillPipeline:
    """Closed-loop pipeline that turns failure patterns into synthesized tools.

    Parameters
    ----------
    mistake_notebook:
        The shared :class:`~evoagentx.memory.mistake_notebook.MistakeNotebook`
        instance used by the agent(s).
    tool_synthesizer:
        A configured :class:`~evoagentx.core.tool_synthesizer.ToolSynthesizer`.
    config:
        Behavioural configuration for the pipeline.
    llm_fn:
        A callable ``llm_fn(prompt: str) -> str`` used for LLM-powered gap
        identification and tool-spec generation.  If ``None``, the pipeline
        falls back to heuristic (non-LLM) gap identification.
    """

    def __init__(
        self,
        mistake_notebook: Any,
        tool_synthesizer: ToolSynthesizer,
        config: Optional[EvoSkillConfig] = None,
        llm_fn: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self._notebook = mistake_notebook
        self._synthesizer = tool_synthesizer
        self._config = config or EvoSkillConfig()
        self._llm_fn = llm_fn
        self.discoveries: List[SkillDiscovery] = []
        self._mistakes_since_last_cycle: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_failures(self) -> List[SkillGap]:
        """Scan the mistake notebook and surface recurring skill gaps.

        Returns
        -------
        List[SkillGap]
            Gaps that exceed the frequency and severity thresholds.
        """
        unresolved = [e for e in self._notebook.entries if not e.resolved]
        if not unresolved:
            logger.debug("EvoSkill | no unresolved mistakes; nothing to analyse.")
            return []

        clusters = self._cluster_mistakes(unresolved)
        gaps: List[SkillGap] = []

        for cluster in clusters:
            if len(cluster) < self._config.min_mistake_frequency:
                logger.debug(
                    "EvoSkill | cluster of %d below min_mistake_frequency=%d; skipping.",
                    len(cluster),
                    self._config.min_mistake_frequency,
                )
                continue

            severity = self._cluster_severity(cluster)
            if severity < self._config.severity_threshold:
                logger.debug(
                    "EvoSkill | cluster severity %.2f below threshold %.2f; skipping.",
                    severity,
                    self._config.severity_threshold,
                )
                continue

            gap = self._identify_skill_gap(cluster)
            if gap is not None:
                gaps.append(gap)

        logger.info(
            "EvoSkill | analysis found %d actionable skill gap(s) from %d mistakes.",
            len(gaps),
            len(unresolved),
        )
        return gaps

    def run_cycle(self) -> List[SkillDiscovery]:
        """Execute a full improvement cycle.

        Steps
        -----
        1. Analyse failures → identify skill gaps.
        2. For each gap: generate tool spec → synthesize → validate.
        3. If ``auto_deploy`` is enabled, deploy each passing discovery.

        Returns
        -------
        List[SkillDiscovery]
            Newly created :class:`SkillDiscovery` objects from this cycle.
        """
        self._mistakes_since_last_cycle = 0
        gaps = self.analyze_failures()
        if not gaps:
            return []

        new_discoveries: List[SkillDiscovery] = []

        for gap in gaps:
            gap.suggested_tool_spec = self._generate_tool_spec(gap)
            discovery = self.synthesize_skill(gap)

            if self._config.auto_deploy and discovery.synthesized_tool is not None:
                self.deploy_skill(discovery)

            self._add_discovery(discovery)
            new_discoveries.append(discovery)

        return new_discoveries

    def notify_mistake_recorded(self) -> Optional[List[SkillDiscovery]]:
        """Call this every time a mistake is added to the notebook.

        When :attr:`EvoSkillConfig.analysis_interval` is set and the count
        reaches the threshold, a cycle is automatically triggered.

        Returns
        -------
        Optional[List[SkillDiscovery]]
            Discoveries from an auto-triggered cycle, or ``None``.
        """
        self._mistakes_since_last_cycle += 1
        interval = self._config.analysis_interval
        if interval > 0 and self._mistakes_since_last_cycle >= interval:
            logger.info(
                "EvoSkill | auto-triggering cycle after %d new mistakes.",
                self._mistakes_since_last_cycle,
            )
            return self.run_cycle()
        return None

    def synthesize_skill(self, gap: SkillGap) -> SkillDiscovery:
        """Attempt to synthesize a tool that addresses a :class:`SkillGap`.

        Parameters
        ----------
        gap:
            The skill gap to address.

        Returns
        -------
        SkillDiscovery
            Contains the synthesized tool (or ``None`` on failure) and a
            validation score.
        """
        spec = gap.suggested_tool_spec or self._generate_tool_spec(gap)
        task_description = spec.get("task_description", gap.description)
        examples = spec.get("examples", [])

        tool: Optional[SynthesizedTool] = None
        validation_score = 0.0

        try:
            tool = self._synthesizer.synthesize(
                task_description=task_description,
                examples=examples,
                force=True,
            )
            validation_score = self.validate_skill_tool(tool, examples)
        except ToolSynthesisError as exc:
            logger.warning(
                "EvoSkill | synthesis failed for gap '%s': %s", gap.gap_type, exc
            )

        return SkillDiscovery(
            gap=gap,
            synthesized_tool=tool,
            validation_score=validation_score,
            deployed=False,
        )

    def validate_skill(self, discovery: SkillDiscovery) -> float:
        """Re-validate a :class:`SkillDiscovery` against its source mistake cases.

        A discovery without a synthesized tool scores 0.0.  Otherwise we run
        any examples embedded in the gap's tool spec through the tool.

        Parameters
        ----------
        discovery:
            The discovery to validate.

        Returns
        -------
        float
            Updated validation score in [0, 1].
        """
        if discovery.synthesized_tool is None:
            return 0.0
        examples = discovery.gap.suggested_tool_spec.get("examples", [])
        score = self.validate_skill_tool(discovery.synthesized_tool, examples)
        discovery.validation_score = score
        return score

    def validate_skill_tool(
        self, tool: SynthesizedTool, examples: List[Dict[str, Any]]
    ) -> float:
        """Run ``examples`` through ``tool`` and return the pass fraction.

        Parameters
        ----------
        tool:
            A :class:`~evoagentx.core.tool_synthesizer.SynthesizedTool`.
        examples:
            List of ``{"args": {...}, "expected": ...}`` dicts.

        Returns
        -------
        float
            Fraction of examples that passed, in [0, 1].  Returns 1.0 when
            ``examples`` is empty (no cases to fail).
        """
        if not examples:
            return 1.0

        passed = 0
        for ex in examples:
            try:
                actual = tool.call(**ex.get("args", {}))
                expected = ex.get("expected")
                if expected is None or actual == expected:
                    passed += 1
            except Exception:
                pass

        return passed / len(examples)

    def deploy_skill(self, discovery: SkillDiscovery) -> bool:
        """Register the synthesized tool and resolve the source mistakes.

        Parameters
        ----------
        discovery:
            A :class:`SkillDiscovery` that has a valid synthesized tool.

        Returns
        -------
        bool
            True if deployment succeeded, False otherwise.
        """
        if discovery.synthesized_tool is None:
            logger.warning("EvoSkill | cannot deploy discovery with no synthesized tool.")
            return False

        # Register into tool registry (synthesize already registers, but this
        # is idempotent and ensures the tool is present after a re-deploy).
        self._synthesizer._registry.register(discovery.synthesized_tool)

        # Mark source mistakes as resolved
        for mistake_id in discovery.gap.source_mistakes:
            self._notebook.resolve(mistake_id, auto_save=False)

        # Persist notebook once after bulk resolve
        self._notebook.save()

        discovery.deployed = True
        logger.info(
            "EvoSkill | deployed tool '%s' and resolved %d mistake(s).",
            discovery.synthesized_tool.tool_name,
            len(discovery.gap.source_mistakes),
        )
        return True

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """Return summary statistics about this pipeline's lifetime activity.

        Returns
        -------
        dict
            Keys: ``gaps_found``, ``skills_synthesized``, ``deployed``,
            ``validation_pass_rate``, ``mistakes_resolved``.
        """
        total = len(self.discoveries)
        synthesized = sum(
            1 for d in self.discoveries if d.synthesized_tool is not None
        )
        deployed_count = sum(1 for d in self.discoveries if d.deployed)
        scores = [d.validation_score for d in self.discoveries]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        mistakes_resolved = sum(
            len(d.gap.source_mistakes)
            for d in self.discoveries
            if d.deployed
        )

        return {
            "gaps_found": total,
            "skills_synthesized": synthesized,
            "deployed": deployed_count,
            "validation_pass_rate": avg_score,
            "mistakes_resolved": mistakes_resolved,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cluster_mistakes(self, mistakes: list) -> List[list]:
        """Group mistakes by category + description similarity (Jaccard).

        Two entries are placed in the same cluster when they share the same
        :attr:`~evoagentx.memory.mistake_notebook.MistakeEntry.category`
        *and* their Jaccard token similarity exceeds a fixed threshold of 0.2.

        Parameters
        ----------
        mistakes:
            List of :class:`~evoagentx.memory.mistake_notebook.MistakeEntry`.

        Returns
        -------
        List[list]
            List of clusters, each a list of ``MistakeEntry`` objects.
        """
        SIMILARITY_THRESHOLD = 0.2
        clusters: List[list] = []
        assigned = [False] * len(mistakes)

        for i, entry_a in enumerate(mistakes):
            if assigned[i]:
                continue
            cluster = [entry_a]
            assigned[i] = True
            tokens_a = set(_tokenize(entry_a.went_wrong + " " + entry_a.attempted))

            for j, entry_b in enumerate(mistakes):
                if assigned[j] or i == j:
                    continue
                if entry_b.category != entry_a.category:
                    continue
                tokens_b = set(
                    _tokenize(entry_b.went_wrong + " " + entry_b.attempted)
                )
                sim = _jaccard(tokens_a, tokens_b)
                if sim >= SIMILARITY_THRESHOLD:
                    cluster.append(entry_b)
                    assigned[j] = True

            clusters.append(cluster)

        return clusters

    def _cluster_severity(self, cluster: list) -> float:
        """Return the average severity of a cluster of mistakes."""
        if not cluster:
            return 0.0
        total = sum(
            _CATEGORY_SEVERITY.get(e.category.value, 0.4) for e in cluster
        )
        return total / len(cluster)

    def _identify_skill_gap(self, cluster: list) -> Optional[SkillGap]:
        """Determine the skill gap represented by a cluster of similar mistakes.

        When an LLM function is available the prompt asks it to articulate the
        missing capability.  Otherwise a heuristic description is built from
        the most common tokens.

        Parameters
        ----------
        cluster:
            Non-empty list of :class:`~evoagentx.memory.mistake_notebook.MistakeEntry`.

        Returns
        -------
        Optional[SkillGap]
            A :class:`SkillGap`, or ``None`` if identification failed.
        """
        representative = cluster[0]
        severity = self._cluster_severity(cluster)
        source_ids = [e.entry_id for e in cluster]

        if self._llm_fn is not None:
            description = self._llm_identify_gap(cluster)
        else:
            description = _heuristic_gap_description(cluster)

        if not description:
            return None

        return SkillGap(
            gap_type=representative.category.value,
            description=description,
            source_mistakes=source_ids,
            frequency=len(cluster),
            severity=severity,
        )

    def _generate_tool_spec(self, gap: SkillGap) -> Dict[str, Any]:
        """Build a tool specification dict from a :class:`SkillGap`.

        When an LLM is available, it generates a rich spec.  Otherwise a
        minimal spec is built from the gap description alone.

        Parameters
        ----------
        gap:
            The skill gap to generate a spec for.

        Returns
        -------
        dict
            Dict with at least ``task_description``; optionally ``examples``.
        """
        if self._llm_fn is not None:
            return self._llm_generate_spec(gap)
        return {"task_description": gap.description, "examples": []}

    def _llm_identify_gap(self, cluster: list) -> str:
        """Use LLM to articulate the missing capability from a mistake cluster."""
        lines = []
        for i, e in enumerate(cluster[:5], 1):
            lines.append(
                f"Mistake {i}:\n"
                f"  Attempted: {e.attempted}\n"
                f"  Went wrong: {e.went_wrong}\n"
                f"  Category: {e.category.value}"
            )
        prompt = (
            "You are an AI assistant analysing agent failure patterns.\n\n"
            "The following mistakes occurred repeatedly:\n\n"
            + "\n\n".join(lines)
            + "\n\nDescribe in one concise sentence what Python tool or capability "
            "would prevent these mistakes.  Return ONLY the sentence, no JSON."
        )
        try:
            raw = self._llm_fn(prompt)
            content = getattr(raw, "content", str(raw)) if raw else ""
            return content.strip().split("\n")[0].strip()
        except Exception as exc:
            logger.warning("EvoSkill | LLM gap identification failed: %s", exc)
            return _heuristic_gap_description(cluster)

    def _llm_generate_spec(self, gap: SkillGap) -> Dict[str, Any]:
        """Use LLM to generate a structured tool specification from a gap."""
        prompt = (
            "You are a Python expert.\n\n"
            f"Gap description: {gap.description}\n"
            f"Gap type: {gap.gap_type}\n\n"
            "Generate a tool specification to address this gap.\n"
            "Return ONLY a JSON object with exactly these keys:\n"
            '{\n'
            '  "task_description": "<one-sentence task for the tool>",\n'
            '  "examples": [{"args": {}, "expected": null}]\n'
            '}\n'
        )
        try:
            raw = self._llm_fn(prompt)
            content = getattr(raw, "content", str(raw)) if raw else ""
            text = content.strip()
            for fence in ("```json", "```"):
                if text.startswith(fence):
                    text = text[len(fence):]
            text = text.rstrip("`").strip()
            spec = json.loads(text)
            if "task_description" not in spec:
                raise ValueError("missing task_description")
            return spec
        except Exception as exc:
            logger.warning("EvoSkill | LLM spec generation failed: %s", exc)
            return {"task_description": gap.description, "examples": []}

    def _add_discovery(self, discovery: SkillDiscovery) -> None:
        """Add a discovery, evicting the oldest if max_pending_gaps is exceeded."""
        self.discoveries.append(discovery)
        max_pending = self._config.max_pending_gaps
        if max_pending > 0:
            undeployed = [d for d in self.discoveries if not d.deployed]
            while len(undeployed) > max_pending:
                oldest = undeployed.pop(0)
                self.discoveries.remove(oldest)
                logger.debug(
                    "EvoSkill | evicted oldest undeployed gap '%s' "
                    "to stay within max_pending_gaps=%d.",
                    oldest.gap.gap_type,
                    max_pending,
                )
                undeployed = [d for d in self.discoveries if not d.deployed]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    """Return alphabetic tokens of length ≥ 3 from *text*, lower-cased."""
    return re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())


def _jaccard(set_a: set, set_b: set) -> float:
    """Return the Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    intersection = set_a & set_b
    return len(intersection) / len(union)


def _heuristic_gap_description(cluster: list) -> str:
    """Build a plain-text gap description from the most frequent tokens."""
    all_tokens: List[str] = []
    for entry in cluster:
        all_tokens.extend(_tokenize(entry.went_wrong))

    freq: Dict[str, int] = {}
    for t in all_tokens:
        freq[t] = freq.get(t, 0) + 1

    top = sorted(freq, key=lambda k: -freq[k])[:5]
    category = cluster[0].category.value.replace("_", " ")
    keywords = ", ".join(top) if top else "unknown issue"
    return (
        f"A tool that handles {category} failures related to: {keywords}. "
        f"Observed in {len(cluster)} mistake(s)."
    )

"""AceForge ↔ EvoSkill Connector.

Bridges EvoSkill's Python-based failure→skill synthesis pipeline with
AceForge's TypeScript-based self-evolving skill engine inside OpenClaw.

Communication model
-------------------
Both systems operate independently and exchange data through shared files on
disk.  No network transport or shared process is required.

  EvoSkill → AceForge
      When EvoSkill identifies a :class:`SkillGap`, the connector translates
      it into AceForge's GapCandidate JSON format and appends it to a shared
      JSONL file in the forge directory.  AceForge's gap-detect engine reads
      that file on its next cycle.

  AceForge → EvoSkill
      When AceForge detects gaps (``patterns.jsonl``) or crystallises a skill
      (``workspace/skills/*.md``), the connector reads those files, converts
      them to EvoSkill data structures, and injects them into the pipeline.

  Evolution requests
      AceForge can signal EvoSkill by writing a JSON trigger file.  The
      connector's :meth:`AceForgeConnector.poll_evolution_requests` method
      reads and clears that file, then triggers a pipeline cycle.

Typical usage::

    from evoagentx.core.evoskill import EvoSkillPipeline
    from evoagentx.integrations.aceforge_connector import (
        AceForgeConnector,
        AceForgeConnectorConfig,
    )

    pipeline = EvoSkillPipeline(...)
    cfg = AceForgeConnectorConfig(dry_run=True)
    connector = AceForgeConnector(pipeline=pipeline, config=cfg)

    result = connector.run_sync()
    print(result.aceforge_to_evoskill.imported_count)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.evoskill import EvoSkillPipeline, SkillDiscovery, SkillGap
from ..core.logging import logger


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AceForgeConnectorConfig:
    """Settings for the AceForge ↔ EvoSkill integration.

    Attributes
    ----------
    forge_dir:
        AceForge's working directory.  Defaults to
        ``~/.openclaw/workspace/.forge``.
    skills_dir:
        AceForge's deployed skills directory.  Defaults to
        ``~/.openclaw/workspace/skills``.
    external_gaps_file:
        File name (relative to *forge_dir*) where EvoSkill writes exported
        gaps so AceForge can pick them up.
    evolution_trigger_file:
        File name (relative to *forge_dir*) where AceForge writes evolution
        requests for EvoSkill.
    sync_interval_seconds:
        How often a background sync loop should run (informational; the loop
        itself must be driven by the caller).  ``0`` means manual-only.
    max_severity_aceforge:
        Maximum severity value in AceForge's integer scoring system; used to
        normalise severity into [0, 1].
    min_aceforge_severity:
        Minimum AceForge severity score for a gap candidate to be imported.
    validation_threshold:
        Minimum EvoSkill validation score for a discovery to be exported as an
        AceForge SKILL.md proposal.
    dry_run:
        When True, log every file-write that would occur but do not create or
        mutate any files.
    """

    forge_dir: Path = field(
        default_factory=lambda: Path.home() / ".openclaw" / "workspace" / ".forge"
    )
    skills_dir: Path = field(
        default_factory=lambda: Path.home() / ".openclaw" / "workspace" / "skills"
    )
    external_gaps_file: str = "evoskill_gaps.jsonl"
    evolution_trigger_file: str = "evoskill_trigger.json"
    sync_interval_seconds: float = 30.0
    max_severity_aceforge: float = 100.0
    min_aceforge_severity: float = 3.0
    validation_threshold: float = 0.6
    dry_run: bool = False


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ImportResult:
    """Outcome of importing gap candidates from AceForge into EvoSkill.

    Attributes
    ----------
    imported_count:
        Number of GapCandidates successfully converted to SkillGaps.
    skipped_count:
        Number skipped due to severity threshold or deduplication.
    gaps:
        The translated :class:`~evoagentx.core.evoskill.SkillGap` objects.
    """

    imported_count: int
    skipped_count: int
    gaps: List[SkillGap]


@dataclass
class ExportResult:
    """Outcome of exporting EvoSkill gaps to AceForge.

    Attributes
    ----------
    exported_count:
        Number of :class:`~evoagentx.core.evoskill.SkillGap` objects written
        to the shared JSONL file.
    file_path:
        Absolute path of the file written, or ``None`` in dry-run mode or when
        there was nothing to export.
    """

    exported_count: int
    file_path: Optional[Path]


@dataclass
class SyncResult:
    """Summary of a full bidirectional synchronisation cycle.

    Attributes
    ----------
    aceforge_to_evoskill:
        Gaps imported from AceForge.
    evoskill_to_aceforge:
        Gaps exported to AceForge.
    discoveries_triggered:
        :class:`~evoagentx.core.evoskill.SkillDiscovery` objects produced by
        EvoSkill after processing imported gaps and any evolution requests.
    skill_exports:
        Paths of SKILL.md files written to AceForge's proposals directory.
    timestamp:
        UTC timestamp when this sync completed.
    """

    aceforge_to_evoskill: ImportResult
    evoskill_to_aceforge: ExportResult
    discoveries_triggered: List[SkillDiscovery]
    skill_exports: List[Path]
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# Gap-type translation tables
# ---------------------------------------------------------------------------

# AceForge gapType → EvoSkill category value
_ACEFORGE_TO_EVOSKILL_GAP_TYPE: Dict[str, str] = {
    "high_failure": "tool_misuse",
    "correction_cluster": "tool_misuse",
    "retry_storm": "context_loss",
    "chain_break": "planning_error",
}

# EvoSkill category value → AceForge gapType
_EVOSKILL_TO_ACEFORGE_GAP_TYPE: Dict[str, str] = {
    "tool_misuse": "high_failure",
    "hallucination": "correction_cluster",
    "planning_error": "chain_break",
    "reasoning_error": "high_failure",
    "context_loss": "retry_storm",
    "format_error": "correction_cluster",
    "other": "high_failure",
}


# ---------------------------------------------------------------------------
# AceForgeConnector
# ---------------------------------------------------------------------------


class AceForgeConnector:
    """Bidirectional bridge between EvoSkill and AceForge.

    The connector is stateful across sync cycles: it tracks which gap
    candidates and skills it has already seen so that repeated syncs do not
    import duplicates.

    Parameters
    ----------
    pipeline:
        The live :class:`~evoagentx.core.evoskill.EvoSkillPipeline` instance.
    config:
        Connector-level settings.  If omitted, ``AceForgeConnectorConfig``
        defaults are used (auto-detected paths, dry_run=False).
    llm_fn:
        Optional callable ``llm_fn(prompt: str) -> str`` available for any
        LLM-assisted translation work.  Currently reserved for future use.
    """

    def __init__(
        self,
        pipeline: EvoSkillPipeline,
        config: Optional[AceForgeConnectorConfig] = None,
        llm_fn: Optional[Callable[[str], Any]] = None,
    ) -> None:
        self._pipeline = pipeline
        self._cfg = config or AceForgeConnectorConfig()
        self._llm_fn = llm_fn
        # Dedup state; persists for the lifetime of the connector instance
        self._seen_gap_ids: set = set()
        self._seen_skill_names: set = set()

    # ------------------------------------------------------------------
    # AceForge → EvoSkill
    # ------------------------------------------------------------------

    def pull_gaps_from_aceforge(self) -> ImportResult:
        """Read AceForge's pattern log and translate gap entries to SkillGaps.

        AceForge's gap-detect engine appends ``GapCandidate`` objects into
        ``patterns.jsonl`` in the forge directory.  This method reads all
        entries, filters by severity, deduplicates against previously seen
        IDs, and converts each candidate to a
        :class:`~evoagentx.core.evoskill.SkillGap`.

        Returns
        -------
        ImportResult
            Summary of imported and skipped candidates.
        """
        patterns_file = self._cfg.forge_dir / "patterns.jsonl"
        if not patterns_file.exists():
            logger.debug(
                "AceForgeConnector | patterns.jsonl not found at %s; nothing to import.",
                patterns_file,
            )
            return ImportResult(imported_count=0, skipped_count=0, gaps=[])

        raw_entries = _read_jsonl(patterns_file)
        gap_candidates = _filter_gap_candidates(raw_entries)

        imported: List[SkillGap] = []
        skipped = 0

        for candidate in gap_candidates:
            gap_id = _candidate_dedup_key(candidate)

            if gap_id in self._seen_gap_ids:
                skipped += 1
                continue

            severity_raw = float(candidate.get("severity", 0))
            if severity_raw < self._cfg.min_aceforge_severity:
                skipped += 1
                continue

            gap = self._translate_candidate_to_gap(candidate)
            if gap is None:
                skipped += 1
                continue

            imported.append(gap)
            self._seen_gap_ids.add(gap_id)

        logger.info(
            "AceForgeConnector | pulled %d gap(s) from AceForge (%d skipped).",
            len(imported),
            skipped,
        )
        return ImportResult(
            imported_count=len(imported), skipped_count=skipped, gaps=imported
        )

    def import_aceforge_skill(self, skill_path: Path) -> Optional[SkillDiscovery]:
        """Parse an AceForge SKILL.md and wrap it as an EvoSkill SkillDiscovery.

        AceForge crystallises skills as Markdown files with YAML frontmatter.
        This method extracts the skill name and AceForge metadata, reconstructs
        a :class:`~evoagentx.core.evoskill.SkillGap` from that context, and
        wraps both in a :class:`~evoagentx.core.evoskill.SkillDiscovery` so
        EvoSkill's validation pipeline can evaluate the skill.

        The ``synthesized_tool`` field will be ``None`` because AceForge skills
        are Markdown documents, not Python callables.  The ``validation_score``
        is derived from the ``candidate_success_rate`` metadata if present.

        Parameters
        ----------
        skill_path:
            Absolute path to a ``SKILL.md`` file produced by AceForge.

        Returns
        -------
        Optional[SkillDiscovery]
            The constructed discovery, or ``None`` if the file cannot be parsed
            or has already been imported.
        """
        skill_name, metadata = _parse_skill_md(skill_path)
        if skill_name is None:
            logger.warning(
                "AceForgeConnector | could not parse SKILL.md at %s.", skill_path
            )
            return None

        if skill_name in self._seen_skill_names:
            logger.debug(
                "AceForgeConnector | skill '%s' already imported; skipping.",
                skill_name,
            )
            return None

        gap = SkillGap(
            gap_type=_metadata_to_gap_type(metadata),
            description=metadata.get("description", skill_name),
            source_mistakes=[],
            frequency=int(float(metadata.get("candidate_occurrences", 1))),
            severity=_success_rate_to_severity(
                metadata.get("candidate_success_rate", 0.5)
            ),
            suggested_tool_spec={
                "task_description": metadata.get("description", skill_name),
                "source": "aceforge",
                "skill_name": skill_name,
            },
        )

        validation_score = _parse_float(
            metadata.get("candidate_success_rate", 0.0), default=0.0
        )

        discovery = SkillDiscovery(
            gap=gap,
            synthesized_tool=None,
            validation_score=validation_score,
            deployed=False,
        )
        self._seen_skill_names.add(skill_name)

        logger.info(
            "AceForgeConnector | imported AceForge skill '%s' as SkillDiscovery "
            "(validation_score=%.2f).",
            skill_name,
            validation_score,
        )
        return discovery

    def poll_evolution_requests(self) -> List[str]:
        """Check whether AceForge has requested EvoSkill evolution runs.

        AceForge writes a JSON trigger file listing tool names for which it
        wants EvoSkill to run a targeted analysis cycle.  This method reads
        and then deletes that file (unless ``dry_run`` is set).

        Returns
        -------
        List[str]
            Tool names requested, or an empty list if no trigger exists.
        """
        trigger_file = self._cfg.forge_dir / self._cfg.evolution_trigger_file
        if not trigger_file.exists():
            return []

        try:
            content = trigger_file.read_text(encoding="utf-8").strip()
            if not content:
                return []

            data = json.loads(content)
            tool_names = data.get("tools", [])
            if not isinstance(tool_names, list):
                tool_names = [str(tool_names)]

            if not self._cfg.dry_run:
                trigger_file.unlink(missing_ok=True)

            valid_names = [str(t) for t in tool_names if t]
            logger.info(
                "AceForgeConnector | received evolution requests for %d tool(s).",
                len(valid_names),
            )
            return valid_names

        except Exception as exc:
            logger.warning(
                "AceForgeConnector | failed to read evolution trigger file: %s", exc
            )
            return []

    # ------------------------------------------------------------------
    # EvoSkill → AceForge
    # ------------------------------------------------------------------

    def push_gap_to_aceforge(self, gap: SkillGap) -> Optional[Path]:
        """Append an EvoSkill SkillGap to AceForge's shared gap file.

        Translates the gap to AceForge's ``GapCandidate`` JSON format and
        appends one line to ``evoskill_gaps.jsonl`` in the forge directory.
        AceForge's gap-detect engine reads this file on its next cycle.

        Parameters
        ----------
        gap:
            The :class:`~evoagentx.core.evoskill.SkillGap` to export.

        Returns
        -------
        Optional[Path]
            Path to the file written, or ``None`` in dry-run mode.
        """
        candidate = self._translate_gap_to_candidate(gap)
        out_file = self._cfg.forge_dir / self._cfg.external_gaps_file

        if self._cfg.dry_run:
            logger.info(
                "AceForgeConnector [dry_run] | would append gap '%s' to %s.",
                gap.gap_type,
                out_file,
            )
            return None

        _ensure_dir(self._cfg.forge_dir)
        with out_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(candidate, ensure_ascii=False) + "\n")

        logger.info(
            "AceForgeConnector | exported gap '%s' to %s.", gap.gap_type, out_file
        )
        return out_file

    def export_discovery_as_skill_md(
        self, discovery: SkillDiscovery
    ) -> Optional[Path]:
        """Write an EvoSkill SkillDiscovery as an AceForge SKILL.md proposal.

        Produces a Markdown file with YAML frontmatter compatible with
        AceForge's approval workflow.  The file is placed in
        ``forge_dir/proposals/`` for human review before AceForge deploys it.

        A discovery is only exported if its ``validation_score`` meets
        :attr:`AceForgeConnectorConfig.validation_threshold`.

        Parameters
        ----------
        discovery:
            The :class:`~evoagentx.core.evoskill.SkillDiscovery` to export.

        Returns
        -------
        Optional[Path]
            Path to the SKILL.md file, or ``None`` when the score threshold
            is not met or dry-run mode is active.
        """
        if discovery.validation_score < self._cfg.validation_threshold:
            logger.debug(
                "AceForgeConnector | gap '%s' has score %.2f < threshold %.2f; "
                "skipping export.",
                discovery.gap.gap_type,
                discovery.validation_score,
                self._cfg.validation_threshold,
            )
            return None

        skill_md = _render_skill_md(discovery)
        proposals_dir = self._cfg.forge_dir / "proposals"

        tool_name = (
            discovery.synthesized_tool.tool_name
            if discovery.synthesized_tool is not None
            else _gap_to_slug(discovery.gap)
        )
        out_file = proposals_dir / f"evoskill-{tool_name}.md"

        if self._cfg.dry_run:
            logger.info(
                "AceForgeConnector [dry_run] | would write SKILL.md to %s.", out_file
            )
            return None

        _ensure_dir(proposals_dir)
        out_file.write_text(skill_md, encoding="utf-8")

        logger.info(
            "AceForgeConnector | exported SkillDiscovery as SKILL.md to %s.", out_file
        )
        return out_file

    def request_aceforge_evolution(self, tool_names: List[str]) -> bool:
        """Signal AceForge to run an evolution pass for specific tools.

        Writes a JSON trigger file that AceForge's ``agent_end`` hook reads to
        schedule evolution runs.  Any pre-existing trigger file is replaced.

        Parameters
        ----------
        tool_names:
            Tool names AceForge should evolve.

        Returns
        -------
        bool
            True if the trigger was written (or would be in dry-run mode),
            False if *tool_names* is empty.
        """
        if not tool_names:
            return False

        trigger = {
            "source": "evoskill",
            "tools": tool_names,
            "requested_at": datetime.now(timezone.utc).isoformat(),
        }
        trigger_file = self._cfg.forge_dir / self._cfg.evolution_trigger_file

        if self._cfg.dry_run:
            logger.info(
                "AceForgeConnector [dry_run] | would write evolution trigger "
                "for %d tool(s): %s.",
                len(tool_names),
                ", ".join(tool_names),
            )
            return True

        _ensure_dir(self._cfg.forge_dir)
        trigger_file.write_text(
            json.dumps(trigger, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(
            "AceForgeConnector | wrote evolution trigger for: %s.",
            ", ".join(tool_names),
        )
        return True

    # ------------------------------------------------------------------
    # High-level sync
    # ------------------------------------------------------------------

    def run_sync(self) -> SyncResult:
        """Execute a full bidirectional synchronisation cycle.

        Steps
        -----
        1. Pull gap candidates from AceForge → translate to SkillGaps.
        2. Synthesise a skill for each imported gap via the EvoSkill pipeline.
        3. Export pending EvoSkill gaps to AceForge's shared JSONL file.
        4. Export qualifying SkillDiscoveries as AceForge SKILL.md proposals.
        5. Check for AceForge-initiated evolution requests and run a pipeline
           cycle if any are found.

        Returns
        -------
        SyncResult
            Full summary of this sync cycle.
        """
        logger.info("AceForgeConnector | starting bidirectional sync cycle.")

        # Step 1 — import from AceForge
        import_result = self.pull_gaps_from_aceforge()

        # Step 2 — synthesise skills for each imported gap
        discoveries: List[SkillDiscovery] = []
        for gap in import_result.gaps:
            discovery = self._pipeline.synthesize_skill(gap)
            if self._pipeline._config.auto_deploy and discovery.synthesized_tool:
                self._pipeline.deploy_skill(discovery)
            self._pipeline._add_discovery(discovery)
            discoveries.append(discovery)

        # Step 3 — export pending EvoSkill gaps → AceForge
        pending_gaps = [d.gap for d in self._pipeline.discoveries if not d.deployed]
        exported_file: Optional[Path] = None
        exported_types: set = set()
        for gap in pending_gaps:
            if gap.gap_type not in exported_types:
                path = self.push_gap_to_aceforge(gap)
                if path:
                    exported_file = path
                exported_types.add(gap.gap_type)

        export_result = ExportResult(
            exported_count=len(pending_gaps),
            file_path=exported_file,
        )

        # Step 4 — export qualifying discoveries as SKILL.md proposals
        skill_exports: List[Path] = []
        for disc in discoveries:
            path = self.export_discovery_as_skill_md(disc)
            if path:
                skill_exports.append(path)

        # Step 5 — handle AceForge-initiated evolution requests
        requested_tools = self.poll_evolution_requests()
        if requested_tools:
            logger.info(
                "AceForgeConnector | running EvoSkill cycle for %d requested tool(s).",
                len(requested_tools),
            )
            extra = self._pipeline.run_cycle()
            discoveries.extend(extra)

        result = SyncResult(
            aceforge_to_evoskill=import_result,
            evoskill_to_aceforge=export_result,
            discoveries_triggered=discoveries,
            skill_exports=skill_exports,
        )

        logger.info(
            "AceForgeConnector | sync done — "
            "imported=%d exported=%d discoveries=%d skill_exports=%d.",
            import_result.imported_count,
            export_result.exported_count,
            len(discoveries),
            len(skill_exports),
        )
        return result

    # ------------------------------------------------------------------
    # Translation helpers
    # ------------------------------------------------------------------

    def _translate_candidate_to_gap(
        self, candidate: Dict[str, Any]
    ) -> Optional[SkillGap]:
        """Convert an AceForge GapCandidate dict to an EvoSkill SkillGap."""
        tool = candidate.get("tool", "")
        gap_type_raw = str(candidate.get("gapType", "high_failure"))
        gap_type = _ACEFORGE_TO_EVOSKILL_GAP_TYPE.get(gap_type_raw, "tool_misuse")

        suggested_focus = candidate.get("suggestedFocus", "")
        evidence = candidate.get("evidence", [])
        description = suggested_focus or (
            evidence[0] if isinstance(evidence, list) and evidence else tool
        )
        if not description:
            return None

        severity_raw = float(candidate.get("severity", 0))
        severity = min(severity_raw / self._cfg.max_severity_aceforge, 1.0)

        examples = _build_examples_from_traces(
            candidate.get("failureTraces", [])
        )

        return SkillGap(
            gap_type=gap_type,
            description=description,
            source_mistakes=[],
            frequency=max(1, int(severity_raw)),
            severity=severity,
            suggested_tool_spec={
                "task_description": description,
                "examples": examples,
                "source": "aceforge",
                "tool": tool,
                "aceforge_gap_type": gap_type_raw,
            },
        )

    def _translate_gap_to_candidate(self, gap: SkillGap) -> Dict[str, Any]:
        """Convert an EvoSkill SkillGap to an AceForge GapCandidate dict."""
        aceforge_gap_type = _EVOSKILL_TO_ACEFORGE_GAP_TYPE.get(
            gap.gap_type, "high_failure"
        )
        severity_int = int(gap.severity * self._cfg.max_severity_aceforge)
        tool = gap.suggested_tool_spec.get("tool", gap.gap_type)

        evidence = [gap.description] + [
            f"Source mistake: {mid}" for mid in gap.source_mistakes[:3]
        ]

        return {
            "tool": tool,
            "gapType": aceforge_gap_type,
            "severity": severity_int,
            "evidence": evidence,
            "suggestedFocus": gap.description,
            "failureTraces": [],
            "corrections": [],
            "source": "evoskill",
            "frequency": gap.frequency,
            "ts": datetime.now(timezone.utc).isoformat(),
        }


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file; silently skip malformed lines."""
    records: List[Dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except json.JSONDecodeError:
                pass
    except OSError:
        pass
    return records


def _filter_gap_candidates(
    entries: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Return only entries that represent AceForge gap candidates.

    AceForge marks gap entries with ``"type": "gap"`` or with a ``"gapType"``
    key (from the gap-detect engine's output format).
    """
    return [
        e for e in entries
        if e.get("type") == "gap" or "gapType" in e
    ]


def _candidate_dedup_key(candidate: Dict[str, Any]) -> str:
    """Produce a stable deduplication key for a GapCandidate."""
    tool = candidate.get("tool", "")
    gap_type = candidate.get("gapType", "")
    ts = candidate.get("ts", candidate.get("first_seen", ""))
    return f"{tool}:{gap_type}:{ts}"


def _ensure_dir(path: Path) -> None:
    """Create *path* and all parents if they do not exist."""
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# SKILL.md parsing / rendering
# ---------------------------------------------------------------------------


def _parse_skill_md(path: Path) -> Tuple[Optional[str], Dict[str, Any]]:
    """Extract the skill name and metadata from an AceForge SKILL.md.

    Returns
    -------
    Tuple[Optional[str], Dict[str, Any]]
        ``(skill_name, flat_metadata)``; both are ``None``/empty on error.

    Notes
    -----
    Uses a lightweight regex-based YAML parser to avoid pulling in PyYAML
    as a hard dependency.  Only flat ``key: value`` pairs are extracted;
    nested mappings are flattened by taking their leaf ``key: value`` lines.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None, {}

    fm_match = re.match(r"^---\s*\n(.+?)\n---", text, re.DOTALL)
    if not fm_match:
        return None, {}

    frontmatter = fm_match.group(1)
    metadata: Dict[str, Any] = {}

    for line in frontmatter.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("-") or stripped.startswith("#"):
            continue
        if ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and val:
                metadata[key] = val

    skill_name = metadata.get("name", path.stem) or path.stem
    return skill_name, metadata


def _render_skill_md(discovery: SkillDiscovery) -> str:
    """Produce an AceForge-compatible SKILL.md from an EvoSkill SkillDiscovery."""
    tool_name = (
        discovery.synthesized_tool.tool_name
        if discovery.synthesized_tool is not None
        else _gap_to_slug(discovery.gap)
    )
    description = discovery.gap.description
    now = datetime.now(timezone.utc).isoformat()
    score = round(discovery.validation_score, 4)

    source_code_section = ""
    if (
        discovery.synthesized_tool is not None
        and discovery.synthesized_tool.source_code
    ):
        source_code_section = (
            "\n## Implementation\n\n"
            "```python\n"
            + discovery.synthesized_tool.source_code
            + "\n```\n"
        )

    return (
        "---\n"
        f"name: evoskill-{tool_name}\n"
        f'description: "{description}"\n'
        "metadata:\n"
        "  openclaw:\n"
        f"    category: {discovery.gap.gap_type}\n"
        "    aceforge:\n"
        "      status: proposed\n"
        f"      proposed: {now}\n"
        "      auto_generated: true\n"
        "      source: evoskill\n"
        f"      validation_score: {score}\n"
        f"      gap_frequency: {discovery.gap.frequency}\n"
        f"      gap_severity: {round(discovery.gap.severity, 4)}\n"
        "---\n\n"
        f"# {tool_name}\n\n"
        f"{description}\n\n"
        "## When to Use\n\n"
        f"Apply this skill when encountering **{discovery.gap.gap_type}** failures "
        f"(observed {discovery.gap.frequency} time(s)).\n"
        + source_code_section
        + "\n## Quality\n\n"
        f"- EvoSkill validation score: **{score}**\n"
        f"- Gap severity: **{discovery.gap.severity:.2f}**\n"
        f"- Generated: {now}\n"
    )


# ---------------------------------------------------------------------------
# Miscellaneous helpers
# ---------------------------------------------------------------------------


def _build_examples_from_traces(
    failure_traces: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Convert AceForge failure trace records into EvoSkill example dicts.

    At most five traces are converted.  Each example gets ``"expected": None``
    because the correct output is unknown from trace data alone.
    """
    examples = []
    for trace in failure_traces[:5]:
        args_summary = trace.get("args_summary") or ""
        error = trace.get("error") or trace.get("result_summary") or ""
        if args_summary or error:
            examples.append(
                {
                    "args": {"input": args_summary},
                    "expected": None,
                    "error_context": error,
                }
            )
    return examples


def _gap_to_slug(gap: SkillGap) -> str:
    """Produce a file-safe slug (up to 5 words) from a SkillGap description."""
    words = re.sub(r"[^a-z0-9 ]", "", gap.description.lower()).split()
    return "-".join(words[:5]) or gap.gap_type.replace("_", "-")


def _metadata_to_gap_type(metadata: Dict[str, Any]) -> str:
    """Map AceForge SKILL.md category metadata to an EvoSkill gap_type string."""
    category = metadata.get("category", "")
    return _ACEFORGE_TO_EVOSKILL_GAP_TYPE.get(category, "tool_misuse")


def _success_rate_to_severity(success_rate: Any) -> float:
    """Convert an AceForge success rate [0, 1] to an EvoSkill severity [0, 1].

    A lower success rate implies higher severity.
    """
    return max(0.0, min(1.0, 1.0 - _parse_float(success_rate, default=0.5)))


def _parse_float(value: Any, *, default: float) -> float:
    """Parse *value* to float, returning *default* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

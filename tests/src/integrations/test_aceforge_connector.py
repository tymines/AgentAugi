"""Tests for evoagentx.integrations.aceforge_connector.

Coverage
--------
- Gap import from AceForge patterns.jsonl (filtering, deduplication, severity)
- SKILL.md parsing and SkillDiscovery construction
- Evolution trigger polling and file cleanup
- Gap export to AceForge shared JSONL
- SkillDiscovery export as SKILL.md proposal (threshold enforcement)
- Evolution request writing
- Bidirectional run_sync() cycle
- Dry-run mode (no files written)
- Translation helpers (_candidate_dedup_key, _filter_gap_candidates, etc.)
- File helpers (_read_jsonl, _parse_skill_md, _render_skill_md, _gap_to_slug)
- Edge cases: missing files, malformed JSON, empty inputs
"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from evoagentx.core.evoskill import EvoSkillConfig, EvoSkillPipeline, SkillDiscovery, SkillGap
from evoagentx.core.tool_synthesizer import SynthesizedTool
from evoagentx.integrations.aceforge_connector import (
    AceForgeConnector,
    AceForgeConnectorConfig,
    ExportResult,
    ImportResult,
    SyncResult,
    _build_examples_from_traces,
    _candidate_dedup_key,
    _ensure_dir,
    _filter_gap_candidates,
    _gap_to_slug,
    _metadata_to_gap_type,
    _parse_float,
    _parse_skill_md,
    _read_jsonl,
    _render_skill_md,
    _success_rate_to_severity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def forge_dir(tmp_path: Path) -> Path:
    d = tmp_path / ".forge"
    d.mkdir()
    return d


@pytest.fixture()
def skills_dir(tmp_path: Path) -> Path:
    d = tmp_path / "skills"
    d.mkdir()
    return d


@pytest.fixture()
def connector_config(forge_dir: Path, skills_dir: Path) -> AceForgeConnectorConfig:
    return AceForgeConnectorConfig(
        forge_dir=forge_dir,
        skills_dir=skills_dir,
        dry_run=False,
        min_aceforge_severity=3.0,
        max_severity_aceforge=100.0,
        validation_threshold=0.6,
    )


@pytest.fixture()
def mock_pipeline() -> MagicMock:
    pipeline = MagicMock(spec=EvoSkillPipeline)
    pipeline._config = EvoSkillConfig(auto_deploy=False)
    pipeline.discoveries = []
    # synthesize_skill returns a SkillDiscovery with no tool by default
    pipeline.synthesize_skill.return_value = SkillDiscovery(
        gap=_make_gap("tool_misuse", "test gap"),
        synthesized_tool=None,
        validation_score=0.0,
        deployed=False,
    )
    pipeline.run_cycle.return_value = []
    return pipeline


@pytest.fixture()
def connector(
    mock_pipeline: MagicMock, connector_config: AceForgeConnectorConfig
) -> AceForgeConnector:
    return AceForgeConnector(pipeline=mock_pipeline, config=connector_config)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gap(
    gap_type: str = "tool_misuse",
    description: str = "A test gap",
    frequency: int = 3,
    severity: float = 0.7,
    source_mistakes: Optional[List[str]] = None,
) -> SkillGap:
    return SkillGap(
        gap_type=gap_type,
        description=description,
        source_mistakes=source_mistakes or ["m1", "m2"],
        frequency=frequency,
        severity=severity,
        suggested_tool_spec={"task_description": description},
    )


def _make_discovery(
    gap: Optional[SkillGap] = None,
    validation_score: float = 0.8,
    synthesized_tool: Optional[SynthesizedTool] = None,
    deployed: bool = False,
) -> SkillDiscovery:
    return SkillDiscovery(
        gap=gap or _make_gap(),
        synthesized_tool=synthesized_tool,
        validation_score=validation_score,
        deployed=deployed,
    )


def _write_patterns(forge_dir: Path, records: List[Dict[str, Any]]) -> Path:
    f = forge_dir / "patterns.jsonl"
    with f.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return f


def _gap_candidate(
    tool: str = "bash",
    gap_type: str = "high_failure",
    severity: float = 10.0,
    ts: str = "2026-01-01T00:00:00Z",
) -> Dict[str, Any]:
    return {
        "tool": tool,
        "gapType": gap_type,
        "severity": severity,
        "evidence": [f"{tool} fails frequently"],
        "suggestedFocus": f"Improve {tool} reliability",
        "failureTraces": [],
        "corrections": [],
        "ts": ts,
    }


# ===========================================================================
# _read_jsonl
# ===========================================================================


def test_read_jsonl_valid(tmp_path: Path) -> None:
    f = tmp_path / "data.jsonl"
    f.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")
    result = _read_jsonl(f)
    assert result == [{"a": 1}, {"b": 2}]


def test_read_jsonl_skips_malformed(tmp_path: Path) -> None:
    f = tmp_path / "data.jsonl"
    f.write_text('{"a": 1}\nnot-json\n{"c": 3}\n', encoding="utf-8")
    result = _read_jsonl(f)
    assert len(result) == 2


def test_read_jsonl_missing_file(tmp_path: Path) -> None:
    result = _read_jsonl(tmp_path / "nonexistent.jsonl")
    assert result == []


def test_read_jsonl_skips_non_dict_lines(tmp_path: Path) -> None:
    f = tmp_path / "data.jsonl"
    f.write_text('["list"]\n{"ok": true}\n', encoding="utf-8")
    result = _read_jsonl(f)
    assert result == [{"ok": True}]


# ===========================================================================
# _filter_gap_candidates
# ===========================================================================


def test_filter_gap_candidates_by_gaptype_key() -> None:
    entries = [
        {"tool": "a", "gapType": "high_failure"},
        {"tool": "b", "type": "gap"},
        {"tool": "c", "type": "pattern"},
    ]
    result = _filter_gap_candidates(entries)
    assert len(result) == 2
    assert all("gapType" in r or r.get("type") == "gap" for r in result)


def test_filter_gap_candidates_empty() -> None:
    assert _filter_gap_candidates([]) == []


def test_filter_gap_candidates_none_match() -> None:
    entries = [{"tool": "x", "type": "correction"}]
    assert _filter_gap_candidates(entries) == []


# ===========================================================================
# _candidate_dedup_key
# ===========================================================================


def test_candidate_dedup_key_stable() -> None:
    c = {"tool": "bash", "gapType": "high_failure", "ts": "2026-01-01T00:00:00Z"}
    k1 = _candidate_dedup_key(c)
    k2 = _candidate_dedup_key(c)
    assert k1 == k2


def test_candidate_dedup_key_differs_by_tool() -> None:
    c1 = {"tool": "bash", "gapType": "high_failure", "ts": "t"}
    c2 = {"tool": "python", "gapType": "high_failure", "ts": "t"}
    assert _candidate_dedup_key(c1) != _candidate_dedup_key(c2)


def test_candidate_dedup_key_falls_back_to_first_seen() -> None:
    c = {"tool": "x", "gapType": "y", "first_seen": "2025-12-01"}
    key = _candidate_dedup_key(c)
    assert "2025-12-01" in key


# ===========================================================================
# _parse_skill_md
# ===========================================================================


def test_parse_skill_md_valid(tmp_path: Path) -> None:
    skill = tmp_path / "SKILL.md"
    skill.write_text(
        textwrap.dedent("""\
        ---
        name: auto-bash
        description: "Runs bash commands reliably"
        metadata:
          openclaw:
            category: high_failure
            aceforge:
              status: proposed
              candidate_occurrences: 7
              candidate_success_rate: 0.43
        ---

        # auto-bash
        """),
        encoding="utf-8",
    )
    name, meta = _parse_skill_md(skill)
    assert name == "auto-bash"
    assert meta["description"] == "Runs bash commands reliably"
    assert float(meta["candidate_success_rate"]) == pytest.approx(0.43)


def test_parse_skill_md_no_frontmatter(tmp_path: Path) -> None:
    skill = tmp_path / "SKILL.md"
    skill.write_text("# plain markdown\n", encoding="utf-8")
    name, meta = _parse_skill_md(skill)
    assert name is None
    assert meta == {}


def test_parse_skill_md_missing_file(tmp_path: Path) -> None:
    name, meta = _parse_skill_md(tmp_path / "ghost.md")
    assert name is None
    assert meta == {}


def test_parse_skill_md_uses_stem_as_fallback(tmp_path: Path) -> None:
    skill = tmp_path / "my-skill.md"
    skill.write_text("---\nstatus: proposed\n---\n# body\n", encoding="utf-8")
    name, meta = _parse_skill_md(skill)
    assert name == "my-skill"


# ===========================================================================
# _render_skill_md
# ===========================================================================


def test_render_skill_md_contains_frontmatter() -> None:
    disc = _make_discovery(validation_score=0.75)
    md = _render_skill_md(disc)
    assert md.startswith("---\n")
    assert "evoskill-" in md
    assert "validation_score: 0.75" in md


def test_render_skill_md_includes_source_code() -> None:
    tool = SynthesizedTool(
        tool_name="my_tool",
        description="does stuff",
        parameter_schema={},
        required_params=[],
        source_code="def my_tool(x): return x * 2",
        validation_passed=True,
    )
    disc = _make_discovery(synthesized_tool=tool, validation_score=0.9)
    md = _render_skill_md(disc)
    assert "```python" in md
    assert "def my_tool" in md


def test_render_skill_md_no_source_code_section_when_none() -> None:
    disc = _make_discovery(synthesized_tool=None, validation_score=0.9)
    md = _render_skill_md(disc)
    assert "```python" not in md


# ===========================================================================
# _gap_to_slug
# ===========================================================================


def test_gap_to_slug_basic() -> None:
    gap = _make_gap(description="Handle tool misuse failures efficiently")
    slug = _gap_to_slug(gap)
    assert slug == "handle-tool-misuse-failures-efficiently"


def test_gap_to_slug_strips_special_chars() -> None:
    gap = _make_gap(description="Fix: rate-limit (500 errors)!")
    slug = _gap_to_slug(gap)
    assert ":" not in slug
    assert "(" not in slug


def test_gap_to_slug_max_five_words() -> None:
    gap = _make_gap(description="one two three four five six seven eight")
    slug = _gap_to_slug(gap)
    assert len(slug.split("-")) == 5


def test_gap_to_slug_empty_description_falls_back_to_gap_type() -> None:
    gap = _make_gap(gap_type="planning_error", description="")
    slug = _gap_to_slug(gap)
    assert slug == "planning-error"


# ===========================================================================
# _success_rate_to_severity
# ===========================================================================


def test_success_rate_to_severity_inverts() -> None:
    assert _success_rate_to_severity(0.0) == pytest.approx(1.0)
    assert _success_rate_to_severity(1.0) == pytest.approx(0.0)
    assert _success_rate_to_severity(0.3) == pytest.approx(0.7)


def test_success_rate_to_severity_clamps() -> None:
    assert _success_rate_to_severity(-1.0) == pytest.approx(1.0)
    assert _success_rate_to_severity(2.0) == pytest.approx(0.0)


def test_success_rate_to_severity_bad_input() -> None:
    result = _success_rate_to_severity("not-a-number")
    assert 0.0 <= result <= 1.0


# ===========================================================================
# _build_examples_from_traces
# ===========================================================================


def test_build_examples_from_traces_basic() -> None:
    traces = [
        {"args_summary": "ls -la", "error": "permission denied"},
        {"args_summary": "rm -rf /", "result_summary": "blocked"},
    ]
    examples = _build_examples_from_traces(traces)
    assert len(examples) == 2
    assert examples[0]["args"]["input"] == "ls -la"
    assert examples[0]["expected"] is None


def test_build_examples_max_five() -> None:
    traces = [{"args_summary": f"cmd{i}"} for i in range(10)]
    assert len(_build_examples_from_traces(traces)) == 5


def test_build_examples_empty() -> None:
    assert _build_examples_from_traces([]) == []


# ===========================================================================
# _parse_float
# ===========================================================================


def test_parse_float_valid() -> None:
    assert _parse_float("0.75", default=0.0) == pytest.approx(0.75)
    assert _parse_float(3, default=0.0) == pytest.approx(3.0)


def test_parse_float_invalid_returns_default() -> None:
    assert _parse_float("nan-str", default=0.5) == pytest.approx(0.5)
    assert _parse_float(None, default=0.1) == pytest.approx(0.1)


# ===========================================================================
# pull_gaps_from_aceforge
# ===========================================================================


def test_pull_gaps_no_patterns_file(
    connector: AceForgeConnector,
) -> None:
    result = connector.pull_gaps_from_aceforge()
    assert result.imported_count == 0
    assert result.skipped_count == 0
    assert result.gaps == []


def test_pull_gaps_imports_valid_candidates(
    connector: AceForgeConnector, forge_dir: Path
) -> None:
    _write_patterns(
        forge_dir,
        [
            _gap_candidate("bash", "high_failure", severity=15.0),
            _gap_candidate("python", "retry_storm", severity=8.0),
        ],
    )
    result = connector.pull_gaps_from_aceforge()
    assert result.imported_count == 2
    assert result.skipped_count == 0
    assert all(g.gap_type in ("tool_misuse", "context_loss") for g in result.gaps)


def test_pull_gaps_skips_below_severity_threshold(
    connector: AceForgeConnector, forge_dir: Path
) -> None:
    _write_patterns(
        forge_dir,
        [_gap_candidate("bash", severity=1.0)],  # below min=3.0
    )
    result = connector.pull_gaps_from_aceforge()
    assert result.imported_count == 0
    assert result.skipped_count == 1


def test_pull_gaps_deduplicates_across_calls(
    connector: AceForgeConnector, forge_dir: Path
) -> None:
    candidate = _gap_candidate("bash", ts="2026-01-01T00:00:00Z")
    _write_patterns(forge_dir, [candidate])
    first = connector.pull_gaps_from_aceforge()
    second = connector.pull_gaps_from_aceforge()
    assert first.imported_count == 1
    assert second.imported_count == 0
    assert second.skipped_count == 1


def test_pull_gaps_skips_non_gap_entries(
    connector: AceForgeConnector, forge_dir: Path
) -> None:
    _write_patterns(
        forge_dir,
        [
            {"tool": "x", "type": "correction", "ts": "2026-01-01"},
            _gap_candidate("bash", severity=10.0),
        ],
    )
    result = connector.pull_gaps_from_aceforge()
    assert result.imported_count == 1


def test_pull_gaps_normalises_severity(
    connector: AceForgeConnector, forge_dir: Path
) -> None:
    _write_patterns(forge_dir, [_gap_candidate("bash", severity=50.0)])
    result = connector.pull_gaps_from_aceforge()
    gap = result.gaps[0]
    assert gap.severity == pytest.approx(0.5)  # 50 / 100


def test_pull_gaps_maps_gap_type(
    connector: AceForgeConnector, forge_dir: Path
) -> None:
    _write_patterns(forge_dir, [_gap_candidate("bash", "chain_break", severity=10.0)])
    result = connector.pull_gaps_from_aceforge()
    assert result.gaps[0].gap_type == "planning_error"


def test_pull_gaps_no_description_skips(
    connector: AceForgeConnector, forge_dir: Path
) -> None:
    bad = {
        "gapType": "high_failure",
        "severity": 20.0,
        "suggestedFocus": "",
        "evidence": [],
        "tool": "",
        "ts": "2026-02-01",
    }
    _write_patterns(forge_dir, [bad])
    result = connector.pull_gaps_from_aceforge()
    assert result.imported_count == 0
    assert result.skipped_count == 1


# ===========================================================================
# import_aceforge_skill
# ===========================================================================


def test_import_aceforge_skill_valid(
    connector: AceForgeConnector, tmp_path: Path
) -> None:
    skill = tmp_path / "SKILL.md"
    skill.write_text(
        textwrap.dedent("""\
        ---
        name: auto-bash
        description: "Run bash safely"
        metadata:
          openclaw:
            category: high_failure
            aceforge:
              status: proposed
              candidate_occurrences: 5
              candidate_success_rate: 0.4
        ---
        # body
        """),
        encoding="utf-8",
    )
    disc = connector.import_aceforge_skill(skill)
    assert disc is not None
    assert disc.gap.description == "Run bash safely"
    assert disc.validation_score == pytest.approx(0.4)
    assert disc.synthesized_tool is None


def test_import_aceforge_skill_deduplicates(
    connector: AceForgeConnector, tmp_path: Path
) -> None:
    skill = tmp_path / "SKILL.md"
    skill.write_text(
        "---\nname: auto-bash\ndescription: desc\n---\n# body\n",
        encoding="utf-8",
    )
    first = connector.import_aceforge_skill(skill)
    second = connector.import_aceforge_skill(skill)
    assert first is not None
    assert second is None


def test_import_aceforge_skill_missing_file(
    connector: AceForgeConnector, tmp_path: Path
) -> None:
    result = connector.import_aceforge_skill(tmp_path / "ghost.md")
    assert result is None


def test_import_aceforge_skill_no_frontmatter(
    connector: AceForgeConnector, tmp_path: Path
) -> None:
    skill = tmp_path / "plain.md"
    skill.write_text("# plain markdown\n", encoding="utf-8")
    assert connector.import_aceforge_skill(skill) is None


# ===========================================================================
# poll_evolution_requests
# ===========================================================================


def test_poll_evolution_requests_reads_and_clears(
    connector: AceForgeConnector, forge_dir: Path
) -> None:
    trigger = forge_dir / "evoskill_trigger.json"
    trigger.write_text(
        json.dumps({"source": "aceforge", "tools": ["bash", "python"]}),
        encoding="utf-8",
    )
    tools = connector.poll_evolution_requests()
    assert tools == ["bash", "python"]
    assert not trigger.exists()


def test_poll_evolution_requests_no_file(
    connector: AceForgeConnector,
) -> None:
    assert connector.poll_evolution_requests() == []


def test_poll_evolution_requests_dry_run_preserves_file(
    mock_pipeline: MagicMock, connector_config: AceForgeConnectorConfig,
    forge_dir: Path
) -> None:
    connector_config.dry_run = True
    conn = AceForgeConnector(pipeline=mock_pipeline, config=connector_config)
    trigger = forge_dir / "evoskill_trigger.json"
    trigger.write_text(json.dumps({"tools": ["bash"]}), encoding="utf-8")
    tools = conn.poll_evolution_requests()
    assert tools == ["bash"]
    assert trigger.exists()  # not deleted in dry_run


def test_poll_evolution_requests_scalar_tool(
    connector: AceForgeConnector, forge_dir: Path
) -> None:
    trigger = forge_dir / "evoskill_trigger.json"
    trigger.write_text(json.dumps({"tools": "bash"}), encoding="utf-8")
    tools = connector.poll_evolution_requests()
    assert tools == ["bash"]


def test_poll_evolution_requests_malformed_json(
    connector: AceForgeConnector, forge_dir: Path
) -> None:
    trigger = forge_dir / "evoskill_trigger.json"
    trigger.write_text("not-json", encoding="utf-8")
    tools = connector.poll_evolution_requests()
    assert tools == []


# ===========================================================================
# push_gap_to_aceforge
# ===========================================================================


def test_push_gap_to_aceforge_appends_jsonl(
    connector: AceForgeConnector, forge_dir: Path
) -> None:
    gap = _make_gap("tool_misuse", "bash fails frequently")
    path = connector.push_gap_to_aceforge(gap)
    assert path is not None
    records = _read_jsonl(path)
    assert len(records) == 1
    assert records[0]["source"] == "evoskill"
    assert records[0]["gapType"] == "high_failure"


def test_push_gap_to_aceforge_appends_multiple(
    connector: AceForgeConnector, forge_dir: Path
) -> None:
    connector.push_gap_to_aceforge(_make_gap("tool_misuse", "gap1"))
    connector.push_gap_to_aceforge(_make_gap("planning_error", "gap2"))
    out = forge_dir / "evoskill_gaps.jsonl"
    records = _read_jsonl(out)
    assert len(records) == 2


def test_push_gap_dry_run_writes_nothing(
    mock_pipeline: MagicMock, connector_config: AceForgeConnectorConfig,
    forge_dir: Path
) -> None:
    connector_config.dry_run = True
    conn = AceForgeConnector(pipeline=mock_pipeline, config=connector_config)
    result = conn.push_gap_to_aceforge(_make_gap())
    assert result is None
    assert not (forge_dir / "evoskill_gaps.jsonl").exists()


def test_push_gap_translates_planning_error(
    connector: AceForgeConnector, forge_dir: Path
) -> None:
    connector.push_gap_to_aceforge(_make_gap("planning_error", "chain breaks"))
    records = _read_jsonl(forge_dir / "evoskill_gaps.jsonl")
    assert records[0]["gapType"] == "chain_break"


# ===========================================================================
# export_discovery_as_skill_md
# ===========================================================================


def test_export_discovery_as_skill_md_writes_file(
    connector: AceForgeConnector, forge_dir: Path
) -> None:
    disc = _make_discovery(validation_score=0.9)
    path = connector.export_discovery_as_skill_md(disc)
    assert path is not None
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "evoskill-" in content
    assert "proposed" in content


def test_export_discovery_skips_below_threshold(
    connector: AceForgeConnector,
) -> None:
    disc = _make_discovery(validation_score=0.4)  # below 0.6
    path = connector.export_discovery_as_skill_md(disc)
    assert path is None


def test_export_discovery_dry_run_no_file(
    mock_pipeline: MagicMock, connector_config: AceForgeConnectorConfig,
    forge_dir: Path
) -> None:
    connector_config.dry_run = True
    conn = AceForgeConnector(pipeline=mock_pipeline, config=connector_config)
    disc = _make_discovery(validation_score=0.95)
    path = conn.export_discovery_as_skill_md(disc)
    assert path is None
    proposals = forge_dir / "proposals"
    assert not proposals.exists()


def test_export_discovery_uses_tool_name(
    connector: AceForgeConnector,
) -> None:
    tool = SynthesizedTool(
        tool_name="my_custom_tool",
        description="does stuff",
        parameter_schema={},
        required_params=[],
        source_code="def my_custom_tool(): pass",
        validation_passed=True,
    )
    disc = _make_discovery(synthesized_tool=tool, validation_score=0.8)
    path = connector.export_discovery_as_skill_md(disc)
    assert path is not None
    assert "my_custom_tool" in path.name


def test_export_discovery_includes_source_code(
    connector: AceForgeConnector,
) -> None:
    tool = SynthesizedTool(
        tool_name="inspect_tool",
        description="inspects",
        parameter_schema={},
        required_params=[],
        source_code="def inspect_tool(x): return x",
        validation_passed=True,
    )
    disc = _make_discovery(synthesized_tool=tool, validation_score=0.85)
    path = connector.export_discovery_as_skill_md(disc)
    assert path is not None
    content = path.read_text(encoding="utf-8")
    assert "def inspect_tool" in content


# ===========================================================================
# request_aceforge_evolution
# ===========================================================================


def test_request_aceforge_evolution_writes_trigger(
    connector: AceForgeConnector, forge_dir: Path
) -> None:
    result = connector.request_aceforge_evolution(["bash", "python"])
    assert result is True
    trigger = forge_dir / "evoskill_trigger.json"
    assert trigger.exists()
    data = json.loads(trigger.read_text(encoding="utf-8"))
    assert data["tools"] == ["bash", "python"]
    assert data["source"] == "evoskill"


def test_request_aceforge_evolution_empty_returns_false(
    connector: AceForgeConnector,
) -> None:
    assert connector.request_aceforge_evolution([]) is False


def test_request_aceforge_evolution_dry_run_no_file(
    mock_pipeline: MagicMock, connector_config: AceForgeConnectorConfig,
    forge_dir: Path
) -> None:
    connector_config.dry_run = True
    conn = AceForgeConnector(pipeline=mock_pipeline, config=connector_config)
    result = conn.request_aceforge_evolution(["bash"])
    assert result is True
    assert not (forge_dir / "evoskill_trigger.json").exists()


# ===========================================================================
# run_sync — bidirectional cycle
# ===========================================================================


def test_run_sync_empty_forge_dir(connector: AceForgeConnector) -> None:
    result = connector.run_sync()
    assert isinstance(result, SyncResult)
    assert result.aceforge_to_evoskill.imported_count == 0
    assert result.evoskill_to_aceforge.exported_count == 0
    assert result.discoveries_triggered == []


def test_run_sync_imports_and_synthesises(
    connector: AceForgeConnector,
    mock_pipeline: MagicMock,
    forge_dir: Path,
) -> None:
    _write_patterns(forge_dir, [_gap_candidate("bash", severity=20.0)])
    result = connector.run_sync()
    assert result.aceforge_to_evoskill.imported_count == 1
    mock_pipeline.synthesize_skill.assert_called_once()
    assert len(result.discoveries_triggered) == 1


def test_run_sync_exports_pending_gaps(
    connector: AceForgeConnector,
    mock_pipeline: MagicMock,
    forge_dir: Path,
) -> None:
    pending_gap = _make_gap("tool_misuse", "pending gap")
    pending_disc = _make_discovery(gap=pending_gap, deployed=False)
    mock_pipeline.discoveries = [pending_disc]

    result = connector.run_sync()
    assert result.evoskill_to_aceforge.exported_count == 1
    assert (forge_dir / "evoskill_gaps.jsonl").exists()


def test_run_sync_exports_skill_md_for_qualifying_discoveries(
    connector: AceForgeConnector,
    mock_pipeline: MagicMock,
    forge_dir: Path,
) -> None:
    _write_patterns(forge_dir, [_gap_candidate("bash", severity=15.0)])
    # Make synthesize return a high-score discovery
    mock_pipeline.synthesize_skill.return_value = _make_discovery(
        validation_score=0.9
    )
    result = connector.run_sync()
    assert len(result.skill_exports) == 1


def test_run_sync_skips_skill_md_below_threshold(
    connector: AceForgeConnector,
    mock_pipeline: MagicMock,
    forge_dir: Path,
) -> None:
    _write_patterns(forge_dir, [_gap_candidate("bash", severity=15.0)])
    mock_pipeline.synthesize_skill.return_value = _make_discovery(
        validation_score=0.3  # below threshold
    )
    result = connector.run_sync()
    assert len(result.skill_exports) == 0


def test_run_sync_handles_evolution_trigger(
    connector: AceForgeConnector,
    mock_pipeline: MagicMock,
    forge_dir: Path,
) -> None:
    trigger = forge_dir / "evoskill_trigger.json"
    trigger.write_text(
        json.dumps({"source": "aceforge", "tools": ["bash"]}),
        encoding="utf-8",
    )
    mock_pipeline.run_cycle.return_value = [_make_discovery()]
    result = connector.run_sync()
    mock_pipeline.run_cycle.assert_called_once()
    assert len(result.discoveries_triggered) == 1


def test_run_sync_auto_deploy_calls_deploy(
    mock_pipeline: MagicMock,
    connector_config: AceForgeConnectorConfig,
    forge_dir: Path,
) -> None:
    mock_pipeline._config = EvoSkillConfig(auto_deploy=True)
    tool = SynthesizedTool(
        tool_name="auto_tool",
        description="auto",
        parameter_schema={},
        required_params=[],
        source_code="def auto_tool(): pass",
        validation_passed=True,
    )
    discovery = _make_discovery(synthesized_tool=tool, validation_score=0.8)
    mock_pipeline.synthesize_skill.return_value = discovery

    conn = AceForgeConnector(pipeline=mock_pipeline, config=connector_config)
    _write_patterns(forge_dir, [_gap_candidate("bash", severity=20.0)])
    conn.run_sync()
    mock_pipeline.deploy_skill.assert_called_once_with(discovery)


def test_run_sync_result_has_timestamp(connector: AceForgeConnector) -> None:
    result = connector.run_sync()
    assert isinstance(result.timestamp, datetime)
    assert result.timestamp.tzinfo is not None


# ===========================================================================
# Translation symmetry
# ===========================================================================


def test_gap_round_trips_through_translation(
    connector: AceForgeConnector,
) -> None:
    original = _make_gap("context_loss", "agent loses context mid-task", frequency=5)
    candidate = connector._translate_gap_to_candidate(original)
    reconstructed = connector._translate_candidate_to_gap(candidate)
    assert reconstructed is not None
    assert reconstructed.description == original.description


def test_translate_candidate_all_gap_types(
    connector: AceForgeConnector,
) -> None:
    types = ["high_failure", "correction_cluster", "retry_storm", "chain_break"]
    for gt in types:
        c = _gap_candidate("tool", gap_type=gt, severity=10.0)
        gap = connector._translate_candidate_to_gap(c)
        assert gap is not None
        assert gap.gap_type in (
            "tool_misuse", "context_loss", "planning_error"
        )


def test_translate_gap_all_evoskill_types(
    connector: AceForgeConnector,
) -> None:
    evoskill_types = [
        "tool_misuse", "hallucination", "planning_error",
        "reasoning_error", "context_loss", "format_error", "other",
    ]
    for gt in evoskill_types:
        gap = _make_gap(gt, "some description")
        candidate = connector._translate_gap_to_candidate(gap)
        assert candidate["gapType"] in (
            "high_failure", "correction_cluster", "chain_break", "retry_storm"
        )


# ===========================================================================
# _ensure_dir
# ===========================================================================


def test_ensure_dir_creates_nested(tmp_path: Path) -> None:
    deep = tmp_path / "a" / "b" / "c"
    _ensure_dir(deep)
    assert deep.is_dir()


def test_ensure_dir_idempotent(tmp_path: Path) -> None:
    d = tmp_path / "existing"
    d.mkdir()
    _ensure_dir(d)  # should not raise
    assert d.is_dir()


# ===========================================================================
# AceForgeConnectorConfig defaults
# ===========================================================================


def test_connector_config_defaults() -> None:
    cfg = AceForgeConnectorConfig()
    assert cfg.forge_dir == Path.home() / ".openclaw" / "workspace" / ".forge"
    assert cfg.dry_run is False
    assert cfg.sync_interval_seconds == 30.0
    assert cfg.validation_threshold == pytest.approx(0.6)


# ===========================================================================
# __init__.py re-exports
# ===========================================================================


def test_module_exports() -> None:
    from evoagentx.integrations import (  # noqa: F401
        AceForgeConnector,
        AceForgeConnectorConfig,
        ExportResult,
        ImportResult,
        SyncResult,
    )

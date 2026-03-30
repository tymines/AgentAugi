"""Comprehensive tests for evoagentx.core.evoskill."""

import json
import os
import sys
import unittest
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from evoagentx.memory.mistake_notebook import MistakeCategory, MistakeEntry, MistakeNotebook
from evoagentx.core.tool_synthesizer import SynthesizedTool, ToolRegistry, ToolSynthesizer
from evoagentx.core.evoskill import (
    EvoSkillConfig,
    EvoSkillPipeline,
    SkillDiscovery,
    SkillGap,
    _heuristic_gap_description,
    _jaccard,
    _tokenize,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_DOUBLE_SOURCE = """\
def double_number(n):
    \"\"\"Return n * 2.\"\"\"
    return n * 2
"""

_DOUBLE_SPEC_JSON = json.dumps({
    "tool_name": "double_number",
    "description": "Returns double of input.",
    "parameter_schema": {"n": {"type": "integer", "description": "input"}},
    "required_params": ["n"],
    "source_code": _DOUBLE_SOURCE,
})


def _make_llm(content: str) -> MagicMock:
    llm = MagicMock()
    resp = MagicMock()
    resp.content = content
    llm.generate.return_value = resp
    return llm


def _make_entry(
    attempted: str = "call tool X",
    went_wrong: str = "tool X failed with wrong parameter",
    fix: str = "use correct parameter",
    category: MistakeCategory = MistakeCategory.TOOL_MISUSE,
    tags: Optional[List[str]] = None,
) -> MistakeEntry:
    return MistakeEntry(
        attempted=attempted,
        went_wrong=went_wrong,
        fix=fix,
        category=category,
        tags=tags or [],
    )


def _make_notebook_with_entries(
    n: int,
    category: MistakeCategory = MistakeCategory.TOOL_MISUSE,
    went_wrong_template: str = "tool X failed with wrong parameter value",
) -> MistakeNotebook:
    nb = MistakeNotebook()
    for i in range(n):
        nb.record(
            _make_entry(
                attempted=f"call tool X with param {i}",
                went_wrong=went_wrong_template,
                category=category,
            ),
            auto_save=False,
        )
    return nb


def _make_synthesizer(llm_content: str = _DOUBLE_SPEC_JSON) -> ToolSynthesizer:
    llm = _make_llm(llm_content)
    registry = ToolRegistry()
    return ToolSynthesizer(llm=llm, registry=registry, max_retries=1)


def _make_pipeline(
    nb: Optional[MistakeNotebook] = None,
    synth: Optional[ToolSynthesizer] = None,
    config: Optional[EvoSkillConfig] = None,
    llm_fn=None,
) -> EvoSkillPipeline:
    nb = nb or MistakeNotebook()
    synth = synth or _make_synthesizer()
    config = config or EvoSkillConfig(min_mistake_frequency=3, auto_deploy=False)
    return EvoSkillPipeline(
        mistake_notebook=nb,
        tool_synthesizer=synth,
        config=config,
        llm_fn=llm_fn,
    )


# ---------------------------------------------------------------------------
# SkillGap dataclass
# ---------------------------------------------------------------------------

class TestSkillGap(unittest.TestCase):

    def test_basic_creation(self):
        gap = SkillGap(
            gap_type="tool_misuse",
            description="A tool for correct parameter validation",
            source_mistakes=["id1", "id2"],
            frequency=2,
            severity=0.8,
        )
        self.assertEqual(gap.gap_type, "tool_misuse")
        self.assertEqual(gap.frequency, 2)
        self.assertAlmostEqual(gap.severity, 0.8)
        self.assertEqual(gap.source_mistakes, ["id1", "id2"])
        self.assertEqual(gap.suggested_tool_spec, {})

    def test_suggested_tool_spec_populated(self):
        spec = {"task_description": "validate parameters", "examples": []}
        gap = SkillGap(
            gap_type="format_error",
            description="validate output format",
            source_mistakes=[],
            frequency=5,
            severity=0.3,
            suggested_tool_spec=spec,
        )
        self.assertEqual(gap.suggested_tool_spec, spec)

    def test_severity_range(self):
        for sev in (0.0, 0.5, 1.0):
            gap = SkillGap(
                gap_type="other", description="d", source_mistakes=[],
                frequency=1, severity=sev,
            )
            self.assertGreaterEqual(gap.severity, 0.0)
            self.assertLessEqual(gap.severity, 1.0)


# ---------------------------------------------------------------------------
# SkillDiscovery dataclass
# ---------------------------------------------------------------------------

class TestSkillDiscovery(unittest.TestCase):

    def _make_gap(self) -> SkillGap:
        return SkillGap(
            gap_type="tool_misuse", description="d",
            source_mistakes=["x"], frequency=3, severity=0.8,
        )

    def test_basic_creation(self):
        disc = SkillDiscovery(
            gap=self._make_gap(),
            synthesized_tool=None,
            validation_score=0.0,
            deployed=False,
        )
        self.assertFalse(disc.deployed)
        self.assertIsNone(disc.synthesized_tool)
        self.assertIsInstance(disc.created_at, datetime)

    def test_created_at_defaults_to_now(self):
        before = datetime.now()
        disc = SkillDiscovery(
            gap=self._make_gap(), synthesized_tool=None,
            validation_score=0.5, deployed=False,
        )
        after = datetime.now()
        self.assertGreaterEqual(disc.created_at, before)
        self.assertLessEqual(disc.created_at, after)

    def test_deployed_flag(self):
        disc = SkillDiscovery(
            gap=self._make_gap(), synthesized_tool=None,
            validation_score=1.0, deployed=True,
        )
        self.assertTrue(disc.deployed)


# ---------------------------------------------------------------------------
# EvoSkillConfig
# ---------------------------------------------------------------------------

class TestEvoSkillConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = EvoSkillConfig()
        self.assertEqual(cfg.min_mistake_frequency, 3)
        self.assertAlmostEqual(cfg.severity_threshold, 0.5)
        self.assertFalse(cfg.auto_deploy)
        self.assertEqual(cfg.analysis_interval, 10)
        self.assertEqual(cfg.max_pending_gaps, 20)

    def test_custom_values(self):
        cfg = EvoSkillConfig(
            min_mistake_frequency=5,
            severity_threshold=0.7,
            auto_deploy=True,
            analysis_interval=20,
            max_pending_gaps=10,
        )
        self.assertEqual(cfg.min_mistake_frequency, 5)
        self.assertTrue(cfg.auto_deploy)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestTokenizeAndJaccard(unittest.TestCase):

    def test_tokenize_basic(self):
        tokens = _tokenize("Tool failed with wrong parameter")
        self.assertIn("tool", tokens)
        self.assertIn("failed", tokens)
        self.assertIn("wrong", tokens)
        self.assertIn("parameter", tokens)

    def test_tokenize_filters_short(self):
        tokens = _tokenize("is a ok bad fine")
        # "is", "a", "ok" are too short (<3 chars); only "bad" and "fine" qualify
        for t in tokens:
            self.assertGreaterEqual(len(t), 3)

    def test_jaccard_identical(self):
        s = {"a", "b", "c"}
        self.assertAlmostEqual(_jaccard(s, s), 1.0)

    def test_jaccard_disjoint(self):
        self.assertAlmostEqual(_jaccard({"a", "b"}, {"c", "d"}), 0.0)

    def test_jaccard_partial(self):
        sim = _jaccard({"a", "b", "c"}, {"b", "c", "d"})
        # intersection=2, union=4 → 0.5
        self.assertAlmostEqual(sim, 0.5)

    def test_jaccard_both_empty(self):
        self.assertAlmostEqual(_jaccard(set(), set()), 1.0)

    def test_heuristic_gap_description(self):
        cluster = [
            _make_entry(went_wrong="parameter validation failed badly"),
            _make_entry(went_wrong="parameter validation failed completely"),
        ]
        desc = _heuristic_gap_description(cluster)
        self.assertIsInstance(desc, str)
        self.assertGreater(len(desc), 10)
        self.assertIn("tool_misuse", desc.lower().replace(" ", "_").replace(" ", "_"))


# ---------------------------------------------------------------------------
# Mistake clustering
# ---------------------------------------------------------------------------

class TestClusterMistakes(unittest.TestCase):

    def test_same_category_similar_text_clusters(self):
        pipeline = _make_pipeline()
        entries = [
            _make_entry(went_wrong="tool X wrong parameter value"),
            _make_entry(went_wrong="tool X wrong parameter type"),
            _make_entry(went_wrong="tool X wrong parameter missing"),
        ]
        clusters = pipeline._cluster_mistakes(entries)
        # All should end up in one cluster (same category, high overlap)
        sizes = sorted(len(c) for c in clusters)
        self.assertEqual(sum(sizes), 3)
        self.assertIn(3, sizes)

    def test_different_category_no_cluster(self):
        pipeline = _make_pipeline()
        entries = [
            _make_entry(went_wrong="network timeout error", category=MistakeCategory.TOOL_MISUSE),
            _make_entry(went_wrong="network timeout error", category=MistakeCategory.HALLUCINATION),
        ]
        clusters = pipeline._cluster_mistakes(entries)
        # Different categories → separate clusters
        self.assertEqual(len(clusters), 2)

    def test_empty_list(self):
        pipeline = _make_pipeline()
        self.assertEqual(pipeline._cluster_mistakes([]), [])

    def test_single_entry(self):
        pipeline = _make_pipeline()
        entry = _make_entry()
        clusters = pipeline._cluster_mistakes([entry])
        self.assertEqual(len(clusters), 1)
        self.assertEqual(len(clusters[0]), 1)


# ---------------------------------------------------------------------------
# Gap identification
# ---------------------------------------------------------------------------

class TestIdentifySkillGap(unittest.TestCase):

    def test_heuristic_gap_no_llm(self):
        pipeline = _make_pipeline(llm_fn=None)
        cluster = [
            _make_entry(went_wrong="tool X fails parameter validation"),
            _make_entry(went_wrong="tool X fails parameter validation error"),
            _make_entry(went_wrong="tool X fails parameter validation crash"),
        ]
        gap = pipeline._identify_skill_gap(cluster)
        self.assertIsNotNone(gap)
        self.assertEqual(gap.frequency, 3)
        self.assertIsInstance(gap.description, str)
        self.assertEqual(gap.gap_type, "tool_misuse")
        self.assertEqual(len(gap.source_mistakes), 3)

    def test_gap_with_llm_fn(self):
        llm_resp = MagicMock()
        llm_resp.content = "A tool that validates function parameters before execution."

        def mock_llm(prompt: str):
            return llm_resp

        pipeline = _make_pipeline(llm_fn=mock_llm)
        cluster = [_make_entry() for _ in range(3)]
        gap = pipeline._identify_skill_gap(cluster)
        self.assertIsNotNone(gap)
        self.assertIn("validates", gap.description)

    def test_severity_set_correctly(self):
        pipeline = _make_pipeline()
        cluster = [
            _make_entry(category=MistakeCategory.HALLUCINATION),
            _make_entry(category=MistakeCategory.HALLUCINATION),
        ]
        gap = pipeline._identify_skill_gap(cluster)
        self.assertIsNotNone(gap)
        # HALLUCINATION severity = 0.9
        self.assertAlmostEqual(gap.severity, 0.9)


# ---------------------------------------------------------------------------
# Tool spec generation
# ---------------------------------------------------------------------------

class TestGenerateToolSpec(unittest.TestCase):

    def test_heuristic_spec_no_llm(self):
        pipeline = _make_pipeline(llm_fn=None)
        gap = SkillGap(
            gap_type="tool_misuse",
            description="Validate function parameters",
            source_mistakes=[],
            frequency=3,
            severity=0.8,
        )
        spec = pipeline._generate_tool_spec(gap)
        self.assertIn("task_description", spec)
        self.assertEqual(spec["task_description"], gap.description)

    def test_llm_spec_generation(self):
        expected_spec = {
            "task_description": "Validate parameters before calling a tool",
            "examples": [{"args": {"n": 5}, "expected": True}],
        }
        llm_resp = MagicMock()
        llm_resp.content = json.dumps(expected_spec)

        def mock_llm(prompt: str):
            return llm_resp

        pipeline = _make_pipeline(llm_fn=mock_llm)
        gap = SkillGap(
            gap_type="tool_misuse",
            description="Validate parameters",
            source_mistakes=[],
            frequency=3,
            severity=0.8,
        )
        spec = pipeline._generate_tool_spec(gap)
        self.assertEqual(spec["task_description"], expected_spec["task_description"])
        self.assertEqual(spec["examples"], expected_spec["examples"])

    def test_llm_spec_fallback_on_bad_json(self):
        llm_resp = MagicMock()
        llm_resp.content = "not valid json at all"

        def mock_llm(prompt: str):
            return llm_resp

        pipeline = _make_pipeline(llm_fn=mock_llm)
        gap = SkillGap(
            gap_type="other",
            description="Do something",
            source_mistakes=[],
            frequency=3,
            severity=0.6,
        )
        spec = pipeline._generate_tool_spec(gap)
        # Should fall back gracefully
        self.assertIn("task_description", spec)


# ---------------------------------------------------------------------------
# Frequency and severity thresholds
# ---------------------------------------------------------------------------

class TestThresholds(unittest.TestCase):

    def test_below_frequency_threshold_no_gaps(self):
        """Only 2 mistakes when threshold is 3 → no gaps."""
        nb = _make_notebook_with_entries(2)
        pipeline = _make_pipeline(
            nb=nb,
            config=EvoSkillConfig(min_mistake_frequency=3, severity_threshold=0.0),
        )
        gaps = pipeline.analyze_failures()
        self.assertEqual(gaps, [])

    def test_at_frequency_threshold_gap_found(self):
        """Exactly 3 similar mistakes with threshold=3 → gap found."""
        nb = _make_notebook_with_entries(
            3,
            category=MistakeCategory.TOOL_MISUSE,
            went_wrong_template="tool X fails with wrong parameter type error",
        )
        pipeline = _make_pipeline(
            nb=nb,
            config=EvoSkillConfig(min_mistake_frequency=3, severity_threshold=0.0),
        )
        gaps = pipeline.analyze_failures()
        self.assertGreater(len(gaps), 0)

    def test_below_severity_threshold_no_gaps(self):
        """FORMAT_ERROR has severity 0.3; threshold 0.5 → no gaps."""
        nb = _make_notebook_with_entries(
            5,
            category=MistakeCategory.FORMAT_ERROR,
            went_wrong_template="output format was wrong completely invalid",
        )
        pipeline = _make_pipeline(
            nb=nb,
            config=EvoSkillConfig(min_mistake_frequency=3, severity_threshold=0.5),
        )
        gaps = pipeline.analyze_failures()
        self.assertEqual(gaps, [])

    def test_high_severity_category_passes_threshold(self):
        """HALLUCINATION severity 0.9 > threshold 0.5 → gap found."""
        nb = _make_notebook_with_entries(
            4,
            category=MistakeCategory.HALLUCINATION,
            went_wrong_template="agent hallucinated wrong information completely fabricated",
        )
        pipeline = _make_pipeline(
            nb=nb,
            config=EvoSkillConfig(min_mistake_frequency=3, severity_threshold=0.5),
        )
        gaps = pipeline.analyze_failures()
        self.assertGreater(len(gaps), 0)

    def test_empty_notebook_returns_no_gaps(self):
        pipeline = _make_pipeline(nb=MistakeNotebook())
        self.assertEqual(pipeline.analyze_failures(), [])


# ---------------------------------------------------------------------------
# Validate skill
# ---------------------------------------------------------------------------

class TestValidateSkill(unittest.TestCase):

    def _make_double_tool(self) -> SynthesizedTool:
        from evoagentx.core.tool_synthesizer import _compile_function
        fn = _compile_function(_DOUBLE_SOURCE, "double_number")
        return SynthesizedTool(
            tool_name="double_number",
            description="double",
            parameter_schema={},
            required_params=["n"],
            source_code=_DOUBLE_SOURCE,
            callable_fn=fn,
            validation_passed=True,
        )

    def test_validate_all_pass(self):
        pipeline = _make_pipeline()
        tool = self._make_double_tool()
        examples = [
            {"args": {"n": 2}, "expected": 4},
            {"args": {"n": 5}, "expected": 10},
        ]
        score = pipeline.validate_skill_tool(tool, examples)
        self.assertAlmostEqual(score, 1.0)

    def test_validate_partial_fail(self):
        pipeline = _make_pipeline()
        tool = self._make_double_tool()
        examples = [
            {"args": {"n": 2}, "expected": 4},   # pass
            {"args": {"n": 5}, "expected": 99},   # fail (expects 10)
        ]
        score = pipeline.validate_skill_tool(tool, examples)
        self.assertAlmostEqual(score, 0.5)

    def test_validate_no_examples_returns_one(self):
        pipeline = _make_pipeline()
        tool = self._make_double_tool()
        score = pipeline.validate_skill_tool(tool, [])
        self.assertAlmostEqual(score, 1.0)

    def test_validate_discovery_no_tool(self):
        pipeline = _make_pipeline()
        gap = SkillGap(
            gap_type="other", description="d", source_mistakes=[],
            frequency=3, severity=0.6,
        )
        disc = SkillDiscovery(gap=gap, synthesized_tool=None,
                              validation_score=0.5, deployed=False)
        score = pipeline.validate_skill(disc)
        self.assertAlmostEqual(score, 0.0)


# ---------------------------------------------------------------------------
# Auto-deploy vs manual deploy
# ---------------------------------------------------------------------------

class TestDeploySkill(unittest.TestCase):

    def _make_discovery_with_tool(self, nb: MistakeNotebook) -> SkillDiscovery:
        from evoagentx.core.tool_synthesizer import _compile_function
        fn = _compile_function(_DOUBLE_SOURCE, "double_number")
        tool = SynthesizedTool(
            tool_name="double_number",
            description="double",
            parameter_schema={},
            required_params=["n"],
            source_code=_DOUBLE_SOURCE,
            callable_fn=fn,
            validation_passed=True,
        )
        entry = _make_entry()
        nb.record(entry, auto_save=False)
        gap = SkillGap(
            gap_type="tool_misuse",
            description="double numbers",
            source_mistakes=[entry.entry_id],
            frequency=1,
            severity=0.8,
        )
        return SkillDiscovery(gap=gap, synthesized_tool=tool,
                              validation_score=1.0, deployed=False)

    def test_deploy_resolves_source_mistakes(self):
        nb = MistakeNotebook()
        pipeline = _make_pipeline(nb=nb)
        disc = self._make_discovery_with_tool(nb)
        mistake_id = disc.gap.source_mistakes[0]

        # Mistake is unresolved before deploy
        entry = next(e for e in nb.entries if e.entry_id == mistake_id)
        self.assertFalse(entry.resolved)

        pipeline.deploy_skill(disc)

        self.assertTrue(entry.resolved)

    def test_deploy_marks_discovery_deployed(self):
        nb = MistakeNotebook()
        pipeline = _make_pipeline(nb=nb)
        disc = self._make_discovery_with_tool(nb)
        pipeline.deploy_skill(disc)
        self.assertTrue(disc.deployed)

    def test_deploy_registers_tool(self):
        nb = MistakeNotebook()
        synth = _make_synthesizer()
        pipeline = _make_pipeline(nb=nb, synth=synth)
        disc = self._make_discovery_with_tool(nb)
        pipeline.deploy_skill(disc)
        self.assertTrue(synth._registry.has("double_number"))

    def test_deploy_no_tool_returns_false(self):
        nb = MistakeNotebook()
        pipeline = _make_pipeline(nb=nb)
        gap = SkillGap(
            gap_type="other", description="d", source_mistakes=[],
            frequency=3, severity=0.6,
        )
        disc = SkillDiscovery(gap=gap, synthesized_tool=None,
                              validation_score=0.0, deployed=False)
        result = pipeline.deploy_skill(disc)
        self.assertFalse(result)
        self.assertFalse(disc.deployed)

    def test_auto_deploy_in_run_cycle(self):
        """auto_deploy=True should deploy immediately after synthesis."""
        nb = _make_notebook_with_entries(
            4,
            category=MistakeCategory.TOOL_MISUSE,
            went_wrong_template="tool X wrong parameter type error invalid",
        )
        synth = _make_synthesizer(_DOUBLE_SPEC_JSON)
        config = EvoSkillConfig(
            min_mistake_frequency=3,
            severity_threshold=0.0,
            auto_deploy=True,
        )
        pipeline = EvoSkillPipeline(
            mistake_notebook=nb,
            tool_synthesizer=synth,
            config=config,
            llm_fn=None,
        )
        discoveries = pipeline.run_cycle()
        deployed = [d for d in discoveries if d.deployed]
        self.assertGreater(len(deployed), 0)


# ---------------------------------------------------------------------------
# Stats tracking
# ---------------------------------------------------------------------------

class TestStats(unittest.TestCase):

    def test_initial_stats_zeros(self):
        pipeline = _make_pipeline()
        s = pipeline.stats()
        self.assertEqual(s["gaps_found"], 0)
        self.assertEqual(s["skills_synthesized"], 0)
        self.assertEqual(s["deployed"], 0)
        self.assertAlmostEqual(s["validation_pass_rate"], 0.0)
        self.assertEqual(s["mistakes_resolved"], 0)

    def test_stats_after_failed_synthesis(self):
        """A discovery with no tool still counts as a gap found."""
        pipeline = _make_pipeline()
        gap = SkillGap(
            gap_type="other", description="d", source_mistakes=[],
            frequency=3, severity=0.6,
        )
        disc = SkillDiscovery(gap=gap, synthesized_tool=None,
                              validation_score=0.0, deployed=False)
        pipeline._add_discovery(disc)
        s = pipeline.stats()
        self.assertEqual(s["gaps_found"], 1)
        self.assertEqual(s["skills_synthesized"], 0)

    def test_mistakes_resolved_counts_deployed(self):
        from evoagentx.core.tool_synthesizer import _compile_function
        nb = MistakeNotebook()
        fn = _compile_function(_DOUBLE_SOURCE, "double_number")
        tool = SynthesizedTool(
            tool_name="double_number",
            description="double",
            parameter_schema={},
            required_params=["n"],
            source_code=_DOUBLE_SOURCE,
            callable_fn=fn,
            validation_passed=True,
        )
        entries = [_make_entry() for _ in range(3)]
        for e in entries:
            nb.record(e, auto_save=False)
        gap = SkillGap(
            gap_type="tool_misuse",
            description="d",
            source_mistakes=[e.entry_id for e in entries],
            frequency=3,
            severity=0.8,
        )
        pipeline = _make_pipeline(nb=nb)
        disc = SkillDiscovery(gap=gap, synthesized_tool=tool,
                              validation_score=1.0, deployed=False)
        pipeline.deploy_skill(disc)
        pipeline._add_discovery(disc)
        s = pipeline.stats()
        self.assertEqual(s["mistakes_resolved"], 3)
        self.assertEqual(s["deployed"], 1)


# ---------------------------------------------------------------------------
# Auto-interval triggering
# ---------------------------------------------------------------------------

class TestAutoInterval(unittest.TestCase):

    def test_no_cycle_below_interval(self):
        pipeline = _make_pipeline(
            config=EvoSkillConfig(analysis_interval=5)
        )
        for _ in range(4):
            result = pipeline.notify_mistake_recorded()
            self.assertIsNone(result)

    def test_cycle_triggered_at_interval(self):
        nb = _make_notebook_with_entries(
            5,
            went_wrong_template="tool failed with wrong parameter type error",
        )
        pipeline = _make_pipeline(
            nb=nb,
            config=EvoSkillConfig(
                analysis_interval=3,
                min_mistake_frequency=3,
                severity_threshold=0.0,
            ),
        )
        # Simulate recording 3 new mistakes
        for i in range(2):
            r = pipeline.notify_mistake_recorded()
            self.assertIsNone(r)
        r = pipeline.notify_mistake_recorded()
        # Third call should trigger the cycle (returns a list, even if empty)
        self.assertIsNotNone(r)
        self.assertIsInstance(r, list)

    def test_interval_zero_never_auto_triggers(self):
        nb = _make_notebook_with_entries(20)
        pipeline = _make_pipeline(
            nb=nb,
            config=EvoSkillConfig(analysis_interval=0),
        )
        for _ in range(25):
            result = pipeline.notify_mistake_recorded()
            self.assertIsNone(result)


# ---------------------------------------------------------------------------
# max_pending_gaps eviction
# ---------------------------------------------------------------------------

class TestMaxPendingGaps(unittest.TestCase):

    def test_evicts_oldest_undeployed(self):
        pipeline = _make_pipeline(
            config=EvoSkillConfig(max_pending_gaps=2)
        )
        for i in range(4):
            gap = SkillGap(
                gap_type="other",
                description=f"gap {i}",
                source_mistakes=[],
                frequency=3,
                severity=0.6,
            )
            disc = SkillDiscovery(gap=gap, synthesized_tool=None,
                                  validation_score=0.0, deployed=False)
            pipeline._add_discovery(disc)

        undeployed = [d for d in pipeline.discoveries if not d.deployed]
        self.assertLessEqual(len(undeployed), 2)


# ---------------------------------------------------------------------------
# Integration: MistakeNotebook ↔ ToolSynthesizer
# ---------------------------------------------------------------------------

class TestIntegration(unittest.TestCase):

    def test_full_cycle_synthesizes_and_resolves(self):
        """End-to-end: enough mistakes → synthesize tool → resolve mistakes."""
        nb = _make_notebook_with_entries(
            4,
            category=MistakeCategory.TOOL_MISUSE,
            went_wrong_template="tool X fails with wrong parameter type error badly",
        )
        synth = _make_synthesizer(_DOUBLE_SPEC_JSON)
        config = EvoSkillConfig(
            min_mistake_frequency=3,
            severity_threshold=0.0,
            auto_deploy=True,
        )
        pipeline = EvoSkillPipeline(
            mistake_notebook=nb,
            tool_synthesizer=synth,
            config=config,
            llm_fn=None,
        )
        discoveries = pipeline.run_cycle()
        self.assertIsInstance(discoveries, list)

        if discoveries:
            deployed = [d for d in discoveries if d.deployed]
            self.assertGreater(len(deployed), 0)
            # Source mistakes should be resolved
            for disc in deployed:
                for mid in disc.gap.source_mistakes:
                    entry = next((e for e in nb.entries if e.entry_id == mid), None)
                    if entry:
                        self.assertTrue(entry.resolved)

    def test_insufficient_mistakes_no_discoveries(self):
        """Only 1 mistake with threshold=3 → no discoveries."""
        nb = _make_notebook_with_entries(1)
        pipeline = _make_pipeline(nb=nb)
        discoveries = pipeline.run_cycle()
        self.assertEqual(discoveries, [])

    def test_synthesizer_failure_produces_none_tool_discovery(self):
        """Synthesis failure produces SkillDiscovery with synthesized_tool=None."""
        nb = _make_notebook_with_entries(
            4,
            went_wrong_template="tool X fails with wrong parameter type error badly",
        )
        # LLM returns garbage → synthesizer will fail
        bad_synth = _make_synthesizer("not valid json at all !!!")
        config = EvoSkillConfig(
            min_mistake_frequency=3,
            severity_threshold=0.0,
            auto_deploy=False,
        )
        pipeline = EvoSkillPipeline(
            mistake_notebook=nb,
            tool_synthesizer=bad_synth,
            config=config,
        )
        discoveries = pipeline.run_cycle()
        for disc in discoveries:
            self.assertIsNone(disc.synthesized_tool)
            self.assertFalse(disc.deployed)


if __name__ == "__main__":
    unittest.main()

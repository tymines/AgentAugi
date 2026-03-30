"""Unit tests for evoagentx.core.design_archive."""

import json
import os
import tempfile
import unittest
from copy import deepcopy
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

from evoagentx.core.design_archive import (
    DesignArchive,
    DesignEntry,
    PerformanceProfile,
    _structural_similarity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entry(
    config: Dict[str, Any],
    score: float = 0.7,
    task_type: str = "qa",
    success: bool = True,
    source: str = "manual",
    rationale: str = "test design",
) -> DesignEntry:
    return DesignEntry(
        config=config,
        performance=PerformanceProfile(
            scores={task_type: score},
            n_trials={task_type: 1},
        ),
        success=success,
        task_types=[task_type],
        source=source,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# PerformanceProfile
# ---------------------------------------------------------------------------

class TestPerformanceProfile(unittest.TestCase):

    def test_mean_score_empty(self):
        p = PerformanceProfile()
        self.assertAlmostEqual(p.mean_score(), 0.0)

    def test_mean_score(self):
        p = PerformanceProfile(scores={"qa": 0.8, "coding": 0.6})
        self.assertAlmostEqual(p.mean_score(), 0.7)

    def test_score_for_missing_task(self):
        p = PerformanceProfile(scores={"qa": 0.9})
        self.assertAlmostEqual(p.score_for("coding"), 0.0)

    def test_update_running_mean(self):
        p = PerformanceProfile()
        p.update("qa", 0.8, n=1)
        p.update("qa", 0.4, n=1)
        # Running mean: (0.8 * 1 + 0.4 * 1) / 2 = 0.6
        self.assertAlmostEqual(p.score_for("qa"), 0.6)

    def test_update_weighted_by_n(self):
        p = PerformanceProfile()
        p.update("qa", 0.8, n=3)
        p.update("qa", 0.2, n=1)
        # Running mean: (0.8 * 3 + 0.2 * 1) / 4 = 0.65
        self.assertAlmostEqual(p.score_for("qa"), 0.65)


# ---------------------------------------------------------------------------
# DesignEntry serialization
# ---------------------------------------------------------------------------

class TestDesignEntrySerialization(unittest.TestCase):

    def test_to_dict_round_trip(self):
        entry = _make_entry({"agent_type": "ReAct", "max_steps": 5})
        d = entry.to_dict()
        restored = DesignEntry.from_dict(d)
        self.assertEqual(restored.config, entry.config)
        self.assertEqual(restored.success, entry.success)
        self.assertAlmostEqual(
            restored.performance.score_for("qa"),
            entry.performance.score_for("qa"),
        )

    def test_design_id_preserved(self):
        entry = _make_entry({"a": "b"})
        d = entry.to_dict()
        restored = DesignEntry.from_dict(d)
        self.assertEqual(restored.design_id, entry.design_id)


# ---------------------------------------------------------------------------
# Structural similarity
# ---------------------------------------------------------------------------

class TestStructuralSimilarity(unittest.TestCase):

    def test_identical_configs(self):
        cfg = {"a": 1, "b": "x"}
        self.assertAlmostEqual(_structural_similarity(cfg, cfg), 1.0)

    def test_completely_different_keys(self):
        self.assertAlmostEqual(_structural_similarity({"a": 1}, {"b": 2}), 0.0)

    def test_same_key_different_value(self):
        sim = _structural_similarity({"a": 1}, {"a": 2})
        # Same key, different value → partial credit (0.3)
        self.assertAlmostEqual(sim, 0.3)

    def test_partial_overlap(self):
        cfg_a = {"a": 1, "b": 2, "c": 3}
        cfg_b = {"a": 1, "b": 99, "d": 4}
        # a matches (1.0), b same key diff val (0.3), c/d no match (0.0)
        # all_keys = {a, b, c, d} -> 4 keys
        # score = (1.0 + 0.3 + 0.0 + 0.0) / 4 = 0.325
        sim = _structural_similarity(cfg_a, cfg_b)
        self.assertAlmostEqual(sim, 0.325)

    def test_empty_configs(self):
        self.assertAlmostEqual(_structural_similarity({}, {}), 1.0)


# ---------------------------------------------------------------------------
# DesignArchive — add / deduplication
# ---------------------------------------------------------------------------

class TestDesignArchiveAdd(unittest.TestCase):

    def test_add_new_entry(self):
        archive = DesignArchive(similarity_threshold=0.9)
        entry = _make_entry({"agent_type": "CoT"})
        inserted = archive.add(entry)
        self.assertTrue(inserted)
        self.assertEqual(len(archive), 1)

    def test_duplicate_rejected(self):
        archive = DesignArchive(similarity_threshold=0.9)
        cfg = {"agent_type": "ReAct", "max_steps": 5}
        entry1 = _make_entry(cfg)
        entry2 = _make_entry(deepcopy(cfg))
        archive.add(entry1)
        inserted = archive.add(entry2)
        self.assertFalse(inserted)
        self.assertEqual(len(archive), 1)

    def test_duplicate_updates_performance(self):
        archive = DesignArchive(similarity_threshold=0.9)
        cfg = {"agent_type": "CoT"}
        entry1 = _make_entry(cfg, score=0.6)
        entry2 = _make_entry(deepcopy(cfg), score=0.8)
        archive.add(entry1)
        archive.add(entry2)
        # Running mean should be updated
        stored = list(archive)[0]
        self.assertAlmostEqual(stored.performance.score_for("qa"), 0.7)

    def test_dissimilar_entries_both_inserted(self):
        archive = DesignArchive(similarity_threshold=0.9)
        archive.add(_make_entry({"agent_type": "ReAct"}))
        archive.add(_make_entry({"agent_type": "CoT", "tools": ["search"]}))
        self.assertEqual(len(archive), 2)

    def test_invalid_threshold_raises(self):
        with self.assertRaises(ValueError):
            DesignArchive(similarity_threshold=0.0)


# ---------------------------------------------------------------------------
# DesignArchive — search
# ---------------------------------------------------------------------------

class TestDesignArchiveSearch(unittest.TestCase):

    def _populated_archive(self) -> DesignArchive:
        archive = DesignArchive(similarity_threshold=0.99)
        archive.add(_make_entry({"agent_type": "ReAct", "max_steps": 5}, score=0.9))
        archive.add(_make_entry({"agent_type": "CoT", "depth": 3}, score=0.7))
        archive.add(_make_entry({"agent_type": "Plan", "beam": 2}, score=0.3, success=False))
        return archive

    def test_search_returns_k_results(self):
        archive = self._populated_archive()
        results = archive.search({"agent_type": "ReAct"}, k=2)
        self.assertEqual(len(results), 2)

    def test_search_sorted_by_similarity(self):
        archive = self._populated_archive()
        results = archive.search({"agent_type": "ReAct", "max_steps": 5}, k=3)
        sims = [sim for _, sim in results]
        self.assertEqual(sims, sorted(sims, reverse=True))

    def test_search_exclude_failures(self):
        archive = self._populated_archive()
        results = archive.search({"agent_type": "Plan"}, k=5, include_failures=False)
        for entry, _ in results:
            self.assertTrue(entry.success)


# ---------------------------------------------------------------------------
# DesignArchive — top_k and failed_designs
# ---------------------------------------------------------------------------

class TestDesignArchiveRanking(unittest.TestCase):

    def _populated_archive(self) -> DesignArchive:
        archive = DesignArchive(similarity_threshold=0.99)
        for score, typ in [(0.9, "qa"), (0.7, "qa"), (0.5, "coding")]:
            archive.add(_make_entry(
                {"agent_type": str(score)},
                score=score,
                task_type=typ,
                success=score > 0.4,
            ))
        return archive

    def test_top_k_returns_k_best(self):
        archive = self._populated_archive()
        top = archive.top_k(k=2, task_type="qa")
        self.assertEqual(len(top), 2)

    def test_top_k_sorted_descending(self):
        archive = self._populated_archive()
        top = archive.top_k(k=3)
        scores = [e.performance.mean_score() for e in top]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_failed_designs_only_returns_failures(self):
        archive = self._populated_archive()
        failures = archive.failed_designs()
        for e in failures:
            self.assertFalse(e.success)

    def test_failed_designs_task_type_filter(self):
        archive = self._populated_archive()
        failures = archive.failed_designs(task_type="qa")
        for e in failures:
            self.assertTrue("qa" in e.task_types or "qa" in e.performance.scores)


# ---------------------------------------------------------------------------
# DesignArchive — persistence
# ---------------------------------------------------------------------------

class TestDesignArchivePersistence(unittest.TestCase):

    def test_save_load_round_trip(self):
        archive = DesignArchive(similarity_threshold=0.9)
        archive.add(_make_entry({"a": "b"}, score=0.75))
        archive.add(_make_entry({"c": "d"}, score=0.55))

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            archive.save(path)
            restored = DesignArchive.load(path, similarity_threshold=0.9)
            self.assertEqual(len(restored), len(archive))
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# meta_agent_search
# ---------------------------------------------------------------------------

class TestMetaAgentSearch(unittest.TestCase):

    def _make_llm(self, config_response: Dict[str, Any]) -> Any:
        response_json = json.dumps({
            "config": config_response,
            "rationale": "test rationale",
            "expected_strengths": ["s1"],
            "expected_weaknesses": ["w1"],
        })

        class FakeResp:
            content = response_json

        llm = MagicMock()
        llm.generate.return_value = FakeResp()
        return llm

    def test_meta_agent_search_adds_entries(self):
        llm = self._make_llm({"agent_type": "novel", "tools": ["calc"]})
        archive = DesignArchive(similarity_threshold=0.99, llm=llm)
        evaluator = MagicMock(return_value=0.8)
        new_entries = archive.meta_agent_search(
            task_type="qa",
            evaluator=evaluator,
            n_iterations=2,
        )
        self.assertGreater(len(new_entries), 0)

    def test_meta_agent_search_requires_llm(self):
        archive = DesignArchive(similarity_threshold=0.9)
        with self.assertRaises(RuntimeError):
            archive.meta_agent_search("qa", evaluator=MagicMock(), n_iterations=1)

    def test_meta_agent_search_handles_llm_error(self):
        """Should silently skip iterations where LLM fails."""
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("LLM down")
        archive = DesignArchive(similarity_threshold=0.99, llm=llm)
        new_entries = archive.meta_agent_search(
            task_type="qa",
            evaluator=MagicMock(return_value=0.5),
            n_iterations=3,
        )
        # All iterations skipped → no new entries
        self.assertEqual(len(new_entries), 0)


if __name__ == "__main__":
    unittest.main()

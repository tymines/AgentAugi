"""Unit tests for evoagentx.memory.reasoning_bank."""

import json
import math
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# Import the module directly to avoid triggering evoagentx.memory.__init__,
# which pulls in optional heavy dependencies (feedparser, etc.) not installed
# in every environment.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "reasoning_bank",
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..",
        "evoagentx", "memory", "reasoning_bank.py",
    ),
)
_mod = _ilu.module_from_spec(_spec)
sys.modules["reasoning_bank"] = _mod  # required so @dataclass can resolve the module
_spec.loader.exec_module(_mod)

ReasoningBank = _mod.ReasoningBank
ReasoningEntry = _mod.ReasoningEntry
_cosine_similarity = _mod._cosine_similarity
_jaccard_similarity = _mod._jaccard_similarity
_tokenise = _mod._tokenise


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _store_entry(
    bank: ReasoningBank,
    task: str = "default task",
    steps: list = None,
    outcome: str = "success",
    quality: float = 0.8,
) -> ReasoningEntry:
    return bank.store(
        task=task,
        steps=steps or ["step 1", "step 2"],
        outcome=outcome,
        quality_score=quality,
    )


# ---------------------------------------------------------------------------
# Similarity utilities
# ---------------------------------------------------------------------------

class TestTokenise(unittest.TestCase):

    def test_lowercases(self):
        tokens = _tokenise("Hello World")
        self.assertIn("hello", tokens)
        self.assertIn("world", tokens)

    def test_strips_punctuation(self):
        tokens = _tokenise("foo, bar!")
        self.assertNotIn(",", tokens)
        self.assertNotIn("!", tokens)

    def test_empty_string(self):
        self.assertEqual(_tokenise(""), [])


class TestJaccardSimilarity(unittest.TestCase):

    def test_identical_strings(self):
        self.assertAlmostEqual(_jaccard_similarity("hello world", "hello world"), 1.0)

    def test_disjoint_strings(self):
        self.assertAlmostEqual(_jaccard_similarity("foo bar", "baz qux"), 0.0)

    def test_partial_overlap(self):
        sim = _jaccard_similarity("hello world", "hello there")
        # intersection={'hello'}, union={'hello','world','there'} → 1/3
        self.assertAlmostEqual(sim, 1.0 / 3.0, places=5)

    def test_both_empty(self):
        self.assertAlmostEqual(_jaccard_similarity("", ""), 1.0)

    def test_one_empty(self):
        self.assertAlmostEqual(_jaccard_similarity("", "foo"), 0.0)


class TestCosineSimilarity(unittest.TestCase):

    def test_identical_vectors(self):
        v = [1.0, 0.5, 0.25]
        self.assertAlmostEqual(_cosine_similarity(v, v), 1.0)

    def test_orthogonal_vectors(self):
        self.assertAlmostEqual(_cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_zero_vector(self):
        self.assertAlmostEqual(_cosine_similarity([0.0, 0.0], [1.0, 2.0]), 0.0)

    def test_dimension_mismatch_raises(self):
        with self.assertRaises(ValueError):
            _cosine_similarity([1.0], [1.0, 2.0])

    def test_known_value(self):
        # [1, 0] · [1, 1] / (1 * sqrt(2)) = 1/sqrt(2)
        result = _cosine_similarity([1.0, 0.0], [1.0, 1.0])
        self.assertAlmostEqual(result, 1.0 / math.sqrt(2), places=5)


# ---------------------------------------------------------------------------
# ReasoningEntry
# ---------------------------------------------------------------------------

class TestReasoningEntry(unittest.TestCase):

    def _make_entry(self) -> ReasoningEntry:
        return ReasoningEntry(
            entry_id="1",
            task="Sort a list",
            steps=["Parse input", "Call sorted()", "Return result"],
            outcome="Sorted list returned",
            quality_score=0.9,
        )

    def test_step_summary_shows_first_n(self):
        entry = self._make_entry()
        summary = entry.step_summary(max_steps=2)
        self.assertIn("Parse input", summary)
        self.assertIn("Call sorted()", summary)
        # Third step is truncated
        self.assertNotIn("Return result", summary)
        self.assertIn("more steps", summary)

    def test_step_summary_all_steps(self):
        entry = self._make_entry()
        summary = entry.step_summary(max_steps=10)
        self.assertIn("Return result", summary)
        self.assertNotIn("more steps", summary)

    def test_to_prompt_excerpt_contains_task(self):
        entry = self._make_entry()
        excerpt = entry.to_prompt_excerpt()
        self.assertIn("Sort a list", excerpt)
        self.assertIn("Sorted list returned", excerpt)
        self.assertIn("0.90", excerpt)


# ---------------------------------------------------------------------------
# ReasoningBank — basic storage
# ---------------------------------------------------------------------------

class TestReasoningBankStore(unittest.TestCase):

    def setUp(self):
        self.bank = ReasoningBank(min_quality=0.5)

    def test_store_returns_entry(self):
        entry = _store_entry(self.bank)
        self.assertIsNotNone(entry)
        self.assertIsInstance(entry, ReasoningEntry)

    def test_store_below_quality_returns_none(self):
        result = _store_entry(self.bank, quality=0.3)
        self.assertIsNone(result)

    def test_size_increments(self):
        self.assertEqual(self.bank.size(), 0)
        _store_entry(self.bank)
        self.assertEqual(self.bank.size(), 1)
        _store_entry(self.bank, task="another task")
        self.assertEqual(self.bank.size(), 2)

    def test_entry_id_assigned(self):
        e1 = _store_entry(self.bank, task="task 1")
        e2 = _store_entry(self.bank, task="task 2")
        self.assertNotEqual(e1.entry_id, e2.entry_id)

    def test_quality_clamped_to_one(self):
        entry = _store_entry(self.bank, quality=1.5)
        self.assertAlmostEqual(entry.quality_score, 1.0)

    def test_quality_clamped_to_zero(self):
        # quality=-0.1 → clamped to 0.0 → below min_quality 0.5 → rejected
        result = _store_entry(self.bank, quality=-0.1)
        self.assertIsNone(result)

    def test_metadata_stored(self):
        entry = self.bank.store(
            task="test",
            steps=[],
            outcome="ok",
            quality_score=0.8,
            metadata={"source": "eval_run"},
        )
        self.assertEqual(entry.metadata["source"], "eval_run")


# ---------------------------------------------------------------------------
# ReasoningBank — capacity eviction
# ---------------------------------------------------------------------------

class TestReasoningBankCapacity(unittest.TestCase):

    def test_evicts_lowest_quality_when_full(self):
        bank = ReasoningBank(min_quality=0.5, max_entries=3)
        # Fill with entries of known quality
        _store_entry(bank, task="A", quality=0.9)
        _store_entry(bank, task="B", quality=0.6)
        _store_entry(bank, task="C", quality=0.8)
        self.assertEqual(bank.size(), 3)

        # Adding a 4th should evict the lowest (0.6)
        _store_entry(bank, task="D", quality=0.75)
        self.assertEqual(bank.size(), 3)

        # "B" (quality 0.6) should be gone
        results = bank.retrieve_similar("B")
        tasks = [e.task for e in results]
        self.assertNotIn("B", tasks)


# ---------------------------------------------------------------------------
# ReasoningBank — retrieval
# ---------------------------------------------------------------------------

class TestReasoningBankRetrieval(unittest.TestCase):

    def setUp(self):
        self.bank = ReasoningBank(min_quality=0.5)
        _store_entry(self.bank, task="sort a list of integers ascending", quality=0.9)
        _store_entry(self.bank, task="compute the fibonacci sequence", quality=0.85)
        _store_entry(self.bank, task="reverse a string", quality=0.7)

    def test_retrieve_returns_list(self):
        results = self.bank.retrieve_similar("sort numbers")
        self.assertIsInstance(results, list)

    def test_retrieve_returns_at_most_k(self):
        results = self.bank.retrieve_similar("sort numbers", k=2)
        self.assertLessEqual(len(results, ), 2)

    def test_most_relevant_first(self):
        # "sort" entries should rank higher for a sorting query
        results = self.bank.retrieve_similar("sort a list", k=3)
        self.assertTrue(len(results) > 0)
        # The sort-related entry should be near the top
        top_task = results[0].task
        self.assertIn("sort", top_task.lower())

    def test_min_similarity_filters(self):
        # min_similarity=1.0 (perfect match only) should return empty for non-identical
        results = self.bank.retrieve_similar("something completely different", min_similarity=1.0)
        self.assertEqual(results, [])

    def test_empty_bank_returns_empty_list(self):
        empty_bank = ReasoningBank()
        results = empty_bank.retrieve_similar("anything")
        self.assertEqual(results, [])

    def test_cosine_similarity_path(self):
        """When embed_fn is supplied, cosine similarity is used."""
        calls = [0]

        def fake_embed(text: str):
            # Deterministic fake embedding: count unique chars as a 3D vector
            calls[0] += 1
            chars = list(set(text.lower()))
            return [len(chars) / 30.0, len(text) / 100.0, 0.5]

        bank = ReasoningBank(min_quality=0.5, embed_fn=fake_embed)
        _store_entry(bank, task="sorting algorithm", quality=0.9)
        results = bank.retrieve_similar("sort a list")
        self.assertTrue(calls[0] > 0)
        self.assertIsInstance(results, list)


# ---------------------------------------------------------------------------
# ReasoningBank — pruning
# ---------------------------------------------------------------------------

class TestReasoningBankPrune(unittest.TestCase):

    def test_prune_removes_below_threshold(self):
        bank = ReasoningBank(min_quality=0.4)
        _store_entry(bank, task="high", quality=0.9)
        _store_entry(bank, task="medium", quality=0.6)
        _store_entry(bank, task="low", quality=0.5)  # meets initial min but prunable

        removed = bank.prune(quality_threshold=0.65)
        self.assertEqual(removed, 2)  # 0.6 and 0.5 pruned
        self.assertEqual(bank.size(), 1)

    def test_prune_uses_min_quality_by_default(self):
        bank = ReasoningBank(min_quality=0.7)
        # Temporarily lower min to store a low-quality entry
        bank._min_quality = 0.3
        _store_entry(bank, task="low", quality=0.5)
        bank._min_quality = 0.7  # restore

        bank.prune()
        self.assertEqual(bank.size(), 0)

    def test_prune_returns_count(self):
        bank = ReasoningBank(min_quality=0.4)
        _store_entry(bank, task="a", quality=0.9)
        _store_entry(bank, task="b", quality=0.9)
        removed = bank.prune(quality_threshold=0.95)
        self.assertEqual(removed, 2)


# ---------------------------------------------------------------------------
# ReasoningBank — stats
# ---------------------------------------------------------------------------

class TestReasoningBankStats(unittest.TestCase):

    def test_empty_bank_stats(self):
        bank = ReasoningBank()
        s = bank.stats()
        self.assertEqual(s["count"], 0)
        self.assertAlmostEqual(s["mean_quality"], 0.0)

    def test_stats_with_entries(self):
        bank = ReasoningBank(min_quality=0.5)
        _store_entry(bank, quality=0.8)
        _store_entry(bank, task="b", quality=0.6)
        s = bank.stats()
        self.assertEqual(s["count"], 2)
        self.assertAlmostEqual(s["mean_quality"], 0.7)
        self.assertAlmostEqual(s["min_quality"], 0.6)
        self.assertAlmostEqual(s["max_quality"], 0.8)

    def test_get_by_id_found(self):
        bank = ReasoningBank(min_quality=0.5)
        entry = _store_entry(bank)
        found = bank.get_by_id(entry.entry_id)
        self.assertIs(found, entry)

    def test_get_by_id_missing(self):
        bank = ReasoningBank()
        self.assertIsNone(bank.get_by_id("999"))


# ---------------------------------------------------------------------------
# ReasoningBank — persistence
# ---------------------------------------------------------------------------

class TestReasoningBankPersistence(unittest.TestCase):

    def test_save_and_load_round_trip(self):
        bank = ReasoningBank(min_quality=0.5)
        _store_entry(bank, task="sort integers", quality=0.9)
        _store_entry(bank, task="compute fibonacci", quality=0.85)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            bank.save(path)
            new_bank = ReasoningBank(min_quality=0.5)
            loaded = new_bank.load(path)
            self.assertEqual(loaded, 2)
            self.assertEqual(new_bank.size(), 2)
            tasks = {e.task for e in new_bank.retrieve_similar("anything", k=10)}
            self.assertIn("sort integers", tasks)
            self.assertIn("compute fibonacci", tasks)
        finally:
            os.unlink(path)

    def test_load_nonexistent_file_returns_zero(self):
        bank = ReasoningBank()
        result = bank.load("/tmp/nonexistent_reasoning_bank_xyz.json")
        self.assertEqual(result, 0)

    def test_persist_path_auto_saves(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            bank = ReasoningBank(min_quality=0.5, persist_path=path)
            _store_entry(bank, task="auto saved task", quality=0.9)

            # Load into a fresh bank
            new_bank = ReasoningBank(min_quality=0.5)
            new_bank.load(path)
            tasks = {e.task for e in new_bank.retrieve_similar("auto saved", k=5)}
            self.assertIn("auto saved task", tasks)
        finally:
            os.unlink(path)

    def test_persist_path_loaded_on_construction(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            bank1 = ReasoningBank(min_quality=0.5, persist_path=path)
            _store_entry(bank1, task="pre-loaded task", quality=0.9)

            # New instance pointing at same path should auto-load
            bank2 = ReasoningBank(min_quality=0.5, persist_path=path)
            self.assertEqual(bank2.size(), 1)
            self.assertEqual(bank2._entries[0].task, "pre-loaded task")
        finally:
            os.unlink(path)

    def test_load_skips_malformed_entries(self):
        """Entries with missing required keys are silently skipped."""
        data = {
            "version": 1,
            "next_id": 3,
            "min_quality": 0.5,
            "max_entries": 100,
            "entries": [
                {
                    "entry_id": "1",
                    "task": "valid task",
                    "steps": ["step"],
                    "outcome": "ok",
                    "quality_score": 0.8,
                    "metadata": {},
                    "created_at": 0.0,
                },
                {
                    # missing quality_score — should be skipped
                    "entry_id": "2",
                    "task": "broken entry",
                },
            ],
        }
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            json.dump(data, f)
            path = f.name

        try:
            bank = ReasoningBank(min_quality=0.5)
            bank.load(path)
            self.assertEqual(bank.size(), 1)
            self.assertEqual(bank._entries[0].task, "valid task")
        finally:
            os.unlink(path)

    def test_no_duplicate_on_repeated_load(self):
        bank = ReasoningBank(min_quality=0.5)
        _store_entry(bank, task="unique task", quality=0.9)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            bank.save(path)
            bank.load(path)  # load again into same bank
            self.assertEqual(bank.size(), 1)  # no duplicate
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

class TestReasoningBankContextManager(unittest.TestCase):

    def test_context_manager_saves_on_exit(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            with ReasoningBank(min_quality=0.5, persist_path=path) as bank:
                _store_entry(bank, task="ctx task", quality=0.8)

            # File should have been written
            self.assertTrue(os.path.getsize(path) > 0)

            new_bank = ReasoningBank(min_quality=0.5)
            new_bank.load(path)
            self.assertEqual(new_bank.size(), 1)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()

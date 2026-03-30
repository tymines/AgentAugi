"""Unit tests for evoagentx.memory.mistake_notebook."""

import os
import json
import tempfile
import unittest
from datetime import datetime, timedelta

from evoagentx.memory.mistake_notebook import (
    MistakeCategory,
    MistakeEntry,
    MistakeNotebook,
)


def _make_entry(
    attempted: str = "call API",
    went_wrong: str = "timeout",
    fix: str = "add retry logic",
    category: MistakeCategory = MistakeCategory.TOOL_MISUSE,
    resolved: bool = False,
    task_type: str = None,
    tags: list = None,
) -> MistakeEntry:
    return MistakeEntry(
        attempted=attempted,
        went_wrong=went_wrong,
        fix=fix,
        category=category,
        resolved=resolved,
        task_type=task_type,
        tags=tags or [],
    )


class TestMistakeEntry(unittest.TestCase):
    """Tests for MistakeEntry helper methods."""

    def test_combined_text_includes_all_fields(self):
        entry = _make_entry(
            attempted="call database",
            went_wrong="connection refused",
            fix="check host configuration",
            tags=["postgres", "networking"],
        )
        combined = entry.combined_text()
        self.assertIn("database", combined)
        self.assertIn("connection", combined)
        self.assertIn("check host", combined)
        self.assertIn("postgres", combined)

    def test_keywords_returns_unique_tokens(self):
        entry = _make_entry(
            attempted="database database database",
            went_wrong="error error",
            fix="fix fix",
        )
        kw = entry.keywords()
        self.assertEqual(kw.count("database"), 1)

    def test_keywords_minimum_length_filter(self):
        entry = _make_entry(attempted="a bb ccc dddd")
        kw = entry.keywords()
        # Only tokens ≥ 3 chars should appear
        for token in kw:
            self.assertGreaterEqual(len(token), 3)


class TestMistakeNotebookWrite(unittest.TestCase):
    """Tests for MistakeNotebook.record and resolve."""

    def setUp(self):
        self.nb = MistakeNotebook()

    def test_record_stores_entry(self):
        entry = _make_entry()
        eid = self.nb.record(entry)
        self.assertEqual(len(self.nb.entries), 1)
        self.assertEqual(eid, entry.entry_id)

    def test_duplicate_entry_id_not_stored(self):
        entry = _make_entry()
        self.nb.record(entry)
        self.nb.record(entry)
        self.assertEqual(len(self.nb.entries), 1)

    def test_max_entries_prunes_oldest_unresolved(self):
        nb = MistakeNotebook(max_entries=3)
        for i in range(5):
            nb.record(_make_entry(attempted=f"task {i}", went_wrong=f"err {i}", fix=f"fix {i}"), auto_save=False)
        self.assertLessEqual(len(nb.entries), 3)

    def test_max_entries_all_resolved_prunes_oldest(self):
        nb = MistakeNotebook(max_entries=2)
        for i in range(3):
            entry = _make_entry(
                attempted=f"a{i}", went_wrong=f"w{i}", fix=f"f{i}", resolved=True
            )
            nb.record(entry, auto_save=False)
        self.assertLessEqual(len(nb.entries), 2)

    def test_resolve_marks_entry(self):
        entry = _make_entry()
        self.nb.record(entry, auto_save=False)
        result = self.nb.resolve(entry.entry_id, auto_save=False)
        self.assertTrue(result)
        self.assertTrue(self.nb.entries[0].resolved)

    def test_resolve_unknown_id_returns_false(self):
        result = self.nb.resolve("nonexistent-id", auto_save=False)
        self.assertFalse(result)


class TestMistakeNotebookRead(unittest.TestCase):
    """Tests for MistakeNotebook.consult and format_for_prompt."""

    def setUp(self):
        self.nb = MistakeNotebook()
        entries = [
            _make_entry(
                attempted="authenticate user with JWT token",
                went_wrong="token validation failed",
                fix="validate signature and expiry",
                category=MistakeCategory.TOOL_MISUSE,
                task_type="auth",
            ),
            _make_entry(
                attempted="connect to database with pool",
                went_wrong="pool exhausted under load",
                fix="increase max pool size",
                category=MistakeCategory.PLANNING_ERROR,
                task_type="db",
            ),
            _make_entry(
                attempted="parse JSON from API response",
                went_wrong="null field caused KeyError",
                fix="use dict.get() with defaults",
                category=MistakeCategory.REASONING_ERROR,
                task_type="api",
            ),
        ]
        for e in entries:
            self.nb.record(e, auto_save=False)

    def test_consult_returns_relevant_entries(self):
        results = self.nb.consult("JWT token authentication", top_k=2)
        self.assertGreater(len(results), 0)
        texts = [e.attempted for e in results]
        self.assertTrue(any("JWT" in t or "authenticate" in t for t in texts))

    def test_consult_empty_when_no_entries(self):
        nb = MistakeNotebook()
        results = nb.consult("anything")
        self.assertEqual(results, [])

    def test_consult_excludes_resolved_by_default(self):
        self.nb.entries[0].resolved = True
        results = self.nb.consult("token authentication", top_k=5)
        for e in results:
            self.assertFalse(e.resolved)

    def test_consult_includes_resolved_when_flag_set(self):
        self.nb.entries[0].resolved = True
        results = self.nb.consult("token authentication", top_k=5, include_resolved=True)
        found_resolved = any(e.resolved for e in results)
        self.assertTrue(found_resolved)

    def test_consult_category_filter(self):
        results = self.nb.consult(
            "query", category_filter=MistakeCategory.PLANNING_ERROR, top_k=5
        )
        for e in results:
            self.assertEqual(e.category, MistakeCategory.PLANNING_ERROR)

    def test_consult_task_type_filter(self):
        results = self.nb.consult("query", task_type_filter="db", top_k=5)
        for e in results:
            self.assertEqual(e.task_type, "db")

    def test_format_for_prompt_produces_string(self):
        text = self.nb.format_for_prompt("JWT token")
        if text:
            self.assertIn("Avoid repeating these past mistakes", text)

    def test_format_for_prompt_empty_when_no_match(self):
        text = self.nb.format_for_prompt("zzz yyy xxx completely unrelated")
        self.assertEqual(text, "")

    def test_stats_counts_by_category(self):
        s = self.nb.stats()
        self.assertEqual(s["total"], 3)
        self.assertEqual(s["unresolved"], 3)
        self.assertEqual(s["resolved"], 0)
        self.assertIn("by_category", s)
        self.assertIn(MistakeCategory.TOOL_MISUSE.value, s["by_category"])


class TestMistakeNotebookCleanup(unittest.TestCase):
    """Tests for cleanup_stale."""

    def _entry_with_old_timestamp(self, days_ago: int, resolved: bool = False) -> MistakeEntry:
        entry = _make_entry(
            attempted=f"old task {days_ago}d",
            went_wrong="something",
            fix="something else",
            resolved=resolved,
        )
        old_ts = datetime.now() - timedelta(days=days_ago)
        entry.timestamp = old_ts.strftime("%Y-%m-%d %H:%M:%S")
        return entry

    def test_cleanup_removes_old_entries(self):
        nb = MistakeNotebook()
        nb.record(self._entry_with_old_timestamp(40), auto_save=False)
        nb.record(self._entry_with_old_timestamp(10), auto_save=False)
        removed = nb.cleanup_stale(max_age_days=30, auto_save=False)
        self.assertEqual(removed, 1)
        self.assertEqual(len(nb.entries), 1)

    def test_cleanup_resolved_only_flag(self):
        nb = MistakeNotebook()
        nb.record(self._entry_with_old_timestamp(40, resolved=True), auto_save=False)
        nb.record(self._entry_with_old_timestamp(40, resolved=False), auto_save=False)
        removed = nb.cleanup_stale(max_age_days=30, resolved_only=True, auto_save=False)
        self.assertEqual(removed, 1)
        # The unresolved one should remain
        self.assertFalse(nb.entries[0].resolved)

    def test_cleanup_returns_zero_when_nothing_stale(self):
        nb = MistakeNotebook()
        nb.record(_make_entry(), auto_save=False)
        removed = nb.cleanup_stale(max_age_days=30, auto_save=False)
        self.assertEqual(removed, 0)


class TestMistakeNotebookPersistence(unittest.TestCase):
    """Tests for MistakeNotebook save/load."""

    def test_save_and_load_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            path = tf.name
        try:
            nb = MistakeNotebook(persistence_path=path)
            entry = _make_entry(attempted="persist test")
            nb.record(entry, auto_save=False)
            nb.save()

            nb2 = MistakeNotebook(persistence_path=path)
            nb2.load()
            self.assertEqual(len(nb2.entries), 1)
            self.assertEqual(nb2.entries[0].attempted, "persist test")
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_load_no_file_noop(self):
        nb = MistakeNotebook(persistence_path="/nonexistent/path.json")
        nb.load()  # Should not raise
        self.assertEqual(len(nb.entries), 0)

    def test_load_deduplicates(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            path = tf.name
        try:
            entry = _make_entry(attempted="dedup test")
            nb = MistakeNotebook(persistence_path=path)
            nb.record(entry, auto_save=False)
            nb.save()

            nb2 = MistakeNotebook(persistence_path=path)
            nb2.record(entry, auto_save=False)  # already there
            nb2.load()
            self.assertEqual(len(nb2.entries), 1)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_record_auto_saves(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            path = tf.name
        try:
            os.unlink(path)
            nb = MistakeNotebook(persistence_path=path)
            nb.record(_make_entry(), auto_save=True)
            self.assertTrue(os.path.exists(path))
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_save_noop_when_no_path(self):
        nb = MistakeNotebook()
        nb.record(_make_entry(), auto_save=False)
        nb.save()  # Should not raise

    def test_persisted_json_is_valid(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            path = tf.name
        try:
            nb = MistakeNotebook(persistence_path=path)
            nb.record(_make_entry(tags=["t1", "t2"]), auto_save=True)
            with open(path) as f:
                data = json.load(f)
            self.assertIsInstance(data, list)
            self.assertEqual(data[0]["tags"], ["t1", "t2"])
        finally:
            if os.path.exists(path):
                os.unlink(path)


if __name__ == "__main__":
    unittest.main()

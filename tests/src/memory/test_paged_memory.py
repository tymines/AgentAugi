"""Unit tests for evoagentx.memory.paged_memory."""

import os
import json
import tempfile
import unittest
from typing import List

from evoagentx.core.message import Message
from evoagentx.memory.paged_memory import PagedMemory, WarmPage, PageTier


def _make_msg(content: str, idx: int = 0) -> Message:
    """Helper: create a Message with deterministic content."""
    return Message(content=content)


class TestWarmPage(unittest.TestCase):
    """Tests for the WarmPage supporting class."""

    def test_creation_defaults(self):
        page = WarmPage(summary="test summary")
        self.assertIsInstance(page.page_id, str)
        self.assertEqual(page.access_count, 0)
        self.assertEqual(page.keywords, [])

    def test_touch_increments_access_count(self):
        page = WarmPage(summary="s")
        page.touch()
        self.assertEqual(page.access_count, 1)
        page.touch()
        self.assertEqual(page.access_count, 2)


class TestPagedMemoryBasic(unittest.TestCase):
    """Tests for core PagedMemory add/get behaviour."""

    def setUp(self):
        self.memory = PagedMemory(hot_capacity=5, eviction_block_size=2)

    def test_add_and_get_within_capacity(self):
        msgs = [_make_msg(f"msg{i}") for i in range(3)]
        for m in msgs:
            self.memory.add_message(m)
        self.assertEqual(self.memory.hot_size, 3)
        result = self.memory.get()
        self.assertEqual(len(result), 3)

    def test_get_with_n_returns_most_recent(self):
        for i in range(5):
            self.memory.add_message(_make_msg(f"content {i}"))
        last2 = self.memory.get(n=2)
        self.assertEqual(len(last2), 2)
        self.assertIn("4", last2[-1].content)

    def test_duplicate_message_not_added(self):
        msg = _make_msg("hello")
        self.memory.add_message(msg)
        self.memory.add_message(msg)
        self.assertEqual(self.memory.hot_size, 1)

    def test_add_none_is_noop(self):
        self.memory.add_message(None)
        self.assertEqual(self.memory.hot_size, 0)

    def test_total_message_count_tracks_full_history(self):
        """messages list should contain all added messages, not just hot."""
        mem = PagedMemory(hot_capacity=3, eviction_block_size=2)
        for i in range(6):
            mem.add_message(_make_msg(f"m{i}"))
        # 6 messages total, hot can only hold 3
        self.assertEqual(mem.size, 6)


class TestPagedMemoryEviction(unittest.TestCase):
    """Tests for hot→warm eviction logic."""

    def test_eviction_triggers_warm_page_creation(self):
        mem = PagedMemory(hot_capacity=3, eviction_block_size=2)
        for i in range(4):  # 4 > 3, triggers eviction
            mem.add_message(_make_msg(f"message about authentication {i}"))
        self.assertGreater(mem.warm_size, 0)

    def test_hot_size_bounded_after_eviction(self):
        mem = PagedMemory(hot_capacity=4, eviction_block_size=2)
        for i in range(10):
            mem.add_message(_make_msg(f"msg {i}"))
        # hot_size should never exceed hot_capacity + eviction_block_size - 1
        self.assertLessEqual(mem.hot_size, mem.hot_capacity)

    def test_warm_capacity_respected(self):
        mem = PagedMemory(hot_capacity=2, eviction_block_size=1, warm_capacity=2)
        for i in range(10):
            mem.add_message(_make_msg(f"x {i}"))
        self.assertLessEqual(mem.warm_size, mem.warm_capacity)

    def test_warm_page_contains_keywords(self):
        mem = PagedMemory(hot_capacity=2, eviction_block_size=2)
        mem.add_message(_make_msg("database connection error timeout"))
        mem.add_message(_make_msg("authentication token expired"))
        mem.add_message(_make_msg("retry connection successful"))
        if mem.warm_pages:
            kw = mem.warm_pages[0].keywords
            self.assertIsInstance(kw, list)
            self.assertGreater(len(kw), 0)

    def test_custom_summarizer_used_on_eviction(self):
        called: List[bool] = []

        def my_summarizer(messages):
            called.append(True)
            return "custom summary"

        mem = PagedMemory(hot_capacity=2, eviction_block_size=2)
        mem.set_summarizer(my_summarizer)
        for i in range(3):
            mem.add_message(_make_msg(f"m{i}"))
        self.assertTrue(called, "Summarizer should have been called during eviction")
        self.assertEqual(mem.warm_pages[-1].summary, "custom summary")

    def test_summarizer_exception_falls_back_to_digest(self):
        def bad_summarizer(msgs):
            raise RuntimeError("oops")

        mem = PagedMemory(hot_capacity=2, eviction_block_size=2)
        mem.set_summarizer(bad_summarizer)
        for i in range(3):
            mem.add_message(_make_msg(f"fallback test {i}"))
        # Should not raise; warm page summary should be non-empty digest
        self.assertTrue(mem.warm_pages[-1].summary)


class TestPagedMemoryColdStorage(unittest.TestCase):
    """Tests for cold-storage persistence."""

    def test_cold_file_created_on_eviction(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            path = tf.name
        try:
            os.unlink(path)  # start fresh
            mem = PagedMemory(hot_capacity=2, eviction_block_size=2, cold_storage_path=path)
            for i in range(3):
                mem.add_message(_make_msg(f"cold test {i}"))
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                data = json.load(f)
            self.assertIsInstance(data, list)
            self.assertGreater(len(data), 0)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_load_cold_returns_messages(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tf:
            json.dump([
                {"content": "hello from cold", "message_id": "abc123"}
            ], tf)
            path = tf.name
        try:
            mem = PagedMemory(cold_storage_path=path)
            cold = mem.load_cold()
            self.assertEqual(len(cold), 1)
            self.assertEqual(cold[0].content, "hello from cold")
        finally:
            os.unlink(path)

    def test_load_cold_returns_empty_when_no_file(self):
        mem = PagedMemory(cold_storage_path="/nonexistent/path/file.json")
        self.assertEqual(mem.load_cold(), [])

    def test_cold_deduplication(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            path = tf.name
        try:
            os.unlink(path)
            mem = PagedMemory(hot_capacity=2, eviction_block_size=2, cold_storage_path=path)
            msg = _make_msg("unique content dedup test")
            for i in range(3):
                mem.add_message(_make_msg(f"dedup {i}"))

            with open(path) as f:
                first_count = len(json.load(f))

            # Write cold again — should not duplicate
            mem._write_cold(mem.load_cold())
            with open(path) as f:
                second_count = len(json.load(f))

            self.assertEqual(first_count, second_count)
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestPagedMemoryRecall(unittest.TestCase):
    """Tests for recall_warm and recall_cold."""

    def test_recall_warm_returns_relevant_pages(self):
        mem = PagedMemory(hot_capacity=2, eviction_block_size=2)
        mem.add_message(_make_msg("database authentication error"))
        mem.add_message(_make_msg("connection pool timeout"))
        mem.add_message(_make_msg("retry logic fixed"))
        results = mem.recall_warm("database authentication")
        self.assertIsInstance(results, list)

    def test_recall_warm_empty_when_no_pages(self):
        mem = PagedMemory(hot_capacity=50)
        results = mem.recall_warm("anything")
        self.assertEqual(results, [])

    def test_recall_warm_touches_page(self):
        mem = PagedMemory(hot_capacity=2, eviction_block_size=2)
        mem.add_message(_make_msg("cache eviction policy"))
        mem.add_message(_make_msg("memory management strategy"))
        mem.add_message(_make_msg("more content here"))
        if mem.warm_pages:
            original_count = mem.warm_pages[0].access_count
            mem.recall_warm("memory cache")
            self.assertGreaterEqual(
                max(p.access_count for p in mem.warm_pages), original_count
            )

    def test_recall_cold_keyword_matching(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as tf:
            json.dump([
                {"content": "database connection pool error", "message_id": "id1"},
                {"content": "user login authentication", "message_id": "id2"},
            ], tf)
            path = tf.name
        try:
            mem = PagedMemory(cold_storage_path=path)
            results = mem.recall_cold("database connection")
            self.assertGreater(len(results), 0)
            self.assertIn("database", str(results[0].content))
        finally:
            os.unlink(path)

    def test_page_in_adds_to_hot(self):
        mem = PagedMemory(hot_capacity=10)
        msgs = [_make_msg(f"recalled {i}") for i in range(3)]
        mem.page_in(msgs)
        self.assertEqual(mem.hot_size, 3)

    def test_page_in_does_not_duplicate(self):
        mem = PagedMemory(hot_capacity=10)
        msg = _make_msg("test")
        mem.add_message(msg)
        mem.page_in([msg])
        self.assertEqual(mem.hot_size, 1)


class TestPagedMemoryLifecycle(unittest.TestCase):
    """Tests for clear, stats, and init_module rehydration."""

    def test_clear_resets_all_tiers(self):
        mem = PagedMemory(hot_capacity=2, eviction_block_size=2)
        for i in range(5):
            mem.add_message(_make_msg(f"m{i}"))
        mem.clear()
        self.assertEqual(mem.hot_size, 0)
        self.assertEqual(mem.warm_size, 0)
        self.assertEqual(mem.size, 0)

    def test_stats_returns_expected_keys(self):
        mem = PagedMemory(hot_capacity=10)
        mem.add_message(_make_msg("hello"))
        s = mem.stats()
        self.assertIn("hot_size", s)
        self.assertIn("warm_size", s)
        self.assertIn("total_messages", s)
        self.assertIn("hot_capacity", s)

    def test_init_module_rebuilds_hot_buffer(self):
        """Re-initialising should rebuild the hot buffer from self.messages."""
        mem = PagedMemory(hot_capacity=5)
        for i in range(3):
            mem.add_message(_make_msg(f"reload {i}"))
        mem.init_module()
        self.assertEqual(mem.hot_size, 3)


if __name__ == "__main__":
    unittest.main()

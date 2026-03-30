"""Unit tests for evoagentx.memory.reflexion."""

import os
import json
import tempfile
import unittest
from typing import List

from evoagentx.memory.reflexion import (
    Episode,
    ReflexionMemory,
    ReflexionAgent,
    TaskOutcome,
)


class TestEpisode(unittest.TestCase):
    """Tests for the Episode data class."""

    def test_creation_defaults(self):
        ep = Episode(
            task_description="solve the problem",
            reflection="I should have checked inputs first",
        )
        self.assertIsInstance(ep.episode_id, str)
        self.assertEqual(ep.outcome, TaskOutcome.UNKNOWN)
        self.assertEqual(ep.attempt_summary, "")
        self.assertIsInstance(ep.timestamp, str)

    def test_keywords_extracts_tokens(self):
        ep = Episode(
            task_description="authenticate the user with JWT token",
            reflection="the token was expired, validate expiry before use",
        )
        kw = ep.keywords()
        self.assertIn("token", kw)
        self.assertIn("authenticate", kw)

    def test_keywords_deduplication(self):
        ep = Episode(
            task_description="token token token",
            reflection="token repeated",
        )
        kw = ep.keywords()
        self.assertEqual(kw.count("token"), 1)


class TestReflexionMemoryWrite(unittest.TestCase):
    """Tests for ReflexionMemory.add_episode."""

    def setUp(self):
        self.mem = ReflexionMemory()

    def test_add_episode_stores_correctly(self):
        ep = Episode(
            task_description="task A",
            reflection="reflection A",
            outcome=TaskOutcome.SUCCESS,
        )
        self.mem.add_episode(ep)
        self.assertEqual(len(self.mem.episodes), 1)

    def test_duplicate_episode_id_not_stored(self):
        ep = Episode(task_description="t", reflection="r")
        self.mem.add_episode(ep)
        self.mem.add_episode(ep)
        self.assertEqual(len(self.mem.episodes), 1)

    def test_max_episodes_prunes_oldest(self):
        mem = ReflexionMemory(max_episodes=3)
        for i in range(5):
            mem.add_episode(Episode(task_description=f"task {i}", reflection=f"ref {i}"))
        self.assertEqual(len(mem.episodes), 3)
        # Oldest should have been pruned; most recent should remain
        descriptions = [e.task_description for e in mem.episodes]
        self.assertIn("task 4", descriptions)
        self.assertNotIn("task 0", descriptions)


class TestReflexionMemoryRead(unittest.TestCase):
    """Tests for ReflexionMemory.find_similar and get_reflections_for_task."""

    def setUp(self):
        self.mem = ReflexionMemory()
        episodes = [
            Episode(
                task_description="authenticate user via OAuth token",
                reflection="always validate token expiry",
                outcome=TaskOutcome.FAILURE,
                task_type="auth",
            ),
            Episode(
                task_description="connect to database with connection pool",
                reflection="check pool size configuration",
                outcome=TaskOutcome.SUCCESS,
                task_type="db",
            ),
            Episode(
                task_description="parse JSON response from API",
                reflection="handle null fields gracefully",
                outcome=TaskOutcome.PARTIAL,
                task_type="api",
            ),
        ]
        for ep in episodes:
            self.mem.add_episode(ep)

    def test_find_similar_returns_relevant_episodes(self):
        results = self.mem.find_similar("authenticate user token", top_k=2)
        self.assertGreater(len(results), 0)
        descriptions = [e.task_description for e in results]
        self.assertTrue(any("authenticate" in d for d in descriptions))

    def test_find_similar_empty_when_no_episodes(self):
        empty = ReflexionMemory()
        results = empty.find_similar("anything")
        self.assertEqual(results, [])

    def test_outcome_filter_restricts_results(self):
        results = self.mem.find_similar(
            "user token", outcome_filter=TaskOutcome.SUCCESS, top_k=5
        )
        for ep in results:
            self.assertEqual(ep.outcome, TaskOutcome.SUCCESS)

    def test_task_type_filter_restricts_results(self):
        results = self.mem.find_similar("query", task_type_filter="db", top_k=5)
        for ep in results:
            self.assertEqual(ep.task_type, "db")

    def test_get_reflections_for_task_formats_string(self):
        text = self.mem.get_reflections_for_task("authenticate token", top_k=2)
        if text:
            self.assertIn("Past reflections", text)
            self.assertIn("authenticate", text.lower())

    def test_get_reflections_empty_when_no_match(self):
        # Query with no overlapping keywords
        text = self.mem.get_reflections_for_task("zzz yyy xxx", top_k=3)
        self.assertEqual(text, "")

    def test_recent_returns_last_n(self):
        recent = self.mem.recent(n=2)
        self.assertEqual(len(recent), 2)
        # Most recently added should be last
        self.assertEqual(recent[-1].task_type, "api")

    def test_stats_counts_by_outcome(self):
        s = self.mem.stats()
        self.assertIn("total", s)
        self.assertEqual(s["total"], 3)
        self.assertEqual(s[TaskOutcome.FAILURE.value], 1)
        self.assertEqual(s[TaskOutcome.SUCCESS.value], 1)
        self.assertEqual(s[TaskOutcome.PARTIAL.value], 1)


class TestReflexionMemoryPersistence(unittest.TestCase):
    """Tests for ReflexionMemory save/load."""

    def test_save_and_load_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            path = tf.name
        try:
            mem = ReflexionMemory(persistence_path=path)
            ep = Episode(
                task_description="test task",
                reflection="test reflection",
                outcome=TaskOutcome.SUCCESS,
            )
            mem.add_episode(ep)
            mem.save()

            mem2 = ReflexionMemory(persistence_path=path)
            mem2.load()
            self.assertEqual(len(mem2.episodes), 1)
            self.assertEqual(mem2.episodes[0].task_description, "test task")
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_load_no_file_noop(self):
        mem = ReflexionMemory(persistence_path="/nonexistent/path/r.json")
        mem.load()  # Should not raise
        self.assertEqual(len(mem.episodes), 0)

    def test_load_deduplicates_with_existing(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            path = tf.name
        try:
            ep = Episode(task_description="t", reflection="r")
            mem = ReflexionMemory(persistence_path=path)
            mem.add_episode(ep)
            mem.save()

            # Load into a memory that already has the same episode
            mem2 = ReflexionMemory(persistence_path=path)
            mem2.add_episode(ep)
            mem2.load()
            self.assertEqual(len(mem2.episodes), 1)  # no duplicate
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_add_episode_auto_saves(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            path = tf.name
        try:
            os.unlink(path)
            mem = ReflexionMemory(persistence_path=path)
            mem.add_episode(Episode(task_description="auto", reflection="save"))
            self.assertTrue(os.path.exists(path))
        finally:
            if os.path.exists(path):
                os.unlink(path)


class MockAgent:
    """Minimal agent stub for ReflexionAgent tests."""

    def __init__(self, should_raise: bool = False):
        self.system_prompt: str = "base prompt"
        self.should_raise = should_raise
        self.execute_called = False

    def execute(self, **kwargs):
        self.execute_called = True
        if self.should_raise:
            raise RuntimeError("agent failed")
        return "result"

    async def async_execute(self, **kwargs):
        self.execute_called = True
        if self.should_raise:
            raise RuntimeError("async agent failed")
        return "async result"


class TestReflexionAgent(unittest.TestCase):
    """Tests for the ReflexionAgent wrapper."""

    def test_execute_stores_episode_on_success(self):
        agent = MockAgent()
        mem = ReflexionMemory()
        wrapper = ReflexionAgent(agent=agent, memory=mem)
        result = wrapper.execute(task="do something useful")
        self.assertEqual(result, "result")
        self.assertEqual(len(mem.episodes), 1)
        self.assertEqual(mem.episodes[0].outcome, TaskOutcome.SUCCESS)

    def test_execute_stores_failure_episode_on_exception(self):
        agent = MockAgent(should_raise=True)
        mem = ReflexionMemory()
        wrapper = ReflexionAgent(agent=agent, memory=mem)
        with self.assertRaises(RuntimeError):
            wrapper.execute(task="risky task")
        self.assertEqual(len(mem.episodes), 1)
        self.assertEqual(mem.episodes[0].outcome, TaskOutcome.FAILURE)

    def test_execute_injects_reflections_into_system_prompt(self):
        # Seed memory with a relevant past episode
        mem = ReflexionMemory()
        mem.add_episode(Episode(
            task_description="do something useful action",
            reflection="always validate inputs first",
            outcome=TaskOutcome.FAILURE,
        ))

        agent = MockAgent()
        original_prompt = agent.system_prompt
        wrapper = ReflexionAgent(agent=agent, memory=mem, top_k_reflections=2)
        wrapper.execute(task="do something useful")
        # System prompt should be restored after execution
        self.assertEqual(agent.system_prompt, original_prompt)

    def test_custom_reflect_fn_used(self):
        called: List[bool] = []

        def my_reflect(task, summary, outcome, result):
            called.append(True)
            return "custom reflection"

        agent = MockAgent()
        mem = ReflexionMemory()
        wrapper = ReflexionAgent(agent=agent, memory=mem, reflect_fn=my_reflect)
        wrapper.execute(task="test custom reflect")
        self.assertTrue(called)
        self.assertEqual(mem.episodes[0].reflection, "custom reflection")

    def test_task_type_propagated_to_episode(self):
        agent = MockAgent()
        mem = ReflexionMemory()
        wrapper = ReflexionAgent(agent=agent, memory=mem, task_type="code_gen")
        wrapper.execute(task="generate code")
        self.assertEqual(mem.episodes[0].task_type, "code_gen")

    def test_async_execute_stores_episode(self):
        import asyncio
        agent = MockAgent()
        mem = ReflexionMemory()
        wrapper = ReflexionAgent(agent=agent, memory=mem)
        result = asyncio.run(wrapper.async_execute(task="async task"))
        self.assertEqual(result, "async result")
        self.assertEqual(len(mem.episodes), 1)
        self.assertEqual(mem.episodes[0].outcome, TaskOutcome.SUCCESS)


if __name__ == "__main__":
    unittest.main()

"""Unit tests for evoagentx.core.plan_cache."""

import json
import os
import tempfile
import time
import unittest
from typing import List, Optional

from evoagentx.core.plan_cache import (
    PlanCache,
    PlanTemplate,
    PlanStep,
    _compute_structural_hash,
    _jaccard_similarity,
    _cosine_similarity,
    _lcs_length,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _steps(action_types: List[str], tool_names: Optional[List[Optional[str]]] = None) -> List[PlanStep]:
    """Build a list of PlanStep objects from action_type strings."""
    if tool_names is None:
        tool_names = [None] * len(action_types)
    return [
        PlanStep(
            action_type=at,
            description=f"Execute {at}",
            tool_name=tn,
            parameters={"key": at},
            estimated_cost=0.01,
        )
        for at, tn in zip(action_types, tool_names)
    ]


def _char_embed(text: str) -> List[float]:
    """Deterministic 26-dimensional character-frequency embedding."""
    counts = [0.0] * 26
    for ch in text.lower():
        if "a" <= ch <= "z":
            counts[ord(ch) - ord("a")] += 1.0
    total = max(1.0, sum(counts))
    return [c / total for c in counts]


# ---------------------------------------------------------------------------
# PlanStep
# ---------------------------------------------------------------------------

class TestPlanStep(unittest.TestCase):

    def test_minimal_creation(self):
        step = PlanStep(action_type="search", description="search the web")
        self.assertEqual(step.action_type, "search")
        self.assertIsNone(step.tool_name)
        self.assertEqual(step.parameters, {})
        self.assertEqual(step.estimated_cost, 0.0)

    def test_full_creation(self):
        step = PlanStep(
            action_type="code_exec",
            description="run code",
            tool_name="python_repl",
            parameters={"code": "print('hi')"},
            estimated_cost=0.05,
        )
        self.assertEqual(step.tool_name, "python_repl")
        self.assertEqual(step.parameters["code"], "print('hi')")
        self.assertEqual(step.estimated_cost, 0.05)

    def test_parameters_default_is_empty_dict(self):
        step = PlanStep("foo", "bar")
        step.parameters["x"] = 1
        step2 = PlanStep("foo", "bar")
        # Each instance gets its own default dict
        self.assertEqual(step2.parameters, {})


# ---------------------------------------------------------------------------
# PlanTemplate
# ---------------------------------------------------------------------------

class TestPlanTemplate(unittest.TestCase):

    def test_structural_hash_auto_computed(self):
        steps = _steps(["search", "summarise"])
        tmpl = PlanTemplate(task_description="test", steps=steps)
        expected = _compute_structural_hash(steps)
        self.assertEqual(tmpl.structural_hash, expected)

    def test_total_cost_auto_summed(self):
        steps = [
            PlanStep("a", "desc a", estimated_cost=0.10),
            PlanStep("b", "desc b", estimated_cost=0.20),
        ]
        tmpl = PlanTemplate(task_description="cost test", steps=steps)
        self.assertAlmostEqual(tmpl.total_cost, 0.30)

    def test_explicit_hash_not_overwritten(self):
        steps = _steps(["search"])
        tmpl = PlanTemplate(task_description="t", steps=steps, structural_hash="custom")
        self.assertEqual(tmpl.structural_hash, "custom")

    def test_default_success_rate_is_one(self):
        tmpl = PlanTemplate(task_description="t", steps=_steps(["a"]))
        self.assertEqual(tmpl.success_rate, 1.0)


# ---------------------------------------------------------------------------
# _compute_structural_hash
# ---------------------------------------------------------------------------

class TestComputeStructuralHash(unittest.TestCase):

    def test_same_structure_same_hash(self):
        steps_a = _steps(["search", "summarise"], ["web_search", None])
        steps_b = _steps(["search", "summarise"], ["web_search", None])
        self.assertEqual(
            _compute_structural_hash(steps_a),
            _compute_structural_hash(steps_b),
        )

    def test_different_action_types_different_hash(self):
        steps_a = _steps(["search"])
        steps_b = _steps(["execute"])
        self.assertNotEqual(
            _compute_structural_hash(steps_a),
            _compute_structural_hash(steps_b),
        )

    def test_different_tool_names_different_hash(self):
        steps_a = _steps(["search"], ["web_search"])
        steps_b = _steps(["search"], ["db_search"])
        self.assertNotEqual(
            _compute_structural_hash(steps_a),
            _compute_structural_hash(steps_b),
        )

    def test_empty_steps_hash_is_stable(self):
        h1 = _compute_structural_hash([])
        h2 = _compute_structural_hash([])
        self.assertEqual(h1, h2)

    def test_hash_ignores_parameters_and_description(self):
        steps_a = [PlanStep("search", "find X", parameters={"q": "weather"})]
        steps_b = [PlanStep("search", "find Y", parameters={"q": "news"})]
        self.assertEqual(
            _compute_structural_hash(steps_a),
            _compute_structural_hash(steps_b),
        )

    def test_hash_is_64_hex_chars(self):
        h = _compute_structural_hash(_steps(["a", "b"]))
        self.assertEqual(len(h), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in h))


# ---------------------------------------------------------------------------
# PlanCache.store
# ---------------------------------------------------------------------------

class TestPlanCacheStore(unittest.TestCase):

    def setUp(self):
        self.cache = PlanCache()

    def test_store_returns_template(self):
        steps = _steps(["search"])
        tmpl = self.cache.store("find Paris weather", steps)
        self.assertIsInstance(tmpl, PlanTemplate)

    def test_store_increments_size(self):
        self.cache.store("task A", _steps(["a"]))
        self.cache.store("task B", _steps(["b"]))
        self.assertEqual(len(self.cache), 2)

    def test_duplicate_task_updates_existing(self):
        steps = _steps(["search"])
        self.cache.store("same task", steps, outcome="success")
        self.cache.store("same task", steps, outcome="failure")
        # Should still be 1 template, not 2
        self.assertEqual(len(self.cache), 1)

    def test_duplicate_updates_success_rate(self):
        steps = _steps(["search"])
        self.cache.store("same task", steps, outcome="success")
        tmpl = self.cache.store("same task", steps, outcome="failure")
        # Initial success_rate = 1.0 (from first store as times_used=0)
        # After second store with failure: rolling update
        self.assertLess(tmpl.success_rate, 1.0)
        self.assertGreater(tmpl.success_rate, 0.0)

    def test_failure_outcome_sets_success_rate_zero(self):
        steps = _steps(["search"])
        tmpl = self.cache.store("task", steps, outcome="failure")
        self.assertEqual(tmpl.success_rate, 0.0)

    def test_auto_prune_on_overflow(self):
        cache = PlanCache(max_templates=3)
        for i in range(5):
            cache.store(f"task {i}", _steps([f"action_{i}"]))
        self.assertLessEqual(len(cache), 3)


# ---------------------------------------------------------------------------
# PlanCache.retrieve
# ---------------------------------------------------------------------------

class TestPlanCacheRetrieve(unittest.TestCase):

    def setUp(self):
        self.cache = PlanCache(similarity_threshold=0.3)

    def test_empty_cache_returns_none(self):
        result = self.cache.retrieve("any task")
        self.assertIsNone(result)

    def test_exact_task_description_found(self):
        steps = _steps(["search"])
        self.cache.store("find weather in Paris", steps)
        result = self.cache.retrieve("find weather in Paris")
        self.assertIsNotNone(result)
        self.assertEqual(result.task_description, "find weather in Paris")

    def test_similar_task_found_above_threshold(self):
        steps = _steps(["search"])
        self.cache.store("find weather in Paris", steps)
        # "find weather in London" shares "find weather in"
        result = self.cache.retrieve("find weather in London")
        self.assertIsNotNone(result)

    def test_dissimilar_task_below_threshold_returns_none(self):
        cache = PlanCache(similarity_threshold=0.9)
        cache.store("find weather in Paris", _steps(["search"]))
        result = cache.retrieve("compile the Rust source code")
        self.assertIsNone(result)

    def test_retrieve_increments_times_used(self):
        steps = _steps(["search"])
        self.cache.store("find weather in Paris", steps)
        # times_used=1 after the initial store
        result = self.cache.retrieve("find weather in Paris")
        self.assertEqual(result.times_used, 2)
        self.cache.retrieve("find weather in Paris")
        result2 = self.cache.retrieve("find weather in Paris")
        self.assertEqual(result2.times_used, 4)

    def test_retrieve_updates_last_used(self):
        steps = _steps(["search"])
        self.cache.store("task", steps)
        before = time.time()
        result = self.cache.retrieve("task")
        self.assertGreaterEqual(result.last_used, before)

    def test_similarity_threshold_override(self):
        cache = PlanCache(similarity_threshold=0.1)
        cache.store("task alpha", _steps(["x"]))
        # High threshold override should miss
        result = cache.retrieve("task beta gamma delta extra", similarity_threshold=0.99)
        self.assertIsNone(result)

    def test_retrieve_with_embed_fn(self):
        cache = PlanCache(embed_fn=_char_embed, similarity_threshold=0.5)
        cache.store("find weather forecast", _steps(["search"]))
        result = cache.retrieve("check weather forecast today")
        # Both have "weather" and "forecast" — embeddings should be similar
        self.assertIsNotNone(result)

    def test_retrieve_no_match_increments_miss_stats(self):
        self.cache.retrieve("no match here ever")
        s = self.cache.stats()
        self.assertEqual(s["misses"], 1)
        self.assertEqual(s["hits"], 0)


# ---------------------------------------------------------------------------
# PlanCache.structural_similarity
# ---------------------------------------------------------------------------

class TestStructuralSimilarity(unittest.TestCase):

    def setUp(self):
        self.cache = PlanCache()

    def test_identical_plans_score_one(self):
        steps = _steps(["search", "summarise"])
        self.assertAlmostEqual(
            self.cache.structural_similarity(steps, list(steps)), 1.0
        )

    def test_empty_plans_score_one(self):
        self.assertAlmostEqual(
            self.cache.structural_similarity([], []), 1.0
        )

    def test_one_empty_scores_zero(self):
        self.assertAlmostEqual(
            self.cache.structural_similarity(_steps(["search"]), []), 0.0
        )

    def test_disjoint_action_types_score_less(self):
        a = _steps(["search", "rank", "summarise"])
        b = _steps(["execute", "compile", "deploy"])
        score = self.cache.structural_similarity(a, b)
        self.assertLess(score, 0.5)

    def test_partial_overlap_between_zero_and_one(self):
        a = _steps(["search", "rank", "summarise"])
        b = _steps(["search", "filter", "summarise"])
        score = self.cache.structural_similarity(a, b)
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_tool_names_affect_similarity(self):
        a = _steps(["search"], ["web_search"])
        b = _steps(["search"], ["db_search"])
        self.assertLess(self.cache.structural_similarity(a, b), 1.0)

    def test_similarity_is_symmetric(self):
        a = _steps(["a", "b", "c"])
        b = _steps(["a", "x", "c"])
        self.assertAlmostEqual(
            self.cache.structural_similarity(a, b),
            self.cache.structural_similarity(b, a),
        )


# ---------------------------------------------------------------------------
# PlanCache.adapt
# ---------------------------------------------------------------------------

class TestAdapt(unittest.TestCase):

    def setUp(self):
        self.cache = PlanCache()

    def test_adapt_returns_same_length(self):
        steps = _steps(["search", "summarise"])
        self.cache.store("find Paris weather", steps)
        tmpl = self.cache.retrieve("find Paris weather")
        adapted = self.cache.adapt(tmpl, "find London weather")
        self.assertEqual(len(adapted), len(steps))

    def test_adapt_preserves_action_types(self):
        steps = _steps(["search", "rank", "summarise"])
        self.cache.store("task A", steps)
        tmpl = self.cache.retrieve("task A")
        adapted = self.cache.adapt(tmpl, "task B")
        for orig, new in zip(steps, adapted):
            self.assertEqual(orig.action_type, new.action_type)

    def test_adapt_preserves_tool_names(self):
        steps = _steps(["search"], ["web_search"])
        self.cache.store("task A", steps)
        tmpl = self.cache.retrieve("task A")
        adapted = self.cache.adapt(tmpl, "task B")
        self.assertEqual(adapted[0].tool_name, "web_search")

    def test_adapt_copies_parameters_shallowly(self):
        steps = [PlanStep("search", "desc", parameters={"q": "Paris"})]
        self.cache.store("find Paris", steps)
        tmpl = self.cache.retrieve("find Paris")
        adapted = self.cache.adapt(tmpl, "find London")
        # Mutating adapted copy should not affect template
        adapted[0].parameters["q"] = "London"
        self.assertEqual(tmpl.steps[0].parameters["q"], "Paris")

    def test_adapt_does_not_mutate_template(self):
        steps = _steps(["search"])
        self.cache.store("task A", steps)
        tmpl = self.cache.retrieve("task A")
        original_desc = tmpl.steps[0].description
        self.cache.adapt(tmpl, "task B")
        self.assertEqual(tmpl.steps[0].description, original_desc)

    def test_adapt_preserves_estimated_cost(self):
        steps = [PlanStep("a", "desc", estimated_cost=0.42)]
        self.cache.store("task A", steps)
        tmpl = self.cache.retrieve("task A")
        adapted = self.cache.adapt(tmpl, "task B")
        self.assertAlmostEqual(adapted[0].estimated_cost, 0.42)


# ---------------------------------------------------------------------------
# PlanCache.prune
# ---------------------------------------------------------------------------

class TestPrune(unittest.TestCase):

    def test_prune_by_success_rate(self):
        cache = PlanCache()
        cache.store("task pass", _steps(["a"]), outcome="success")
        cache.store("task fail", _steps(["b"]), outcome="failure")
        removed = cache.prune(min_success_rate=0.5)
        self.assertEqual(removed, 1)
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache._templates[0].task_description, "task pass")

    def test_prune_by_max_templates(self):
        cache = PlanCache()
        for i in range(5):
            cache.store(f"task {i}", _steps([f"action_{i}"]))
        removed = cache.prune(max_templates=3)
        self.assertEqual(len(cache), 3)

    def test_prune_keeps_most_recently_used(self):
        cache = PlanCache()
        for i in range(4):
            tmpl = cache.store(f"task {i}", _steps([f"action_{i}"]))
        # Touch the last two templates
        cache._templates[2].last_used = time.time() + 100
        cache._templates[3].last_used = time.time() + 200
        cache.prune(max_templates=2)
        kept_descs = {t.task_description for t in cache._templates}
        self.assertIn("task 2", kept_descs)
        self.assertIn("task 3", kept_descs)

    def test_prune_returns_count_removed(self):
        cache = PlanCache()
        for i in range(3):
            cache.store(f"t{i}", _steps([f"a{i}"]))
        n = cache.prune(max_templates=1)
        self.assertEqual(n, 2)

    def test_prune_no_op_when_under_limit(self):
        cache = PlanCache()
        cache.store("single task", _steps(["x"]))
        removed = cache.prune(max_templates=10, min_success_rate=0.0)
        self.assertEqual(removed, 0)
        self.assertEqual(len(cache), 1)


# ---------------------------------------------------------------------------
# JSON persistence (save / load)
# ---------------------------------------------------------------------------

class TestSaveLoad(unittest.TestCase):

    def test_save_and_load_round_trip(self):
        cache = PlanCache()
        steps = [
            PlanStep("search", "search step", tool_name="web", estimated_cost=0.01),
            PlanStep("summarise", "summarise step", estimated_cost=0.02),
        ]
        cache.store("original task", steps)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cache.save(path)
            cache2 = PlanCache()
            cache2.load(path)
            self.assertEqual(len(cache2), 1)
            tmpl = cache2._templates[0]
            self.assertEqual(tmpl.task_description, "original task")
            self.assertEqual(len(tmpl.steps), 2)
            self.assertEqual(tmpl.steps[0].action_type, "search")
            self.assertEqual(tmpl.steps[0].tool_name, "web")
            self.assertAlmostEqual(tmpl.total_cost, 0.03)
        finally:
            os.unlink(path)

    def test_stats_persisted(self):
        cache = PlanCache(similarity_threshold=0.1)
        cache.store("task", _steps(["a"]))
        cache.retrieve("task")  # creates a hit

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cache.save(path)
            cache2 = PlanCache()
            cache2.load(path)
            self.assertEqual(cache2._hits, 1)
            self.assertEqual(cache2._total_queries, 1)
        finally:
            os.unlink(path)

    def test_load_nonexistent_raises(self):
        cache = PlanCache()
        with self.assertRaises(FileNotFoundError):
            cache.load("/nonexistent/path/plan_cache.json")

    def test_load_wrong_version_raises(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"version": 99, "templates": []}, f)
            path = f.name
        try:
            cache = PlanCache()
            with self.assertRaises(ValueError):
                cache.load(path)
        finally:
            os.unlink(path)

    def test_load_replaces_existing_templates(self):
        cache = PlanCache()
        cache.store("old task", _steps(["old"]))
        steps = _steps(["new"])
        cache2 = PlanCache()
        cache2.store("new task", steps)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cache2.save(path)
            cache.load(path)
            self.assertEqual(len(cache), 1)
            self.assertEqual(cache._templates[0].task_description, "new task")
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# stats()
# ---------------------------------------------------------------------------

class TestStats(unittest.TestCase):

    def test_initial_stats_all_zero(self):
        cache = PlanCache()
        s = cache.stats()
        self.assertEqual(s["total_queries"], 0)
        self.assertEqual(s["hits"], 0)
        self.assertEqual(s["misses"], 0)
        self.assertAlmostEqual(s["hit_rate"], 0.0)
        self.assertAlmostEqual(s["total_cost_saved"], 0.0)

    def test_hit_rate_calculation(self):
        cache = PlanCache(similarity_threshold=0.1)
        cache.store("find weather", _steps(["a"]))
        cache.retrieve("find weather")   # hit
        cache.retrieve("xyz abc 123")    # miss
        s = cache.stats()
        self.assertEqual(s["total_queries"], 2)
        self.assertEqual(s["hits"], 1)
        self.assertEqual(s["misses"], 1)
        self.assertAlmostEqual(s["hit_rate"], 0.5)

    def test_total_cost_saved_accumulates(self):
        cache = PlanCache(similarity_threshold=0.1)
        steps = [PlanStep("a", "desc", estimated_cost=0.10)]
        cache.store("task", steps)
        cache.retrieve("task")
        cache.retrieve("task")
        s = cache.stats()
        self.assertAlmostEqual(s["total_cost_saved"], 0.20)

    def test_num_templates_reflects_store(self):
        cache = PlanCache()
        self.assertEqual(cache.stats()["num_templates"], 0)
        cache.store("t1", _steps(["a"]))
        cache.store("t2", _steps(["b"]))
        self.assertEqual(cache.stats()["num_templates"], 2)

    def test_avg_template_cost_correct(self):
        cache = PlanCache()
        cache.store("t1", [PlanStep("a", "d", estimated_cost=0.10)])
        cache.store("t2", [PlanStep("b", "d", estimated_cost=0.30)])
        s = cache.stats()
        self.assertAlmostEqual(s["avg_template_cost"], 0.20)


# ---------------------------------------------------------------------------
# Jaccard and cosine helpers
# ---------------------------------------------------------------------------

class TestHelpers(unittest.TestCase):

    def test_jaccard_identical(self):
        self.assertAlmostEqual(_jaccard_similarity("foo bar", "foo bar"), 1.0)

    def test_jaccard_disjoint(self):
        self.assertAlmostEqual(_jaccard_similarity("foo", "bar"), 0.0)

    def test_jaccard_partial(self):
        score = _jaccard_similarity("the cat sat", "the dog sat")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_jaccard_empty_strings(self):
        self.assertAlmostEqual(_jaccard_similarity("", ""), 1.0)

    def test_cosine_identical(self):
        v = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(_cosine_similarity(v, v), 1.0)

    def test_cosine_orthogonal(self):
        self.assertAlmostEqual(_cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_cosine_zero_vector(self):
        self.assertAlmostEqual(_cosine_similarity([0.0, 0.0], [1.0, 0.0]), 0.0)

    def test_cosine_dimension_mismatch_raises(self):
        with self.assertRaises(ValueError):
            _cosine_similarity([1.0, 2.0], [1.0])

    def test_lcs_identical(self):
        seq = ["a", "b", "c"]
        self.assertEqual(_lcs_length(seq, seq), 3)

    def test_lcs_disjoint(self):
        self.assertEqual(_lcs_length(["a", "b"], ["c", "d"]), 0)

    def test_lcs_partial(self):
        self.assertEqual(_lcs_length(["a", "b", "c"], ["a", "x", "c"]), 2)

    def test_lcs_empty(self):
        self.assertEqual(_lcs_length([], ["a"]), 0)
        self.assertEqual(_lcs_length(["a"], []), 0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):

    def test_retrieve_from_empty_cache(self):
        cache = PlanCache()
        self.assertIsNone(cache.retrieve("anything"))

    def test_store_empty_steps(self):
        cache = PlanCache()
        tmpl = cache.store("empty plan task", [])
        self.assertEqual(len(tmpl.steps), 0)
        self.assertAlmostEqual(tmpl.total_cost, 0.0)

    def test_retrieve_identical_plans(self):
        cache = PlanCache(similarity_threshold=0.1)
        steps = _steps(["a"])
        cache.store("exact same task", steps)
        cache.store("exact same task", steps, outcome="success")
        result = cache.retrieve("exact same task")
        self.assertIsNotNone(result)
        # Should still be 1 template
        self.assertEqual(len(cache), 1)

    def test_no_match_above_high_threshold(self):
        cache = PlanCache(similarity_threshold=0.99)
        cache.store("a b c d e f", _steps(["x"]))
        result = cache.retrieve("completely different words here")
        self.assertIsNone(result)

    def test_cost_tracker_integration(self):
        """PlanCache should call cost_tracker.record_savings on hit."""

        class FakeTracker:
            def __init__(self):
                self.saved = 0.0
            def record_savings(self, amount: float):
                self.saved += amount

        tracker = FakeTracker()
        cache = PlanCache(similarity_threshold=0.1, cost_tracker=tracker)
        steps = [PlanStep("a", "desc", estimated_cost=0.50)]
        cache.store("task", steps)
        cache.retrieve("task")
        self.assertAlmostEqual(tracker.saved, 0.50)

    def test_ttl_expiry_prevents_retrieval(self):
        cache = PlanCache(similarity_threshold=0.1, ttl_seconds=0.001)
        cache.store("task", _steps(["a"]))
        time.sleep(0.05)
        result = cache.retrieve("task")
        self.assertIsNone(result)

    def test_len_reflects_store_count(self):
        cache = PlanCache()
        self.assertEqual(len(cache), 0)
        cache.store("t1", _steps(["a"]))
        self.assertEqual(len(cache), 1)
        cache.store("t2", _steps(["b"]))
        self.assertEqual(len(cache), 2)


# ---------------------------------------------------------------------------
# GeminiEmbedder unit tests (mock-based, no network)
# ---------------------------------------------------------------------------

class TestGeminiEmbedderUnit(unittest.TestCase):
    """Mock-based tests for GeminiEmbedder that never hit the network."""

    def _make_fake_client(self, vector: List[float]):
        """Return a mock genai.Client whose embed_content returns *vector*."""
        import types

        class FakeEmbedding:
            values = vector

        class FakeResult:
            embeddings = [FakeEmbedding()]

        class FakeModels:
            def embed_content(self, model, contents):
                return FakeResult()

        client = types.SimpleNamespace(models=FakeModels())
        return client

    def test_missing_api_key_raises(self):
        """GeminiEmbedder should raise ValueError when no key is available."""
        from evoagentx.core.gemini_embedder import GeminiEmbedder
        # Temporarily remove env var if present
        saved = os.environ.pop("GEMINI_API_KEY", None)
        saved2 = os.environ.pop("GOOGLE_API_KEY", None)
        try:
            with self.assertRaises((ValueError, ImportError)):
                GeminiEmbedder(api_key=None)
        finally:
            if saved is not None:
                os.environ["GEMINI_API_KEY"] = saved
            if saved2 is not None:
                os.environ["GOOGLE_API_KEY"] = saved2

    def test_call_returns_vector(self):
        """__call__ should return the vector from the API response."""
        from evoagentx.core.gemini_embedder import GeminiEmbedder
        embedder = GeminiEmbedder.__new__(GeminiEmbedder)
        embedder._api_key = "fake"
        embedder._model = "models/gemini-embedding-001"
        embedder._cache = {}
        embedder._cache_order = []
        embedder._cache_size = 10
        embedder._client = self._make_fake_client([0.1, 0.2, 0.3])

        result = embedder("hello world")
        self.assertEqual(result, [0.1, 0.2, 0.3])

    def test_repeated_call_uses_cache(self):
        """Second call for same text should not re-call the API."""
        from evoagentx.core.gemini_embedder import GeminiEmbedder

        call_count = [0]

        class CountingModels:
            def embed_content(self, model, contents):
                call_count[0] += 1
                import types
                class FE:
                    values = [1.0, 2.0]
                class FR:
                    embeddings = [FE()]
                return FR()

        import types
        client = types.SimpleNamespace(models=CountingModels())

        embedder = GeminiEmbedder.__new__(GeminiEmbedder)
        embedder._api_key = "fake"
        embedder._model = "models/gemini-embedding-001"
        embedder._cache = {}
        embedder._cache_order = []
        embedder._cache_size = 10
        embedder._client = client

        embedder("same text")
        embedder("same text")
        self.assertEqual(call_count[0], 1)

    def test_cache_eviction(self):
        """Cache evicts oldest entries when full."""
        from evoagentx.core.gemini_embedder import GeminiEmbedder

        embedder = GeminiEmbedder.__new__(GeminiEmbedder)
        embedder._api_key = "fake"
        embedder._model = "m"
        embedder._cache = {}
        embedder._cache_order = []
        embedder._cache_size = 4

        counter = [0]

        class Mods:
            def embed_content(self, model, contents):
                counter[0] += 1
                import types
                class FE:
                    values = [float(counter[0])]
                class FR:
                    embeddings = [FE()]
                return FR()

        import types
        embedder._client = types.SimpleNamespace(models=Mods())

        for i in range(6):
            embedder(f"text {i}")

        # After 6 inserts into a cache of size 4, half should have been evicted
        self.assertLessEqual(len(embedder._cache), 4)


class TestPlanCacheWithGeminiEmbedder(unittest.TestCase):
    """PlanCache integration tests using a fake embedder (no network calls)."""

    @staticmethod
    def _make_embedder(dim: int = 8):
        """Return a fake embed_fn that hashes text to a dim-dimensional vector."""
        import hashlib

        def embed(text: str) -> List[float]:
            digest = hashlib.md5(text.encode()).digest()
            # Stretch to `dim` floats in [0, 1]
            values = [(b / 255.0) for b in digest]
            while len(values) < dim:
                values.extend(values)
            return values[:dim]

        return embed

    def test_embedding_stored_on_template(self):
        """store() should populate template.embedding when embed_fn is set."""
        cache = PlanCache(embed_fn=self._make_embedder())
        tmpl = cache.store("find weather in Paris", _steps(["search"]))
        self.assertIsNotNone(tmpl.embedding)
        self.assertIsInstance(tmpl.embedding, list)
        self.assertGreater(len(tmpl.embedding), 0)

    def test_no_embedding_without_embed_fn(self):
        """template.embedding should be None when no embed_fn is provided."""
        cache = PlanCache()
        tmpl = cache.store("find weather in Paris", _steps(["search"]))
        self.assertIsNone(tmpl.embedding)

    def test_retrieve_uses_cosine_with_embed_fn(self):
        """retrieve() should use cosine similarity (via embed_fn) not Jaccard."""
        embed = self._make_embedder()
        cache = PlanCache(embed_fn=embed, similarity_threshold=1e-6)
        cache.store("find weather in Paris", _steps(["search"]))
        # Should hit — threshold is tiny so any positive cosine score matches
        result = cache.retrieve("find weather in Paris")
        self.assertIsNotNone(result)

    def test_embedding_round_trip_via_save_load(self):
        """Embeddings should survive a save/load cycle."""
        embed = self._make_embedder()
        cache = PlanCache(embed_fn=embed)
        cache.store("task with embedding", _steps(["search"]))
        original_emb = cache._templates[0].embedding

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            cache.save(path)
            cache2 = PlanCache()
            cache2.load(path)
            loaded_emb = cache2._templates[0].embedding
            self.assertIsNotNone(loaded_emb)
            self.assertEqual(original_emb, loaded_emb)
        finally:
            os.unlink(path)

    def test_old_cache_file_without_embedding_loads_cleanly(self):
        """Loading a cache file without 'embedding' keys should not raise."""
        payload = {
            "version": 1,
            "stats": {"total_queries": 0, "hits": 0, "total_cost_saved": 0.0},
            "templates": [
                {
                    "task_description": "old task",
                    "steps": [{"action_type": "search", "description": "d",
                                "tool_name": None, "parameters": {}, "estimated_cost": 0.0}],
                    "total_cost": 0.0,
                    "success_rate": 1.0,
                    "times_used": 1,
                    "last_used": 1000000.0,
                    "structural_hash": "abc",
                    # No "embedding" key — simulates pre-Gemini cache file
                }
            ],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(payload, f)
            path = f.name
        try:
            cache = PlanCache()
            cache.load(path)
            self.assertEqual(len(cache), 1)
            self.assertIsNone(cache._templates[0].embedding)
        finally:
            os.unlink(path)

    def test_backfill_embedding_on_retrieve_for_old_template(self):
        """Templates loaded without embeddings get backfilled on first retrieve."""
        embed = self._make_embedder()
        # Load a template that has no embedding
        cache = PlanCache(embed_fn=embed, similarity_threshold=1e-6)
        # Manually insert a template with no embedding
        tmpl = PlanTemplate(
            task_description="old task no embed",
            steps=_steps(["search"]),
        )
        self.assertIsNone(tmpl.embedding)
        cache._templates.append(tmpl)

        # retrieve() should backfill the embedding
        cache.retrieve("old task no embed")
        self.assertIsNotNone(cache._templates[0].embedding)


# ---------------------------------------------------------------------------
# Live Gemini test (skipped unless GEMINI_API_KEY is set)
# ---------------------------------------------------------------------------

@unittest.skipUnless(
    os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
    "GEMINI_API_KEY / GOOGLE_API_KEY not set — skipping live Gemini test",
)
class TestGeminiEmbedderLive(unittest.TestCase):
    """Integration test that calls the real Gemini API."""

    def test_live_embedding_dimensions(self):
        """Verify gemini-embedding-001 returns 3072-dimensional vectors."""
        from evoagentx.core.gemini_embedder import GeminiEmbedder, _EMBEDDING_DIMS
        embedder = GeminiEmbedder()
        vec = embedder("search the web for Paris weather forecast")
        self.assertEqual(len(vec), _EMBEDDING_DIMS)
        self.assertIsInstance(vec[0], float)
        print(f"\n[live] Gemini embedding dims: {len(vec)}, first 3: {vec[:3]}")

    def test_live_similar_queries_score_higher_than_dissimilar(self):
        """Semantically similar tasks should score higher than dissimilar ones."""
        from evoagentx.core.gemini_embedder import make_gemini_plan_cache
        cache = make_gemini_plan_cache(similarity_threshold=1e-6)

        steps = _steps(["search", "summarise"])
        cache.store("find the weather forecast for Paris", steps)

        similar = cache.retrieve("what is the weather in London?")
        dissimilar_score_template = cache._templates[0]

        # Both should be findable at threshold=0.0
        self.assertIsNotNone(similar)

        # Score the similar vs a dissimilar query manually
        from evoagentx.core.plan_cache import _cosine_similarity
        from evoagentx.core.gemini_embedder import GeminiEmbedder
        embedder = GeminiEmbedder()
        sim_score = _cosine_similarity(
            embedder("what is the weather in London?"),
            embedder("find the weather forecast for Paris"),
        )
        dis_score = _cosine_similarity(
            embedder("compile and deploy the Rust web server"),
            embedder("find the weather forecast for Paris"),
        )
        self.assertGreater(sim_score, dis_score)
        print(f"\n[live] similar={sim_score:.4f}  dissimilar={dis_score:.4f}")


if __name__ == "__main__":
    unittest.main()

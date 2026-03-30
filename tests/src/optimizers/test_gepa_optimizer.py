"""Unit tests for evoagentx.optimizers.gepa_optimizer."""

import unittest
from copy import deepcopy
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from evoagentx.optimizers.gepa_optimizer import (
    GEPACandidate,
    GEPAHistory,
    GEPAOptimizer,
)
from evoagentx.optimizers.engine.registry import ParamRegistry


# ---------------------------------------------------------------------------
# Minimal fake LLM for testing
# ---------------------------------------------------------------------------

class _FakeLLMResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    """Always returns the first argument (prompt seed) as the response."""

    def __init__(self, response: str = "improved prompt") -> None:
        self._response = response
        self.call_count = 0

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs):
        self.call_count += 1
        return _FakeLLMResponse(self._response)


# ---------------------------------------------------------------------------
# Registry fixture
# ---------------------------------------------------------------------------

def _make_registry() -> ParamRegistry:
    """Build a minimal ParamRegistry with one optimisable parameter."""
    registry = MagicMock(spec=ParamRegistry)
    registry.fields = {"instruction": None}
    registry.names.return_value = ["instruction"]
    registry.get.return_value = "Answer the question."
    registry.set.return_value = None
    return registry


def _make_optimizer(
    population_size: int = 4,
    generations: int = 2,
    examples: Optional[List[Dict[str, Any]]] = None,
    llm_response: str = "improved prompt",
) -> GEPAOptimizer:
    registry = _make_registry()
    llm = _FakeLLM(response=llm_response)
    program = MagicMock(return_value={"answer": "42"})
    evaluator = MagicMock(return_value=0.7)

    return GEPAOptimizer(
        registry=registry,
        program=program,
        evaluator=evaluator,
        llm=llm,
        population_size=population_size,
        generations=generations,
        tournament_sample_size=2,
        training_examples=examples or [{"q": "what is 2+2?"}],
        seed=42,
    )


# ---------------------------------------------------------------------------
# GEPACandidate
# ---------------------------------------------------------------------------

class TestGEPACandidate(unittest.TestCase):

    def test_defaults(self):
        cand = GEPACandidate(config={"p": "v"})
        self.assertIsNone(cand.score)
        self.assertEqual(cand.generation, 0)
        self.assertEqual(cand.lineage, "seed")

    def test_score_assignment(self):
        cand = GEPACandidate(config={"p": "v"}, score=0.85)
        self.assertAlmostEqual(cand.score, 0.85)


# ---------------------------------------------------------------------------
# GEPAHistory
# ---------------------------------------------------------------------------

class TestGEPAHistory(unittest.TestCase):

    def test_empty_defaults(self):
        h = GEPAHistory()
        self.assertEqual(h.best_score_per_generation, [])
        self.assertIsNone(h.best_config)
        self.assertEqual(h.total_evaluations, 0)
        self.assertEqual(h.evaluation_calls_saved, 0)


# ---------------------------------------------------------------------------
# Validation guards
# ---------------------------------------------------------------------------

class TestGEPAOptimizerValidation(unittest.TestCase):

    def test_population_size_too_small(self):
        registry = _make_registry()
        with self.assertRaises(ValueError):
            GEPAOptimizer(
                registry=registry,
                program=MagicMock(),
                evaluator=MagicMock(),
                llm=_FakeLLM(),
                population_size=1,
            )

    def test_elite_fraction_out_of_range(self):
        registry = _make_registry()
        with self.assertRaises(ValueError):
            GEPAOptimizer(
                registry=registry,
                program=MagicMock(),
                evaluator=MagicMock(),
                llm=_FakeLLM(),
                population_size=4,
                elite_fraction=0.0,
            )

    def test_crossover_rate_out_of_range(self):
        registry = _make_registry()
        with self.assertRaises(ValueError):
            GEPAOptimizer(
                registry=registry,
                program=MagicMock(),
                evaluator=MagicMock(),
                llm=_FakeLLM(),
                population_size=4,
                crossover_rate=1.5,
            )


# ---------------------------------------------------------------------------
# Seeding and population size
# ---------------------------------------------------------------------------

class TestGEPASeeding(unittest.TestCase):

    def test_seed_population_size(self):
        opt = _make_optimizer(population_size=6)
        pop = opt._seed_population()
        self.assertEqual(len(pop), 6)

    def test_first_seed_is_base(self):
        opt = _make_optimizer(population_size=4)
        pop = opt._seed_population()
        self.assertEqual(pop[0].lineage, "seed_base")

    def test_variants_have_different_lineages(self):
        opt = _make_optimizer(population_size=4)
        pop = opt._seed_population()
        lineages = [c.lineage for c in pop[1:]]
        self.assertTrue(all("seed_variant" in l for l in lineages))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class TestGEPAEvaluation(unittest.TestCase):

    def test_score_candidate_clamps_to_01(self):
        opt = _make_optimizer()
        # Evaluator returns value > 1 — should be clamped
        opt.evaluator = MagicMock(return_value=2.5)
        score = opt._score_candidate(
            config={"instruction": "test"},
            examples=[{"q": "x"}],
        )
        self.assertLessEqual(score, 1.0)
        self.assertGreaterEqual(score, 0.0)

    def test_evaluate_population_fills_scores(self):
        opt = _make_optimizer(population_size=3)
        pop = opt._seed_population()
        opt._evaluate_population(pop, generation=0)
        for cand in pop:
            self.assertIsNotNone(cand.score)

    def test_already_scored_candidates_not_re_evaluated(self):
        opt = _make_optimizer(population_size=3)
        pop = opt._seed_population()
        pop[0].score = 0.99  # pre-scored
        eval_calls_before = opt.evaluator.call_count
        opt._evaluate_population(pop, generation=0)
        # First candidate should not trigger an evaluator call
        # (only the remaining 2 should)
        calls_made = opt.evaluator.call_count - eval_calls_before
        self.assertLessEqual(calls_made, 2)


# ---------------------------------------------------------------------------
# Tournament selection
# ---------------------------------------------------------------------------

class TestTournamentSelect(unittest.TestCase):

    def test_selects_from_population(self):
        opt = _make_optimizer()
        candidates = [
            GEPACandidate(config={"p": "a"}, score=0.3),
            GEPACandidate(config={"p": "b"}, score=0.9),
            GEPACandidate(config={"p": "c"}, score=0.5),
        ]
        winner = opt._tournament_select(candidates, k=3)
        self.assertAlmostEqual(winner.score, 0.9)

    def test_handles_population_smaller_than_k(self):
        opt = _make_optimizer()
        candidates = [GEPACandidate(config={"p": "a"}, score=0.5)]
        winner = opt._tournament_select(candidates, k=10)
        self.assertAlmostEqual(winner.score, 0.5)


# ---------------------------------------------------------------------------
# Evolution operators
# ---------------------------------------------------------------------------

class TestGEPAEvolution(unittest.TestCase):

    def test_evolve_preserves_population_size(self):
        opt = _make_optimizer(population_size=6)
        pop = opt._seed_population()
        for c in pop:
            c.score = 0.5
        new_pop = opt._evolve(pop, generation=1)
        self.assertEqual(len(new_pop), 6)

    def test_elites_have_scores(self):
        opt = _make_optimizer(population_size=4, generations=1)
        pop = opt._seed_population()
        for i, c in enumerate(pop):
            c.score = float(i) * 0.25
        pop.sort(key=lambda c: c.score, reverse=True)
        new_pop = opt._evolve(pop, generation=1)
        n_elite = max(1, int(4 * opt.elite_fraction))
        # Elite candidates carry their scores forward
        for elite in new_pop[:n_elite]:
            self.assertIsNotNone(elite.score)

    def test_crossover_child_has_no_score(self):
        opt = _make_optimizer(population_size=4)
        opt.crossover_rate = 1.0  # Force crossover
        parent_a = GEPACandidate(config={"instruction": "foo"}, score=0.6, generation=0)
        parent_b = GEPACandidate(config={"instruction": "bar"}, score=0.8, generation=0)
        child = opt._crossover(parent_a, parent_b, generation=1)
        self.assertIsNone(child.score)
        self.assertIn("crossover", child.lineage)

    def test_mutation_child_has_no_score(self):
        opt = _make_optimizer(population_size=4)
        parent = GEPACandidate(config={"instruction": "foo"}, score=0.4, generation=0)
        child = opt._mutate(parent, generation=1)
        self.assertIsNone(child.score)
        self.assertIn("mutation", child.lineage)


# ---------------------------------------------------------------------------
# Full optimize() integration
# ---------------------------------------------------------------------------

class TestGEPAOptimize(unittest.TestCase):

    def test_optimize_returns_config_and_history(self):
        opt = _make_optimizer(population_size=4, generations=2)
        best_cfg, history = opt.optimize()
        self.assertIsInstance(best_cfg, dict)
        self.assertIsInstance(history, GEPAHistory)

    def test_history_has_generation_records(self):
        opt = _make_optimizer(population_size=4, generations=3)
        _, history = opt.optimize()
        # Should record one entry per generation (0..3 = 4 entries)
        self.assertEqual(len(history.best_score_per_generation), 4)

    def test_best_config_is_set(self):
        opt = _make_optimizer(population_size=4, generations=2)
        _, history = opt.optimize()
        self.assertIsNotNone(history.best_config)

    def test_total_evaluations_positive(self):
        opt = _make_optimizer(population_size=4, generations=2)
        _, history = opt.optimize()
        self.assertGreater(history.total_evaluations, 0)

    def test_optimize_no_examples(self):
        """Optimize should work when no training examples are provided."""
        opt = _make_optimizer(population_size=3, generations=1, examples=[])
        best_cfg, history = opt.optimize()
        self.assertIsNotNone(best_cfg)


if __name__ == "__main__":
    unittest.main()

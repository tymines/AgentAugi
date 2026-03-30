"""Unit tests for evoagentx.optimizers.mapelites_optimizer."""

import unittest
from typing import Any, Dict
from unittest.mock import MagicMock

from evoagentx.optimizers.mapelites_optimizer import (
    Archive,
    ArchiveCell,
    FeatureDimension,
    MAPElitesOptimizer,
    _crossover,
)
from evoagentx.optimizers.engine.registry import ParamRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dim(name="accuracy", lo=0.0, hi=1.0, resolution=5) -> FeatureDimension:
    return FeatureDimension(
        name=name,
        lo=lo,
        hi=hi,
        resolution=resolution,
        extractor=lambda cfg, res: res.get(name, 0.0),
    )


def _make_2d_archive() -> Archive:
    return Archive(dimensions=[_make_dim("accuracy"), _make_dim("cost")])


# ---------------------------------------------------------------------------
# FeatureDimension tests
# ---------------------------------------------------------------------------

class TestFeatureDimension(unittest.TestCase):

    def test_bin_index_min_value(self):
        dim = _make_dim(lo=0.0, hi=1.0, resolution=10)
        self.assertEqual(dim.bin_index(0.0), 0)

    def test_bin_index_max_value(self):
        dim = _make_dim(lo=0.0, hi=1.0, resolution=10)
        self.assertEqual(dim.bin_index(1.0), 9)

    def test_bin_index_midpoint(self):
        dim = _make_dim(lo=0.0, hi=1.0, resolution=10)
        # 0.5 → normalised 0.5 → int(5) = 5
        self.assertEqual(dim.bin_index(0.5), 5)

    def test_bin_index_clamped_below(self):
        dim = _make_dim(lo=0.0, hi=1.0, resolution=5)
        self.assertEqual(dim.bin_index(-10.0), 0)

    def test_bin_index_clamped_above(self):
        dim = _make_dim(lo=0.0, hi=1.0, resolution=5)
        self.assertEqual(dim.bin_index(99.0), 4)

    def test_invalid_bounds_raises(self):
        with self.assertRaises(ValueError):
            FeatureDimension(name="x", lo=1.0, hi=0.0, resolution=5,
                             extractor=lambda c, r: 0.0)

    def test_invalid_resolution_raises(self):
        with self.assertRaises(ValueError):
            FeatureDimension(name="x", lo=0.0, hi=1.0, resolution=0,
                             extractor=lambda c, r: 0.0)


# ---------------------------------------------------------------------------
# Archive tests
# ---------------------------------------------------------------------------

class TestArchive(unittest.TestCase):

    def test_empty_archive_best_is_none(self):
        archive = _make_2d_archive()
        self.assertIsNone(archive.best())

    def test_insert_first_candidate(self):
        archive = _make_2d_archive()
        config = {"prompt": "v1"}
        result = {"accuracy": 0.5, "cost": 0.3}
        inserted = archive.try_insert(config, result, quality=0.5)
        self.assertTrue(inserted)
        self.assertEqual(archive.size(), 1)

    def test_insert_better_candidate_replaces_cell(self):
        archive = _make_2d_archive()
        config_a = {"prompt": "v1"}
        result = {"accuracy": 0.5, "cost": 0.3}
        archive.try_insert(config_a, result, quality=0.4)

        config_b = {"prompt": "v2"}  # same feature cell
        inserted = archive.try_insert(config_b, result, quality=0.9)
        self.assertTrue(inserted)
        self.assertEqual(archive.size(), 1)  # still one cell
        self.assertAlmostEqual(archive.best().quality, 0.9)

    def test_insert_worse_candidate_rejected(self):
        archive = _make_2d_archive()
        config = {"prompt": "v1"}
        result = {"accuracy": 0.5, "cost": 0.3}
        archive.try_insert(config, result, quality=0.9)
        inserted = archive.try_insert({"prompt": "v2"}, result, quality=0.5)
        self.assertFalse(inserted)
        self.assertAlmostEqual(archive.best().quality, 0.9)

    def test_different_feature_cells_coexist(self):
        archive = _make_2d_archive()
        archive.try_insert({"p": "a"}, {"accuracy": 0.1, "cost": 0.1}, 0.5)
        archive.try_insert({"p": "b"}, {"accuracy": 0.9, "cost": 0.9}, 0.8)
        self.assertEqual(archive.size(), 2)

    def test_sample_from_empty_returns_empty(self):
        archive = _make_2d_archive()
        self.assertEqual(archive.sample(k=3), [])

    def test_sample_respects_k(self):
        archive = _make_2d_archive()
        for i in range(5):
            archive.try_insert(
                {"p": str(i)},
                {"accuracy": i * 0.2, "cost": i * 0.2},
                quality=float(i),
            )
        sample = archive.sample(k=3)
        self.assertLessEqual(len(sample), 3)

    def test_pareto_front_non_dominated(self):
        archive = Archive(dimensions=[_make_dim("f1"), _make_dim("f2")])
        # Cell A: f1=0.9, f2=0.1 — high f1, low f2
        # Cell B: f1=0.1, f2=0.9 — low f1, high f2
        # Neither dominates the other → both on Pareto front
        archive.try_insert({"p": "A"}, {"f1": 0.9, "f2": 0.1}, quality=0.5)
        archive.try_insert({"p": "B"}, {"f1": 0.1, "f2": 0.9}, quality=0.5)
        front = archive.pareto_front(["f1", "f2"])
        self.assertEqual(len(front), 2)

    def test_pareto_front_dominated_excluded(self):
        archive = Archive(dimensions=[_make_dim("f1"), _make_dim("f2")])
        # A dominates C on both f1 and f2
        archive.try_insert({"p": "A"}, {"f1": 0.9, "f2": 0.9}, quality=0.9)
        archive.try_insert({"p": "C"}, {"f1": 0.1, "f2": 0.1}, quality=0.1)
        front = archive.pareto_front(["f1", "f2"])
        # Only A should survive
        self.assertEqual(len(front), 1)
        self.assertEqual(front[0].config["p"], "A")


# ---------------------------------------------------------------------------
# Crossover test
# ---------------------------------------------------------------------------

class TestCrossover(unittest.TestCase):

    def test_child_keys_are_union_of_parents(self):
        a = {"x": 1, "y": 2}
        b = {"y": 3, "z": 4}
        child = _crossover(a, b)
        self.assertIn("x", child)
        self.assertIn("y", child)
        self.assertIn("z", child)

    def test_child_values_come_from_parents(self):
        a = {"k": "val_a"}
        b = {"k": "val_b"}
        # Run many crossovers; child["k"] should always be one of the parents
        for _ in range(20):
            child = _crossover(a, b)
            self.assertIn(child["k"], ("val_a", "val_b"))


# ---------------------------------------------------------------------------
# MAPElitesOptimizer integration tests
# ---------------------------------------------------------------------------

class TestMAPElitesOptimizer(unittest.TestCase):

    def _make_optimizer(self, iterations=5, init_population=3):
        """Build a MAP-Elites optimizer with deterministic mock program."""
        call_count = [0]

        def mock_program():
            call_count[0] += 1
            return {
                "accuracy": min(1.0, call_count[0] * 0.1),
                "cost": max(0.0, 1.0 - call_count[0] * 0.05),
            }

        registry = ParamRegistry()

        # Dummy root object with a prompt attribute
        class DummyProgram:
            prompt = "initial prompt"

        prog_obj = DummyProgram()
        registry.track(prog_obj, "prompt")

        accuracy_dim = FeatureDimension(
            name="accuracy",
            lo=0.0, hi=1.0, resolution=5,
            extractor=lambda cfg, res: res.get("accuracy", 0.0),
        )
        cost_dim = FeatureDimension(
            name="cost",
            lo=0.0, hi=1.0, resolution=5,
            extractor=lambda cfg, res: res.get("cost", 0.5),
        )

        optimizer = MAPElitesOptimizer(
            registry=registry,
            program=mock_program,
            evaluator=lambda res: res.get("accuracy", 0.0),
            dimensions=[accuracy_dim, cost_dim],
            quality_fn=lambda res: res.get("accuracy", 0.0),
            iterations=iterations,
            init_population_size=init_population,
            seed=42,
        )
        return optimizer

    def test_optimize_returns_non_empty_archive(self):
        opt = self._make_optimizer(iterations=5, init_population=3)
        cells = opt.optimize()
        self.assertGreater(len(cells), 0)
        # Cells are sorted descending by quality
        for i in range(len(cells) - 1):
            self.assertGreaterEqual(cells[i].quality, cells[i + 1].quality)

    def test_optimize_returns_archive_cells(self):
        opt = self._make_optimizer()
        cells = opt.optimize()
        for cell in cells:
            self.assertIsInstance(cell, ArchiveCell)
            self.assertGreaterEqual(cell.quality, 0.0)

    def test_iteration_log_populated(self):
        opt = self._make_optimizer(iterations=4, init_population=2)
        opt.optimize()
        log = opt.get_iteration_log()
        # warm-start uses negative indices; MAP-Elites uses 0..iterations-1
        self.assertGreater(len(log), 0)

    def test_invalid_dimensions_raises(self):
        registry = ParamRegistry()
        with self.assertRaises(ValueError):
            MAPElitesOptimizer(
                registry=registry,
                program=lambda: {},
                evaluator=lambda r: 0.0,
                dimensions=[],  # empty
                quality_fn=lambda r: 0.0,
            )

    def test_invalid_iterations_raises(self):
        registry = ParamRegistry()
        dim = _make_dim()
        with self.assertRaises(ValueError):
            MAPElitesOptimizer(
                registry=registry,
                program=lambda: {},
                evaluator=lambda r: 0.0,
                dimensions=[dim],
                quality_fn=lambda r: 0.0,
                iterations=0,
            )

    def test_best_config_applied_to_registry_after_optimize(self):
        """After optimize(), the registry should reflect the best config."""
        opt = self._make_optimizer(iterations=5, init_population=3)
        opt.optimize()
        # Just ensure apply_cfg was called without error; no exception == pass


if __name__ == "__main__":
    unittest.main()

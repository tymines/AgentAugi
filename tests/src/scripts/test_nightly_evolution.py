"""
Tests for scripts/nightly_evolution.py

Covers:
  - Config loading (defaults, YAML override, CLI overrides)
  - LockFile acquire/release semantics
  - Notification backends (file and slack)
  - Report writing (JSON + human-readable text)
  - JitRL trajectory recording helper
  - PlanCache strategy caching helper
  - Dry-run evaluation function
  - Graceful shutdown flag
  - Full dry-run end-to-end execution (no LLM, no real AgentAugi imports needed)
  - CLI argument parsing
"""

from __future__ import annotations

import fcntl
import importlib
import json
import os
import sys
import tempfile
import textwrap
import time
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Ensure repo root is importable
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]  # tests/src/scripts -> repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Lazy import of the module under test
# (avoids hard dependency on evoagentx at import time)
# ---------------------------------------------------------------------------

def _load_module():
    """Import scripts.nightly_evolution as a module, adding scripts/ to path."""
    scripts_dir = str(_REPO_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import importlib.util
    mod_name = "nightly_evolution"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(
        mod_name,
        str(_REPO_ROOT / "scripts" / "nightly_evolution.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass __module__ lookups succeed
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


ne = _load_module()


# ---------------------------------------------------------------------------
# Minimal stubs replicating AgentAugi dataclass shapes
# ---------------------------------------------------------------------------

@dataclass
class _StubGenome:
    name: str
    source_code: str = "stub strategy"
    config: Dict[str, Any] = field(default_factory=dict)
    fitness_history: List[float] = field(default_factory=list)
    generation: int = 0
    parent_name: Optional[str] = None

    @property
    def best_fitness(self) -> float:
        return max(self.fitness_history) if self.fitness_history else float("-inf")

    def record_fitness(self, score: float) -> None:
        self.fitness_history.append(score)


@dataclass
class _StubPopulation:
    _genomes: Dict[str, _StubGenome] = field(default_factory=dict)

    def genomes(self) -> List[_StubGenome]:
        return list(self._genomes.values())

    def add(self, genome: _StubGenome) -> None:
        self._genomes[genome.name] = genome

    def diversity_score(self) -> float:
        return 0.75


class _StubHyperOptimizer:
    def __init__(self, config, base_optimizers, evaluation_fn):
        self.config = config
        self.evaluation_fn = evaluation_fn
        self._generation = 0
        self._pop_genomes: List[_StubGenome] = [
            _StubGenome(name="seed_optimizer", fitness_history=[0.6])
        ]
        self._load_called = False
        self._save_called = False

    class _FakePop:
        def __init__(self, genomes):
            self._genomes = {g.name: g for g in genomes}

        def diversity_score(self):
            return 0.75

    @property
    def population(self):
        return self._FakePop(self._pop_genomes)

    def best_optimizer(self):
        return self._pop_genomes[0] if self._pop_genomes else None

    def evolve_generation(self, task_suite):
        self._generation += 1
        return {
            "generation": self._generation,
            "best_fitness": 0.75,
            "best_name": "seed_optimizer",
            "diversity": 0.75,
            "population_size": len(self._pop_genomes),
            "elapsed_seconds": 0.01,
        }

    def load(self, path: str):
        self._load_called = True

    def save(self, path: str):
        self._save_called = True


class _StubCostTracker:
    def __init__(self):
        self._cost = 0.0
        self._budget = None
        self._session_id = None

    def set_budget(self, max_usd, session_id=None):
        self._budget = max_usd

    def session_cost(self) -> float:
        return self._cost

    def total_cost(self, session_id=None) -> float:
        return self._cost

    def session(self, session_id: str):
        """Return a no-op context manager."""
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            self._session_id = session_id
            yield

        return _ctx()


class _StubJitRLMemory:
    def __init__(self, config=None, persistence_path=None):
        self.trajectories = []
        self.saved = False

    def record_trajectory(self, traj):
        self.trajectories.append(traj)

    def save(self, path=None):
        self.saved = True

    def load(self, path=None):
        pass


class _StubPlanCache:
    def __init__(self):
        self.stored = []
        self.saved = False

    def store(self, task_description, steps, outcome="success"):
        self.stored.append((task_description, steps, outcome))

    def save(self, path: str):
        self.saved = True

    def load(self, path: str):
        pass


@dataclass
class _StubPlanStep:
    action_type: str
    description: str
    tool_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    estimated_cost: float = 0.0


@dataclass
class _StubTrajectoryStep:
    action_type: str
    reward: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _StubTrajectoryStatistics:
    steps: List[_StubTrajectoryStep]
    outcome: str = "success"
    total_reward: float = 0.0


# ---------------------------------------------------------------------------
# Test: EvolutionConfig defaults
# ---------------------------------------------------------------------------

class TestEvolutionConfigDefaults(unittest.TestCase):
    def test_default_values(self):
        cfg = ne.EvolutionConfig()
        self.assertEqual(cfg.generations, 20)
        self.assertEqual(cfg.population_size, 8)
        self.assertAlmostEqual(cfg.mutation_rate, 0.35)
        self.assertAlmostEqual(cfg.crossover_rate, 0.25)
        self.assertEqual(cfg.tournament_size, 3)
        self.assertEqual(cfg.evaluation_budget, 50)
        self.assertEqual(cfg.fitness_metric, "accuracy")
        self.assertAlmostEqual(cfg.max_usd, 10.00)
        self.assertFalse(cfg.dry_run)
        self.assertEqual(cfg.notification_backend, "file")


# ---------------------------------------------------------------------------
# Test: load_config from YAML
# ---------------------------------------------------------------------------

class TestLoadConfig(unittest.TestCase):
    def test_load_yaml_overrides_defaults(self):
        yaml_content = textwrap.dedent("""\
            evolution:
              generations: 5
              population_size: 4
              mutation_rate: 0.5
              fitness_metric: "f1"
            budget:
              max_usd: 2.50
            notification:
              backend: slack
              slack:
                webhook_url: "https://hooks.slack.com/test"
                channel: "#test"
        """)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            cfg = ne.load_config(path, {})
            self.assertEqual(cfg.generations, 5)
            self.assertEqual(cfg.population_size, 4)
            self.assertAlmostEqual(cfg.mutation_rate, 0.5)
            self.assertEqual(cfg.fitness_metric, "f1")
            self.assertAlmostEqual(cfg.max_usd, 2.50)
            self.assertEqual(cfg.notification_backend, "slack")
            self.assertEqual(cfg.slack_channel, "#test")
        finally:
            os.unlink(path)

    def test_cli_overrides_yaml(self):
        yaml_content = "evolution:\n  generations: 10\n"
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            cfg = ne.load_config(path, {"generations": 3, "dry_run": True})
            self.assertEqual(cfg.generations, 3)
            self.assertTrue(cfg.dry_run)
        finally:
            os.unlink(path)

    def test_none_path_returns_defaults(self):
        cfg = ne.load_config(None, {})
        self.assertIsInstance(cfg, ne.EvolutionConfig)
        self.assertEqual(cfg.generations, 20)


# ---------------------------------------------------------------------------
# Test: LockFile
# ---------------------------------------------------------------------------

class TestLockFile(unittest.TestCase):
    def test_acquire_and_release(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, "test.lock")
            lock = ne.LockFile(lock_path)
            self.assertTrue(lock.acquire())
            self.assertTrue(os.path.exists(lock_path))
            lock.release()
            self.assertFalse(os.path.exists(lock_path))

    def test_double_acquire_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, "test.lock")
            lock1 = ne.LockFile(lock_path)
            lock2 = ne.LockFile(lock_path)
            self.assertTrue(lock1.acquire())
            try:
                # Second acquire should fail (LOCK_NB)
                acquired = lock2.acquire()
                self.assertFalse(acquired)
            finally:
                lock1.release()

    def test_release_without_acquire_is_safe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = os.path.join(tmpdir, "test.lock")
            lock = ne.LockFile(lock_path)
            lock.release()  # Should not raise


# ---------------------------------------------------------------------------
# Test: Dry-run evaluation function
# ---------------------------------------------------------------------------

class TestDryRunEvalFn(unittest.TestCase):
    def test_returns_float_in_range(self):
        genome = _StubGenome(name="test_genome")
        score = ne._dry_run_evaluation_fn(genome, [])
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_deterministic(self):
        genome = _StubGenome(name="deterministic_genome")
        s1 = ne._dry_run_evaluation_fn(genome, [])
        s2 = ne._dry_run_evaluation_fn(genome, [])
        self.assertEqual(s1, s2)

    def test_different_genomes_may_differ(self):
        g1 = _StubGenome(name="alpha")
        g2 = _StubGenome(name="beta")
        # Not guaranteed to differ, but with two distinct names it typically will
        s1 = ne._dry_run_evaluation_fn(g1, [])
        s2 = ne._dry_run_evaluation_fn(g2, [])
        # At minimum both are valid floats
        self.assertIsInstance(s1, float)
        self.assertIsInstance(s2, float)


# ---------------------------------------------------------------------------
# Test: Notification — file backend
# ---------------------------------------------------------------------------

class TestNotificationFileBackend(unittest.TestCase):
    def test_appends_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            notif_path = os.path.join(tmpdir, "notif.log")
            cfg = ne.EvolutionConfig()
            cfg.notification_backend = "file"
            cfg.notification_file_path = notif_path

            ne.send_notification(cfg, "run completed")
            ne.send_notification(cfg, "second message")

            content = Path(notif_path).read_text(encoding="utf-8")
            self.assertIn("run completed", content)
            self.assertIn("second message", content)
            # Each line should contain a timestamp bracket
            for line in content.strip().splitlines():
                self.assertTrue(line.startswith("["))

    def test_creates_parent_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            deep_path = os.path.join(tmpdir, "subdir", "notif.log")
            cfg = ne.EvolutionConfig()
            cfg.notification_backend = "file"
            cfg.notification_file_path = deep_path

            ne.send_notification(cfg, "test")
            self.assertTrue(os.path.exists(deep_path))


# ---------------------------------------------------------------------------
# Test: Notification — slack backend (mocked HTTP)
# ---------------------------------------------------------------------------

class TestNotificationSlackBackend(unittest.TestCase):
    def test_sends_post_request(self):
        cfg = ne.EvolutionConfig()
        cfg.notification_backend = "slack"
        cfg.slack_webhook_url = "https://hooks.slack.com/services/TEST/HOOK"
        cfg.slack_channel = "#ch"

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_conn = MagicMock()
        mock_conn.getresponse.return_value = mock_resp

        with patch("http.client.HTTPSConnection", return_value=mock_conn):
            ne.send_notification(cfg, "test slack")

        mock_conn.request.assert_called_once()
        args, kwargs = mock_conn.request.call_args
        self.assertEqual(args[0], "POST")
        body = kwargs.get("body", args[2] if len(args) > 2 else None)
        decoded = json.loads(body.decode("utf-8"))
        self.assertIn("test slack", decoded["text"])

    def test_handles_missing_url_gracefully(self):
        cfg = ne.EvolutionConfig()
        cfg.notification_backend = "slack"
        cfg.slack_webhook_url = ""
        # Should log a warning but not raise
        ne.send_notification(cfg, "no url")

    def test_handles_http_error_gracefully(self):
        cfg = ne.EvolutionConfig()
        cfg.notification_backend = "slack"
        cfg.slack_webhook_url = "https://hooks.slack.com/services/TEST/HOOK"

        with patch("http.client.HTTPSConnection", side_effect=OSError("timeout")):
            # Should not raise
            ne.send_notification(cfg, "error test")


# ---------------------------------------------------------------------------
# Test: Report writing
# ---------------------------------------------------------------------------

class TestWriteReport(unittest.TestCase):
    def _make_report(self, tmpdir: str) -> ne.EvolutionReport:
        return ne.EvolutionReport(
            run_id="evo_20260329_020000",
            started_at="2026-03-29T02:00:00+00:00",
            finished_at="2026-03-29T04:00:00+00:00",
            status="completed",
            generations_run=5,
            generations_planned=5,
            best_genome_name="SEWOptimizer_mut_3",
            best_fitness=0.8765,
            final_diversity=0.625,
            total_cost_usd=1.23,
            budget_usd=10.00,
            generation_log=[
                {"generation": i + 1, "best_fitness": 0.5 + i * 0.05,
                 "diversity": 0.6, "elapsed_seconds": 1.2}
                for i in range(5)
            ],
            dry_run=False,
        )

    def test_creates_json_and_txt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = self._make_report(tmpdir)
            json_path, txt_path = ne._write_report(report, tmpdir)
            self.assertTrue(json_path.exists())
            self.assertTrue(txt_path.exists())

    def test_json_is_valid_and_contains_key_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = self._make_report(tmpdir)
            json_path, _ = ne._write_report(report, tmpdir)
            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(data["run_id"], "evo_20260329_020000")
            self.assertEqual(data["status"], "completed")
            self.assertAlmostEqual(data["best_fitness"], 0.8765)
            self.assertEqual(data["generations_run"], 5)
            self.assertIsInstance(data["generation_log"], list)

    def test_txt_contains_run_id_and_status(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = self._make_report(tmpdir)
            _, txt_path = ne._write_report(report, tmpdir)
            text = txt_path.read_text(encoding="utf-8")
            self.assertIn("evo_20260329_020000", text)
            self.assertIn("COMPLETED", text)
            self.assertIn("0.8765", text)

    def test_dry_run_label_in_txt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = self._make_report(tmpdir)
            report.dry_run = True
            _, txt_path = ne._write_report(report, tmpdir)
            text = txt_path.read_text(encoding="utf-8")
            self.assertIn("DRY-RUN", text)

    def test_report_with_nan_fitness(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = self._make_report(tmpdir)
            report.best_genome_name = None
            report.best_fitness = float("-inf")
            _, txt_path = ne._write_report(report, tmpdir)
            text = txt_path.read_text(encoding="utf-8")
            self.assertIn("N/A", text)


# ---------------------------------------------------------------------------
# Test: JitRL recording helper
# ---------------------------------------------------------------------------

class TestRecordGenerationToJitrl(unittest.TestCase):
    def test_records_correct_number_of_steps(self):
        memory = _StubJitRLMemory()
        gen_log = [
            {"generation": 1, "best_fitness": 0.5},
            {"generation": 2, "best_fitness": 0.7},
            {"generation": 3, "best_fitness": 0.8},
        ]
        ne._record_generation_to_jitrl(
            memory, gen_log, _StubTrajectoryStatistics, _StubTrajectoryStep
        )
        self.assertEqual(len(memory.trajectories), 1)
        traj = memory.trajectories[0]
        self.assertEqual(len(traj.steps), 3)

    def test_outcome_success_when_high_reward(self):
        memory = _StubJitRLMemory()
        gen_log = [{"generation": 1, "best_fitness": 0.9}]
        ne._record_generation_to_jitrl(
            memory, gen_log, _StubTrajectoryStatistics, _StubTrajectoryStep
        )
        self.assertEqual(memory.trajectories[0].outcome, "success")

    def test_outcome_partial_when_low_reward(self):
        memory = _StubJitRLMemory()
        gen_log = [{"generation": 1, "best_fitness": 0.2}]
        ne._record_generation_to_jitrl(
            memory, gen_log, _StubTrajectoryStatistics, _StubTrajectoryStep
        )
        self.assertEqual(memory.trajectories[0].outcome, "partial")

    def test_empty_log_does_nothing(self):
        memory = _StubJitRLMemory()
        ne._record_generation_to_jitrl(
            memory, [], _StubTrajectoryStatistics, _StubTrajectoryStep
        )
        self.assertEqual(len(memory.trajectories), 0)

    def test_neg_inf_fitness_maps_to_zero_reward(self):
        memory = _StubJitRLMemory()
        gen_log = [{"generation": 1, "best_fitness": float("-inf")}]
        ne._record_generation_to_jitrl(
            memory, gen_log, _StubTrajectoryStatistics, _StubTrajectoryStep
        )
        step = memory.trajectories[0].steps[0]
        self.assertEqual(step.reward, 0.0)


# ---------------------------------------------------------------------------
# Test: PlanCache strategy caching helper
# ---------------------------------------------------------------------------

class TestCacheBestStrategy(unittest.TestCase):
    def test_stores_genome_strategy(self):
        cache = _StubPlanCache()
        genome = _StubGenome(
            name="SEWOptimizer_mut_2",
            source_code="Use adaptive learning rate with momentum.",
            fitness_history=[0.6, 0.75, 0.82],
        )
        ne._cache_best_strategy(cache, genome, "evo_20260329_020000", _StubPlanStep)
        self.assertEqual(len(cache.stored), 1)
        task_desc, steps, outcome = cache.stored[0]
        self.assertIn("SEWOptimizer_mut_2", task_desc)
        self.assertEqual(outcome, "success")
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].tool_name, "HyperOptimizer")

    def test_does_nothing_when_genome_is_none(self):
        cache = _StubPlanCache()
        ne._cache_best_strategy(cache, None, "run_id", _StubPlanStep)
        self.assertEqual(len(cache.stored), 0)

    def test_truncates_long_source_code(self):
        cache = _StubPlanCache()
        genome = _StubGenome(
            name="g",
            source_code="x" * 2000,
            fitness_history=[0.9],
        )
        ne._cache_best_strategy(cache, genome, "run", _StubPlanStep)
        step = cache.stored[0][1][0]
        self.assertLessEqual(len(step.description), 500)


# ---------------------------------------------------------------------------
# Test: Shutdown flag
# ---------------------------------------------------------------------------

class TestShutdownFlag(unittest.TestCase):
    def test_initially_not_requested(self):
        s = ne._ShutdownFlag()
        self.assertFalse(s.requested)

    def test_request_sets_flag(self):
        s = ne._ShutdownFlag()
        s.request(signal.SIGTERM, None)
        self.assertTrue(s.requested)


# ---------------------------------------------------------------------------
# Test: CLI argument parsing
# ---------------------------------------------------------------------------

import signal  # noqa: E402 (needed above, re-import for clarity)


class TestCLIParsing(unittest.TestCase):
    def test_defaults(self):
        args = ne._parse_args([])
        self.assertFalse(args.dry_run)
        self.assertIsNone(args.generations)

    def test_dry_run_flag(self):
        args = ne._parse_args(["--dry-run"])
        self.assertTrue(args.dry_run)

    def test_generations_override(self):
        args = ne._parse_args(["--generations", "7"])
        self.assertEqual(args.generations, 7)

    def test_config_path(self):
        args = ne._parse_args(["--config", "/tmp/my.yaml"])
        self.assertEqual(args.config, "/tmp/my.yaml")


# ---------------------------------------------------------------------------
# Test: Full dry-run end-to-end execution
# ---------------------------------------------------------------------------

class TestDryRunEndToEnd(unittest.TestCase):
    """
    Exercises the full run_evolution() path with all AgentAugi imports mocked.
    No LLM calls are made; all IO uses temporary directories.
    """

    def _make_cfg(self, tmpdir: str) -> ne.EvolutionConfig:
        cfg = ne.EvolutionConfig()
        cfg.dry_run = True
        cfg.generations = 2
        cfg.population_size = 2
        cfg.population_state = os.path.join(tmpdir, "pop.json")
        cfg.jitrl_state = os.path.join(tmpdir, "jitrl.json")
        cfg.plan_cache_path = os.path.join(tmpdir, "plan_cache.json")
        cfg.reports_dir = os.path.join(tmpdir, "reports")
        cfg.lock_file = os.path.join(tmpdir, "test.lock")
        cfg.notification_backend = "file"
        cfg.notification_file_path = os.path.join(tmpdir, "notif.log")
        return cfg

    def _mock_agentaugi(self, stub_hyper: _StubHyperOptimizer):
        """Return a patch target for _import_agentaugi."""
        class _FakeCostBudgetExceeded(Exception):
            pass

        stub_tracker = _StubCostTracker()
        stub_jitrl = _StubJitRLMemory()
        stub_cache = _StubPlanCache()

        class _FakeHyperOptimizerCls:
            def __new__(cls, config=None, base_optimizers=None, evaluation_fn=None, **kw):
                stub_hyper.config = config
                stub_hyper.evaluation_fn = evaluation_fn
                return stub_hyper

        class _FakeJitRLMemoryCls:
            def __new__(cls, config=None, persistence_path=None, **kw):
                return stub_jitrl

        def _fake_import():
            return (
                (
                    _FakeHyperOptimizerCls,
                    MagicMock(),  # HyperOptimizerConfig
                    MagicMock(),  # OptimizerGenome
                ),
                (lambda: stub_tracker, _FakeCostBudgetExceeded),
                (_StubPlanCache, _StubPlanStep),
                (
                    _FakeJitRLMemoryCls,
                    MagicMock(),  # JitRLConfig
                    _StubTrajectoryStatistics,
                    _StubTrajectoryStep,
                ),
            )

        return _fake_import, stub_tracker, stub_jitrl, stub_cache

    def test_dry_run_returns_0(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._make_cfg(tmpdir)
            stub_hyper = _StubHyperOptimizer(
                config=MagicMock(),
                base_optimizers=[],
                evaluation_fn=ne._dry_run_evaluation_fn,
            )
            fake_import, _, _, _ = self._mock_agentaugi(stub_hyper)

            with patch.object(ne, "_import_agentaugi", fake_import), \
                 patch.object(ne, "_resolve_seed_optimizers", return_value=[]), \
                 patch.object(ne, "_build_meta_llm_fn", return_value=None):
                exit_code = ne.run_evolution(cfg)

            self.assertEqual(exit_code, 0)

    def test_report_is_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._make_cfg(tmpdir)
            stub_hyper = _StubHyperOptimizer(
                config=MagicMock(),
                base_optimizers=[],
                evaluation_fn=ne._dry_run_evaluation_fn,
            )
            fake_import, _, _, _ = self._mock_agentaugi(stub_hyper)

            with patch.object(ne, "_import_agentaugi", fake_import), \
                 patch.object(ne, "_resolve_seed_optimizers", return_value=[]), \
                 patch.object(ne, "_build_meta_llm_fn", return_value=None):
                ne.run_evolution(cfg)

            reports = list(Path(cfg.reports_dir).glob("*.json"))
            self.assertGreaterEqual(len(reports), 1)

    def test_notification_file_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._make_cfg(tmpdir)
            stub_hyper = _StubHyperOptimizer(
                config=MagicMock(),
                base_optimizers=[],
                evaluation_fn=ne._dry_run_evaluation_fn,
            )
            fake_import, _, _, _ = self._mock_agentaugi(stub_hyper)

            with patch.object(ne, "_import_agentaugi", fake_import), \
                 patch.object(ne, "_resolve_seed_optimizers", return_value=[]), \
                 patch.object(ne, "_build_meta_llm_fn", return_value=None):
                ne.run_evolution(cfg)

            self.assertTrue(os.path.exists(cfg.notification_file_path))
            content = Path(cfg.notification_file_path).read_text(encoding="utf-8")
            self.assertTrue(len(content) > 0)

    def test_lock_released_on_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._make_cfg(tmpdir)
            stub_hyper = _StubHyperOptimizer(
                config=MagicMock(),
                base_optimizers=[],
                evaluation_fn=ne._dry_run_evaluation_fn,
            )
            fake_import, _, _, _ = self._mock_agentaugi(stub_hyper)

            with patch.object(ne, "_import_agentaugi", fake_import), \
                 patch.object(ne, "_resolve_seed_optimizers", return_value=[]), \
                 patch.object(ne, "_build_meta_llm_fn", return_value=None):
                ne.run_evolution(cfg)

            # After run completes, lock file should be gone
            self.assertFalse(os.path.exists(cfg.lock_file))

    def test_concurrent_run_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._make_cfg(tmpdir)
            # Acquire the lock externally to simulate a concurrent run
            lock = ne.LockFile(cfg.lock_file)
            lock.acquire()
            try:
                exit_code = ne.run_evolution(cfg)
                self.assertEqual(exit_code, 1)
            finally:
                lock.release()

    def test_shutdown_flag_stops_after_generation(self):
        """Patching shutdown flag to trigger after first generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = self._make_cfg(tmpdir)
            cfg.generations = 10  # Would run 10, but shutdown fires after 1

            call_count = 0

            class _EarlyStopHyper(_StubHyperOptimizer):
                def evolve_generation(self, task_suite):
                    nonlocal call_count
                    call_count += 1
                    return super().evolve_generation(task_suite)

            stub_hyper = _EarlyStopHyper(
                config=MagicMock(),
                base_optimizers=[],
                evaluation_fn=ne._dry_run_evaluation_fn,
            )
            fake_import, _, _, _ = self._mock_agentaugi(stub_hyper)

            original_run = ne._inner_run

            def patched_inner_run(cfg, run_id, started_at, shutdown, gen_log):
                # Trigger shutdown after first generation by monkey-patching
                original_evolve = stub_hyper.evolve_generation

                def _once_then_shutdown(task_suite):
                    result = original_evolve(task_suite)
                    shutdown.requested = True  # signal after 1 gen
                    stub_hyper.evolve_generation = original_evolve
                    return result

                stub_hyper.evolve_generation = _once_then_shutdown
                return original_run(cfg, run_id, started_at, shutdown, gen_log)

            with patch.object(ne, "_import_agentaugi", fake_import), \
                 patch.object(ne, "_resolve_seed_optimizers", return_value=[]), \
                 patch.object(ne, "_build_meta_llm_fn", return_value=None), \
                 patch.object(ne, "_inner_run", patched_inner_run):
                exit_code = ne.run_evolution(cfg)

            self.assertLessEqual(call_count, 2)


# ---------------------------------------------------------------------------
# Test: _resolve_path
# ---------------------------------------------------------------------------

class TestResolvePath(unittest.TestCase):
    def test_absolute_path_unchanged(self):
        p = ne._resolve_path("/tmp/foo/bar.json")
        self.assertEqual(str(p), "/tmp/foo/bar.json")

    def test_relative_path_anchored_to_repo_root(self):
        p = ne._resolve_path("data/evolution/pop.json")
        self.assertTrue(p.is_absolute())
        self.assertTrue(str(p).endswith("data/evolution/pop.json"))


# ---------------------------------------------------------------------------
# Test: _build_task_suite
# ---------------------------------------------------------------------------

class TestBuildTaskSuite(unittest.TestCase):
    def test_returns_non_empty_list(self):
        tasks = ne._build_task_suite()
        self.assertIsInstance(tasks, list)
        self.assertGreater(len(tasks), 0)

    def test_each_task_has_task_key(self):
        for task in ne._build_task_suite():
            self.assertIn("task", task)


if __name__ == "__main__":
    unittest.main()

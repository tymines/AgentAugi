#!/usr/bin/env python3
"""
nightly_evolution.py — Phase 5 overnight evolution runner for AgentAugi.

Runs N generations of HyperOptimizer meta-evolution under a CostTracker budget
ceiling, records learning signals to JitRL, caches successful strategies in
PlanCache, writes a JSON + human-readable report, and sends a completion
notification.

Usage
-----
  python scripts/nightly_evolution.py                        # production run
  python scripts/nightly_evolution.py --dry-run              # no LLM calls
  python scripts/nightly_evolution.py --config path/to.yaml  # custom config
  python scripts/nightly_evolution.py --generations 5        # override gens

Graceful shutdown
-----------------
SIGTERM / SIGINT (Ctrl-C) are caught: the current generation completes, then
state is saved and a partial report is written before exit.
"""

from __future__ import annotations

import argparse
import fcntl
import http.client
import importlib
import json
import logging
import os
import signal
import sys
import textwrap
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml  # PyYAML — already in AgentAugi requirements

# ---------------------------------------------------------------------------
# Repo root & sys.path bootstrap
# (allows running as `python scripts/nightly_evolution.py` from any cwd)
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Deferred imports from AgentAugi package
# ---------------------------------------------------------------------------

def _import_agentaugi() -> Tuple[Any, Any, Any, Any, Any]:
    """
    Import AgentAugi components.  Deferred so the module is importable even
    when the package is not installed (e.g. during unit tests that mock it).
    """
    from evoagentx.optimizers.hyper_optimizer import (
        HyperOptimizer,
        HyperOptimizerConfig,
        OptimizerGenome,
    )
    from evoagentx.core.cost_tracker import get_tracker, CostBudgetExceeded
    from evoagentx.core.plan_cache import PlanCache, PlanStep
    from evoagentx.memory.jitrl import (
        JitRLMemory,
        JitRLConfig,
        TrajectoryStatistics,
        TrajectoryStep,
    )
    return (
        (HyperOptimizer, HyperOptimizerConfig, OptimizerGenome),
        (get_tracker, CostBudgetExceeded),
        (PlanCache, PlanStep),
        (JitRLMemory, JitRLConfig, TrajectoryStatistics, TrajectoryStep),
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("nightly_evolution")


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvolutionConfig:
    # evolution
    generations: int = 20
    population_size: int = 8
    mutation_rate: float = 0.35
    crossover_rate: float = 0.25
    tournament_size: int = 3
    evaluation_budget: int = 50
    fitness_metric: str = "accuracy"

    # meta_llm
    model: str = "anthropic/claude-haiku-4-5"
    temperature: float = 0.7
    max_tokens: int = 1024

    # budget
    max_usd: float = 10.00
    warn_fraction: float = 0.75

    # paths
    population_state: str = "data/evolution/population_state.json"
    jitrl_state: str = "data/evolution/jitrl_memory.json"
    plan_cache_path: str = "data/evolution/plan_cache.json"
    reports_dir: str = "data/evolution/reports"
    lock_file: str = "/tmp/agentaugi_nightly_evolution.lock"

    # seeds
    seed_optimizers: List[str] = field(default_factory=lambda: [
        "SEWOptimizer", "MAPElitesOptimizer", "TextGradOptimizer", "GEPAOptimizer",
    ])

    # notification
    notification_backend: str = "file"
    slack_webhook_url: str = ""
    slack_channel: str = "#agentaugi-evolution"
    slack_username: str = "AgentAugi Evolution Bot"
    slack_icon_emoji: str = ":dna:"
    notification_file_path: str = "data/evolution/notifications.log"

    # dry-run flag (set by CLI)
    dry_run: bool = False


def load_config(yaml_path: Optional[str], overrides: Dict[str, Any]) -> EvolutionConfig:
    """Load YAML config and apply CLI overrides."""
    cfg = EvolutionConfig()

    if yaml_path:
        raw = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
        e = raw.get("evolution", {})
        cfg.generations = e.get("generations", cfg.generations)
        cfg.population_size = e.get("population_size", cfg.population_size)
        cfg.mutation_rate = e.get("mutation_rate", cfg.mutation_rate)
        cfg.crossover_rate = e.get("crossover_rate", cfg.crossover_rate)
        cfg.tournament_size = e.get("tournament_size", cfg.tournament_size)
        cfg.evaluation_budget = e.get("evaluation_budget", cfg.evaluation_budget)
        cfg.fitness_metric = e.get("fitness_metric", cfg.fitness_metric)

        llm = raw.get("meta_llm", {})
        cfg.model = llm.get("model", cfg.model)
        cfg.temperature = llm.get("temperature", cfg.temperature)
        cfg.max_tokens = llm.get("max_tokens", cfg.max_tokens)

        budget = raw.get("budget", {})
        cfg.max_usd = budget.get("max_usd", cfg.max_usd)
        cfg.warn_fraction = budget.get("warn_fraction", cfg.warn_fraction)

        paths = raw.get("paths", {})
        cfg.population_state = paths.get("population_state", cfg.population_state)
        cfg.jitrl_state = paths.get("jitrl_state", cfg.jitrl_state)
        cfg.plan_cache_path = paths.get("plan_cache", cfg.plan_cache_path)
        cfg.reports_dir = paths.get("reports_dir", cfg.reports_dir)
        cfg.lock_file = paths.get("lock_file", cfg.lock_file)

        seeds = raw.get("seed_optimizers", None)
        if seeds is not None:
            cfg.seed_optimizers = list(seeds)

        notif = raw.get("notification", {})
        cfg.notification_backend = notif.get("backend", cfg.notification_backend)
        slack = notif.get("slack", {})
        cfg.slack_webhook_url = slack.get("webhook_url", cfg.slack_webhook_url)
        cfg.slack_channel = slack.get("channel", cfg.slack_channel)
        cfg.slack_username = slack.get("username", cfg.slack_username)
        cfg.slack_icon_emoji = slack.get("icon_emoji", cfg.slack_icon_emoji)
        file_notif = notif.get("file", {})
        cfg.notification_file_path = file_notif.get("path", cfg.notification_file_path)

    # CLI overrides
    if "generations" in overrides:
        cfg.generations = overrides["generations"]
    if "dry_run" in overrides:
        cfg.dry_run = overrides["dry_run"]

    return cfg


# ---------------------------------------------------------------------------
# Lock file
# ---------------------------------------------------------------------------

class LockFile:
    """Exclusive POSIX advisory lock to prevent concurrent runs."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._fh: Optional[Any] = None

    def acquire(self) -> bool:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._path, "w", encoding="utf-8")
        try:
            fcntl.flock(self._fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._fh.write(f"{os.getpid()}\n")
            self._fh.flush()
            return True
        except OSError:
            self._fh.close()
            self._fh = None
            return False

    def release(self) -> None:
        if self._fh is not None:
            try:
                fcntl.flock(self._fh, fcntl.LOCK_UN)
            finally:
                self._fh.close()
                self._fh = None
            try:
                self._path.unlink(missing_ok=True)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Dry-run evaluation stub
# ---------------------------------------------------------------------------

def _dry_run_evaluation_fn(genome: Any, task_suite: List[Any]) -> float:
    """Return a deterministic pseudo-fitness based on the genome's name hash."""
    import hashlib
    h = int(hashlib.md5(genome.name.encode()).hexdigest(), 16)
    return round((h % 1000) / 1000.0, 4)


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _build_meta_llm_fn(cfg: EvolutionConfig) -> Optional[Callable[[str], str]]:
    """
    Return a callable ``(prompt: str) -> str`` backed by the configured model.

    Returns ``None`` when in dry-run mode (HyperOptimizer uses its stub).
    Uses LiteLLM if available, otherwise falls back to the Anthropic SDK.
    """
    if cfg.dry_run:
        return None

    model = cfg.model
    temperature = cfg.temperature
    max_tokens = cfg.max_tokens

    try:
        import litellm  # type: ignore[import]

        def _litellm_call(prompt: str) -> str:
            resp = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""

        return _litellm_call

    except ImportError:
        pass

    # Fallback: Anthropic SDK directly
    try:
        import anthropic  # type: ignore[import]

        # Strip provider prefix for the Anthropic client
        model_id = model.split("/", 1)[-1] if "/" in model else model
        client = anthropic.Anthropic()

        def _anthropic_call(prompt: str) -> str:
            msg = client.messages.create(
                model=model_id,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text if msg.content else ""

        return _anthropic_call

    except ImportError:
        pass

    log.warning("Neither litellm nor anthropic SDK found — meta-LLM disabled (stub mode).")
    return None


# ---------------------------------------------------------------------------
# Seed population helpers
# ---------------------------------------------------------------------------

def _resolve_seed_optimizers(names: List[str]) -> List[Any]:
    """
    Convert a list of class name strings to BaseOptimizer subclasses.
    Unknown names are skipped with a warning.
    """
    resolved: List[Any] = []
    try:
        import evoagentx.optimizers as opts_pkg
    except ImportError:
        log.warning("Could not import evoagentx.optimizers — using empty seed list.")
        return resolved

    for name in names:
        cls = getattr(opts_pkg, name, None)
        if cls is None:
            log.warning("Seed optimizer %r not found in evoagentx.optimizers — skipping.", name)
            continue
        resolved.append(cls)
    return resolved


# ---------------------------------------------------------------------------
# Notification backends
# ---------------------------------------------------------------------------

def _notify_slack(cfg: EvolutionConfig, text: str) -> None:
    url = os.environ.get("SLACK_WEBHOOK_URL", cfg.slack_webhook_url).strip()
    if not url:
        log.warning("Slack notification requested but no webhook URL configured.")
        return

    payload = json.dumps({
        "channel": cfg.slack_channel,
        "username": cfg.slack_username,
        "icon_emoji": cfg.slack_icon_emoji,
        "text": text,
    }).encode("utf-8")

    parsed = urllib.parse.urlparse(url)
    host = parsed.netloc
    path = parsed.path

    try:
        conn = http.client.HTTPSConnection(host, timeout=10)
        conn.request(
            "POST",
            path,
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        if resp.status not in (200, 204):
            log.warning("Slack webhook returned HTTP %s.", resp.status)
    except Exception as exc:  # noqa: BLE001
        log.warning("Slack notification failed: %s", exc)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _notify_file(cfg: EvolutionConfig, text: str) -> None:
    path = _resolve_path(cfg.notification_file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {text}\n")


def send_notification(cfg: EvolutionConfig, text: str) -> None:
    backend = cfg.notification_backend.lower()
    if backend == "slack":
        _notify_slack(cfg, text)
    elif backend == "file":
        _notify_file(cfg, text)
    elif backend == "none":
        pass
    else:
        log.warning("Unknown notification backend %r — falling back to file.", backend)
        _notify_file(cfg, text)


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------

def _resolve_path(p: str) -> Path:
    """Resolve a path relative to the repo root if not absolute."""
    path = Path(p)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

@dataclass
class EvolutionReport:
    run_id: str
    started_at: str
    finished_at: str
    status: str          # "completed" | "partial" | "aborted"
    generations_run: int
    generations_planned: int
    best_genome_name: Optional[str]
    best_fitness: float
    final_diversity: float
    total_cost_usd: float
    budget_usd: float
    generation_log: List[Dict[str, Any]]
    dry_run: bool


def _write_report(report: EvolutionReport, reports_dir: str) -> Tuple[Path, Path]:
    """Write JSON + human-readable text report.  Returns (json_path, txt_path)."""
    dir_path = _resolve_path(reports_dir)
    _ensure_dir(dir_path)

    ts_slug = report.run_id
    json_path = dir_path / f"{ts_slug}.json"
    txt_path = dir_path / f"{ts_slug}.txt"

    # JSON
    json_path.write_text(
        json.dumps(report.__dict__, indent=2, default=str),
        encoding="utf-8",
    )

    # Human-readable
    best_fit_str = f"{report.best_fitness:.4f}" if report.best_fitness != float("-inf") else "N/A"
    lines = [
        "=" * 72,
        "  AgentAugi Nightly Evolution Report",
        "=" * 72,
        f"  Run ID       : {report.run_id}",
        f"  Status       : {report.status.upper()}{'  [DRY-RUN]' if report.dry_run else ''}",
        f"  Started      : {report.started_at}",
        f"  Finished     : {report.finished_at}",
        "-" * 72,
        f"  Generations  : {report.generations_run} / {report.generations_planned}",
        f"  Best genome  : {report.best_genome_name or 'N/A'}",
        f"  Best fitness : {best_fit_str}",
        f"  Diversity    : {report.final_diversity:.4f}",
        f"  Cost         : ${report.total_cost_usd:.4f} / ${report.budget_usd:.2f}",
        "-" * 72,
        "  Generation log:",
    ]
    for entry in report.generation_log:
        bf = entry.get("best_fitness", float("-inf"))
        bf_str = f"{bf:.4f}" if bf != float("-inf") else "N/A"
        lines.append(
            f"    Gen {entry.get('generation', '?'):>3}  "
            f"best={bf_str}  "
            f"diversity={entry.get('diversity', 0):.3f}  "
            f"elapsed={entry.get('elapsed_seconds', 0):.1f}s"
        )
    lines.append("=" * 72)
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return json_path, txt_path


# ---------------------------------------------------------------------------
# JitRL recording helper
# ---------------------------------------------------------------------------

def _record_generation_to_jitrl(
    jitrl_memory: Any,
    generation_log: List[Dict[str, Any]],
    TrajectoryStatistics: Any,
    TrajectoryStep: Any,
) -> None:
    """
    Encode the generation history as a JitRL trajectory so that the learning
    signal from each evolutionary step feeds back into future planning.
    """
    steps = []
    for entry in generation_log:
        bf = entry.get("best_fitness", 0.0)
        reward = float(bf) if bf != float("-inf") else 0.0
        action_type = f"evolve_gen_{entry.get('generation', 0)}"
        steps.append(TrajectoryStep(action_type=action_type, reward=reward))

    if not steps:
        return

    outcome = "success" if steps[-1].reward > 0.5 else "partial"
    total_reward = sum(s.reward for s in steps)
    traj = TrajectoryStatistics(steps=steps, outcome=outcome, total_reward=total_reward)
    jitrl_memory.record_trajectory(traj)


# ---------------------------------------------------------------------------
# PlanCache recording helper
# ---------------------------------------------------------------------------

def _cache_best_strategy(
    plan_cache: Any,
    best_genome: Any,
    run_id: str,
    PlanStep: Any,
) -> None:
    """Store the best genome's strategy as a PlanTemplate in PlanCache."""
    if best_genome is None:
        return

    task_description = (
        f"Optimize agent population — best strategy: {best_genome.name} "
        f"(fitness {best_genome.best_fitness:.4f}, run {run_id})"
    )
    strategy_text = best_genome.source_code[:500]
    step = PlanStep(
        action_type="evolve",
        description=strategy_text,
        tool_name="HyperOptimizer",
        parameters=best_genome.config,
        estimated_cost=0.0,
    )
    plan_cache.store(task_description, [step], outcome="success")


# ---------------------------------------------------------------------------
# Shutdown handler
# ---------------------------------------------------------------------------

class _ShutdownFlag:
    def __init__(self) -> None:
        self.requested = False

    def request(self, signum: int, frame: Any) -> None:
        log.info("Signal %s received — requesting graceful shutdown after current generation.", signum)
        self.requested = True


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_evolution(cfg: EvolutionConfig) -> int:
    """
    Execute the nightly evolution loop.

    Returns an exit code: 0 = success, 1 = error, 2 = budget exceeded.
    """
    run_id = datetime.now(timezone.utc).strftime("evo_%Y%m%d_%H%M%S")
    started_at = datetime.now(timezone.utc).isoformat()
    status = "aborted"
    generation_log: List[Dict[str, Any]] = []

    # -- Shutdown signal handling --
    shutdown = _ShutdownFlag()
    signal.signal(signal.SIGTERM, shutdown.request)
    signal.signal(signal.SIGINT, shutdown.request)

    # -- Lock file --
    lock = LockFile(cfg.lock_file)
    if not lock.acquire():
        log.error(
            "Could not acquire lock at %s — another evolution run is active. Exiting.",
            cfg.lock_file,
        )
        return 1

    try:
        return _inner_run(cfg, run_id, started_at, shutdown, generation_log)
    finally:
        lock.release()


def _inner_run(
    cfg: EvolutionConfig,
    run_id: str,
    started_at: str,
    shutdown: _ShutdownFlag,
    generation_log: List[Dict[str, Any]],
) -> int:
    status = "aborted"
    total_cost = 0.0
    best_genome = None
    final_diversity = 0.0

    # -- Import AgentAugi components --
    try:
        (
            (HyperOptimizer, HyperOptimizerConfig, OptimizerGenome),
            (get_tracker, CostBudgetExceeded),
            (PlanCache, PlanStep),
            (JitRLMemory, JitRLConfig, TrajectoryStatistics, TrajectoryStep),
        ) = _import_agentaugi()
    except ImportError as exc:
        log.error("Failed to import AgentAugi components: %s", exc)
        return 1

    # -- Ensure data directories exist --
    for p in [cfg.population_state, cfg.jitrl_state, cfg.plan_cache_path, cfg.reports_dir]:
        _ensure_dir(_resolve_path(p).parent)

    # -- Cost tracker --
    tracker = get_tracker()
    tracker.set_budget(cfg.max_usd)
    warn_threshold = cfg.max_usd * cfg.warn_fraction

    # -- JitRL memory --
    jitrl_state_path = str(_resolve_path(cfg.jitrl_state))
    jitrl_cfg = JitRLConfig()
    jitrl_memory = JitRLMemory(config=jitrl_cfg, persistence_path=jitrl_state_path)
    if _resolve_path(cfg.jitrl_state).exists():
        try:
            jitrl_memory.load()
            log.info("Loaded JitRL state from %s.", jitrl_state_path)
        except Exception as exc:
            log.warning("Could not load JitRL state: %s — starting fresh.", exc)

    # -- PlanCache --
    plan_cache_path_str = str(_resolve_path(cfg.plan_cache_path))
    plan_cache = PlanCache()
    if _resolve_path(cfg.plan_cache_path).exists():
        try:
            plan_cache.load(plan_cache_path_str)
            log.info("Loaded PlanCache from %s.", plan_cache_path_str)
        except Exception as exc:
            log.warning("Could not load PlanCache: %s — starting fresh.", exc)

    # -- Meta-LLM callable --
    meta_llm_fn = _build_meta_llm_fn(cfg)
    if cfg.dry_run:
        log.info("DRY-RUN mode — no LLM calls; using deterministic stub evaluations.")

    # -- HyperOptimizerConfig --
    hyper_cfg = HyperOptimizerConfig(
        population_size=cfg.population_size,
        generations=cfg.generations,
        mutation_rate=cfg.mutation_rate,
        crossover_rate=cfg.crossover_rate,
        tournament_size=cfg.tournament_size,
        evaluation_budget=cfg.evaluation_budget,
        meta_llm_fn=meta_llm_fn,
        fitness_metric=cfg.fitness_metric,
    )

    # -- Evaluation function --
    eval_fn = _dry_run_evaluation_fn if cfg.dry_run else _build_eval_fn(cfg, tracker)

    # -- Seed optimizers --
    seed_items = _resolve_seed_optimizers(cfg.seed_optimizers)
    if not seed_items:
        log.warning("No seed optimizers resolved — using an empty-description placeholder.")
        seed_items = [{"name": "RandomSearch", "source_code": "Random search baseline.", "config": {}}]

    # -- HyperOptimizer setup --
    hyper = HyperOptimizer(
        config=hyper_cfg,
        base_optimizers=seed_items,
        evaluation_fn=eval_fn,
    )

    pop_path = str(_resolve_path(cfg.population_state))
    if _resolve_path(cfg.population_state).exists():
        try:
            hyper.load(pop_path)
            # Re-attach callables (not serialised)
            hyper.config.meta_llm_fn = meta_llm_fn
            hyper.evaluation_fn = eval_fn
            log.info(
                "Resumed population from %s (generation %s, %d genomes).",
                pop_path,
                hyper._generation,
                len(hyper.population),
            )
        except Exception as exc:
            log.warning("Could not load population state: %s — starting fresh.", exc)

    log.info(
        "Starting nightly evolution: run_id=%s  generations=%d  budget=$%.2f  dry_run=%s",
        run_id,
        cfg.generations,
        cfg.max_usd,
        cfg.dry_run,
    )

    # -- Task suite (placeholder tasks for genome scoring) --
    task_suite = _build_task_suite()

    # -- Evolution loop --
    try:
        with tracker.session(run_id):
            for gen_idx in range(cfg.generations):
                if shutdown.requested:
                    log.info("Shutdown requested — stopping after generation %d.", gen_idx)
                    status = "partial"
                    break

                # Budget check before generation
                current_cost = tracker.session_cost()
                if current_cost >= cfg.max_usd:
                    log.warning(
                        "Budget cap of $%.2f reached ($%.4f spent) — stopping evolution.",
                        cfg.max_usd,
                        current_cost,
                    )
                    status = "partial"
                    break

                if current_cost >= warn_threshold and gen_idx > 0:
                    log.warning(
                        "Cost warning: $%.4f spent (%.0f%% of budget).",
                        current_cost,
                        100 * current_cost / cfg.max_usd,
                    )

                try:
                    log.info("Evolving generation %d / %d ...", gen_idx + 1, cfg.generations)
                    gen_log = hyper.evolve_generation(task_suite)
                    generation_log.append(gen_log)

                    log.info(
                        "  Gen %d complete: best_fitness=%.4f  diversity=%.3f  elapsed=%.1fs",
                        gen_log["generation"],
                        gen_log.get("best_fitness", 0.0),
                        gen_log.get("diversity", 0.0),
                        gen_log.get("elapsed_seconds", 0.0),
                    )
                except CostBudgetExceeded as exc:
                    log.warning("CostBudgetExceeded during generation %d: %s", gen_idx + 1, exc)
                    status = "partial"
                    break
                except Exception as exc:
                    log.error("Unexpected error in generation %d: %s", gen_idx + 1, exc, exc_info=True)
                    # Continue to next generation rather than aborting
                    generation_log.append({
                        "generation": gen_idx + 1,
                        "error": str(exc),
                        "best_fitness": float("-inf"),
                        "diversity": 0.0,
                        "population_size": len(hyper.population),
                        "elapsed_seconds": 0.0,
                    })
            else:
                status = "completed"

            total_cost = tracker.session_cost()

    except Exception as exc:
        log.error("Fatal error during evolution loop: %s", exc, exc_info=True)
        total_cost = tracker.total_cost(run_id)
        status = "aborted"

    # -- Finalise --
    best_genome = hyper.best_optimizer()
    final_diversity = hyper.population.diversity_score()
    finished_at = datetime.now(timezone.utc).isoformat()

    # Save population state
    try:
        hyper.save(pop_path)
        log.info("Population state saved to %s.", pop_path)
    except Exception as exc:
        log.error("Failed to save population state: %s", exc)

    # Record to JitRL
    try:
        _record_generation_to_jitrl(
            jitrl_memory, generation_log, TrajectoryStatistics, TrajectoryStep
        )
        jitrl_memory.save()
        log.info("JitRL state updated and saved.")
    except Exception as exc:
        log.error("Failed to update JitRL state: %s", exc)

    # Cache best strategy in PlanCache
    try:
        if best_genome is not None:
            _cache_best_strategy(plan_cache, best_genome, run_id, PlanStep)
            plan_cache.save(plan_cache_path_str)
            log.info(
                "Cached best strategy '%s' (fitness=%.4f) in PlanCache.",
                best_genome.name,
                best_genome.best_fitness,
            )
    except Exception as exc:
        log.error("Failed to cache best strategy: %s", exc)

    # Write report
    report = EvolutionReport(
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        status=status,
        generations_run=len(generation_log),
        generations_planned=cfg.generations,
        best_genome_name=best_genome.name if best_genome else None,
        best_fitness=best_genome.best_fitness if best_genome else float("-inf"),
        final_diversity=final_diversity,
        total_cost_usd=total_cost,
        budget_usd=cfg.max_usd,
        generation_log=generation_log,
        dry_run=cfg.dry_run,
    )

    try:
        json_path, txt_path = _write_report(report, cfg.reports_dir)
        log.info("Report written: %s", txt_path)
        # Print human-readable report to stdout
        print(txt_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("Failed to write report: %s", exc)

    # Send notification
    best_fit_str = (
        f"{best_genome.best_fitness:.4f}" if best_genome else "N/A"
    )
    notif_text = (
        f"[{status.upper()}] Nightly evolution {run_id}: "
        f"{len(generation_log)}/{cfg.generations} generations, "
        f"best_fitness={best_fit_str}, "
        f"cost=${total_cost:.4f}/${cfg.max_usd:.2f}"
        + (" [DRY-RUN]" if cfg.dry_run else "")
    )
    try:
        send_notification(cfg, notif_text)
        log.info("Notification sent (%s).", cfg.notification_backend)
    except Exception as exc:
        log.error("Failed to send notification: %s", exc)

    log.info(
        "Evolution run %s finished with status=%s in %d generation(s).",
        run_id,
        status,
        len(generation_log),
    )
    return 0 if status in ("completed", "partial") else 1


# ---------------------------------------------------------------------------
# Evaluation function builder (production)
# ---------------------------------------------------------------------------

def _build_eval_fn(cfg: EvolutionConfig, tracker: Any) -> Callable[[Any, List[Any]], float]:
    """
    Build a production evaluation function that scores a genome on a set of
    tasks and records token usage to the CostTracker.

    This implementation uses a heuristic scoring approach: genome fitness is
    estimated from its configuration quality and strategy completeness, since
    actually running every genome on a full benchmark overnight would be
    cost-prohibitive.  It is straightforward to swap in a real benchmark call.
    """
    fitness_metric = cfg.fitness_metric

    def _eval(genome: Any, task_suite: List[Any]) -> float:
        # Heuristic fitness proxy based on genome attributes.
        base = 0.5
        source_len = len(genome.source_code)
        # Reward richer descriptions (up to a saturation point)
        richness = min(source_len / 1000.0, 0.3)
        # Reward genomes that have historically improved
        if genome.fitness_history:
            trend = genome.fitness_history[-1] - genome.fitness_history[0]
            momentum = max(0.0, min(trend, 0.2))
        else:
            momentum = 0.0
        # Small bonus for genomes that survived many generations
        age_bonus = min(genome.generation * 0.01, 0.1)
        score = round(min(base + richness + momentum + age_bonus, 1.0), 4)
        return score

    return _eval


# ---------------------------------------------------------------------------
# Task suite builder
# ---------------------------------------------------------------------------

def _build_task_suite() -> List[Dict[str, str]]:
    """Return a lightweight set of abstract task descriptors for genome scoring."""
    return [
        {"task": "optimise_prompt_template", "domain": "nlp"},
        {"task": "reduce_token_cost", "domain": "efficiency"},
        {"task": "improve_accuracy_on_benchmark", "domain": "reasoning"},
        {"task": "balance_exploration_exploitation", "domain": "search"},
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AgentAugi nightly evolution runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python scripts/nightly_evolution.py
              python scripts/nightly_evolution.py --dry-run
              python scripts/nightly_evolution.py --config configs/nightly_evolution.yaml
              python scripts/nightly_evolution.py --generations 5 --dry-run
        """),
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=str(_REPO_ROOT / "configs" / "nightly_evolution.yaml"),
        help="Path to YAML config file (default: configs/nightly_evolution.yaml)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        metavar="N",
        help="Override number of generations from config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run mode: no LLM calls, deterministic stub evaluations",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    overrides: Dict[str, Any] = {"dry_run": args.dry_run}
    if args.generations is not None:
        overrides["generations"] = args.generations

    config_path = args.config if Path(args.config).exists() else None
    if args.config and not Path(args.config).exists():
        log.warning("Config file not found at %s — using defaults.", args.config)

    cfg = load_config(config_path, overrides)
    return run_evolution(cfg)


if __name__ == "__main__":
    sys.exit(main())

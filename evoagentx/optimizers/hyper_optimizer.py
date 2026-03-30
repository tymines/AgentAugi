"""
HyperAgents — meta-optimization layer where optimization algorithms themselves evolve.

Inspired by ADAS (Automated Design of Agentic Systems) but applied to the optimizer
level rather than the agent level. OptimizerGenomes encode optimization *strategies*
as natural language descriptions + configuration, and evolve via LLM-guided mutation
and crossover operators.

Architecture:
    OptimizerGenome      — A single optimizer strategy with fitness tracking
    OptimizerPopulation  — Collection of genomes with selection operators
    HyperOptimizerConfig — Hyperparameters for the meta-evolutionary loop
    HyperOptimizer       — Orchestrates the full evolutionary process
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import re
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

# Lazy import: the engine's registry → prompts → tools chain requires several
# optional packages (ddgs, overdue, …).  We defer the import so that
# hyper_optimizer is importable without them, and only fail at call-time
# when the full engine is actually needed.
try:
    from .engine.base import BaseOptimizer
    from .engine.registry import ParamRegistry
    _ENGINE_AVAILABLE = True
except ImportError:
    _ENGINE_AVAILABLE = False
    BaseOptimizer = object  # type: ignore[assignment,misc]
    ParamRegistry = object  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class OptimizerGenome:
    """
    Encodes a single optimizer strategy.

    The strategy is captured as a natural-language description of the
    optimisation algorithm rather than raw source code — this lets the
    meta-LLM reason about *behaviour*, not syntax.  A ``structural_hash``
    provides cheap deduplication.

    Attributes
    ----------
    name : str
        Human-readable identifier, unique within a population.
    source_code : str
        Natural-language description of the optimisation strategy (what it
        does, how it balances exploration vs. exploitation, etc.).
    config : dict
        Numeric hyper-parameters for the strategy (learning rate, temperature,
        population size, …).
    fitness_history : list of float
        Fitness scores recorded in each generation this genome was evaluated.
        Earlier entries correspond to earlier generations.
    generation : int
        Generation in which this genome was created (0 = seed population).
    parent_name : str or None
        Name of the parent genome (mutation) or primary parent (crossover).
        ``None`` for seed genomes.
    structural_hash : str
        SHA-256 hash of ``source_code`` (lowercased, whitespace-normalised),
        used for fast duplicate detection.
    """

    name: str
    source_code: str
    config: Dict[str, Any]
    fitness_history: List[float] = field(default_factory=list)
    generation: int = 0
    parent_name: Optional[str] = None
    structural_hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.structural_hash = _hash_strategy(self.source_code)

    # ------------------------------------------------------------------
    # Fitness helpers
    # ------------------------------------------------------------------

    @property
    def best_fitness(self) -> float:
        """Return the best (highest) fitness seen so far, or -inf if never evaluated."""
        return max(self.fitness_history) if self.fitness_history else float("-inf")

    @property
    def mean_fitness(self) -> float:
        """Return the mean fitness across all evaluations, or -inf if never evaluated."""
        if not self.fitness_history:
            return float("-inf")
        return sum(self.fitness_history) / len(self.fitness_history)

    def record_fitness(self, score: float) -> None:
        """Append *score* to ``fitness_history``."""
        self.fitness_history.append(score)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source_code": self.source_code,
            "config": self.config,
            "fitness_history": list(self.fitness_history),
            "generation": self.generation,
            "parent_name": self.parent_name,
            "structural_hash": self.structural_hash,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OptimizerGenome":
        genome = cls(
            name=d["name"],
            source_code=d["source_code"],
            config=d.get("config", {}),
            fitness_history=list(d.get("fitness_history", [])),
            generation=d.get("generation", 0),
            parent_name=d.get("parent_name"),
        )
        # Restore the stored hash (regenerated in __post_init__ but can be
        # overwritten here for round-trip fidelity).
        stored_hash = d.get("structural_hash", "")
        if stored_hash:
            genome.structural_hash = stored_hash
        return genome

    def __repr__(self) -> str:
        return (
            f"OptimizerGenome(name={self.name!r}, gen={self.generation}, "
            f"best_fitness={self.best_fitness:.4f})"
        )


def _hash_strategy(text: str) -> str:
    """Return an 8-hex-char fingerprint of a normalised strategy string."""
    normalised = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(normalised.encode()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------


class OptimizerPopulation:
    """
    Manages a mutable collection of :class:`OptimizerGenome` objects.

    Supports tournament selection, diversity measurement, and named lookup.
    """

    def __init__(self) -> None:
        self._genomes: Dict[str, OptimizerGenome] = {}

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    @property
    def genomes(self) -> List[OptimizerGenome]:
        """Return genomes as a list (insertion order)."""
        return list(self._genomes.values())

    def add(self, genome: OptimizerGenome) -> None:
        """Add *genome* to the population, overwriting any genome with the same name."""
        self._genomes[genome.name] = genome

    def remove(self, name: str) -> Optional[OptimizerGenome]:
        """Remove and return the genome named *name*, or ``None`` if absent."""
        return self._genomes.pop(name, None)

    def get(self, name: str) -> Optional[OptimizerGenome]:
        """Return the genome named *name*, or ``None`` if absent."""
        return self._genomes.get(name)

    def __len__(self) -> int:
        return len(self._genomes)

    def __contains__(self, name: str) -> bool:
        return name in self._genomes

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def get_best(self, n: int = 1) -> List[OptimizerGenome]:
        """
        Return the *n* genomes with the highest ``best_fitness``, ranked descending.

        Genomes that have never been evaluated sort to the bottom.
        """
        sorted_genomes = sorted(
            self._genomes.values(),
            key=lambda g: g.best_fitness,
            reverse=True,
        )
        return sorted_genomes[:n]

    def tournament_select(self, k: int = 3) -> OptimizerGenome:
        """
        Run a k-way tournament and return the winner (highest ``best_fitness``).

        If the population is smaller than *k*, all genomes participate.

        Parameters
        ----------
        k : int
            Tournament size.

        Returns
        -------
        OptimizerGenome
            The genome with the highest ``best_fitness`` among the *k* entrants.

        Raises
        ------
        ValueError
            If the population is empty.
        """
        if not self._genomes:
            raise ValueError("Cannot select from an empty population.")
        k = min(k, len(self._genomes))
        entrants = random.sample(list(self._genomes.values()), k)
        return max(entrants, key=lambda g: g.best_fitness)

    # ------------------------------------------------------------------
    # Diversity
    # ------------------------------------------------------------------

    def diversity_score(self) -> float:
        """
        Estimate population diversity as the fraction of *unique* structural
        hashes among all genomes.

        Returns
        -------
        float
            A value in [0, 1] where 1.0 means every genome has a distinct
            strategy hash and 0.0 means all genomes are identical.
        """
        if len(self._genomes) <= 1:
            return 0.0
        hashes = [g.structural_hash for g in self._genomes.values()]
        unique_fraction = len(set(hashes)) / len(hashes)
        return unique_fraction

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_list(self) -> List[Dict[str, Any]]:
        return [g.to_dict() for g in self._genomes.values()]

    @classmethod
    def from_list(cls, items: List[Dict[str, Any]]) -> "OptimizerPopulation":
        pop = cls()
        for d in items:
            pop.add(OptimizerGenome.from_dict(d))
        return pop


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class HyperOptimizerConfig:
    """
    Hyper-parameters for the evolutionary meta-optimisation loop.

    Attributes
    ----------
    population_size : int
        Number of genomes maintained per generation.
    generations : int
        Total number of evolutionary generations to run.
    mutation_rate : float
        Probability that any surviving genome is mutated to produce an offspring.
    crossover_rate : float
        Probability that offspring are created by crossover rather than
        pure mutation when there are ≥ 2 parents available.
    tournament_size : int
        Number of candidates drawn per tournament during parent selection.
    evaluation_budget : int
        Maximum number of task evaluations allowed per genome per generation.
        Set to a large value (e.g. 10 000) for unconstrained runs.
    meta_llm_fn : callable or None
        The language model callable used for mutation and crossover.
        Signature: ``(prompt: str) -> str``.
        When ``None`` the class falls back to a trivial echo-based stub (useful
        for unit tests that do not need real LLM calls).
    fitness_metric : str
        Name of the metric extracted from evaluation results; default
        ``"accuracy"``.  The evaluation function should return a dict that
        contains this key as a ``float``.
    """

    population_size: int = 5
    generations: int = 10
    mutation_rate: float = 0.3
    crossover_rate: float = 0.2
    tournament_size: int = 3
    evaluation_budget: int = 100
    meta_llm_fn: Optional[Callable[[str], str]] = None
    fitness_metric: str = "accuracy"

    def __post_init__(self) -> None:
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be in [0, 1].")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in [0, 1].")
        if self.population_size < 1:
            raise ValueError("population_size must be at least 1.")
        if self.generations < 1:
            raise ValueError("generations must be at least 1.")
        if self.tournament_size < 1:
            raise ValueError("tournament_size must be at least 1.")
        if self.evaluation_budget < 1:
            raise ValueError("evaluation_budget must be at least 1.")


# ---------------------------------------------------------------------------
# Safety validation
# ---------------------------------------------------------------------------

_FORBIDDEN_AST_NODES = (
    ast.Delete,        # del statements could destroy state
)

_FORBIDDEN_CALLS = frozenset({
    "exec", "eval", "compile",           # dynamic code execution
    "open", "__import__", "importlib",   # file system / import access
    "subprocess", "os.system",           # shell access
    "shutil", "socket",                  # network / file system
    "exit", "quit",                      # process termination
})

_MAX_GENERATED_SOURCE_LEN = 8_000  # characters


def _validate_generated_code(source: str) -> Tuple[bool, str]:
    """
    Lightweight static analysis of LLM-generated optimizer code.

    Checks for:
    - Excessive length
    - Forbidden AST node types
    - Calls to dangerous built-ins

    Returns
    -------
    (ok, reason)
        ``ok`` is ``True`` when the code passes all checks.
        ``reason`` is an empty string on success or a human-readable
        explanation on failure.
    """
    if len(source) > _MAX_GENERATED_SOURCE_LEN:
        return False, f"Source too long ({len(source)} chars > {_MAX_GENERATED_SOURCE_LEN})."

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, f"Syntax error: {exc}"

    # Check for forbidden node types
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_AST_NODES):
            return False, f"Forbidden AST node: {type(node).__name__}"

    # Check for forbidden function calls
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = _extract_call_name(node.func)
            if func_name in _FORBIDDEN_CALLS:
                return False, f"Forbidden call: {func_name}()"

    return True, ""


def _extract_call_name(func_node: ast.expr) -> str:
    """Return a dotted string name from a Call's func node, best-effort."""
    if isinstance(func_node, ast.Name):
        return func_node.id
    if isinstance(func_node, ast.Attribute):
        return f"{_extract_call_name(func_node.value)}.{func_node.attr}"
    return ""


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

_MUTATION_PROMPT = textwrap.dedent("""\
You are an expert in optimization algorithms and meta-learning.

Below is an optimizer strategy description.  Your task is to produce a
**mutated variant** that changes one or two aspects of the strategy while
preserving its core idea.  You may modify:

  - The selection mechanism (e.g. tournament → roulette wheel → rank-based)
  - The exploration/exploitation balance (e.g. add simulated annealing, ε-greedy)
  - The gradient/update rule (e.g. momentum, adaptive step sizes)
  - Additional heuristics (e.g. restarts, population diversity penalties)

Output ONLY the new strategy description in plain text — no code, no JSON.
Keep it concise (150–300 words).

--- ORIGINAL STRATEGY ---
{source_code}

--- CONFIG ---
{config}

--- MUTATED STRATEGY ---
""")

_CROSSOVER_PROMPT = textwrap.dedent("""\
You are an expert in optimization algorithms and meta-learning.

Below are two optimizer strategies.  Your task is to create a NEW strategy
that **combines the best ideas** from both parents.  The child should be
coherent — not a mere concatenation.  Focus on complementary strengths.

Output ONLY the new strategy description in plain text — no code, no JSON.
Keep it concise (150–300 words).

--- PARENT A: {name_a} ---
{source_a}

--- PARENT B: {name_b} ---
{source_b}

--- CHILD STRATEGY ---
""")

_GENOME_TO_OPTIMIZER_PROMPT = textwrap.dedent("""\
You are an expert Python programmer and optimization researcher.

Convert the following optimizer strategy description into a Python class that
extends ``BaseOptimizer`` from ``evoagentx.optimizers.engine.base``.

Requirements:
- Class name: ``GeneratedOptimizer``
- Constructor: ``__init__(self, registry, program=None, evaluator=None, **kwargs)``
- Implement ``optimize(self)`` returning ``(best_cfg, history)``
- Use only the Python standard library (no subprocess, exec, eval, open, socket)
- History should be a list of dicts with keys ``cfg`` and ``score``
- Apply configs using ``self.apply_cfg(cfg)`` and run via ``self.program()``
- Score results via ``self.evaluator(output)``

--- STRATEGY DESCRIPTION ---
{source_code}

--- CONFIG DEFAULTS ---
{config}

--- PYTHON CODE (class only, no imports besides BaseOptimizer) ---
""")


# ---------------------------------------------------------------------------
# Core HyperOptimizer
# ---------------------------------------------------------------------------


class HyperOptimizer:
    """
    Evolutionary meta-optimizer that evolves optimizer *strategies*.

    Each genome in the population encodes an optimization strategy as a
    natural-language description.  The meta-LLM generates offspring through
    mutation and crossover.  Offspring are compiled into callable
    :class:`BaseOptimizer` subclasses, evaluated on a task suite, and subject
    to tournament selection.

    Parameters
    ----------
    config : HyperOptimizerConfig
        Evolutionary hyper-parameters and the meta-LLM callable.
    base_optimizers : list
        Seed material.  Each element may be:

        - A :class:`BaseOptimizer` *instance* — strategy extracted from its
          class docstring and ``__init__`` signature.
        - A :class:`BaseOptimizer` *class* (type) — same extraction logic.
        - A :class:`dict` with keys ``name`` (str), ``source_code`` (str),
          and optionally ``config`` (dict) — used directly.
    evaluation_fn : callable
        ``(genome: OptimizerGenome, task_suite: list) -> float``
        External scoring function.  Should return a fitness value in [0, 1]
        where higher is better.
    """

    def __init__(
        self,
        config: HyperOptimizerConfig,
        base_optimizers: List[Any],
        evaluation_fn: Callable[["OptimizerGenome", List[Any]], float],
    ) -> None:
        self.config = config
        self.evaluation_fn = evaluation_fn
        self._population = OptimizerPopulation()
        self._lineage: Dict[str, List[str]] = {}  # name -> [ancestor names]
        self._generation: int = 0

        # Seed the population from the provided base optimizers.
        self.seed_population(base_optimizers)

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def seed_population(self, base_optimizers: List[Any]) -> None:
        """
        Populate the initial generation from *base_optimizers*.

        Each element is converted to an :class:`OptimizerGenome` whose
        ``source_code`` captures the strategy in natural language.
        Duplicate hashes are skipped.
        """
        seen_hashes: set = set()

        for idx, item in enumerate(base_optimizers):
            genome = self._item_to_genome(item, idx)
            if genome.structural_hash in seen_hashes:
                continue
            seen_hashes.add(genome.structural_hash)
            self._population.add(genome)
            self._lineage[genome.name] = []

    def _item_to_genome(self, item: Any, idx: int) -> OptimizerGenome:
        """Convert one seed item to an :class:`OptimizerGenome`."""
        if isinstance(item, dict):
            name = item.get("name", f"seed_{idx}")
            source_code = item.get("source_code", "No strategy description provided.")
            cfg = dict(item.get("config", {}))
            return OptimizerGenome(name=name, source_code=source_code, config=cfg)

        # Accept both instances and classes.
        if isinstance(item, type) and issubclass(item, BaseOptimizer):
            cls = item
        elif isinstance(item, BaseOptimizer):
            cls = type(item)
        else:
            # Fallback: treat as unknown optimizer, use repr as description.
            name = f"unknown_{idx}"
            source_code = f"Unknown optimizer of type {type(item).__name__}."
            return OptimizerGenome(name=name, source_code=source_code, config={})

        name = cls.__name__
        doc = textwrap.dedent(cls.__doc__ or "").strip()
        source_code = (
            f"Optimizer: {name}\n\n"
            + (doc if doc else "No documentation available.")
        )
        return OptimizerGenome(name=name, source_code=source_code, config={})

    # ------------------------------------------------------------------
    # LLM façade
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        """
        Route a prompt through the meta-LLM.

        Falls back to a trivial stub when ``config.meta_llm_fn`` is ``None``
        (useful for testing without a real API key).
        """
        if self.config.meta_llm_fn is not None:
            return self.config.meta_llm_fn(prompt)
        # Stub: return a slightly perturbed copy of the last non-empty block.
        lines = [ln for ln in prompt.splitlines() if ln.strip()]
        stub = lines[-1] if lines else "Random search with uniform sampling."
        return stub + " [stub mutant]"

    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------

    def mutate(self, genome: OptimizerGenome) -> OptimizerGenome:
        """
        Produce a mutated child genome via the meta-LLM.

        The child inherits the parent's ``config`` (with small numeric
        perturbations) and records the parent as its lineage.

        Parameters
        ----------
        genome : OptimizerGenome
            The parent genome to mutate.

        Returns
        -------
        OptimizerGenome
            A new genome; does **not** modify *genome* in place.
        """
        prompt = _MUTATION_PROMPT.format(
            source_code=genome.source_code,
            config=json.dumps(genome.config, indent=2),
        )
        new_strategy = self._call_llm(prompt).strip()
        if not new_strategy:
            new_strategy = genome.source_code + " [mutant]"

        new_config = _perturb_config(genome.config)
        child_name = f"{genome.name}_mut_{self._generation}"
        child = OptimizerGenome(
            name=child_name,
            source_code=new_strategy,
            config=new_config,
            generation=self._generation,
            parent_name=genome.name,
        )
        self._lineage[child_name] = self._lineage.get(genome.name, []) + [genome.name]
        return child

    def crossover(
        self, genome_a: OptimizerGenome, genome_b: OptimizerGenome
    ) -> OptimizerGenome:
        """
        Produce a child genome by combining two parent strategies.

        The child's numeric config is a random convex combination of the
        parents' configs.

        Parameters
        ----------
        genome_a : OptimizerGenome
            Primary parent (name recorded in ``parent_name``).
        genome_b : OptimizerGenome
            Secondary parent.

        Returns
        -------
        OptimizerGenome
            A new genome; does **not** modify either parent in place.
        """
        prompt = _CROSSOVER_PROMPT.format(
            name_a=genome_a.name,
            source_a=genome_a.source_code,
            name_b=genome_b.name,
            source_b=genome_b.source_code,
        )
        child_strategy = self._call_llm(prompt).strip()
        if not child_strategy:
            child_strategy = genome_a.source_code + " + " + genome_b.source_code

        child_config = _blend_configs(genome_a.config, genome_b.config)
        child_name = f"{genome_a.name}x{genome_b.name}_{self._generation}"
        child = OptimizerGenome(
            name=child_name,
            source_code=child_strategy,
            config=child_config,
            generation=self._generation,
            parent_name=genome_a.name,
        )
        ancestors_a = self._lineage.get(genome_a.name, []) + [genome_a.name]
        ancestors_b = self._lineage.get(genome_b.name, []) + [genome_b.name]
        self._lineage[child_name] = list(dict.fromkeys(ancestors_a + ancestors_b))
        return child

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_genome(
        self, genome: OptimizerGenome, task_suite: List[Any]
    ) -> float:
        """
        Evaluate *genome* on *task_suite* and record the fitness.

        The fitness is obtained by calling ``self.evaluation_fn(genome,
        task_suite)``.  The result is appended to ``genome.fitness_history``.

        Parameters
        ----------
        genome : OptimizerGenome
        task_suite : list
            Opaque task descriptors passed directly to ``evaluation_fn``.

        Returns
        -------
        float
            The observed fitness score.
        """
        score = self.evaluation_fn(genome, task_suite)
        genome.record_fitness(float(score))
        return float(score)

    # ------------------------------------------------------------------
    # Generational step
    # ------------------------------------------------------------------

    def evolve_generation(self, task_suite: List[Any]) -> Dict[str, Any]:
        """
        Run one complete evolutionary generation.

        Steps
        -----
        1. Evaluate any unevaluated genomes in the current population.
        2. Generate offspring via mutation and crossover.
        3. Evaluate offspring.
        4. Select survivors by truncation (keep top ``population_size``).
        5. Increment generation counter.

        Parameters
        ----------
        task_suite : list
            Tasks used to score each genome.

        Returns
        -------
        dict
            A log entry with keys ``generation``, ``best_fitness``,
            ``diversity``, ``population_size``, and ``elapsed_seconds``.
        """
        t0 = time.monotonic()
        self._generation += 1

        # Evaluate unevaluated incumbents.
        for genome in self._population.genomes:
            if not genome.fitness_history:
                self.evaluate_genome(genome, task_suite)

        # Generate offspring.
        offspring: List[OptimizerGenome] = []
        target_offspring = max(1, self.config.population_size)

        while len(offspring) < target_offspring:
            use_crossover = (
                len(self._population) >= 2
                and random.random() < self.config.crossover_rate
            )
            if use_crossover:
                parent_a = self._population.tournament_select(self.config.tournament_size)
                parent_b = self._population.tournament_select(self.config.tournament_size)
                # Avoid self-crossover.
                attempts = 0
                while parent_b.name == parent_a.name and attempts < 5:
                    parent_b = self._population.tournament_select(self.config.tournament_size)
                    attempts += 1
                child = self.crossover(parent_a, parent_b)
            else:
                parent = self._population.tournament_select(self.config.tournament_size)
                child = self.mutate(parent)

            # Evaluate child.
            self.evaluate_genome(child, task_suite)
            offspring.append(child)

        # Merge pool and select survivors.
        combined = self._population.genomes + offspring
        combined.sort(key=lambda g: g.best_fitness, reverse=True)
        survivors = combined[: self.config.population_size]

        # Rebuild population.
        self._population = OptimizerPopulation()
        for genome in survivors:
            self._population.add(genome)

        best = survivors[0] if survivors else None
        elapsed = time.monotonic() - t0

        return {
            "generation": self._generation,
            "best_fitness": best.best_fitness if best else float("-inf"),
            "best_name": best.name if best else None,
            "diversity": self._population.diversity_score(),
            "population_size": len(self._population),
            "elapsed_seconds": elapsed,
        }

    # ------------------------------------------------------------------
    # Full evolutionary run
    # ------------------------------------------------------------------

    def run(self, task_suite: List[Any]) -> List[Dict[str, Any]]:
        """
        Execute the complete evolutionary loop.

        Runs ``config.generations`` calls to :meth:`evolve_generation` and
        returns a log of per-generation statistics.

        Parameters
        ----------
        task_suite : list
            Tasks passed to the evaluation function each generation.

        Returns
        -------
        list of dict
            Generation-level statistics (same format as returned by
            :meth:`evolve_generation`).
        """
        history = []
        for _ in range(self.config.generations):
            log = self.evolve_generation(task_suite)
            history.append(log)
        return history

    # ------------------------------------------------------------------
    # Genome → callable optimizer
    # ------------------------------------------------------------------

    def _genome_to_optimizer(
        self,
        genome: OptimizerGenome,
        registry: ParamRegistry,
        program: Optional[Callable] = None,
        evaluator: Optional[Callable] = None,
    ) -> BaseOptimizer:
        """
        Compile a genome's strategy into a callable :class:`BaseOptimizer`.

        When ``config.meta_llm_fn`` is available the method asks the meta-LLM
        to generate Python code from the strategy description, validates it,
        and executes it in a restricted namespace to obtain the class.

        When ``meta_llm_fn`` is ``None`` (or validation fails) it falls back
        to a :class:`_StrategyWrapperOptimizer` that wraps the strategy
        description as a plain random-search optimizer.

        Parameters
        ----------
        genome : OptimizerGenome
        registry : ParamRegistry
        program : callable or None
        evaluator : callable or None

        Returns
        -------
        BaseOptimizer
        """
        if self.config.meta_llm_fn is not None:
            try:
                return self._compile_genome(genome, registry, program, evaluator)
            except Exception:
                pass  # fall through to wrapper

        return _StrategyWrapperOptimizer(
            registry=registry,
            program=program,
            evaluator=evaluator,
            genome=genome,
        )

    def _compile_genome(
        self,
        genome: OptimizerGenome,
        registry: ParamRegistry,
        program: Optional[Callable],
        evaluator: Optional[Callable],
    ) -> BaseOptimizer:
        """
        Use the meta-LLM to generate Python code and instantiate it.

        Raises
        ------
        RuntimeError
            If validation fails or instantiation raises an error.
        """
        prompt = _GENOME_TO_OPTIMIZER_PROMPT.format(
            source_code=genome.source_code,
            config=json.dumps(genome.config, indent=2),
        )
        raw_code = self.config.meta_llm_fn(prompt)

        # Extract the class body (handle fenced code blocks).
        code = _extract_code_block(raw_code)

        ok, reason = self._validate_optimizer(code)
        if not ok:
            raise RuntimeError(f"Generated code failed validation: {reason}")

        # Build the full module: import BaseOptimizer, then paste the class.
        full_source = (
            "from evoagentx.optimizers.engine.base import BaseOptimizer\n\n"
            + code
        )
        ns: Dict[str, Any] = {}
        exec(full_source, ns)  # noqa: S102

        cls = ns.get("GeneratedOptimizer")
        if cls is None or not (isinstance(cls, type) and issubclass(cls, BaseOptimizer)):
            raise RuntimeError("Generated code did not define GeneratedOptimizer(BaseOptimizer).")

        return cls(registry=registry, program=program, evaluator=evaluator, **genome.config)

    def _validate_optimizer(self, source: str) -> Tuple[bool, str]:
        """
        Validate LLM-generated optimizer code for safety.

        Delegates to the module-level :func:`_validate_generated_code` helper
        and additionally checks that the code defines ``GeneratedOptimizer``.

        Parameters
        ----------
        source : str
            Raw Python source to validate.

        Returns
        -------
        (ok, reason)
        """
        ok, reason = _validate_generated_code(source)
        if not ok:
            return False, reason

        # Require the class to be present.
        if "GeneratedOptimizer" not in source:
            return False, "Code must define class GeneratedOptimizer."

        return True, ""

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def best_optimizer(self) -> Optional[OptimizerGenome]:
        """Return the highest-fitness genome in the current population."""
        best = self._population.get_best(1)
        return best[0] if best else None

    def lineage(self, genome_name: str) -> List[str]:
        """
        Return the list of ancestor genome names for *genome_name*.

        The list is in oldest-first order.  Returns an empty list for seed
        genomes or unknown names.
        """
        return list(self._lineage.get(genome_name, []))

    @property
    def population(self) -> OptimizerPopulation:
        """Read-only view of the current population."""
        return self._population

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serialise the entire evolutionary state to a JSON file at *path*.

        The saved state includes the population, lineage graph, current
        generation index, and config (excluding the LLM callable).
        """
        data = {
            "generation": self._generation,
            "config": {
                "population_size": self.config.population_size,
                "generations": self.config.generations,
                "mutation_rate": self.config.mutation_rate,
                "crossover_rate": self.config.crossover_rate,
                "tournament_size": self.config.tournament_size,
                "evaluation_budget": self.config.evaluation_budget,
                "fitness_metric": self.config.fitness_metric,
            },
            "population": self._population.to_list(),
            "lineage": self._lineage,
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: str) -> None:
        """
        Restore evolutionary state from a JSON file previously written by
        :meth:`save`.

        The ``meta_llm_fn`` and ``evaluation_fn`` are **not** serialised and
        must be re-attached after loading.

        Parameters
        ----------
        path : str
            Path to the JSON file.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self._generation = data.get("generation", 0)
        self._population = OptimizerPopulation.from_list(data.get("population", []))
        self._lineage = data.get("lineage", {})

        saved_cfg = data.get("config", {})
        self.config.population_size = saved_cfg.get("population_size", self.config.population_size)
        self.config.generations = saved_cfg.get("generations", self.config.generations)
        self.config.mutation_rate = saved_cfg.get("mutation_rate", self.config.mutation_rate)
        self.config.crossover_rate = saved_cfg.get("crossover_rate", self.config.crossover_rate)
        self.config.tournament_size = saved_cfg.get("tournament_size", self.config.tournament_size)
        self.config.evaluation_budget = saved_cfg.get("evaluation_budget", self.config.evaluation_budget)
        self.config.fitness_metric = saved_cfg.get("fitness_metric", self.config.fitness_metric)


# ---------------------------------------------------------------------------
# Helper: fallback optimizer wrapping a genome strategy
# ---------------------------------------------------------------------------


class _StrategyWrapperOptimizer(BaseOptimizer):
    """
    Minimal optimizer that wraps an :class:`OptimizerGenome` strategy.

    Used as a fallback when the meta-LLM is unavailable or code generation
    fails.  Implements a simple random-search loop whose hyper-parameters are
    drawn from ``genome.config``.
    """

    def __init__(
        self,
        registry: ParamRegistry,
        program: Optional[Callable],
        evaluator: Optional[Callable],
        genome: OptimizerGenome,
    ) -> None:
        super().__init__(registry=registry, program=program, evaluator=evaluator)
        self.genome = genome
        self.n_trials: int = int(genome.config.get("n_trials", 10))

    def optimize(self) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Random-search fallback: sample configs uniformly at random."""
        best_score = float("-inf")
        best_cfg: Dict[str, Any] = {}
        history: List[Dict[str, Any]] = []

        current_cfg = self.get_current_cfg()
        names = list(current_cfg.keys())

        for _ in range(self.n_trials):
            # Perturb the current config slightly.
            cfg = _perturb_config(current_cfg)
            self.apply_cfg(cfg)

            if self.program is not None and self.evaluator is not None:
                output = self.program()
                score = float(self.evaluator(output))
            else:
                score = random.random()

            history.append({"cfg": dict(cfg), "score": score})

            if score > best_score:
                best_score = score
                best_cfg = dict(cfg)

        if best_cfg:
            self.apply_cfg(best_cfg)
        return best_cfg, history


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _perturb_config(config: Dict[str, Any], sigma: float = 0.1) -> Dict[str, Any]:
    """
    Return a shallow copy of *config* with numeric values perturbed by
    Gaussian noise (relative sigma).

    Non-numeric values are copied unchanged.
    """
    out: Dict[str, Any] = {}
    for k, v in config.items():
        if isinstance(v, float):
            out[k] = v + random.gauss(0, abs(v) * sigma + 1e-8)
        elif isinstance(v, int):
            out[k] = v  # keep ints as-is to avoid type confusion
        else:
            out[k] = v
    return out


def _blend_configs(
    cfg_a: Dict[str, Any], cfg_b: Dict[str, Any], alpha: Optional[float] = None
) -> Dict[str, Any]:
    """
    Return a convex combination of two configs for shared numeric keys.

    *alpha* is drawn uniformly from (0, 1) when ``None``.  Non-numeric or
    non-shared keys are copied from *cfg_a* if present, otherwise from
    *cfg_b*.
    """
    if alpha is None:
        alpha = random.random()
    alpha = max(0.0, min(1.0, alpha))

    out: Dict[str, Any] = {}
    all_keys = set(cfg_a) | set(cfg_b)
    for k in all_keys:
        a_val = cfg_a.get(k)
        b_val = cfg_b.get(k)
        if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
            out[k] = alpha * a_val + (1.0 - alpha) * b_val
        elif k in cfg_a:
            out[k] = a_val
        else:
            out[k] = b_val
    return out


def _extract_code_block(text: str) -> str:
    """
    Extract the content of a fenced ```python … ``` block, or return the
    full text if no fence is found.
    """
    pattern = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()

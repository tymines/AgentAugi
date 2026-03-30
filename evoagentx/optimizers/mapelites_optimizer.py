"""MAP-Elites quality-diversity optimizer.

MAP-Elites (Mouret & Clune, 2015) maintains an *archive* — a multi-dimensional
grid where each cell stores the single best solution found in that region of
*behavioural feature space*.  The result is a diverse set of high-quality
solutions rather than a single converged optimum.

Why it matters for AgentAugi
-----------------------------
EvoPrompt can suffer from mode collapse (Issue #212): the population converges
to a single dominant prompt variant and stops exploring.  MAP-Elites prevents
this by design: each archive cell can only be displaced by a *better* occupant
in the *same* feature region, so diversity across feature dimensions is
structurally guaranteed.

Feature dimensions
------------------
Feature dimensions are user-configurable.  Each ``FeatureDimension`` defines:
  - A *name* (e.g. "accuracy", "cost", "latency", "verbosity").
  - A *bounds* pair ``(lo, hi)`` for the expected value range.
  - A *resolution* (number of discrete bins along that axis).
  - An *extractor* function ``(candidate, eval_result) → float`` that reads the
    feature value from an evaluation result dict.

Archive
-------
The archive is a dict mapping a tuple of bin indices (one per dimension) to the
best candidate configuration known for that cell.  ``ArchiveCell`` stores the
candidate config dict, its evaluation result dict, and the composite quality
score used for cell replacement decisions.

Integration with existing architecture
---------------------------------------
``MAPElitesOptimizer`` extends ``BaseOptimizer`` from
``evoagentx.optimizers.engine.base``, following the same interface as
``EvopromptOptimizer``.  It accepts a ``program`` callable and a
``ParamRegistry`` and exposes an ``optimize()`` method.

Usage
-----
    >>> from evoagentx.optimizers.mapelites_optimizer import (
    ...     MAPElitesOptimizer, FeatureDimension
    ... )
    >>> accuracy_dim = FeatureDimension(
    ...     name="accuracy", lo=0.0, hi=1.0, resolution=5,
    ...     extractor=lambda cfg, res: res.get("accuracy", 0.0),
    ... )
    >>> cost_dim = FeatureDimension(
    ...     name="cost", lo=0.0, hi=1.0, resolution=5,
    ...     extractor=lambda cfg, res: res.get("cost_normalised", 0.5),
    ... )
    >>> optimizer = MAPElitesOptimizer(
    ...     registry=registry,
    ...     program=my_program,
    ...     evaluator=my_evaluator,
    ...     dimensions=[accuracy_dim, cost_dim],
    ...     quality_fn=lambda res: res.get("accuracy", 0.0),
    ...     iterations=100,
    ... )
    >>> pareto_front = optimizer.optimize()
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.logging import logger
from .engine.base import BaseOptimizer
from .engine.registry import ParamRegistry


# ---------------------------------------------------------------------------
# Feature dimensions
# ---------------------------------------------------------------------------

@dataclass
class FeatureDimension:
    """Describes one axis of the MAP-Elites archive grid.

    Attributes:
        name: Human-readable label (e.g. ``"accuracy"``, ``"cost"``).
        lo: Lower bound of the expected feature range (inclusive).
        hi: Upper bound of the expected feature range (inclusive).
        resolution: Number of discrete bins along this axis.  A finer
            resolution gives more granular diversity preservation at the
            cost of a sparser archive (more cells, fewer occupants per cell).
        extractor: Callable ``(candidate_config, eval_result) → float`` that
            reads this feature's value from an evaluation result dict.
    """

    name: str
    lo: float
    hi: float
    resolution: int
    extractor: Callable[[Dict[str, Any], Dict[str, Any]], float]

    def __post_init__(self) -> None:
        if self.lo >= self.hi:
            raise ValueError(
                f"FeatureDimension '{self.name}': lo ({self.lo}) must be < hi ({self.hi})"
            )
        if self.resolution < 1:
            raise ValueError(
                f"FeatureDimension '{self.name}': resolution must be >= 1, got {self.resolution}"
            )

    def bin_index(self, value: float) -> int:
        """Map a continuous feature value to its discrete bin index.

        Values outside ``[lo, hi]`` are clamped to the boundary bins.

        Args:
            value: Continuous feature value to discretise.

        Returns:
            Integer bin index in ``[0, resolution - 1]``.
        """
        clamped = max(self.lo, min(self.hi, value))
        # Normalise to [0, 1) and scale
        span = self.hi - self.lo
        normalised = (clamped - self.lo) / span
        # Use resolution bins: [0, res-1]
        idx = int(normalised * self.resolution)
        return min(idx, self.resolution - 1)


# ---------------------------------------------------------------------------
# Archive cell and archive
# ---------------------------------------------------------------------------

@dataclass
class ArchiveCell:
    """A single cell in the MAP-Elites archive.

    Attributes:
        config: The candidate configuration dict (parameter name → value).
        eval_result: The evaluation result dict returned by the evaluator.
        quality: The scalar quality score used for cell replacement.
        cell_key: The tuple of bin indices identifying this cell's location.
    """

    config: Dict[str, Any]
    eval_result: Dict[str, Any]
    quality: float
    cell_key: Tuple[int, ...]


class Archive:
    """Multi-dimensional grid storing the best solution per feature cell.

    The archive is the central data structure of MAP-Elites.  Each cell is
    addressed by a tuple of integer bin indices, one per feature dimension.
    A cell is updated only when a new candidate achieves a strictly higher
    quality score than the current occupant.

    Args:
        dimensions: Ordered list of ``FeatureDimension`` objects defining the
            grid axes.
    """

    def __init__(self, dimensions: List[FeatureDimension]) -> None:
        if not dimensions:
            raise ValueError("Archive requires at least one FeatureDimension.")
        self.dimensions = dimensions
        self._cells: Dict[Tuple[int, ...], ArchiveCell] = {}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def cell_key(
        self,
        config: Dict[str, Any],
        eval_result: Dict[str, Any],
    ) -> Tuple[int, ...]:
        """Compute the archive cell key for a given candidate + eval result.

        Args:
            config: Candidate configuration dict.
            eval_result: Evaluation result dict from which feature values are
                extracted.

        Returns:
            Tuple of integer bin indices, one per dimension.
        """
        indices = []
        for dim in self.dimensions:
            value = dim.extractor(config, eval_result)
            indices.append(dim.bin_index(value))
        return tuple(indices)

    def try_insert(
        self,
        config: Dict[str, Any],
        eval_result: Dict[str, Any],
        quality: float,
    ) -> bool:
        """Insert a candidate if it improves on the current cell occupant.

        Args:
            config: Candidate configuration dict.
            eval_result: Evaluation results dict.
            quality: Scalar quality score (higher is better).

        Returns:
            ``True`` if the cell was updated, ``False`` if the existing
            occupant was at least as good.
        """
        key = self.cell_key(config, eval_result)
        existing = self._cells.get(key)
        if existing is None or quality > existing.quality:
            self._cells[key] = ArchiveCell(
                config=dict(config),
                eval_result=dict(eval_result),
                quality=quality,
                cell_key=key,
            )
            return True
        return False

    def sample(self, k: int = 1) -> List[ArchiveCell]:
        """Sample k cells uniformly at random from the occupied cells.

        Args:
            k: Number of cells to sample.

        Returns:
            List of ``ArchiveCell`` objects.  May be shorter than k if the
            archive has fewer occupants.
        """
        occupied = list(self._cells.values())
        if not occupied:
            return []
        return random.sample(occupied, min(k, len(occupied)))

    def best(self) -> Optional[ArchiveCell]:
        """Return the cell with the highest quality score."""
        if not self._cells:
            return None
        return max(self._cells.values(), key=lambda c: c.quality)

    def size(self) -> int:
        """Number of occupied cells."""
        return len(self._cells)

    def all_cells(self) -> List[ArchiveCell]:
        """Return all occupied cells as a list."""
        return list(self._cells.values())

    def pareto_front(self, objectives: List[str]) -> List[ArchiveCell]:
        """Extract approximate Pareto-optimal cells for named objectives.

        A cell is Pareto-optimal if no other cell dominates it on all
        requested objectives simultaneously.

        Args:
            objectives: Keys present in each cell's ``eval_result`` dict.

        Returns:
            List of non-dominated ``ArchiveCell`` objects.
        """
        cells = self.all_cells()
        if not cells:
            return []

        dominated = set()
        for i, a in enumerate(cells):
            for j, b in enumerate(cells):
                if i == j:
                    continue
                # Does b dominate a?
                b_dominates_a = all(
                    b.eval_result.get(obj, 0.0) >= a.eval_result.get(obj, 0.0)
                    for obj in objectives
                ) and any(
                    b.eval_result.get(obj, 0.0) > a.eval_result.get(obj, 0.0)
                    for obj in objectives
                )
                if b_dominates_a:
                    dominated.add(i)
                    break

        return [c for i, c in enumerate(cells) if i not in dominated]


# ---------------------------------------------------------------------------
# Variation operators
# ---------------------------------------------------------------------------

def _crossover(
    parent_a: Dict[str, Any],
    parent_b: Dict[str, Any],
) -> Dict[str, Any]:
    """Uniform crossover between two candidate configurations.

    Each parameter is drawn from parent_a with probability 0.5 and from
    parent_b otherwise.  String-type values use parent_a unchanged (prompt
    text merging is handled separately at the optimizer level).

    Args:
        parent_a: First parent configuration dict.
        parent_b: Second parent configuration dict.

    Returns:
        New child configuration dict.
    """
    child: Dict[str, Any] = {}
    all_keys = set(parent_a) | set(parent_b)
    for key in all_keys:
        a_val = parent_a.get(key)
        b_val = parent_b.get(key)
        if a_val is None:
            child[key] = b_val
        elif b_val is None:
            child[key] = a_val
        else:
            child[key] = a_val if random.random() < 0.5 else b_val
    return child


def _mutate_string(text: str, llm_mutate_fn: Optional[Callable[[str], str]]) -> str:
    """Apply a string mutation using an optional LLM-based mutation function.

    If no mutation function is provided, the original text is returned
    unchanged.  In practice, the caller should supply an LLM-based mutator
    (e.g. "paraphrase / improve this prompt").

    Args:
        text: The string to mutate.
        llm_mutate_fn: Optional callable that takes a string and returns a
            mutated version.

    Returns:
        Mutated (or original) string.
    """
    if llm_mutate_fn is None:
        return text
    try:
        return llm_mutate_fn(text)
    except Exception as exc:
        logger.warning("MAP-Elites string mutation failed: %s", exc)
        return text


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

class MAPElitesOptimizer(BaseOptimizer):
    """Multi-objective quality-diversity optimizer using the MAP-Elites algorithm.

    Maintains an archive of solutions that covers diverse regions of the
    behavioural feature space.  Diversity is structurally guaranteed: a cell
    can only be displaced by a *better* occupant in the *same* region,
    preventing mode collapse.

    Extends ``BaseOptimizer`` (``evoagentx.optimizers.engine.base``) and
    follows the same interface as ``EvopromptOptimizer`` and other optimizers.

    Args:
        registry: ``ParamRegistry`` providing access to the optimizable
            parameters.
        program: Callable that runs the workflow / program and returns a
            result dict.  Receives keyword arguments matching the current
            parameter configuration.
        evaluator: Callable ``(result_dict) → float`` that maps program output
            to a scalar quality score.
        dimensions: List of ``FeatureDimension`` objects defining the archive
            grid axes.
        quality_fn: Callable ``(eval_result_dict) → float`` that extracts the
            scalar quality score from an evaluation result.  This is the value
            used for cell replacement decisions.
        iterations: Number of MAP-Elites iterations (candidate evaluations).
        init_population_size: Number of random solutions used to warm-start
            the archive before the main MAP-Elites loop.
        mutation_fn: Optional callable ``(str) → str`` for LLM-based string
            mutation of candidate prompt parameters.
        seed: Optional random seed for reproducibility.
    """

    def __init__(
        self,
        registry: ParamRegistry,
        program: Callable[..., Dict[str, Any]],
        evaluator: Callable[[Dict[str, Any]], float],
        dimensions: List[FeatureDimension],
        quality_fn: Callable[[Dict[str, Any]], float],
        iterations: int = 100,
        init_population_size: int = 10,
        mutation_fn: Optional[Callable[[str], str]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(registry=registry, program=program, evaluator=evaluator)

        if not dimensions:
            raise ValueError("MAPElitesOptimizer requires at least one FeatureDimension.")
        if iterations < 1:
            raise ValueError("iterations must be >= 1.")

        self.dimensions = dimensions
        self.quality_fn = quality_fn
        self.iterations = iterations
        self.init_population_size = init_population_size
        self.mutation_fn = mutation_fn

        if seed is not None:
            random.seed(seed)

        self.archive = Archive(dimensions=dimensions)
        self._iteration_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_config(self, config: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """Apply a configuration, run the program, and return result + quality.

        Args:
            config: Parameter name → value mapping to apply before running.

        Returns:
            Tuple of ``(eval_result_dict, quality_score)``.
        """
        self.apply_cfg(config)
        try:
            result = self.program()
        except Exception as exc:
            logger.warning("MAPElitesOptimizer: program call failed: %s", exc)
            result = {}

        quality = self.quality_fn(result)
        return result, quality

    def _random_config(self) -> Dict[str, Any]:
        """Sample a random configuration by reading current param values.

        For the warm-start phase we use the current registry values as a
        baseline.  The caller is responsible for perturbing them before
        calling this method.
        """
        return self.get_current_cfg()

    def _vary(self, cells: List[ArchiveCell]) -> Dict[str, Any]:
        """Produce a new candidate through crossover and/or mutation.

        Selection strategy:
        - If at least 2 occupied cells exist, perform crossover between two
          randomly sampled parent cells.
        - Then, apply the string mutation function (if provided) to all
          string-type parameters.

        Args:
            cells: Currently sampled archive cells to use as parents.

        Returns:
            New candidate configuration dict.
        """
        if len(cells) >= 2:
            parent_a = cells[0].config
            parent_b = cells[1].config
            child = _crossover(parent_a, parent_b)
        elif len(cells) == 1:
            child = dict(cells[0].config)
        else:
            # Archive is empty — use current registry state
            child = self.get_current_cfg()

        # Apply string mutations (prompt text)
        if self.mutation_fn is not None:
            for key, val in child.items():
                if isinstance(val, str):
                    child[key] = _mutate_string(val, self.mutation_fn)

        return child

    def _log_iteration(
        self,
        iteration: int,
        inserted: bool,
        quality: float,
        archive_size: int,
    ) -> None:
        """Append a record to the iteration log."""
        self._iteration_log.append(
            {
                "iteration": iteration,
                "inserted": inserted,
                "quality": quality,
                "archive_size": archive_size,
            }
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(self) -> List[ArchiveCell]:
        """Run the MAP-Elites optimisation loop.

        Phase 1 (warm-start): evaluate ``init_population_size`` random
        configurations to seed the archive with diverse initial solutions.

        Phase 2 (MAP-Elites loop): at each iteration, sample two random
        archive cells, produce a child via crossover + mutation, evaluate
        it, and insert into the archive if it improves on the cell it maps
        to.

        Returns:
            List of all ``ArchiveCell`` objects in the final archive,
            sorted by quality descending.
        """
        logger.info(
            "MAPElitesOptimizer: starting optimisation — "
            "%d warm-start samples, %d MAP-Elites iterations.",
            self.init_population_size,
            self.iterations,
        )

        # ---- Phase 1: warm-start ----------------------------------------
        for i in range(self.init_population_size):
            config = self._random_config()
            eval_result, quality = self._evaluate_config(config)
            inserted = self.archive.try_insert(config, eval_result, quality)
            self._log_iteration(
                iteration=-(self.init_population_size - i),
                inserted=inserted,
                quality=quality,
                archive_size=self.archive.size(),
            )
            logger.debug(
                "Warm-start %d/%d: quality=%.4f, archive_size=%d",
                i + 1, self.init_population_size, quality, self.archive.size(),
            )

        if self.archive.size() == 0:
            logger.warning(
                "MAPElitesOptimizer: archive is empty after warm-start. "
                "Check your program and evaluator."
            )
            return []

        # ---- Phase 2: MAP-Elites loop ------------------------------------
        for iteration in range(self.iterations):
            # Sample 2 parent cells for variation
            parents = self.archive.sample(k=2)
            child_config = self._vary(parents)

            eval_result, quality = self._evaluate_config(child_config)
            inserted = self.archive.try_insert(child_config, eval_result, quality)

            self._log_iteration(
                iteration=iteration,
                inserted=inserted,
                quality=quality,
                archive_size=self.archive.size(),
            )

            if (iteration + 1) % max(1, self.iterations // 10) == 0:
                best = self.archive.best()
                logger.info(
                    "MAP-Elites iteration %d/%d: archive_size=%d, "
                    "best_quality=%.4f",
                    iteration + 1,
                    self.iterations,
                    self.archive.size(),
                    best.quality if best else 0.0,
                )

        # Apply the best found configuration back to the registry
        best_cell = self.archive.best()
        if best_cell is not None:
            self.apply_cfg(best_cell.config)
            logger.info(
                "MAPElitesOptimizer: finished. Best quality=%.4f, "
                "archive size=%d.",
                best_cell.quality,
                self.archive.size(),
            )

        return sorted(self.archive.all_cells(), key=lambda c: c.quality, reverse=True)

    def get_iteration_log(self) -> List[Dict[str, Any]]:
        """Return the per-iteration log for post-analysis.

        Each entry is a dict with keys:
            ``iteration``, ``inserted``, ``quality``, ``archive_size``.
        """
        return list(self._iteration_log)

    def get_pareto_front(self, objectives: List[str]) -> List[ArchiveCell]:
        """Extract the approximate Pareto front for named objectives.

        Args:
            objectives: List of keys present in each cell's ``eval_result``
                dict to use as optimisation objectives.

        Returns:
            List of non-dominated ``ArchiveCell`` objects.
        """
        return self.archive.pareto_front(objectives=objectives)

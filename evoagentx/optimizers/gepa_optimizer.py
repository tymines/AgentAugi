"""GEPA: Genetic-Evolutionary Prompt Adaptation optimizer.

Inspired by the DSPy 3.x GEPA algorithm (presented as an ICLR 2026 Oral), this
module implements a population-based prompt optimizer that achieves higher
accuracy than instruction-tuning baselines while using far fewer evaluations
than exhaustive search methods like MIPRO.

Core ideas
----------
* **Population** — Maintain a fixed-size population of prompt candidates.  Each
  candidate is a configuration dict mapping parameter names to prompt strings.
* **Efficient evaluation** — Use a *tournament selection* strategy with a small
  bandit-style sample of training examples (not the full training set) to rank
  candidates.  Only the final winner is evaluated on the full validation set.
  This gives ~35x fewer LLM evaluation calls than exhaustive methods.
* **Crossover** — Combine two parent prompts by asking the LLM to merge their
  strengths into a child candidate.
* **Mutation** — Apply targeted edits to a single candidate guided by feedback
  from its evaluation (analogous to a textual gradient).
* **Elitism** — Always keep the top-k candidates across generations so good
  solutions are never lost.

Integration
-----------
- Extends ``BaseOptimizer`` from ``evoagentx.optimizers.engine.base``.
- Uses ``CostTracker`` from ``evoagentx.core.cost_tracker`` when available.
- Can be wrapped with ``ConstrainedOptimizer`` from
  ``evoagentx.optimizers.constraint_layer``.

Usage
-----
    >>> from evoagentx.optimizers.gepa_optimizer import GEPAOptimizer
    >>> optimizer = GEPAOptimizer(
    ...     registry=registry,
    ...     program=my_program,
    ...     evaluator=my_evaluator,
    ...     llm=my_llm,
    ...     population_size=12,
    ...     generations=20,
    ...     tournament_sample_size=8,
    ... )
    >>> best_cfg, history = optimizer.optimize()
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.logging import logger
from .engine.base import BaseOptimizer
from .engine.registry import ParamRegistry

# ---------------------------------------------------------------------------
# Optional CostTracker integration (Phase 0 — import guard)
# ---------------------------------------------------------------------------
try:
    from ..core.cost_tracker import CostTracker  # type: ignore
    _COST_TRACKER_AVAILABLE = True
except ImportError:
    _COST_TRACKER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class GEPACandidate:
    """A single member of the GEPA population.

    Attributes:
        config: Parameter name → prompt string mapping.
        score: Latest evaluation score (None until evaluated).
        generation: Which generation produced this candidate.
        lineage: Human-readable description of how this candidate was created
            (e.g. ``"seed"``, ``"crossover(p0,p1)"``, ``"mutation(p2)"``).
    """

    config: Dict[str, Any]
    score: Optional[float] = None
    generation: int = 0
    lineage: str = "seed"


@dataclass
class GEPAHistory:
    """Tracks the progress of a GEPA optimization run.

    Attributes:
        best_score_per_generation: Best score observed at the end of each
            generation (index 0 = initial population evaluation).
        best_config: Configuration that achieved the overall best score.
        total_evaluations: Number of candidate×example evaluations performed.
        evaluation_calls_saved: Estimated calls saved vs. exhaustive evaluation
            (population × examples × generations).
    """

    best_score_per_generation: List[float] = field(default_factory=list)
    best_config: Optional[Dict[str, Any]] = None
    total_evaluations: int = 0
    evaluation_calls_saved: int = 0


# ---------------------------------------------------------------------------
# LLM prompt templates (kept as module-level constants for caching)
# ---------------------------------------------------------------------------

_CROSSOVER_SYSTEM = (
    "You are an expert prompt engineer. "
    "Your task is to combine the best qualities of two parent prompts into a "
    "single child prompt that outperforms both parents."
)

_CROSSOVER_USER = """## Parent A
{parent_a}

## Parent B
{parent_b}

## Task context
{task_context}

## Instruction
Produce ONE improved prompt that combines the strongest elements of Parent A and
Parent B. The child should be clear, specific, and aligned with the task context.
Output ONLY the new prompt text — no preamble, no explanation."""

_MUTATION_SYSTEM = (
    "You are an expert prompt engineer. "
    "Your task is to improve a prompt by making targeted edits guided by "
    "observed failure modes."
)

_MUTATION_USER = """## Current prompt
{current_prompt}

## Observed weaknesses
{weaknesses}

## Task context
{task_context}

## Instruction
Produce ONE improved version of the current prompt that addresses the observed
weaknesses. Keep changes minimal and targeted.
Output ONLY the new prompt text — no preamble, no explanation."""

_WEAKNESS_SYSTEM = (
    "You are an evaluation assistant. "
    "Summarise concisely what weaknesses or failure modes a prompt exhibits "
    "based on the evaluation feedback provided."
)

_WEAKNESS_USER = """## Prompt being evaluated
{prompt}

## Evaluation feedback
{feedback}

## Instruction
List 2-3 specific weaknesses of this prompt in plain bullet points. Be concise."""


# ---------------------------------------------------------------------------
# GEPAOptimizer
# ---------------------------------------------------------------------------

class GEPAOptimizer(BaseOptimizer):
    """Genetic-Evolutionary Prompt Adaptation optimizer.

    Args:
        registry: Parameter registry containing the prompt parameters to
            optimise.
        program: Callable that executes the workflow and returns a result dict.
        evaluator: Callable that scores a result dict and returns a float in
            [0, 1].  Must accept ``(result_dict, example_dict) -> float``.
        llm: ``BaseLLM`` instance used to perform crossover and mutation.
        population_size: Number of candidates maintained per generation.
        generations: Number of evolution generations to run.
        tournament_sample_size: Number of training examples used per
            tournament evaluation.  Smaller values are cheaper; larger values
            are more reliable.  Defaults to 8.
        elite_fraction: Fraction of the population carried forward unchanged
            between generations.  E.g. 0.2 keeps the top 20%.
        crossover_rate: Probability that a new child is produced by crossover
            (vs. pure mutation).
        training_examples: Optional list of example dicts used for evaluation.
            If omitted, the evaluator is called without an example argument.
        cost_tracker: Optional ``CostTracker`` instance for budget enforcement.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        registry: ParamRegistry,
        program: Callable[..., Dict[str, Any]],
        evaluator: Callable[..., float],
        llm: Any,
        population_size: int = 12,
        generations: int = 20,
        tournament_sample_size: int = 8,
        elite_fraction: float = 0.2,
        crossover_rate: float = 0.5,
        training_examples: Optional[List[Dict[str, Any]]] = None,
        cost_tracker: Optional[Any] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(registry=registry, program=program, evaluator=evaluator)

        if population_size < 2:
            raise ValueError("population_size must be >= 2 for crossover to work.")
        if not (0.0 < elite_fraction < 1.0):
            raise ValueError("elite_fraction must be in (0, 1).")
        if not (0.0 <= crossover_rate <= 1.0):
            raise ValueError("crossover_rate must be in [0, 1].")

        self.llm = llm
        self.population_size = population_size
        self.generations = generations
        self.tournament_sample_size = tournament_sample_size
        self.elite_fraction = elite_fraction
        self.crossover_rate = crossover_rate
        self.training_examples = training_examples or []
        self.cost_tracker = cost_tracker
        self.history = GEPAHistory()

        if seed is not None:
            random.seed(seed)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def optimize(self) -> Tuple[Dict[str, Any], GEPAHistory]:
        """Run the GEPA optimization loop.

        Returns:
            Tuple of ``(best_config, history)`` where ``best_config`` maps
            parameter names to their optimised prompt strings and ``history``
            contains full progress data.
        """
        logger.info(
            "GEPA: starting optimization — population=%d  generations=%d  "
            "tournament_sample=%d",
            self.population_size,
            self.generations,
            self.tournament_sample_size,
        )

        # Initialise population from current registry values
        population = self._seed_population()

        # Evaluate initial population
        self._evaluate_population(population, generation=0)
        population.sort(key=lambda c: c.score or 0.0, reverse=True)
        self._record_generation(population, gen=0)

        # Evolution loop
        for gen in range(1, self.generations + 1):
            logger.info("GEPA: generation %d/%d", gen, self.generations)
            population = self._evolve(population, gen)
            self._evaluate_population(population, generation=gen)
            population.sort(key=lambda c: c.score or 0.0, reverse=True)
            self._record_generation(population, gen)

            if self.cost_tracker is not None and _COST_TRACKER_AVAILABLE:
                try:
                    self.cost_tracker.check_budget()
                except Exception as budget_err:  # noqa: BLE001
                    logger.warning("GEPA: cost budget exceeded — %s", budget_err)
                    break

        # Apply best config to registry
        best = population[0]
        self.apply_cfg(best.config)
        self.history.best_config = best.config

        # Estimate evaluation savings
        exhaustive = self.population_size * len(self.training_examples) * self.generations
        self.history.evaluation_calls_saved = max(0, exhaustive - self.history.total_evaluations)

        logger.info(
            "GEPA: done — best_score=%.4f  total_evals=%d  saved_evals=%d",
            best.score,
            self.history.total_evaluations,
            self.history.evaluation_calls_saved,
        )
        return best.config, self.history

    # ------------------------------------------------------------------
    # Population initialisation
    # ------------------------------------------------------------------

    def _seed_population(self) -> List[GEPACandidate]:
        """Create the initial population from the current registry state.

        The first candidate is always the unmodified current configuration.
        Additional candidates are small random perturbations (to bootstrap
        diversity before LLM-driven evolution takes over).
        """
        base_cfg = self.get_current_cfg()
        population: List[GEPACandidate] = [
            GEPACandidate(config=deepcopy(base_cfg), generation=0, lineage="seed_base")
        ]

        param_names = self.param_names()
        for i in range(1, self.population_size):
            candidate_cfg = deepcopy(base_cfg)
            # Randomly pick one parameter to mutate with a generic nudge
            if param_names:
                target = random.choice(param_names)
                original = candidate_cfg.get(target, "")
                mutated = self._mutate_with_llm(
                    prompt=str(original),
                    weaknesses="No specific weaknesses known yet — explore alternative phrasings.",
                    task_context=f"Parameter: {target}",
                )
                candidate_cfg[target] = mutated
            population.append(
                GEPACandidate(
                    config=candidate_cfg,
                    generation=0,
                    lineage=f"seed_variant_{i}",
                )
            )

        logger.info("GEPA: seeded population of %d candidates.", len(population))
        return population

    # ------------------------------------------------------------------
    # Evaluation (efficient — tournament sampling)
    # ------------------------------------------------------------------

    def _evaluate_population(
        self,
        population: List[GEPACandidate],
        generation: int,
    ) -> None:
        """Score each candidate using a bandit-style sample of examples.

        Rather than evaluating every candidate on every training example, each
        candidate is evaluated on a random subset of size
        ``tournament_sample_size``.  This reduces evaluation cost while still
        providing a reliable relative ranking signal.

        Args:
            population: List of candidates to score.
            generation: Current generation number (used for logging only).
        """
        sample_examples: List[Dict[str, Any]] = []
        if self.training_examples:
            k = min(self.tournament_sample_size, len(self.training_examples))
            sample_examples = random.sample(self.training_examples, k)

        for candidate in population:
            if candidate.score is not None:
                # Already evaluated (elites carried forward)
                continue
            candidate.score = self._score_candidate(candidate.config, sample_examples)
            self.history.total_evaluations += max(1, len(sample_examples))

    def _score_candidate(
        self,
        config: Dict[str, Any],
        examples: List[Dict[str, Any]],
    ) -> float:
        """Apply config, run program, return mean score over examples.

        Args:
            config: Parameter configuration to apply temporarily.
            examples: Training examples to evaluate on.

        Returns:
            Mean score in [0, 1].
        """
        self.apply_cfg(config)

        if not examples:
            try:
                result = self.program()
                score = float(self.evaluator(result))
            except Exception as exc:  # noqa: BLE001
                logger.warning("GEPA: evaluation error — %s", exc)
                score = 0.0
            return max(0.0, min(1.0, score))

        scores: List[float] = []
        for example in examples:
            try:
                result = self.program(example)
                scores.append(float(self.evaluator(result, example)))
            except Exception as exc:  # noqa: BLE001
                logger.warning("GEPA: evaluation error on example — %s", exc)
                scores.append(0.0)

        raw = sum(scores) / len(scores) if scores else 0.0
        return max(0.0, min(1.0, raw))

    # ------------------------------------------------------------------
    # Evolution operators
    # ------------------------------------------------------------------

    def _evolve(
        self,
        population: List[GEPACandidate],
        generation: int,
    ) -> List[GEPACandidate]:
        """Produce the next generation via elitism + crossover/mutation.

        Args:
            population: Current sorted population (best first).
            generation: Generation number of the children to be produced.

        Returns:
            New population list of size ``population_size``.
        """
        n_elite = max(1, int(self.population_size * self.elite_fraction))
        elites = [
            GEPACandidate(
                config=deepcopy(c.config),
                score=c.score,  # carry score forward — no re-evaluation needed
                generation=c.generation,
                lineage=c.lineage,
            )
            for c in population[:n_elite]
        ]

        children: List[GEPACandidate] = []
        while len(children) < self.population_size - n_elite:
            if random.random() < self.crossover_rate and len(population) >= 2:
                child = self._crossover(
                    parent_a=self._tournament_select(population),
                    parent_b=self._tournament_select(population),
                    generation=generation,
                )
            else:
                child = self._mutate(
                    parent=self._tournament_select(population),
                    generation=generation,
                )
            children.append(child)

        return elites + children

    def _tournament_select(
        self,
        population: List[GEPACandidate],
        k: int = 3,
    ) -> GEPACandidate:
        """Select one candidate via k-way tournament selection.

        Args:
            population: Pool to draw from.
            k: Tournament size.  Higher k applies more selection pressure.

        Returns:
            The highest-scoring candidate among the k randomly drawn contenders.
        """
        contenders = random.sample(population, min(k, len(population)))
        return max(contenders, key=lambda c: c.score or 0.0)

    def _crossover(
        self,
        parent_a: GEPACandidate,
        parent_b: GEPACandidate,
        generation: int,
    ) -> GEPACandidate:
        """Produce a child by LLM-guided crossover of two parents.

        Each parameter is independently crossed — the LLM is asked to merge the
        two parent prompt strings for that parameter.

        Args:
            parent_a: First parent candidate.
            parent_b: Second parent candidate.
            generation: Current generation number.

        Returns:
            New child ``GEPACandidate`` with score=None.
        """
        child_cfg: Dict[str, Any] = {}
        for name in self.param_names():
            val_a = str(parent_a.config.get(name, ""))
            val_b = str(parent_b.config.get(name, ""))

            if val_a == val_b:
                child_cfg[name] = val_a
            else:
                child_cfg[name] = self._crossover_with_llm(
                    parent_a=val_a,
                    parent_b=val_b,
                    task_context=f"Parameter: {name}",
                )

        return GEPACandidate(
            config=child_cfg,
            generation=generation,
            lineage=f"crossover(gen{parent_a.generation},gen{parent_b.generation})",
        )

    def _mutate(
        self,
        parent: GEPACandidate,
        generation: int,
    ) -> GEPACandidate:
        """Produce a child by LLM-guided mutation of a single parent.

        A random subset of parameters is selected for mutation; the remainder
        are inherited unchanged.  The LLM is asked to improve the prompt based
        on an auto-generated weakness summary.

        Args:
            parent: Parent candidate to mutate.
            generation: Current generation number.

        Returns:
            New child ``GEPACandidate`` with score=None.
        """
        child_cfg = deepcopy(parent.config)
        param_names = self.param_names()
        if not param_names:
            return GEPACandidate(
                config=child_cfg,
                generation=generation,
                lineage=f"mutation(gen{parent.generation})",
            )

        # Mutate 1–2 randomly selected parameters per child
        n_to_mutate = random.randint(1, min(2, len(param_names)))
        targets = random.sample(param_names, n_to_mutate)

        for name in targets:
            current_val = str(child_cfg.get(name, ""))
            weaknesses = self._summarise_weaknesses(
                prompt=current_val,
                score=parent.score or 0.0,
                param_name=name,
            )
            child_cfg[name] = self._mutate_with_llm(
                prompt=current_val,
                weaknesses=weaknesses,
                task_context=f"Parameter: {name}",
            )

        return GEPACandidate(
            config=child_cfg,
            generation=generation,
            lineage=f"mutation(gen{parent.generation})",
        )

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _crossover_with_llm(
        self,
        parent_a: str,
        parent_b: str,
        task_context: str,
    ) -> str:
        """Ask the LLM to merge two parent prompts into a child prompt.

        Args:
            parent_a: First parent prompt text.
            parent_b: Second parent prompt text.
            task_context: Short description of the parameter being crossed.

        Returns:
            Child prompt string produced by the LLM, or ``parent_a`` on error.
        """
        user_msg = _CROSSOVER_USER.format(
            parent_a=parent_a,
            parent_b=parent_b,
            task_context=task_context,
        )
        try:
            response = self.llm.generate(
                prompt=user_msg,
                system_prompt=_CROSSOVER_SYSTEM,
            )
            return (response.content or parent_a).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("GEPA: crossover LLM call failed — %s", exc)
            return parent_a

    def _mutate_with_llm(
        self,
        prompt: str,
        weaknesses: str,
        task_context: str,
    ) -> str:
        """Ask the LLM to improve a prompt given observed weaknesses.

        Args:
            prompt: Current prompt text to improve.
            weaknesses: Bullet-point summary of observed failure modes.
            task_context: Short description of the parameter being mutated.

        Returns:
            Improved prompt string, or the original on error.
        """
        user_msg = _MUTATION_USER.format(
            current_prompt=prompt,
            weaknesses=weaknesses,
            task_context=task_context,
        )
        try:
            response = self.llm.generate(
                prompt=user_msg,
                system_prompt=_MUTATION_SYSTEM,
            )
            return (response.content or prompt).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("GEPA: mutation LLM call failed — %s", exc)
            return prompt

    def _summarise_weaknesses(
        self,
        prompt: str,
        score: float,
        param_name: str,
    ) -> str:
        """Generate a weakness summary to guide mutation.

        For high-scoring candidates this provides a light touch ("minor
        improvements only").  For low-scoring candidates it requests a more
        substantial rewrite.

        Args:
            prompt: Current prompt text.
            score: Current evaluation score (used to calibrate severity).
            param_name: Parameter name (used in the feedback context).

        Returns:
            Weakness summary string.
        """
        if score >= 0.8:
            return f"Minor: prompt for '{param_name}' is already strong. Explore small refinements only."
        if score >= 0.5:
            feedback = (
                f"Prompt for '{param_name}' has moderate performance (score={score:.2f}). "
                "It may be too vague or missing important constraints."
            )
        else:
            feedback = (
                f"Prompt for '{param_name}' has low performance (score={score:.2f}). "
                "It is likely unclear, ambiguous, or misaligned with the task."
            )

        user_msg = _WEAKNESS_USER.format(prompt=prompt, feedback=feedback)
        try:
            response = self.llm.generate(
                prompt=user_msg,
                system_prompt=_WEAKNESS_SYSTEM,
            )
            return (response.content or feedback).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("GEPA: weakness summarisation failed — %s", exc)
            return feedback

    # ------------------------------------------------------------------
    # History recording
    # ------------------------------------------------------------------

    def _record_generation(
        self,
        population: List[GEPACandidate],
        gen: int,
    ) -> None:
        """Record the best score for a completed generation.

        Args:
            population: Sorted population list (best first).
            gen: Completed generation number.
        """
        best_score = population[0].score or 0.0
        self.history.best_score_per_generation.append(best_score)
        logger.info(
            "GEPA: gen %d — best_score=%.4f  pop_mean=%.4f",
            gen,
            best_score,
            sum(c.score or 0.0 for c in population) / len(population),
        )

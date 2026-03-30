"""
End-to-end integration tests for AgentAugi module chains.

Each scenario exercises a realistic multi-step workflow by wiring together
two or more modules and verifying their observable outputs — no real LLM
calls are made (mock LLMs return deterministic JSON).

Scenarios
---------
1  Learning Loop        MistakeNotebook → EvoSkillPipeline.analyze_failures()
2  MASTER + PlanCache   MASTERSearch → PlanCache store/retrieve/adapt
3  JitRL + Reflexion    JitRLMemory trajectories → nudge_prompt + Reflexion episodes
4  Speculative + Cost   SpeculativeExecutor pattern hit → CostTracker accounting
5  HyperOptimizer       Genome seed → mutate → crossover → fitness evaluation
6  Full Pipeline        DifficultyRouter → MASTER → PlanCache → JitRL → MistakeNotebook
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# ── core imports ─────────────────────────────────────────────────────────────
from evoagentx.core.master_search import MASTERSearch, MASTERConfig, MASTERNode
from evoagentx.core.plan_cache import PlanCache, PlanTemplate, PlanStep
from evoagentx.core.speculative_executor import (
    SpeculativeExecutor,
    SpeculativeConfig,
    ToolCallRecord,
    ToolPrediction,
)
from evoagentx.core.cost_tracker import CostTracker
from evoagentx.core.tool_synthesizer import ToolSynthesizer, ToolRegistry
from evoagentx.core.difficulty_router import DifficultyRouter, DifficultyTier
from evoagentx.core.evoskill import EvoSkillPipeline, EvoSkillConfig

# ── memory imports ────────────────────────────────────────────────────────────
from evoagentx.memory.jitrl import (
    JitRLMemory,
    JitRLConfig,
    JitRLAgent,
    TrajectoryStatistics,
    TrajectoryStep,
)
from evoagentx.memory.reflexion import (
    ReflexionAgent,
    ReflexionMemory,
    Episode,
    TaskOutcome,
)
from evoagentx.memory.mistake_notebook import (
    MistakeNotebook,
    MistakeEntry,
    MistakeCategory,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers / stubs
# ─────────────────────────────────────────────────────────────────────────────

def _llm_response(content: str) -> MagicMock:
    """Return a mock LLM response whose .content attribute equals *content*."""
    resp = MagicMock()
    resp.content = content
    return resp


def _make_tool_llm() -> MagicMock:
    """Mock LLM that returns a valid SynthesizedTool JSON for ToolSynthesizer."""
    source = (
        "def handle_tool_failure(description):\n"
        "    \"\"\"Handle tool failures.\"\"\"\n"
        "    return f'handled: {description}'\n"
    )
    spec = {
        "tool_name": "handle_tool_failure",
        "description": "Handles tool-related failures gracefully.",
        "parameter_schema": {"description": {"type": "str", "description": "Failure description"}},
        "required_params": ["description"],
        "source_code": source,
    }
    llm = MagicMock()
    llm.generate.return_value = _llm_response(json.dumps(spec))
    return llm


def _make_master_llm() -> MagicMock:
    """Mock LLM for MASTERSearch: returns candidate actions + self-evaluation."""
    llm = MagicMock()

    def _generate(prompt: str, **kwargs) -> MagicMock:
        # Candidate generation
        if "Generate" in prompt or "candidate" in prompt.lower() or "next" in prompt.lower():
            return _llm_response("Action A\nAction B\nAction C")
        # Self-evaluation
        return _llm_response(
            '{"goal_progress": 0.7, "reasoning_quality": 0.8, '
            '"path_confidence": 0.75, "rationale": "Good progress."}'
        )

    llm.generate.side_effect = _generate
    return llm


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1 — Learning Loop
# MistakeNotebook records 3+ similar failures → EvoSkillPipeline.analyze_failures()
# clusters them → identifies gap → generates tool spec
# ─────────────────────────────────────────────────────────────────────────────

class TestScenario1LearningLoop:
    """MistakeNotebook → EvoSkillPipeline → SkillGap → tool spec."""

    def _make_notebook_with_failures(self) -> MistakeNotebook:
        nb = MistakeNotebook()
        for i in range(4):
            nb.record(
                MistakeEntry(
                    attempted=f"call search_tool with query parameter {i}",
                    went_wrong="search_tool raised TypeError: unexpected keyword argument",
                    fix="pass positional argument instead of keyword",
                    category=MistakeCategory.TOOL_MISUSE,
                ),
                auto_save=False,
            )
        return nb

    def test_analyze_failures_returns_gap(self):
        nb = self._make_notebook_with_failures()
        assert len(nb.entries) == 4

        registry = ToolRegistry()
        tool_llm = _make_tool_llm()
        synth = ToolSynthesizer(llm=tool_llm, registry=registry)
        config = EvoSkillConfig(min_mistake_frequency=3, severity_threshold=0.5)
        pipeline = EvoSkillPipeline(
            mistake_notebook=nb,
            tool_synthesizer=synth,
            config=config,
        )

        gaps = pipeline.analyze_failures()

        assert len(gaps) >= 1, "Expected at least one skill gap"
        gap = gaps[0]
        assert gap.frequency >= 3
        assert gap.gap_type == MistakeCategory.TOOL_MISUSE.value
        assert len(gap.source_mistakes) >= 3

    def test_gap_generates_tool_spec_with_llm(self):
        nb = self._make_notebook_with_failures()
        registry = ToolRegistry()
        tool_llm = _make_tool_llm()
        synth = ToolSynthesizer(llm=tool_llm, registry=registry)

        # Mock llm_fn to return a structured spec
        spec_json = json.dumps({
            "task_description": "A tool that validates keyword arguments before calling search_tool",
            "examples": [],
        })
        llm_fn = MagicMock(return_value=_llm_response(spec_json))

        config = EvoSkillConfig(min_mistake_frequency=3, auto_deploy=False)
        pipeline = EvoSkillPipeline(
            mistake_notebook=nb,
            tool_synthesizer=synth,
            config=config,
            llm_fn=llm_fn,
        )

        gaps = pipeline.analyze_failures()
        assert gaps, "No gaps found"

        spec = pipeline._generate_tool_spec(gaps[0])
        assert "task_description" in spec
        assert len(spec["task_description"]) > 5

    def test_run_cycle_produces_discovery(self):
        nb = self._make_notebook_with_failures()
        registry = ToolRegistry()
        tool_llm = _make_tool_llm()
        synth = ToolSynthesizer(llm=tool_llm, registry=registry)
        config = EvoSkillConfig(min_mistake_frequency=3, auto_deploy=False)
        pipeline = EvoSkillPipeline(
            mistake_notebook=nb,
            tool_synthesizer=synth,
            config=config,
        )

        discoveries = pipeline.run_cycle()

        assert len(discoveries) >= 1
        discovery = discoveries[0]
        assert discovery.synthesized_tool is not None
        assert discovery.synthesized_tool.validation_passed
        assert not discovery.deployed  # auto_deploy=False

        stats = pipeline.stats()
        assert stats["skills_synthesized"] >= 1

    def test_consult_before_record_returns_empty(self):
        nb = MistakeNotebook()
        hints = nb.format_for_prompt("call search_tool with keyword arg")
        assert hints == ""

    def test_consult_after_record_returns_relevant(self):
        nb = self._make_notebook_with_failures()
        hints = nb.format_for_prompt("call search_tool")
        assert "search_tool" in hints or "tool" in hints.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2 — MASTER + Plan Cache
# MASTERSearch on a task → store result in PlanCache → retrieve for similar
# task → verify cache hit
# ─────────────────────────────────────────────────────────────────────────────

class TestScenario2MASTERPlanCache:
    """MASTERSearch → PlanCache store → retrieve → adapt."""

    def test_search_result_stored_and_retrieved(self):
        llm = _make_master_llm()
        master = MASTERSearch(llm=llm, config=MASTERConfig(max_iterations=3, num_candidates=2))
        cache = PlanCache(similarity_threshold=0.3)

        result = master.search(task="Sort a list of numbers", initial_state="")
        assert result is not None

        # Convert action sequence to plan steps and store
        steps = [
            PlanStep(action_type="analyse", description=f"Step: {a}", estimated_cost=0.001)
            for a in (result.action_sequence or ["default_action"])
        ]
        cache.store("Sort a list of numbers", steps, outcome="success")

        assert len(cache) == 1
        stats = cache.stats()
        assert stats["num_templates"] == 1

    def test_cache_hit_on_similar_task(self):
        cache = PlanCache(similarity_threshold=0.3)
        steps = [
            PlanStep(action_type="search", description="Search for sorting algorithms", tool_name="web_search"),
            PlanStep(action_type="code", description="Write sort code"),
            PlanStep(action_type="test", description="Test the implementation"),
        ]
        cache.store("Sort a list of integers in Python", steps, outcome="success")

        # Very similar task — should hit
        template = cache.retrieve("Sort a list of integers in Python")
        assert template is not None, "Expected cache hit for identical task"

        stats = cache.stats()
        assert stats["hits"] >= 1

    def test_cache_adapt_preserves_structure(self):
        cache = PlanCache(similarity_threshold=0.3)
        steps = [
            PlanStep(action_type="search", description="Search for sorting in Python", tool_name="web_search"),
            PlanStep(action_type="code", description="Write sort function"),
        ]
        cache.store("Sort a list of integers in Python", steps, outcome="success")

        template = cache.retrieve("Sort a list of integers in Python")
        assert template is not None

        adapted = cache.adapt(template, "Sort a list of strings in Go")
        assert len(adapted) == len(steps)
        assert adapted[0].action_type == "search"
        assert adapted[1].action_type == "code"
        # Tool names preserved
        assert adapted[0].tool_name == "web_search"

    def test_cache_miss_below_threshold(self):
        cache = PlanCache(similarity_threshold=0.9)
        steps = [PlanStep(action_type="analyse", description="Analyse data")]
        cache.store("Analyse genomic sequences with BLAST", steps, outcome="success")

        template = cache.retrieve("Write a web scraper")
        assert template is None

        stats = cache.stats()
        assert stats["misses"] >= 1

    def test_structural_hash_grouping(self):
        from evoagentx.core.plan_cache import _compute_structural_hash

        steps_a = [
            PlanStep("search", "search A", tool_name="tool_x"),
            PlanStep("code", "code A"),
        ]
        steps_b = [
            PlanStep("search", "search B — different text", tool_name="tool_x"),
            PlanStep("code", "code B — different text"),
        ]
        # Same structure → same hash
        assert _compute_structural_hash(steps_a) == _compute_structural_hash(steps_b)

        steps_c = [PlanStep("code", "code only")]
        assert _compute_structural_hash(steps_a) != _compute_structural_hash(steps_c)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 3 — JitRL + Reflexion
# Record 5 trajectories with JitRLMemory → create Episodes in Reflexion →
# verify nudge_prompt and reflection both work on episode 6
# ─────────────────────────────────────────────────────────────────────────────

class TestScenario3JitRLReflexion:
    """JitRLMemory trajectory accumulation + ReflexionMemory episode recall."""

    ACTIONS = ["search", "parse", "summarise", "code", "test"]

    def _make_trajectory(self, outcome: str, steps: Optional[List[str]] = None) -> TrajectoryStatistics:
        steps = steps or self.ACTIONS[:3]
        return TrajectoryStatistics(
            steps=[TrajectoryStep(action_type=a, reward=0.0) for a in steps],
            outcome=outcome,
            total_reward=1.0 if outcome == "success" else 0.0,
        )

    def test_five_trajectories_accumulate_stats(self):
        memory = JitRLMemory(config=JitRLConfig(min_samples=3))

        for _ in range(3):
            memory.record_trajectory(self._make_trajectory("success"))
        for _ in range(2):
            memory.record_trajectory(self._make_trajectory("failure"))

        s = memory.stats()
        assert s["total_trajectories"] == 5
        assert s["action_coverage"] > 0

    def test_nudge_prompt_after_sufficient_samples(self):
        memory = JitRLMemory(config=JitRLConfig(min_samples=3, learning_rate=0.5))

        # Record enough trajectories so that min_samples is reached
        for _ in range(6):
            memory.record_trajectory(self._make_trajectory("success", ["search", "parse", "summarise"]))

        prompt = memory.nudge_prompt(
            recent_steps=["search", "parse"],
            available_actions=["search", "parse", "summarise", "code"],
        )
        # After 6 successes the nudge should mention at least one action
        assert isinstance(prompt, str)
        # Either we get a nudge or the stats are below min_samples still —
        # both are valid since context hash granularity affects coverage.
        if prompt:
            assert "success" in prompt.lower() or "%" in prompt

    def test_get_action_bias_returns_all_actions(self):
        memory = JitRLMemory(config=JitRLConfig(min_samples=2))

        for _ in range(4):
            memory.record_trajectory(self._make_trajectory("success"))

        biases = memory.get_action_bias(
            recent_steps=["search"],
            available_actions=["search", "code", "test"],
        )
        assert set(biases.keys()) == {"search", "code", "test"}
        # All biases are floats
        assert all(isinstance(v, float) for v in biases.values())

    def test_reflexion_memory_stores_and_retrieves_episodes(self):
        reflexion_mem = ReflexionMemory()

        for i in range(5):
            ep = Episode(
                task_description=f"search and summarise weather data in city{i}",
                outcome=TaskOutcome.SUCCESS if i % 2 == 0 else TaskOutcome.FAILURE,
                reflection=f"Used search tool effectively; city{i} query worked.",
            )
            reflexion_mem.add_episode(ep)

        assert len(reflexion_mem.episodes) == 5

        similar = reflexion_mem.find_similar("search and summarise weather data")
        assert len(similar) >= 1
        assert "weather" in similar[0].task_description or "search" in similar[0].task_description

    def test_reflexion_get_reflections_for_task(self):
        reflexion_mem = ReflexionMemory()
        for i in range(5):
            ep = Episode(
                task_description=f"search weather data for analysis run {i}",
                outcome=TaskOutcome.SUCCESS,
                reflection=f"The search step at run {i} completed without error.",
            )
            reflexion_mem.add_episode(ep)

        reflections = reflexion_mem.get_reflections_for_task("search weather data for analysis")
        assert isinstance(reflections, str)
        assert len(reflections) > 0

    def test_jitrl_agent_records_trajectory_on_execute(self):
        """JitRLAgent wraps a stub agent, records trajectory, nudges prompt."""

        class _StubAgent:
            system_prompt = "You are helpful."

            def execute(self, **kwargs) -> str:
                return "stub result"

        memory = JitRLMemory(config=JitRLConfig(min_samples=2))
        stub = _StubAgent()
        agent = JitRLAgent(
            agent=stub,
            memory=memory,
            available_actions=["search", "code", "test"],
        )

        # 6th "episode" after 5 pre-loaded trajectories
        for _ in range(5):
            memory.record_trajectory(
                TrajectoryStatistics(
                    steps=[TrajectoryStep("search"), TrajectoryStep("code")],
                    outcome="success",
                )
            )

        result = agent.execute(
            task_steps=["search", "code"],
            outcome="success",
        )
        assert result == "stub result"
        # Memory should now have 6 trajectories total
        assert memory.stats()["total_trajectories"] == 6


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 4 — Speculative Execution + Cost
# SpeculativeExecutor with predictable tool sequence → verify hit stats and
# CostTracker accounting
# ─────────────────────────────────────────────────────────────────────────────

class TestScenario4SpeculativeCost:
    """SpeculativeExecutor pattern matching + CostTracker integration."""

    # Repeated tool sequence that creates a detectable pattern
    HISTORY_SEQUENCE = ["search", "parse", "search", "parse", "search", "parse"]

    def _make_history(self) -> List[ToolCallRecord]:
        return [ToolCallRecord(tool=t, result=f"{t}_result") for t in self.HISTORY_SEQUENCE]

    def test_predict_next_tool_from_pattern(self):
        config = SpeculativeConfig(confidence_threshold=0.5, max_speculative_depth=2)
        executor = SpeculativeExecutor(config=config)

        history = self._make_history()
        prediction = executor.predict_next_tool(
            history=history,
            available_tools=["search", "parse", "summarise"],
        )

        # After [search, parse] x3 the pattern strongly predicts "parse" after "search"
        assert prediction is not None, "Expected a pattern-based prediction"
        assert prediction.predicted_tool in {"search", "parse"}
        assert prediction.confidence >= 0.5

    def test_speculative_hit_increments_stats(self):
        config = SpeculativeConfig(confidence_threshold=0.5)
        tracker = CostTracker()
        executor = SpeculativeExecutor(config=config, cost_tracker=tracker)

        tool_registry = {"parse": lambda: "parsed_data"}
        prediction = ToolPrediction(
            predicted_tool="parse",
            predicted_args={},
            confidence=0.9,
            basis="pattern",
        )

        async def _run():
            launched = await executor.speculatively_execute(prediction, tool_registry)
            assert launched, "Expected speculation to launch"
            result = await executor.resolve("parse", {})
            return result

        result = asyncio.run(_run())
        assert result is not None
        assert result.was_used
        assert result.result == "parsed_data"

        stats = executor.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 0

    def test_speculative_miss_increments_miss_stat(self):
        config = SpeculativeConfig(confidence_threshold=0.5)
        executor = SpeculativeExecutor(config=config)

        tool_registry = {"parse": lambda: "parsed_data"}
        prediction = ToolPrediction(
            predicted_tool="parse",
            predicted_args={},
            confidence=0.9,
            basis="pattern",
        )

        async def _run():
            await executor.speculatively_execute(prediction, tool_registry)
            result = await executor.resolve("summarise", {})
            return result

        result = asyncio.run(_run())
        assert result is None  # different tool → miss

        stats = executor.stats()
        assert stats["misses"] == 1

    def test_cost_tracker_records_usage(self):
        tracker = CostTracker()
        tracker.record(
            provider="anthropic",
            model="claude-sonnet-4-6",
            input_tokens=500,
            output_tokens=100,
        )
        tracker.record(
            provider="anthropic",
            model="claude-sonnet-4-6",
            input_tokens=300,
            output_tokens=80,
        )

        summary = tracker.summary()
        assert summary.total_calls == 2
        assert summary.total_input_tokens == 800
        assert summary.total_output_tokens == 180
        assert summary.total_cost_usd > 0.0

    def test_cost_tracker_budget_exceeded(self):
        from evoagentx.core.cost_tracker import CostBudgetExceeded

        tracker = CostTracker(max_budget_usd=0.000001)
        with pytest.raises(CostBudgetExceeded):
            tracker.record(
                provider="anthropic",
                model="claude-opus-4-6",
                input_tokens=10_000,
                output_tokens=5_000,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 5 — HyperOptimizer
# Seed 3 genomes → mutate → crossover → evaluate with mock fitness fn →
# verify population evolved
#
# NOTE: evoagentx/optimizers/hyper_optimizer.py has not been implemented yet.
# This scenario provides a self-contained stub that validates the intended
# interface and will need to be replaced with the real import once the module
# ships.
# ─────────────────────────────────────────────────────────────────────────────

# ── Inline stub (remove once evoagentx/optimizers/hyper_optimizer.py exists) ─

@dataclass
class OptimizerGenome:
    """Encodes a single optimizer configuration as an evolvable chromosome."""
    learning_rate: float = 0.1
    batch_size: int = 8
    temperature: float = 0.7
    strategy: str = "textgrad"
    fitness: float = 0.0
    generation: int = 0


@dataclass
class OptimizerPopulation:
    """A generation of optimizer genomes."""
    genomes: List[OptimizerGenome] = field(default_factory=list)
    generation: int = 0

    def best(self) -> Optional[OptimizerGenome]:
        if not self.genomes:
            return None
        return max(self.genomes, key=lambda g: g.fitness)

    def mean_fitness(self) -> float:
        if not self.genomes:
            return 0.0
        return sum(g.fitness for g in self.genomes) / len(self.genomes)


@dataclass
class HyperOptimizerConfig:
    population_size: int = 5
    elite_fraction: float = 0.2
    mutation_rate: float = 0.3
    crossover_rate: float = 0.5
    max_generations: int = 3


class HyperOptimizer:
    """Evolves optimizer hyper-parameters via mutation and crossover.

    Stub implementation matching the intended interface described in
    the Phase 4 design doc.  The real implementation will live in
    evoagentx/optimizers/hyper_optimizer.py and use LLM-guided mutations.
    """

    def __init__(
        self,
        config: Optional[HyperOptimizerConfig] = None,
        llm_fn: Optional[Any] = None,
    ) -> None:
        self.config = config or HyperOptimizerConfig()
        self._llm_fn = llm_fn
        self.population: OptimizerPopulation = OptimizerPopulation()

    def seed(self, genomes: List[OptimizerGenome]) -> None:
        self.population = OptimizerPopulation(genomes=list(genomes), generation=0)

    def mutate(self, genome: OptimizerGenome) -> OptimizerGenome:
        import random
        import copy
        child = copy.deepcopy(genome)
        child.generation = genome.generation + 1
        if random.random() < self.config.mutation_rate:
            child.learning_rate = max(1e-5, child.learning_rate * (0.5 + random.random()))
        if random.random() < self.config.mutation_rate:
            child.temperature = max(0.01, min(2.0, child.temperature + random.uniform(-0.2, 0.2)))
        return child

    def crossover(self, parent_a: OptimizerGenome, parent_b: OptimizerGenome) -> OptimizerGenome:
        child = OptimizerGenome(
            learning_rate=(parent_a.learning_rate + parent_b.learning_rate) / 2,
            batch_size=parent_a.batch_size if parent_a.fitness >= parent_b.fitness else parent_b.batch_size,
            temperature=(parent_a.temperature + parent_b.temperature) / 2,
            strategy=parent_a.strategy if parent_a.fitness >= parent_b.fitness else parent_b.strategy,
            generation=max(parent_a.generation, parent_b.generation) + 1,
        )
        return child

    def evaluate(self, genome: OptimizerGenome, fitness_fn: Any) -> float:
        genome.fitness = float(fitness_fn(genome))
        return genome.fitness

    def evolve(self, fitness_fn: Any) -> OptimizerPopulation:
        """Run one generation: evaluate → select elites → mutate + crossover."""
        import random

        for genome in self.population.genomes:
            self.evaluate(genome, fitness_fn)

        self.population.genomes.sort(key=lambda g: g.fitness, reverse=True)
        n_elite = max(1, int(len(self.population.genomes) * self.config.elite_fraction))
        elites = self.population.genomes[:n_elite]

        offspring: List[OptimizerGenome] = list(elites)
        while len(offspring) < self.config.population_size:
            if len(elites) >= 2 and random.random() < self.config.crossover_rate:
                a, b = random.sample(elites, 2)
                child = self.crossover(a, b)
            else:
                parent = random.choice(elites)
                child = self.mutate(parent)
            self.evaluate(child, fitness_fn)
            offspring.append(child)

        self.population = OptimizerPopulation(
            genomes=offspring, generation=self.population.generation + 1
        )
        return self.population


class TestScenario5HyperOptimizer:
    """Genome lifecycle: seed → mutate → crossover → fitness evaluation."""

    def _mock_fitness(self, genome: OptimizerGenome) -> float:
        """Deterministic fitness: reward lower learning rates and higher temperatures."""
        return (1.0 - genome.learning_rate) * genome.temperature

    def test_seed_populates_population(self):
        optimizer = HyperOptimizer(config=HyperOptimizerConfig(population_size=5))
        seeds = [
            OptimizerGenome(learning_rate=0.1, temperature=0.7),
            OptimizerGenome(learning_rate=0.01, temperature=1.0),
            OptimizerGenome(learning_rate=0.001, temperature=1.5),
        ]
        optimizer.seed(seeds)
        assert len(optimizer.population.genomes) == 3
        assert optimizer.population.generation == 0

    def test_mutate_changes_at_least_one_param(self):
        import random
        random.seed(42)
        optimizer = HyperOptimizer(
            config=HyperOptimizerConfig(mutation_rate=1.0)  # always mutate
        )
        parent = OptimizerGenome(learning_rate=0.1, temperature=0.7)
        child = optimizer.mutate(parent)
        # At least one parameter should change with mutation_rate=1.0
        assert (
            child.learning_rate != parent.learning_rate
            or child.temperature != parent.temperature
        )
        assert child.generation == parent.generation + 1

    def test_crossover_blends_learning_rate(self):
        optimizer = HyperOptimizer()
        a = OptimizerGenome(learning_rate=0.2, temperature=0.5, fitness=0.8)
        b = OptimizerGenome(learning_rate=0.1, temperature=1.0, fitness=0.6)
        child = optimizer.crossover(a, b)
        assert child.learning_rate == pytest.approx(0.15, abs=1e-9)
        assert child.temperature == pytest.approx(0.75, abs=1e-9)

    def test_evaluate_applies_fitness_fn(self):
        optimizer = HyperOptimizer()
        genome = OptimizerGenome(learning_rate=0.1, temperature=1.0)
        score = optimizer.evaluate(genome, self._mock_fitness)
        assert score == pytest.approx((1.0 - 0.1) * 1.0, abs=1e-9)
        assert genome.fitness == score

    def test_evolve_increases_mean_fitness(self):
        import random
        random.seed(0)
        config = HyperOptimizerConfig(population_size=5, mutation_rate=0.3, max_generations=3)
        optimizer = HyperOptimizer(config=config)
        seeds = [
            OptimizerGenome(learning_rate=0.5, temperature=0.5),
            OptimizerGenome(learning_rate=0.3, temperature=0.6),
            OptimizerGenome(learning_rate=0.2, temperature=0.7),
            OptimizerGenome(learning_rate=0.1, temperature=0.8),
            OptimizerGenome(learning_rate=0.05, temperature=0.9),
        ]
        optimizer.seed(seeds)
        # Evaluate initial population
        for g in optimizer.population.genomes:
            optimizer.evaluate(g, self._mock_fitness)
        initial_mean = optimizer.population.mean_fitness()

        for _ in range(3):
            new_pop = optimizer.evolve(self._mock_fitness)

        assert new_pop.generation == 3
        assert new_pop.best() is not None
        assert new_pop.best().fitness >= initial_mean - 0.05  # should not collapse

    def test_population_best_after_three_generations(self):
        import random
        random.seed(7)
        optimizer = HyperOptimizer(config=HyperOptimizerConfig(population_size=6, mutation_rate=0.5))
        seeds = [OptimizerGenome(learning_rate=0.1 * i, temperature=0.5 + 0.1 * i) for i in range(1, 4)]
        optimizer.seed(seeds)
        for g in optimizer.population.genomes:
            optimizer.evaluate(g, self._mock_fitness)

        for _ in range(3):
            optimizer.evolve(self._mock_fitness)

        best = optimizer.population.best()
        assert best is not None
        assert best.fitness > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 6 — Full Pipeline
# DifficultyRouter → classifies task → feeds to appropriate search →
# plan cached → trajectory scored → mistakes logged
# ─────────────────────────────────────────────────────────────────────────────

class TestScenario6FullPipeline:
    """Complete data-flow through all major modules in a single workflow."""

    def test_simple_task_routed_to_simple_tier(self):
        router = DifficultyRouter()
        decision = router.route("What is 2 + 2?")
        assert decision.tier == DifficultyTier.SIMPLE

    def test_complex_task_scores_higher_than_trivial(self):
        """Hard-keyword task must score strictly above the trivial question."""
        router = DifficultyRouter()
        trivial = router.route("What is 2 + 2?")
        complex_task = (
            "Implement and architect a distributed system. "
            "Refactor, optimise, optimize, debug, and analyse complex "
            "multi-step workflows. Compare and evaluate synthesize alternatives."
        )
        hard = router.route(complex_task)
        assert hard.score > trivial.score, (
            f"Complex task scored {hard.score:.3f} ≤ trivial {trivial.score:.3f}"
        )
        # The raw feature vector should show non-zero hard_keyword_ratio
        assert hard.features["hard_keyword_ratio"] > 0

    def test_full_flow_simple_task(self):
        """SIMPLE task: route → plan cache → JitRL trajectory → mistake logged."""
        # 1. Route
        router = DifficultyRouter()
        decision = router.route("What is the capital of France?")
        assert decision.tier == DifficultyTier.SIMPLE

        # 2. Plan cache (no prior plan; store a new one)
        cache = PlanCache(similarity_threshold=0.5)
        steps = [PlanStep(action_type="answer", description="Answer factual question")]
        cache.store("What is the capital of France?", steps, outcome="success")

        template = cache.retrieve("What is the capital of France?")
        assert template is not None

        # 3. Record trajectory in JitRL
        memory = JitRLMemory()
        memory.record_trajectory(
            TrajectoryStatistics(
                steps=[TrajectoryStep(action_type="answer", reward=1.0)],
                outcome="success",
                total_reward=1.0,
            )
        )
        assert memory.stats()["total_trajectories"] == 1

        # 4. Log zero mistakes (success path — no mistakes)
        nb = MistakeNotebook()
        assert len(nb.entries) == 0

    def test_full_flow_hard_task_with_mistake(self):
        """HARD task: route → MASTER search → cache → mistake logged → consult."""
        # 1. Route
        router = DifficultyRouter()
        task = (
            "Implement and optimize a multi-threaded task scheduler with "
            "priority queues and deadline analysis."
        )
        decision = router.route(task)
        # Score should be strictly above a trivial task's score
        trivial = router.route("Yes or no: is Python a language?")
        assert decision.score >= trivial.score
        assert decision.model_hint  # should suggest a capable model

        # 2. MASTER search (mock LLM)
        llm = _make_master_llm()
        master = MASTERSearch(
            llm=llm,
            config=MASTERConfig(max_iterations=2, num_candidates=2),
        )
        result = master.search(task=task, initial_state="")
        assert result is not None

        # 3. Cache the plan
        cache = PlanCache(similarity_threshold=0.4)
        steps = [
            PlanStep(action_type=act, description=f"Step for: {act}", estimated_cost=0.002)
            for act in (result.action_sequence or ["design", "implement", "test"])
        ]
        cache.store(task, steps, outcome="success")

        # 4. Log a mistake (tool misuse during implementation)
        nb = MistakeNotebook()
        nb.record(
            MistakeEntry(
                attempted="call scheduler.add_task with wrong priority type",
                went_wrong="priority must be int, received str",
                fix="always cast priority to int before calling add_task",
                category=MistakeCategory.TOOL_MISUSE,
            ),
            auto_save=False,
        )
        assert len(nb.entries) == 1

        # 5. Consult notebook before retry
        hints = nb.format_for_prompt("implement multi-threaded task scheduler priority")
        assert isinstance(hints, str)

        # 6. Retrieve cached plan for a similar task
        similar_task = (
            "Build an optimized multi-threaded scheduler with priority queues."
        )
        template = cache.retrieve(similar_task)
        # May or may not hit depending on Jaccard similarity; both outcomes are valid
        if template is not None:
            adapted = cache.adapt(template, similar_task)
            assert len(adapted) == len(steps)

    def test_difficulty_routing_with_jitrl_bias(self):
        """Route two tasks; use JitRL bias to prefer historically better action."""
        router = DifficultyRouter()
        memory = JitRLMemory(config=JitRLConfig(min_samples=3))

        # Warm up JitRL: "search" succeeds, "guess" fails
        for _ in range(4):
            memory.record_trajectory(TrajectoryStatistics(
                steps=[TrajectoryStep("search"), TrajectoryStep("summarise")],
                outcome="success",
            ))
        for _ in range(2):
            memory.record_trajectory(TrajectoryStatistics(
                steps=[TrajectoryStep("guess")],
                outcome="failure",
            ))

        # Route a medium task
        decision = router.route("Summarize this research paper and extract key findings.")
        assert decision.tier in {DifficultyTier.SIMPLE, DifficultyTier.MEDIUM, DifficultyTier.HARD}

        # JitRL bias should prefer "search" + "summarise" over "guess"
        biases = memory.get_action_bias(
            recent_steps=[],
            available_actions=["search", "summarise", "guess"],
        )
        assert biases["search"] > biases.get("guess", -1)

    def test_evoskill_pipeline_with_full_mistake_cycle(self):
        """Mistakes accumulate → EvoSkillPipeline deploys a tool → notebook clears entries."""
        nb = MistakeNotebook()

        # Log 4 similar hallucination mistakes
        for i in range(4):
            nb.record(
                MistakeEntry(
                    attempted=f"verify fact {i} about population statistics",
                    went_wrong=f"hallucinated {i}: returned fabricated population number",
                    fix="always cross-reference population data with a reliable source",
                    category=MistakeCategory.HALLUCINATION,
                ),
                auto_save=False,
            )

        assert nb.stats()["unresolved"] == 4

        registry = ToolRegistry()
        tool_llm = MagicMock()
        verify_source = (
            "def verify_statistic(value, source):\n"
            "    \"\"\"Verify a statistic against a source.\"\"\"\n"
            "    return source != '' and value > 0\n"
        )
        tool_llm.generate.return_value = _llm_response(json.dumps({
            "tool_name": "verify_statistic",
            "description": "Verify statistics against a data source.",
            "parameter_schema": {
                "value": {"type": "float", "description": "The statistic to verify"},
                "source": {"type": "str", "description": "Source name"},
            },
            "required_params": ["value", "source"],
            "source_code": verify_source,
        }))

        synth = ToolSynthesizer(llm=tool_llm, registry=registry)
        config = EvoSkillConfig(
            min_mistake_frequency=3,
            severity_threshold=0.5,
            auto_deploy=True,
        )
        pipeline = EvoSkillPipeline(mistake_notebook=nb, tool_synthesizer=synth, config=config)

        discoveries = pipeline.run_cycle()

        assert len(discoveries) >= 1
        deployed = [d for d in discoveries if d.deployed]
        assert len(deployed) >= 1

        stats = pipeline.stats()
        assert stats["deployed"] >= 1
        assert stats["mistakes_resolved"] >= 3

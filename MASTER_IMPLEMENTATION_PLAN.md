# AgentAugi Master Implementation Plan

*Created: March 29, 2026 | Synthesizes IMPROVEMENT_RESEARCH.md original findings + audit gaps*

---

> **CRITICAL: All code in this project is written from scratch.**
> Papers and referenced repositories are used exclusively as conceptual references for understanding algorithms, design patterns, and evaluation methodology. No code is copied, adapted, or translated from any external repository. Every implementation decision is made fresh within AgentAugi's existing architecture.

---

## 1. Current Architecture Baseline

Before building anything, understand what already exists:

| Component | File(s) | What it does |
|-----------|---------|--------------|
| **TextGrad** | `evoagentx/optimizers/textgrad_optimizer.py` | Gradient-based text optimization via LLM feedback |
| **AFlow** | `evoagentx/optimizers/aflow_optimizer.py` | MCTS-based workflow structure search |
| **SEW** | `evoagentx/optimizers/sew_optimizer.py` | Self-evolving workflow generation |
| **MIPRO** | `evoagentx/optimizers/mipro_optimizer.py` | DSPy-based few-shot prompt/instruction optimization |
| **EvoPrompt** | `evoagentx/optimizers/evoprompt_optimizer.py` | GA/DE evolutionary prompt optimization |
| **WorkFlowGraph** | `evoagentx/workflow/workflow_graph.py` | DAG-based workflow execution |
| **LongTermMemory** | `evoagentx/memory/` | Fixed-policy long-term storage |
| **Evaluator** | `evoagentx/evaluators/` | Outcome-based task scoring |
| **ToolRegistry** | `evoagentx/tools/` | 40+ tool definitions with custom interfaces |

**Known issues to fix before new features:**
- `rag/rag.py` FIXME: unbounded memory growth during KV store construction
- Concurrency race conditions in async LLM calls (parallel to upstream PR #225 pattern)
- Incomplete async support (TODOs in agent init and tool serialization)
- Incomplete HITL implementation (TODOs in `hitl/`)

---

## 2. Dependency Graph

Features must be built in dependency order. This diagram shows which items block others:

```
PHASE 0 (Infrastructure — blocks everything else)
├── Evaluation harness (TAU-Bench, SWE-Bench, AgentBench baselines)
├── Cost tracking layer
├── Prompt caching wrapper
└── Async/concurrency bug fixes

PHASE 1A (Core algorithm quality — depends on Phase 0)
├── Process reward models (AgentPRM-style step scoring)
│   ├── → Improves TextGrad gradient quality
│   ├── → Improves EvoPrompt fitness signals
│   └── → Improves AFlow node value estimation
├── MAP-Elites optimizer (multi-objective, diversity preservation)
│   └── → Resolves EvoPrompt mode collapse (Issue #212)
└── Evolution constraints / CEAO layer
    └── → Required before any production deployment

PHASE 1B (Memory foundation — depends on Phase 0, partially parallel with 1A)
├── Paged memory model (Letta-inspired, fixes rag.py FIXME)
│   └── → Required by cross-session learning
├── Reflexion-based episodic memory
│   └── → Depends on paged memory
└── Mistake Notebook
    └── → Standalone, no dependencies

PHASE 2A (Advanced optimization — depends on Phase 1A)
├── GEPA optimizer (DSPy 3.x inspired alternative to MIPRO)
├── metaTextGrad (optimize the TextGrad gradient strategy itself)
│   └── → Depends on TextGrad being stable
├── ADAS-style design archive (Pareto-optimal workflow library)
│   └── → Depends on AFlow + MAP-Elites
└── Alignment drift detection (ATP-inspired)
    └── → Depends on CEAO constraint layer

PHASE 2B (Execution efficiency — depends on Phase 0)
├── Difficulty-aware routing (DAAO-inspired)
│   ├── → Depends on evaluation harness (needs difficulty labels)
│   └── → Doubles as model routing signal for cost
├── Cycle support in WorkFlowGraph
│   └── → Required by LATS runtime planning
├── Parallel tool execution (GAP-inspired dependency graphs)
└── Streaming execution

PHASE 3 (Advanced capabilities — depends on Phases 1+2)
├── LATS runtime tree-search planning
│   └── → Depends on cycle support + process reward models
├── Tool creation (LATM-inspired ToolSynthesizer)
│   └── → Depends on stable ToolRegistry + evaluation harness
├── Cross-session learning via ReasoningBank pattern
│   └── → Depends on paged memory + Reflexion
├── Multi-agent debate quality controls
│   └── → Depends on evaluation harness (need metrics to detect degeneracy)
└── Semantic caching + FrugalGPT model cascading
    └── → Depends on difficulty-aware routing

PHASE 4 (Ecosystem — depends on Phase 3)
├── MCP server/client support
├── A2A protocol inter-agent communication
├── Federated optimization
├── Multimodal prompt evolution
└── Complete HITL (tool call review, multi-turn conversation)
```

---

## 3. Phase 0: Infrastructure (Weeks 1–4) — BLOCKS EVERYTHING

**Goal:** Establish the measurement and cost foundation that every other phase depends on. Nothing else should be built until Phase 0 is complete, because without baselines, claimed improvements cannot be validated.

### 0.1 Evaluation Harness — FIRST TASK

**Effort:** 2 weeks
**Techniques to study:** TAU-bench pass^k metric definition, SWE-Bench task format, AgentBench evaluation protocol, GAIA task structure

**What to build (from scratch):**
- `evoagentx/benchmark/tau_bench.py` — Implement pass^k reliability scoring. Run each task k times, compute pass@1, pass@k. This is AgentAugi's primary metric — a system that passes 60% on attempt 1 but only 25% reliably is worse than one at 50% consistently.
- `evoagentx/benchmark/swe_bench.py` — Adapter for SWE-Bench task format (GitHub issue → code fix evaluation). Tests long-horizon multi-step reasoning.
- `evoagentx/benchmark/agent_bench.py` — Multi-environment evaluation across 8 domains. Needed for multi-dimensional comparison.
- `evoagentx/benchmark/gaia_bench.py` — Real-world reasoning + tool use benchmark.
- `evoagentx/benchmark/baseline_runner.py` — Run all five current optimizers (TextGrad, AFlow, SEW, MIPRO, EvoPrompt) on all benchmarks, save results as the immutable baseline.

**Integration with existing architecture:**
Extends `evoagentx/benchmark/` directory. Uses existing `Evaluator` interface but adds pass^k loop around it. The baseline_runner wraps each existing optimizer with no modifications.

**Success gate:** Can reproduce a number on TAU-bench pass@1 and pass@8 for each of the five existing optimizers. These numbers are frozen as the baseline before any other work begins.

### 0.2 Cost Tracking Layer

**Effort:** 1 week
**Techniques to study:** Token counting patterns, LLM provider pricing APIs, cost budget enforcement patterns

**What to build (from scratch):**
- `evoagentx/core/cost_tracker.py` — Singleton that intercepts every LLM call, records input/output tokens, maps to cost per provider, maintains per-session totals with configurable budget limits
- Integration points: wrap `BaseLLM.call()` and `BaseLLM.async_call()` to record before/after
- `CostBudgetExceeded` exception: raise when cumulative cost exceeds configured threshold

**Integration:** Wrap the existing `BaseLLM` class with a decorator or mixin. All five optimizers inherit this automatically.

### 0.3 Prompt Caching Wrapper

**Effort:** 3 days
**Techniques to study:** Anthropic prompt caching API (cache_control parameter), OpenAI prompt caching behavior, TTL management

**What to build (from scratch):**
- `evoagentx/core/caching_llm.py` — Wraps `BaseLLM`, restructures prompts to place stable content (system prompt, tool definitions, optimization instructions) before variable content (task instance, current candidate prompt). Adds cache_control markers for Anthropic. Reports cache hit rate per session.

**Integration:** Drop-in replacement for `BaseLLM` in optimizer loops. Optimization loops typically share a fixed system prompt across thousands of calls — ideal for caching. Expected 70–90% cost reduction on repeat prefix content.

### 0.4 Async/Concurrency Fixes

**Effort:** 3 days
**Techniques to study:** Python asyncio patterns, race condition detection, thread-safe initialization patterns

**What to fix (audit then fix):**
- Audit all `async def` methods in `evoagentx/models/` for uninitialized instance variables used in concurrent contexts (pattern similar to upstream PR #225)
- Complete TODO async implementations in agent initialization
- Fix tool serialization for async contexts
- Add `asyncio.Lock()` where shared state is mutated concurrently

---

## 4. Phase 1A: Core Algorithm Quality (Weeks 5–12)

**Goal:** Improve the fundamental quality of optimization signals before adding features on top.

### 1A.1 Process Reward Models

**Effort:** 3 weeks
**Techniques to study:** AgentPRM (arXiv:2511.08325) — step-wise "promise" and "progress" signal definitions; ToolRM (arXiv:2510.26167) — function-call quality scoring; Skywork-Reward-V2 architecture for using pre-trained reward models; RLTR (arXiv:2508.19598) process reward integration

**Why this matters:** Current evaluation is outcome-only (final task score). Process reward models evaluate each step. Empirical gap: 78.2% vs 34% on MATH for RAG agents when using process supervision. This would be the single largest quality improvement.

**What to build (from scratch):**
- `evoagentx/evaluators/process_reward.py` — `StepwiseRewardEvaluator` base class
  - `score_tool_call(tool_name, args, context, trajectory_so_far) → float` — was this the right tool at this point?
  - `score_reasoning_step(step_text, context) → float` — is this reasoning sound?
  - `score_trajectory(trajectory) → List[float]` — step scores for a full execution trace
- `evoagentx/evaluators/promise_progress.py` — Implement AgentPRM-style "promise" (estimated future value of current state) and "progress" (advancement made by last action). Two separate signals, both useful for different optimizer components.

**Integration with existing algorithms:**

*TextGrad:* Replace single outcome loss with trajectory-level loss. Each tool call receives a step reward. TextGrad's backward pass propagates through the trajectory sequence, not just from the terminal output. Modify `TextGradOptimizer._compute_gradient()` to accept a `StepwiseRewardEvaluator`.

*EvoPrompt:* Add step reward component to fitness function: `fitness = α × outcome_score + β × mean(step_rewards)`. Modify `EvoPromptOptimizer._evaluate_candidate()`. Start with α=0.7, β=0.3; tune against TAU-bench baseline.

*AFlow:* Use step rewards as intermediate node values in tree search. When MCTS expands a node, use the step reward of the action taken to estimate node quality without waiting for rollout completion. Modify `AFlowOptimizer._mcts_node_value()`.

### 1A.2 MAP-Elites Optimizer

**Effort:** 1 week
**Techniques to study:** MAP-Elites original paper (Mouret & Clune 2015), multi-dimensional archive data structures, Pareto front extraction, diversity metrics for prompt populations

**What to build (from scratch):**
- `evoagentx/optimizers/mapelites_optimizer.py` — `MAPElitesOptimizer(BaseOptimizer)`
  - Feature dimensions: configurable by user (e.g., accuracy dimension, cost dimension, latency dimension)
  - Archive: 2D or 3D grid indexed by discretized feature values, stores best candidate per cell
  - Selection: sample randomly from archive cells (preserves diversity); variation: crossover between two archived solutions
  - `optimize()` loop: evaluate candidate on all feature dimensions, find its cell, replace if better than current cell occupant

**Integration with existing architecture:**
Follows the same `BaseOptimizer` interface as the other five optimizers. Can be used standalone or as a wrapper around EvoPrompt to add diversity. Directly solves Issue #212 (mode collapse) by design — the archive prevents any single cell from dominating.

**Depends on:** Phase 0 evaluation harness (need multi-dimensional scores, not just a single scalar).

### 1A.3 Evolution Constraints (CEAO Layer)

**Effort:** 2 weeks
**Techniques to study:** CEAO (Issue #227) DPO/GRPO-based mutation filtering; DSPy `Assert`/`Suggest` programmatic constraint patterns; alignment constraint formulations from RLHF literature

**What to build (from scratch):**
- `evoagentx/optimizers/constraint_layer.py` — `EvolutionConstraints` wrapper
  - `CostConstraint(budget_per_call, total_budget)` — reject mutations that exceed cost thresholds
  - `HallucinationConstraint(detector_fn, threshold)` — score generated outputs for factual reliability; reject if score < threshold
  - `DriftConstraint(baseline_behavior, semantic_distance_threshold)` — compute semantic similarity between evolved agent behavior and baseline; halt evolution if too far
  - `ConstrainedOptimizer(base_optimizer, constraints)` — wraps any of the five base optimizers, filters candidates through constraint stack before accepting

**Integration:** Wraps any `BaseOptimizer` subclass. `ConstrainedOptimizer(EvoPromptOptimizer(...), constraints=[cost, hallucination, drift])`. Does not modify existing optimizer internals — purely additive.

**Note:** This is a *static* constraint layer — it filters mutations before they're accepted. Alignment drift detection (Phase 2A.4) is a separate *dynamic* monitoring layer that watches deployed behavior over time.

---

## 5. Phase 1B: Memory Foundation (Weeks 5–10, parallel with 1A)

**Goal:** Fix the memory scaling bug, establish the architectural patterns needed for cross-session learning.

### 1B.1 Paged Memory Model (Letta-inspired)

**Effort:** 2 weeks
**Techniques to study:** Letta/MemGPT "LLM as OS" paradigm — fixed in-context page, archive/recall operations; virtual memory paging analogy; retrieval-augmented generation patterns; Zep/Graphiti bi-temporal storage concepts

**Why this matters:** Directly fixes the `rag/rag.py` FIXME ("Computer's memory may increase dramatically"). Without this, any long optimization run will OOM.

**What to build (from scratch):**
- `evoagentx/memory/paged_memory.py` — `PagedMemory`
  - `in_context_page: List[MemoryItem]` — fixed max size (configurable, e.g., 50 items)
  - `archive: VectorStore` — unlimited long-term storage (uses existing storage adapters)
  - `recall(query, k) → List[MemoryItem]` — semantic search into archive, load to in-context page (evicting least-relevant items if page is full)
  - `archive_old(eviction_policy)` — move in-context items to archive based on recency/relevance
  - `memory_op_instructions: str` — LLM-readable instructions for explicit memory management calls

**Integration:** Replace the unbounded KV store growth in `rag/rag.py` with `PagedMemory`. The existing `LongTermMemory` class can delegate to `PagedMemory` as its backend. No interface changes needed for callers.

### 1B.2 Reflexion-based Episodic Memory

**Effort:** 2 weeks
**Techniques to study:** Reflexion (Shinn et al., 2023) verbal self-reflection mechanism; episodic memory storage patterns; retrieval-based in-context learning

**What to build (from scratch):**
- `evoagentx/memory/reflexion_memory.py` — `ReflexionMemory`
  - After each task execution (success or failure), generate a verbal reflection: "What happened, what went wrong, what should be done differently"
  - Store reflection + task context + outcome in `PagedMemory` archive (depends on 1B.1)
  - Before next execution of similar task, retrieve k most relevant past reflections and include in prompt
  - `reflect(task, trajectory, outcome) → ReflectionEntry` — generates and stores the reflection
  - `get_relevant_reflections(task, k) → List[ReflectionEntry]` — retrieves for next attempt

**Integration with existing optimizers:**
Add `ReflexionMemory` as an optional component of all five optimizers. Call `reflect()` at end of each `optimize()` iteration; call `get_relevant_reflections()` at start of each candidate evaluation. The evaluation prompt gains historical context about what didn't work.

### 1B.3 Mistake Notebook

**Effort:** 4 days
**Techniques to study:** Mistake Notebook (arXiv:2512.11485) — lightweight failure pattern extraction and retrieval; categorical error taxonomies

**What to build (from scratch):**
- `evoagentx/memory/mistake_notebook.py` — `MistakeNotebook`
  - Simpler than full Reflexion: records structured error categories (tool call failed, reasoning loop, context overflow, constraint violation, hallucination) with task context
  - `log_mistake(error_type, context, what_was_tried)` — append to notebook
  - `lookup(task_context) → List[MistakeSummary]` — retrieve most relevant past mistakes before attempting similar task
  - Periodic summarization: compress many similar mistakes into a single pattern note

**Integration:** Standalone addition to the optimization loop. Can be active without Reflexion — it is a lighter-weight alternative for early deployment.

---

## 6. Phase 2A: Advanced Optimization (Weeks 13–20)

**Depends on:** Phase 0 complete, Phase 1A process rewards and CEAO layer operational

### 2A.1 GEPA Optimizer

**Effort:** 1 week
**Techniques to study:** DSPy 3.x GEPA design; generalized efficient prompt algorithms; comparison with MIPRO's bootstrap few-shot + instruction optimization approach

**What to build (from scratch):**
- `evoagentx/optimizers/gepa_optimizer.py` — `GEPAOptimizer(BaseOptimizer)`
  - Understand the DSPy 3.x feature/gepa branch conceptually (read, don't copy)
  - Implement the core insight: efficient prompt search by decomposing instruction and example optimization into independent subproblems
  - Benchmark against `MIPROOptimizer` on TAU-bench to validate the "outperforms RL-based" claim before committing to it as primary

**Integration:** Drop-in `BaseOptimizer` subclass. Replace or supplement MIPRO based on benchmark results.

### 2A.2 metaTextGrad

**Effort:** 2 weeks
**Techniques to study:** metaTextGrad (arXiv:2505.18524) — optimizing the optimizer's gradient generation strategy itself; meta-learning concepts; nested optimization loops

**What to build (from scratch):**
- `evoagentx/optimizers/meta_textgrad.py` — `MetaTextGradOptimizer`
  - Outer loop: optimizes the *gradient generation prompt* used by the inner TextGrad optimizer
  - Inner loop: standard TextGrad optimization with current gradient prompt
  - Gradient prompt is itself a TextVariable — it can be evolved
  - `meta_optimize(task_distribution, n_outer_steps, n_inner_steps)` — meta-optimization entry point

**Integration:** Wraps `TextGradOptimizer`. The gradient prompt becomes an optimizable parameter stored in `LongTermMemory`, available across sessions.

**Depends on:** TextGrad working correctly with process reward signals (Phase 1A.1). Only add meta-optimization after the base gradient signals are high-quality.

### 2A.3 ADAS-style Design Archive

**Effort:** 2 weeks
**Techniques to study:** ADAS Meta Agent Search (ICLR 2025) — archive-based agent design discovery; Pareto-optimal solution archiving; bootstrap initialization from archive

**What to build (from scratch):**
- `evoagentx/optimizers/design_archive.py` — `AgentDesignArchive`
  - Store Pareto-optimal workflow + prompt combinations discovered during AFlow/EvoPrompt optimization runs
  - Each entry: workflow structure (as serialized WorkFlowGraph), prompt template, multi-dimensional scores (accuracy, cost, latency), task context
  - `add(design, scores, task_context)` — add if Pareto-dominant on at least one dimension
  - `retrieve_similar(task_context, k) → List[DesignEntry]` — bootstrap new optimization runs from relevant archived designs
  - `get_pareto_front() → List[DesignEntry]` — return current Pareto frontier

**Integration with AFlow:** At start of each AFlow optimization run, retrieve top-k similar designs from archive and use them as the initial MCTS population rather than random initialization. At end of run, add discovered Pareto-optimal designs to archive.

**Differentiation from ADAS:** ADAS archives Python code (full agent implementations). AgentAugi's archive stores prompt templates + workflow graphs — the native representation of its optimization space. Hybrid approach: archive can be used to bootstrap any of the five optimizers.

### 2A.4 Alignment Drift Detection

**Effort:** 1 week
**Techniques to study:** ATP (Alignment Tipping Process, arXiv:2510.04860) — empirical evidence that optimization pressure causes alignment erosion without adversarial prompting; behavioral monitoring patterns

**What to build (from scratch):**
- `evoagentx/safety/drift_monitor.py` — `AlignmentDriftMonitor`
  - Capture behavioral fingerprint of baseline agent: response style, safety constraint adherence rate, cost per task, semantic profile of outputs
  - After each generation of evolution: measure drift from baseline on all dimensions
  - `DriftMetrics(safety_violation_rate, semantic_distance, cost_escalation, constraint_adherence)` — tracked per generation
  - Alert and optionally halt evolution when any metric exceeds threshold
  - Different from CEAO: CEAO filters individual mutations before acceptance; ATP monitor watches aggregate behavioral change across generations

**Integration:** Runs as a monitoring sidecar during any optimizer's evolution loop. Call `monitor.checkpoint(generation_n, agent_sample)` each generation. Does not modify optimizer internals.

---

## 7. Phase 2B: Execution Efficiency (Weeks 13–18, parallel with 2A)

**Depends on:** Phase 0 complete

### 2B.1 Difficulty-Aware Routing (DAAO-inspired)

**Effort:** 2 weeks
**Techniques to study:** DAAO (arXiv:2509.11079) — 11% accuracy improvement + 36% cost reduction via difficulty routing; difficulty classification methods; RouteLLM (arXiv:2406.18665) learned routing

**What to build (from scratch):**
- `evoagentx/workflow/difficulty_router.py` — `DifficultyRouter`
  - `classify_difficulty(task) → DifficultyLevel` — classify task as Easy/Medium/Hard based on: query complexity heuristics, estimated number of tool calls, historical performance on similar tasks (from MistakeNotebook)
  - `route(task, difficulty) → WorkFlowGraph` — select appropriate workflow for difficulty level
  - `WorkflowTier(easy_workflow, medium_workflow, hard_workflow)` — register tiered workflows

**Dual use as cost router:**
The same difficulty signal routes to cheap vs. expensive models. Easy → Claude Haiku / GPT-4o-mini. Hard → Claude Opus / GPT-4. Implement as a second routing layer on top of `DifficultyRouter`: `ModelRouter.route(difficulty) → LLMConfig`.

### 2B.2 Cycle Support in WorkFlowGraph

**Effort:** 2 weeks
**Techniques to study:** LangGraph cyclic workflow patterns; max-iteration cycle bounds; convergence detection in iterative refinement

**What to build (from scratch):**
- Extend `evoagentx/workflow/workflow_graph.py`:
  - Add `allow_cycles: bool` flag to `WorkFlowGraph`
  - Add `max_iterations: int` per cycle-containing subgraph
  - Add `convergence_fn: Optional[Callable]` — user-defined function to detect when iteration has converged (break early)
  - Update topological sort to handle cycles (detect strongly connected components, execute them iteratively)

**This unblocks:** LATS runtime planning (Phase 3.1) which requires cycles for tree search backtracking.

### 2B.3 Parallel Tool Execution (GAP-inspired)

**Effort:** 2 weeks
**Techniques to study:** GAP (arXiv:2510.25320) — task dependency graph learning for parallel tool dispatch; asyncio.gather patterns; dependency analysis

**What to build (from scratch):**
- `evoagentx/workflow/parallel_executor.py` — `ParallelToolExecutor`
  - `DependencyGraph` — directed graph where nodes are tool calls, edges are data dependencies
  - `analyze_dependencies(tool_calls: List[ToolCall]) → DependencyGraph` — static analysis: which tool calls consume outputs of which other calls
  - `execute_parallel(tool_calls, dependency_graph) → Dict[str, Any]` — dispatch independent tool calls concurrently via `asyncio.gather()`, respecting dependency ordering
  - Static declaration path: workflow definitions can declare `depends_on` for each tool call
  - Learned path (Phase 3+ stretch goal): train a small classifier to predict dependencies from tool names and args

**Integration:** Add to `WorkFlowGraph.execute_node()`. Any node with multiple tool calls gets dependency analysis before dispatch. Independent calls run in parallel. Expected 2–3x efficiency improvement for tool-heavy workflows.

### 2B.4 Streaming Execution

**Effort:** 1 week
**Techniques to study:** Python asyncio streaming patterns; SSE (Server-Sent Events) for frontend progress

**What to build (from scratch):**
- `WorkFlowGraph.stream_execute(workflow, task) → AsyncGenerator[ExecutionEvent, None]`
  - Yields `NodeStartEvent`, `ToolCallEvent`, `NodeCompleteEvent`, `WorkflowCompleteEvent`
  - Each event carries: current node name, elapsed time, cumulative cost (from Phase 0.2 tracker), partial result

---

## 8. Phase 3: Advanced Capabilities (Weeks 21–34)

**Depends on:** Phases 1 and 2 substantially complete

### 3.1 LATS Runtime Planning

**Effort:** 3 weeks
**Techniques to study:** LATS (Language Agent Tree Search) — real-time MCTS during task execution rather than offline optimization; UCT node selection; rollout evaluation with LLM

**What to build (from scratch):**
- `evoagentx/agents/lats_agent.py` — `LATSAgent`
  - At each decision point during execution, run a miniature MCTS: expand possible next actions, simulate rollouts, backpropagate values, select best action
  - Uses process reward model (Phase 1A.1) as intermediate node value estimator — no need to wait for full rollout
  - `search_depth` and `n_simulations` configurable; budget-constrained via Phase 0.2 cost tracker
  - Requires cycle support (Phase 2B.2) to allow backtracking

**Depends on:** Cycle support (2B.2) + process reward models (1A.1)

### 3.2 Tool Creation (LATM-inspired ToolSynthesizer)

**Effort:** 3 weeks
**Techniques to study:** LATM — LLMs as Tool Makers (ICLR 2024) — two-phase tool creation: expensive model creates, cheap model executes; tool registration patterns; sandboxed Python execution for safety

**What to build (from scratch):**
- `evoagentx/tools/tool_synthesizer.py` — `ToolSynthesizer`
  - `detect_tool_gap(task, tool_registry) → bool` — check if any existing tool handles this task class adequately
  - `synthesize_tool(task_class_description, examples) → ToolDefinition` — use expensive model to generate Python tool function; include docstring, input schema, output schema
  - `validate_tool(tool_def, test_cases) → ValidationResult` — run tool on test cases in sandbox; check output format, error handling, no side effects
  - `register_tool(tool_def)` — add to `ToolRegistry` if validated; persist across sessions
  - Created tools are subject to the same CEAO constraints as evolved prompts

**Safety note:** Tool synthesis must run in a sandboxed environment. Generated Python code executes only in isolation; no filesystem access, no network calls unless explicitly allowed by configuration.

### 3.3 Cross-Session Learning (ReasoningBank pattern)

**Effort:** 2 weeks
**Techniques to study:** ReasoningBank (arXiv:2509.25140) — persistent repository of successful reasoning traces for retrieval-augmented optimization

**What to build (from scratch):**
- `evoagentx/memory/reasoning_bank.py` — `ReasoningBank`
  - Store successful task trajectories (task description, workflow executed, prompt used, step sequence, outcome) indexed by task embedding
  - `store(task, trajectory, quality_score)` — add if quality above threshold
  - `retrieve_similar(task, k) → List[TrajectoryEntry]` — semantic search, return top-k analogous successful traces
  - Integrated with `PagedMemory` (Phase 1B.1) for bounded in-context retrieval
  - Periodic consolidation: merge similar trajectories into abstract patterns (reduces storage growth)

**Depends on:** PagedMemory (1B.1), Reflexion memory (1B.2)

### 3.4 Multi-Agent Debate Quality Controls

**Effort:** 2 weeks
**Techniques to study:** NeurIPS 2025 convergence detection paper; Liang et al. EMNLP 2024 controlled disagreement; A-HMAD heterogeneous roles; Agent-as-a-Judge (arXiv:2410.10934) — 90% human agreement vs 70% for LLM-as-Judge

**What to build (from scratch):**
- `evoagentx/agents/debate_controller.py` — `DebateController`
  - `ConvergenceDetector` — measure semantic similarity between consecutive debate rounds; stop when similarity exceeds threshold (prevents degeneration-of-thought)
  - `DisagreementRegulator(temperature: float)` — modulate how much critics are prompted to disagree; tunable from 0.0 (consensus) to 1.0 (maximal disagreement); empirically optimal is ~0.3–0.5
  - `HeterogeneousRoles` — predefined critic personas: LogicalReasoningCritic, FactualVerificationCritic, StrategicPlanningCritic, CostEfficiencyCritic; assign one per critic agent in MAR
- `evoagentx/evaluators/agent_judge.py` — `AgentAsJudgeEvaluator`
  - Implements the Agent-as-a-Judge pattern: use a judge agent (with tool access, multi-step reasoning) rather than a single LLM call for evaluation
  - Achieves 90% human agreement; replaces or supplements current `Evaluator` objects

### 3.5 Semantic Caching + FrugalGPT Model Cascading

**Effort:** 2 weeks
**Techniques to study:** FrugalGPT (arXiv:2305.05176) — generation, scoring, early exit cascade; RouteLLM (arXiv:2406.18665) — learned routing; vector similarity thresholds for cache hits

**What to build (from scratch):**
- `evoagentx/core/semantic_cache.py` — `SemanticCache`
  - Embed each LLM query before calling; search cache for queries with cosine similarity > threshold (e.g., 0.95)
  - On hit: return cached response without API call
  - On miss: call LLM, store response + embedding in cache
  - Particularly valuable in EvoPrompt where many candidates differ by only a few tokens
- `evoagentx/core/model_cascade.py` — `FrugalCascade`
  - `CascadePolicy([(cheap_model, confidence_threshold), ..., (expensive_model, 1.0)])`
  - Try cheap model first; if confidence ≥ threshold, return; otherwise escalate to next model
  - Confidence estimation: model's own log-probability, or separate calibration model
  - Use `DifficultyRouter` score (Phase 2B.1) as initial routing signal to skip cascade for clearly hard tasks

---

## 9. Phase 4: Ecosystem and Scale (Weeks 35+)

**Depends on:** Phase 3 substantially complete and stable

### 4.1 MCP Protocol Support

**Effort:** 2 weeks
**Techniques to study:** MCP (Model Context Protocol) specification; tool registration and discovery patterns

**What to build:**
- `evoagentx/protocols/mcp_server.py` — expose AgentAugi's ToolRegistry as an MCP server
- `evoagentx/protocols/mcp_client.py` — consume external MCP tool servers; auto-register discovered tools into ToolRegistry

### 4.2 A2A Protocol Support

**Effort:** 2 weeks
**Techniques to study:** A2A (Agent-to-Agent) protocol specification; inter-agent message passing patterns

**What to build:**
- `evoagentx/protocols/a2a.py` — standardized message format for agent-to-agent communication; enables AgentAugi instances to collaborate with external agents in heterogeneous systems

### 4.3 Federated Optimization

**Effort:** 3 weeks
**Techniques to study:** FedTextGrad (arXiv:2502.19980); pFedDC (arXiv:2506.21144); differential privacy for gradient aggregation

**What to build:**
- `evoagentx/optimizers/federated_optimizer.py` — coordinate TextGrad/EvoPrompt optimization across multiple AgentAugi instances without sharing raw task data; aggregate gradient updates with privacy guarantees

### 4.4 Complete HITL Implementation

**Effort:** 2 weeks
**What to complete:**
- Tool call review UI: present tool calls to human reviewer before execution; human approves/rejects/edits
- Multi-turn conversation in optimization loop: human can steer evolution direction between generations
- Review audit log: record all human interventions for accountability

---

## 10. Consolidated Feature Matrix

| Feature | Phase | Effort | Depends On | Existing Files Affected | Key Reference |
|---------|-------|--------|-----------|------------------------|---------------|
| Evaluation harness (TAU-bench, SWE-bench, GAIA, AgentBench) | 0 | 2 wks | — | `benchmark/` | TAU-bench pass^k, AgentBench arXiv:2308.03688 |
| Cost tracking layer | 0 | 1 wk | — | `models/base_llm.py` | Provider pricing APIs |
| Prompt caching wrapper | 0 | 3 days | — | `models/base_llm.py` | Anthropic cache_control |
| Async/concurrency fixes | 0 | 3 days | — | `models/`, `agents/` | asyncio patterns |
| Process reward models | 1A | 3 wks | Ph0 | `evaluators/` | AgentPRM 2511.08325, ToolRM 2510.26167 |
| MAP-Elites optimizer | 1A | 1 wk | Ph0 | `optimizers/` (new) | Mouret & Clune 2015 |
| CEAO evolution constraints | 1A | 2 wks | Ph0 | `optimizers/` (new) | DPO/GRPO, DSPy Assert |
| Paged memory (fixes rag.py FIXME) | 1B | 2 wks | Ph0 | `memory/`, `rag/rag.py` | Letta/MemGPT architecture |
| Reflexion episodic memory | 1B | 2 wks | 1B.1 | `memory/` | Reflexion (Shinn 2023) |
| Mistake Notebook | 1B | 4 days | — | `memory/` | arXiv:2512.11485 |
| GEPA optimizer | 2A | 1 wk | Ph1A | `optimizers/` (new) | DSPy 3.x GEPA |
| metaTextGrad | 2A | 2 wks | 1A.1 | `optimizers/` | arXiv:2505.18524 |
| ADAS design archive | 2A | 2 wks | 1A.2 | `optimizers/` | ADAS ICLR 2025 |
| Alignment drift detection | 2A | 1 wk | 1A.3 | `safety/` (new) | ATP arXiv:2510.04860 |
| Difficulty-aware routing | 2B | 2 wks | Ph0 | `workflow/` | DAAO arXiv:2509.11079 |
| Cycle support in WorkFlowGraph | 2B | 2 wks | — | `workflow/workflow_graph.py` | LangGraph patterns |
| Parallel tool execution | 2B | 2 wks | Ph0 | `workflow/` | GAP arXiv:2510.25320 |
| Streaming execution | 2B | 1 wk | — | `workflow/workflow_graph.py` | asyncio AsyncGenerator |
| LATS runtime planning | 3 | 3 wks | 2B.2, 1A.1 | `agents/` | LATS paper |
| ToolSynthesizer (LATM) | 3 | 3 wks | Ph1, Ph0 | `tools/` | LATM ICLR 2024 |
| ReasoningBank cross-session | 3 | 2 wks | 1B.1, 1B.2 | `memory/` | arXiv:2509.25140 |
| Debate quality controls | 3 | 2 wks | Ph0 | `agents/` | NeurIPS 2025, Liang EMNLP 2024 |
| Semantic cache + FrugalGPT | 3 | 2 wks | 2B.1 | `core/` | FrugalGPT arXiv:2305.05176 |
| MCP protocol support | 4 | 2 wks | Ph3 | `protocols/` (new) | MCP specification |
| A2A protocol support | 4 | 2 wks | Ph3 | `protocols/` (new) | A2A specification |
| Federated optimization | 4 | 3 wks | Ph3 | `optimizers/` | FedTextGrad arXiv:2502.19980 |
| Complete HITL | 4 | 2 wks | Ph3 | `hitl/` | Existing TODOs |

---

## 11. Success Metrics

Every phase must improve measurable numbers, not just add features.

| Metric | Baseline (Phase 0) | Phase 1 Target | Phase 2 Target | Phase 3 Target |
|--------|-------------------|----------------|----------------|----------------|
| TAU-bench pass@1 | TBD | +10% | +20% | +30% |
| TAU-bench pass@8 | TBD | +15% | +25% | +35% |
| SWE-bench resolve rate | TBD | +5% | +15% | +25% |
| Cost per optimization run | TBD | −50% (caching) | −70% (routing) | −80% (cascade) |
| Evaluation throughput | TBD | 2x (parallel tools) | 4x | 6x |
| Memory per long run | Unbounded | Bounded (paged) | Bounded | Bounded |
| Alignment drift rate | Unmeasured | Measured | Detected | Controlled |

*All targets are relative to Phase 0 baseline. The specific numbers will be filled in after Phase 0 establishes actual baseline values.*

---

## 12. Implementation Principles

1. **Measure before building.** No phase starts until Phase 0 baselines exist. Claiming improvement without measurement is not improvement.

2. **Additive by default.** New components wrap or extend existing ones; they do not replace them until benchmarks confirm superiority. GEPA does not replace MIPRO until benchmarks confirm it.

3. **One optimization signal source.** Do not add multiple competing reward signals without a clear composition strategy. Define α/β weighting (outcome vs. process) in Phase 1A and use it consistently.

4. **Safety gates.** CEAO constraints (Phase 1A.3) and alignment drift monitoring (Phase 2A.4) must be active before any Phase 3 capability that generates new tools or modifies agent behavior at runtime.

5. **Cost accountability.** The cost tracking layer (Phase 0.2) must be active for every LLM call in every phase. No optimization run proceeds without a cost budget.

6. **All code written from scratch.** Papers and repositories are read for conceptual understanding. No code is copied, translated, or adapted from external sources. Every file is an original implementation within AgentAugi's patterns.

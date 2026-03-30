# AgentAugi Improvement Research Report

*Prepared: March 29, 2026 | Pre-OpenClaw Integration Analysis*

---

## Executive Summary

This report synthesizes research across four dimensions—upstream EvoAgentX activity, complementary open-source projects, recent academic papers, and missing-feature analysis—to produce a prioritized roadmap for strengthening AgentAugi before its integration into OpenClaw.

AgentAugi currently implements five optimization algorithms (TextGrad, AFlow, SEW, MIPRO, EvoPrompt) for self-evolving agents with prompt optimization. The research identifies **three tiers of improvements** that could yield an estimated 15–25% accuracy improvement and 30x evaluation efficiency gain:

- **Tier 1 (Critical, 10–14 weeks):** Merge MAP-Elites from upstream, integrate GEPA optimizer from DSPy 3.x, add Reflexion-based episodic memory, implement constraint-based evolution (CEAO)
- **Tier 2 (High, 8–10 weeks):** Add difficulty-aware workflow routing, cost-aware MCTS planning, metaTextGrad meta-optimization, multi-agent reflection loops
- **Tier 3 (Strategic, ongoing):** Federated optimization, lifelong learning, MCP/A2A protocol support, multimodal prompt evolution

**Audit addendum (March 29, 2026):** A post-completion audit identified five critical gaps not addressed in the original report. These are documented in Section 6 and must be incorporated into any implementation plan: (1) no evaluation benchmarks defined — the 15–25% claim is currently unmeasurable; (2) no process reward models — an entire optimization signal category is missing; (3) ADAS (ICLR 2025) is the closest published competitor and is not mentioned; (4) Letta/MemGPT, the leading production memory system, is absent despite a known memory scaling bug; (5) cost optimization infrastructure goes beyond tracking to 80%+ cost reduction via caching and model routing. See Section 6 and MASTER_IMPLEMENTATION_PLAN.md for the integrated response to these gaps.

---

## Table of Contents

1. [EvoAgentX Upstream Activity](#1-evoagentx-upstream-activity)
2. [Complementary Projects to Merge](#2-complementary-projects-to-merge)
3. [Recent Papers from arXiv](#3-recent-papers-from-arxiv)
4. [Missing Features Analysis](#4-missing-features-analysis)
5. [Prioritized Implementation Roadmap](#5-prioritized-implementation-roadmap)
6. [Audit Findings: Critical Gaps](#6-audit-findings-critical-gaps) *(addendum)*
7. [Appendix: Full Paper Catalog](#appendix-full-paper-catalog)

---

## 1. EvoAgentX Upstream Activity

### 1.1 Repository Health Overview

EvoAgentX (github.com/EvoAgentX/EvoAgentX) remains actively developed: 1,050+ commits, 2.7k stars, 227 forks. Development velocity averages ~8 commits/month with a focus on toolkit expansions and bug fixes rather than core algorithm changes. The last release (v0.1.0) was September 2025—no tagged releases since, suggesting rapid iteration without version discipline.

**What this means for AgentAugi:** The core five algorithms are stable upstream. Changes are additive (new tools, new integrations) rather than breaking. A periodic merge strategy (monthly cherry-picks) is safer than tracking HEAD.

### 1.2 Critical Items to Merge Back

#### MAP-Elites Optimizer (PR #222) — MERGE PRIORITY: CRITICAL

A new sixth optimizer has been submitted upstream by contributor `sqsge`. MAP-Elites (Multi-dimensional Archive of Phenotypic Elites) maintains a diverse archive of high-performing solutions across user-defined feature dimensions rather than converging on a single optimum. This directly addresses a known weakness in EvoPrompt—premature convergence and mode collapse (see Issue #212 below).

**Why it matters for AgentAugi:** AgentAugi's EvoPrompt uses GA/DE evolutionary strategies that optimize a single fitness metric. MAP-Elites enables multi-objective optimization—for example, simultaneously optimizing for accuracy, latency, and cost. For OpenClaw, this means agents can be evolved along multiple quality dimensions simultaneously, producing a Pareto front of solutions rather than a single "best" prompt.

**Integration path:** The optimizer follows the standard `Optimizer` base class pattern (`optimize()`, `step()`, `evaluate()`, `convergence_check()`). Drop-in addition to `evoagentx/optimizers/`. Estimated effort: 1–2 days.

#### GEPA Optimizer Branch (feature/gepa) — MERGE PRIORITY: HIGH

An unmerged feature branch contains a GEPA (Generalized Efficient Prompt Algorithm) optimizer derived from DSPy that reportedly outperforms reinforcement-learning-based approaches (Issue #219). The branch is 25 commits behind main with no documented reason for exclusion.

**Why it matters for AgentAugi:** If GEPA genuinely outperforms RL-based optimization, it could replace or supplement MIPRO as the primary DSPy-based optimizer. Community members have questioned the omission without receiving maintainer responses, suggesting bandwidth constraints rather than technical objections.

**Integration path:** Cherry-pick the GEPA optimizer files, rebase onto current main, benchmark against MIPRO on AgentAugi's target tasks. Estimated effort: 3–5 days including benchmarking.

#### SiliconFlowLLM Concurrency Bug Fixes (PR #225, Issues #223–224) — MERGE PRIORITY: HIGH

Race conditions and uninitialized `self.response` bugs in concurrent async calls affect any optimizer that parallelizes evaluations—particularly MIPRO (which uses multi-threaded evaluation) and any future MAP-Elites implementation.

**Why it matters for AgentAugi:** OpenClaw will likely run optimization loops at scale. Concurrent LLM calls are essential for performance. These bugs cause silent failures in parallel evaluation scenarios.

**Integration path:** Direct merge of PR #225 fixes. Also audit AgentAugi's own LLM wrapper implementations for similar patterns. Estimated effort: 1 day.

#### Dependency Management Refactor (PR #221) — MERGE PRIORITY: MEDIUM

Splits monolithic dependencies into optional extras and sanitizes environment configuration. Important for OpenClaw integration where AgentAugi will be one component among many.

**Integration path:** Adopt the modular dependency structure. Estimated effort: 1 day.

### 1.3 Upstream Issues That Affect AgentAugi

#### Constrained Evolutionary Agent Optimization / CEAO (Issue #227)

Contributor `rawathemant246` identified a fundamental problem: unconstrained agent evolution leads to behavioral drift, hallucination, and cost blowup. The proposed solution uses DPO/GRPO (Direct Preference Optimization / Generalized Reward-based Process Optimization) to train mutation policies that generate improved agent variants while controlling cost escalation, hallucination rates, and behavioral drift.

**Why it matters for AgentAugi:** This is the single most important open research direction for production deployment. EvoPrompt and SEW can evolve agents that score well on benchmarks but behave unpredictably in deployment. CEAO adds the guardrails needed for OpenClaw's production use cases.

**Recommended action:** Implement a constraint layer on top of EvoPrompt and SEW that filters mutations based on cost budgets, hallucination detection, and semantic drift thresholds. Start with simple heuristic constraints, evolve toward learned DPO-based filtering.

#### Mode Collapse in Evolution (Issue #212)

Users report cyclic/repetitive output generation—a symptom of premature convergence in evolutionary algorithms. EvoPrompt's GA/DE operators may lack sufficient diversity pressure.

**Recommended action:** Add diversity metrics (semantic similarity between population members), novelty bonuses in fitness evaluation, and the MAP-Elites archive as a diversity-preserving alternative selection mechanism.

#### Incomplete HITL Implementation (TODOs in hitl/)

Human-in-the-loop features have placeholder TODOs for review/edit, tool call review, and multi-turn conversation. These are important for supervised agent evolution in OpenClaw.

**Recommended action:** Complete HITL implementation before OpenClaw integration. Priority: tool call review (safety), then multi-turn conversation (usability).

### 1.4 Upstream Activity Summary Table

| Item | Type | Priority | Effort | Impact |
|------|------|----------|--------|--------|
| MAP-Elites optimizer (PR #222) | New algorithm | Critical | 1–2 days | Multi-objective optimization |
| GEPA optimizer (feature/gepa) | New algorithm | High | 3–5 days | Potentially superior to MIPRO |
| Concurrency bug fixes (PR #225) | Bug fix | High | 1 day | Parallel evaluation reliability |
| CEAO constraints (Issue #227) | Research direction | High | 2–4 weeks | Production safety |
| Dependency refactor (PR #221) | Infrastructure | Medium | 1 day | Cleaner OpenClaw integration |
| Mode collapse fix (Issue #212) | Bug/enhancement | Medium | 1 week | Evolution diversity |
| HITL completion | Feature gap | Medium | 2–3 weeks | Supervised evolution |

---

## 2. Complementary Projects to Merge

### 2.1 DSPy 3.x / GEPA — PRIORITY: CRITICAL

**Project:** stanfordnlp/dspy | **Stars:** 22k+ | **Status:** Active, DSPy 3.x released

AgentAugi already uses MIPRO from DSPy. However, DSPy 3.x introduces GEPA (accepted as ICLR 2026 Oral), which achieves 10%+ accuracy improvement with 35x evaluation efficiency over MIPRO. GEPA uses a generalized efficient prompt algorithm that learns to propose better prompt candidates with far fewer evaluations.

**Why it matters:** MIPRO's Bayesian optimization (via Optuna) requires many evaluation rounds to converge. GEPA's efficiency gain means AgentAugi could optimize prompts in minutes instead of hours for complex workflows, dramatically reducing the cost of the evolution loop.

**How it integrates:** AgentAugi already has `MiproLMWrapper` and `MiproRegistry` for DSPy compatibility. GEPA would follow the same pattern—wrap BaseLLM in a DSPy-compatible interface, implement a `GepaOptimizer` subclass of `Optimizer`. The existing `PromptTuningModule` architecture supports this directly.

**Additional DSPy features to consider:**
- `dspy.Assert` / `dspy.Suggest` — programmatic constraints on LLM outputs, directly useful for CEAO-style constraint enforcement
- `dspy.ChainOfThought` improvements — better structured reasoning for workflow planning agents
- DSPy's caching and tracing infrastructure — reduces redundant API calls during optimization

**Estimated effort:** 4–6 weeks for full GEPA integration and benchmarking. Risk: Low.

### 2.2 Reflexion — PRIORITY: CRITICAL

**Project:** noahshinn/reflexion | **Status:** NeurIPS 2023, production-proven

Reflexion enables agents to learn from failures through verbal self-reflection rather than weight updates. After each attempt, the agent generates a natural-language reflection on what went wrong and stores it in an episodic memory buffer. Future attempts retrieve relevant reflections to avoid repeating mistakes.

**Why it matters for AgentAugi:** TextGrad provides gradient-like feedback for prompt improvement, but it operates at the prompt level. Reflexion operates at the execution trajectory level—it learns from the agent's actual behavior across multiple attempts. This is a fundamentally different and complementary learning signal. Combined with TextGrad, AgentAugi would have both prompt-level optimization (TextGrad) and behavior-level learning (Reflexion).

**How it integrates:** AgentAugi already has `ShortTermMemory` and `LongTermMemory` systems. Reflexion's episodic memory maps directly onto `LongTermMemory` with a reflection-generation step added after each workflow execution. The reflection agent would be a specialized `Agent` subclass that generates verbal feedback, stored via the existing `MemoryManager`.

**Key integration points:**
- After `WorkFlow.async_execute()` completes, trigger reflection agent
- Store reflections in `LongTermMemory` with task-type tags
- Before next execution, retrieve relevant reflections and inject into agent context
- Use reflection quality as an additional signal for TextGrad optimization

**Estimated effort:** 2–4 weeks. Risk: Very low (well-understood technique).

### 2.3 LATS (Language Agent Tree Search) — PRIORITY: HIGH

**Project:** andy-zhou/lats & lapisrocks/LanguageAgentTreeSearch | **Status:** Published ICML 2024

LATS combines Monte Carlo Tree Search with LLM-based value functions and self-reflection for complex reasoning and decision-making. It treats the LLM as both the world model (for state transitions) and the value function (for evaluating states), using MCTS to systematically explore action sequences.

**Why it matters for AgentAugi:** AgentAugi's AFlow already uses MCTS for workflow optimization, but LATS extends tree search to runtime agent planning—not just offline optimization. This means agents could use tree search during execution to plan multi-step actions, not just during the evolution loop. The combination would give AgentAugi both offline evolution (AFlow's MCTS) and online planning (LATS's MCTS).

**How it integrates:** LATS would be implemented as a new `ActionGraph` variant (e.g., `LATSActionGraph`) that uses tree search during `execute()`. It plugs into the existing `WorkFlowNode` → `ActionGraph` → `execute()` pipeline. The MCTS infrastructure from AFlow (`GraphUtils`, `ExperienceUtils`) could be shared.

**Key distinction from AFlow:** AFlow optimizes the workflow graph structure offline. LATS optimizes action sequences at runtime. They operate at different timescales and are fully complementary.

**Estimated effort:** 3–4 weeks. Risk: Medium (requires careful integration with async execution).

### 2.4 LangGraph — PRIORITY: HIGH (patterns to borrow)

**Project:** langchain-ai/langgraph | **Stars:** 10k+ | **Status:** Very active, v0.3+

LangGraph provides a graph-based runtime for agent workflows with first-class support for cycles, branching, human-in-the-loop, persistence, and streaming. While AgentAugi's `WorkFlowGraph` is a DAG (directed acyclic graph), LangGraph supports arbitrary cycles—enabling iterative refinement loops, retry logic, and self-correction patterns natively.

**Why it matters for AgentAugi:** AgentAugi's `WorkFlowGraph` currently enforces DAG structure (`_validate_workflow_structure()` checks for cycles). This prevents implementing iterative refinement patterns (try → evaluate → retry) as first-class workflow constructs. LangGraph's approach shows these patterns are essential for production agent systems.

**What to borrow (not full integration):**
- **Cycle support in WorkFlowGraph:** Allow controlled cycles with max-iteration bounds for retry/refinement patterns
- **Checkpointing/persistence:** LangGraph's `StateGraph` maintains persistent state across interruptions. AgentAugi's `Environment` could adopt this pattern for long-running optimization loops
- **Streaming execution:** LangGraph streams intermediate results. AgentAugi should expose optimization progress as a stream for OpenClaw's UI
- **Conditional edges:** LangGraph's conditional routing based on state is more flexible than AgentAugi's fixed edges

**Estimated effort:** 3–4 weeks for cycle support + streaming. Risk: Medium (DAG-to-graph transition requires careful validation).

### 2.5 CrewAI — PRIORITY: MEDIUM (patterns to borrow)

**Project:** crewAIInc/crewAI | **Stars:** 28k+ | **Status:** Very active

CrewAI's unique contribution is role-based agent orchestration with explicit delegation, memory sharing, and hierarchical/sequential process models. Agents have defined roles, goals, and backstories that shape their behavior.

**What to borrow for AgentAugi:**
- **Role-based agent specialization:** CrewAI's `Agent(role="Senior Researcher", goal="...", backstory="...")` pattern produces more focused agent behavior than generic agents. AgentAugi's `Agent` class could add `role` and `goal` fields that inform prompt construction
- **Delegation patterns:** Agents can delegate subtasks to other agents mid-execution. AgentAugi's `WorkFlowManager` could support dynamic delegation
- **Process models:** CrewAI's hierarchical process (manager agent coordinates workers) could be an alternative to AgentAugi's DAG-based scheduling

**Estimated effort:** 2 weeks for role fields + delegation. Risk: Low.

### 2.6 AutoGen — PRIORITY: MEDIUM (patterns to borrow)

**Project:** microsoft/autogen | **Stars:** 45k+ | **Status:** AutoGen v0.4+ (AgentChat)

AutoGen's conversation-based multi-agent pattern is fundamentally different from AgentAugi's workflow-graph approach. Agents communicate through message passing in group chats rather than through structured data flow in graphs.

**What to borrow for AgentAugi:**
- **Conversational optimization:** Allow optimization agents (TextGrad critic, EvoPrompt evaluator) to have multi-turn conversations rather than single-shot evaluations. This could improve gradient quality in TextGrad
- **Group chat patterns:** For multi-agent workflows, allow agents to discuss and negotiate rather than just pass outputs forward
- **Code execution sandboxing:** AutoGen's Docker-based code execution is more robust than direct Python execution

**Estimated effort:** 2–3 weeks for conversation-based evaluation. Risk: Medium.

### 2.7 Additional Discovered Projects

#### AgentEvolver (Alibaba/ModelScope) — PRIORITY: HIGH

Auto-generates training curricula for agents by progressively creating harder evaluation examples. Achieves 8%+ accuracy improvement through curriculum-based evolution. This addresses a key weakness in AgentAugi: the quality of the benchmark data used to drive optimization. Better training data → better evolved agents.

**Integration:** Add a `CurriculumGenerator` that creates progressively harder evaluation examples for the `Benchmark` class. Use during optimization loops to prevent overfitting to static benchmarks.

#### LoongFlow — PRIORITY: MEDIUM

Extends AFlow-style workflow optimization specifically for long-context multi-step reasoning. Proposes specialized node types for context management, chunked processing, and progressive summarization.

**Integration:** Add long-context node types to `WorkFlowGraph`. Useful for OpenClaw tasks involving large documents or codebases.

#### Trace (Microsoft Research) — PRIORITY: MEDIUM

An AutoDiff-like framework for optimizing arbitrary Python code with LLM-based feedback. More general than TextGrad—can optimize code structure, not just prompts.

**Integration:** Could replace or supplement TextGrad for optimizing `ActionGraph` code, not just prompt strings.

### 2.8 Complementary Projects Summary

| Project | Priority | What to Take | Effort | Expected Impact |
|---------|----------|-------------|--------|-----------------|
| DSPy 3.x / GEPA | Critical | GEPA optimizer, Assert/Suggest | 4–6 weeks | 10%+ accuracy, 35x efficiency |
| Reflexion | Critical | Episodic memory, verbal reflection | 2–4 weeks | 5%+ on multi-turn tasks |
| LATS | High | Runtime tree-search planning | 3–4 weeks | Better complex task execution |
| LangGraph | High | Cycles, streaming, checkpoints | 3–4 weeks | Iterative refinement patterns |
| AgentEvolver | High | Curriculum generation | 4–6 weeks | 8%+ via better training data |
| CrewAI | Medium | Roles, delegation | 2 weeks | More focused agent behavior |
| AutoGen | Medium | Conversational evaluation | 2–3 weeks | Better gradient quality |
| LoongFlow | Medium | Long-context nodes | 2–3 weeks | Large document handling |
| Trace | Medium | Code-level optimization | 3–4 weeks | Beyond prompt optimization |

---

## 3. Recent Papers from arXiv

### 3.1 Papers Directly Extending AgentAugi's Core Algorithms

#### metaTextGrad: Automatically Optimizing Language Model Optimizers
*arXiv:2505.18524, May 2025*

Meta-optimizer that uses LLMs to improve the TextGrad optimization process itself. Rather than using a fixed strategy for generating textual gradients, metaTextGrad learns to produce better gradients based on past optimization trajectories.

**Relevance:** This is a second-order optimization loop—optimizing the optimizer. AgentAugi's TextGrad uses a fixed critic prompt to generate gradients. metaTextGrad would make the critic's strategy itself evolve, compounding improvement over time.

**Actionable:** Add a meta-optimization wrapper around `TextGradEngine` that periodically updates the gradient-generation prompt based on which gradients led to the largest score improvements. Estimated 2 weeks on top of existing TextGrad implementation.

#### DAAO: Difficulty-Aware Agentic Orchestration
*arXiv:2509.11079, September 2025*

Dynamically generates query-specific multi-agent workflows based on predicted query difficulty. Routes simple queries through lightweight pipelines and complex queries through full multi-agent systems. Achieves 11.21% accuracy improvement with 36% cost reduction.

**Relevance:** AgentAugi's `WorkFlowGenerator` creates a single static workflow for a goal. DAAO shows that adaptive workflow selection at inference time is dramatically more efficient. Simple questions don't need a 5-agent pipeline.

**Actionable:** Add a difficulty classifier before `WorkFlow.async_execute()` that routes inputs to pre-optimized workflow variants of different complexity. This is the highest-ROI change for production deployment—most inputs are easy and don't need the full pipeline.

#### BayesFlow: Probability Inference for Workflow Generation
*arXiv:2601.22305, January 2026*

Casts workflow generation as Bayesian inference over a posterior distribution on workflows. Uses parallel look-ahead rollouts and sequential refinement, providing uncertainty estimates on workflow design decisions.

**Relevance:** AFlow uses MCTS for workflow search, which is deterministic in its exploration strategy. BayesFlow provides principled uncertainty quantification—AgentAugi could know *how confident* it is in a generated workflow, enabling better decisions about when to deploy vs. when to continue optimizing.

**Actionable:** Implement as an alternative workflow generation strategy alongside AFlow. Medium priority—most valuable when optimization budget is limited and you need to decide when to stop.

### 3.2 Papers on Self-Evolving Agent Architectures

#### Multi-Agent Evolve (MAE): LLM Self-Improve through Co-evolution
*arXiv:2510.23595, October 2025*

Three-role framework (Proposer, Solver, Judge) where agents co-evolve by providing feedback to each other through reinforcement learning. The key insight: rather than using an external evaluator, agents evaluate each other, creating an internal feedback loop.

**Relevance:** AgentAugi uses external `Evaluator` objects to score optimization candidates. MAE shows that agents within the system can serve as evaluators, reducing dependence on hand-crafted evaluation metrics. This is especially valuable for tasks where defining a good metric is hard.

**Actionable:** Add a `JudgeAgent` that evaluates workflow outputs alongside or instead of the `Evaluator`. Use Judge feedback as the fitness signal for EvoPrompt and the gradient signal for TextGrad. This creates a fully self-contained evolution loop.

#### EvolveR: Self-Evolving Agents through Experience-Driven Lifecycle
*arXiv:2510.16079, October 2025*

Closed-loop framework where agents collect experiences, reflect on them, and adapt behavior iteratively. The key innovation is the lifecycle: Execute → Collect Experience → Reflect → Adapt → Execute (improved).

**Relevance:** AgentAugi's optimization loop is: Execute → Evaluate (score) → Optimize (mutate) → Execute (improved). EvolveR adds the reflection step between evaluation and optimization. This produces richer learning signals than scores alone.

**Actionable:** Insert a reflection phase into the optimization loop. After `evaluate()` and before `step()`, run a reflection agent that analyzes *why* certain variants scored high or low. Feed reflections into the optimizer as additional context. Estimated 2 weeks.

#### SE-Agent: Self-Evolution Trajectory Optimization
*arXiv:2508.02085, August 2025*

Optimizes agent reasoning trajectories (the sequence of actions and thoughts) rather than just prompts. Uses cross-trajectory inspiration to enhance performance.

**Relevance:** AgentAugi optimizes prompts (TextGrad, MIPRO, EvoPrompt) and workflow structure (AFlow, SEW), but not execution trajectories. SE-Agent shows that the *path* an agent takes through a workflow matters as much as the workflow structure.

**Actionable:** Add trajectory recording to `WorkFlow.async_execute()` and a trajectory optimizer that identifies successful action patterns. Feed trajectory insights back into agent prompts. Medium priority.

### 3.3 Papers on Memory and Learning

#### Agentic Memory (AgeMem): Learning Unified Memory Management
*arXiv:2601.01885, January 2026*

Trains store/retrieve/update/summarize/discard memory operations as callable tools via RL. Discovers non-obvious strategies like preemptive summarization before context overflow.

**Relevance:** AgentAugi's `LongTermMemory` uses fixed policies for memory management. AgeMem shows that learned memory policies significantly outperform hand-crafted ones. For long-running optimization loops, efficient memory management is critical.

**Actionable:** Make memory operations (what to store, when to retrieve, when to forget) optimizable parameters within the evolution loop. Allow TextGrad or EvoPrompt to optimize memory management prompts alongside task prompts.

#### Mistake Notebook Learning
*arXiv:2512.11485, December 2025*

Agents self-curate generalizable guidance from failure patterns. Clusters mistakes into "mistake notes," updates only when performance improves. No retraining needed.

**Relevance:** This is a lightweight, training-free complement to Reflexion. Rather than full verbal reflections, agents maintain a concise error pattern library. This is more efficient for long-running optimization and directly usable as additional context for TextGrad.

**Actionable:** Add a `MistakeNotebook` component to AgentAugi's memory system. After failed evaluations, extract error patterns and cluster them. Inject relevant mistake notes into agent prompts before execution. High priority—low effort, high impact.

#### ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory
*arXiv:2509.25140, September 2025*

Agents maintain a searchable repository of past reasoning traces. At inference time, they retrieve and adapt relevant past approaches. Learning is continuous across interactions.

**Relevance:** Combines persistent memory with inference-time reasoning. AgentAugi's optimization currently produces fixed artifacts (optimized prompts/workflows). ReasoningBank enables continuous improvement even after optimization is "done."

**Actionable:** Store successful reasoning traces from optimization runs. At inference time, retrieve relevant traces and inject as few-shot examples. This gives post-optimization performance gains. Medium priority.

### 3.4 Papers on Cost and Efficiency

#### Cost-Augmented MCTS for LLM-Assisted Planning
*arXiv:2505.14656, May 2025*

Extends MCTS with explicit action costs, balancing task completion against budget constraints. Essential for real-world deployments where API calls cost money.

**Relevance:** AFlow's MCTS optimizes for task performance without considering the cost of the workflow it generates. A workflow that achieves 95% accuracy but costs $10 per invocation may be worse than one at 90% accuracy for $0.50. For OpenClaw, cost-performance tradeoffs are critical.

**Actionable:** Add cost tracking to `WorkFlowGraph` nodes. Modify AFlow's reward function to include a cost penalty. Implement Pareto-optimal discovery across accuracy-cost dimensions. High priority for production deployment.

#### Benchmark Test-Time Scaling of General LLM Agents
*arXiv:2602.18998, February 2026*

Two test-time scaling strategies: sequential (extend reasoning) vs. parallel (sample multiple trajectories). Quantifies when each is more efficient.

**Relevance:** AgentAugi could use test-time compute adaptively—spend more compute on harder tasks. This complements DAAO's difficulty-aware routing.

**Actionable:** Add adaptive test-time scaling as a configurable strategy in `WorkFlow`. For hard tasks, run multiple parallel trajectory samples and select the best. Medium priority.

### 3.5 Papers on Reflection and Introspection

#### Multi-Agent Reflexion (MAR)
*arXiv:2512.20845, December 2025*

Multi-agent extension of Reflexion with diverse reasoning personas and a judge model. Separates acting, diagnosing, critiquing, and aggregating into distinct agent roles.

**Relevance:** More sophisticated than single-agent Reflexion. Multiple critic agents with different perspectives produce richer feedback. This feeds directly into TextGrad's gradient quality—better critiques → better gradients.

**Actionable:** Implement as an enhanced reflection stage in the optimization loop. Use multiple critic agents with different evaluation perspectives (correctness, efficiency, clarity, safety). Aggregate their feedback before passing to TextGrad.

#### Agentic Context Engineering (ACE)
*arXiv:2510.04618, October 2025*

Treats agent contexts as evolving "playbooks" that accumulate, refine, and organize strategies through generation, reflection, and curation. Outperforms baselines by +10.6% on agent tasks.

**Relevance:** Goes beyond prompt optimization to optimizing the entire context window—instructions, examples, strategies, constraints. AgentAugi optimizes prompts but not the broader context provided to agents.

**Actionable:** Add a context curation layer that manages libraries of useful examples, strategies, and heuristics. Use SEW-style evolution to optimize context compositions. High priority.

### 3.6 Papers on Tool Integration

#### The Evolution of Tool Use in LLM Agents
*arXiv:2603.22862, March 2026*

Comprehensive review covering the shift toward standardized protocols (Model Context Protocol, Agent2Agent). Discusses inference-time planning for tool selection.

**Relevance:** AgentAugi has 40+ tools but lacks standardized protocol support. For OpenClaw integration, MCP and A2A compatibility would enable AgentAugi to work with any MCP-compatible tool ecosystem.

**Actionable:** Add MCP server/client support to AgentAugi's toolkit system. This makes every MCP-compatible tool instantly available. High priority for OpenClaw.

### 3.7 Key Papers Summary Table

| Paper | Year | Relevance | Component Affected | Priority |
|-------|------|-----------|-------------------|----------|
| metaTextGrad | 2025 | Meta-optimize gradient generation | TextGrad | High |
| DAAO (difficulty-aware routing) | 2025 | Adaptive workflow selection | WorkFlowGenerator | High |
| Multi-Agent Evolve (MAE) | 2025 | Internal feedback loops | Evaluator, Optimizer | High |
| EvolveR | 2025 | Experience-driven lifecycle | Optimizer loop | High |
| AgeMem (learned memory) | 2026 | Optimizable memory management | LongTermMemory | High |
| Mistake Notebook | 2025 | Failure pattern learning | Memory, TextGrad | High |
| Cost-Augmented MCTS | 2025 | Cost-aware optimization | AFlow | High |
| ACE (context engineering) | 2025 | Context-level optimization | PromptTemplate | High |
| MAR (multi-agent reflexion) | 2025 | Richer reflection feedback | Reflection pipeline | High |
| MCP/A2A tool protocols | 2026 | Standardized tool access | Toolkit system | High |
| BayesFlow | 2026 | Uncertainty-aware workflow gen | WorkFlowGenerator | Medium |
| SE-Agent (trajectories) | 2025 | Trajectory optimization | WorkFlow executor | Medium |
| ReasoningBank | 2025 | Reasoning memory at inference | LongTermMemory | Medium |
| Test-time scaling | 2026 | Adaptive compute at inference | WorkFlow | Medium |
| FedTextGrad | 2025 | Distributed optimization | TextGrad | Medium |
| Evolutionary multimodal prompts | 2025 | Vision-language optimization | EvoPrompt | Medium |

---

## 4. Missing Features Analysis

Based on the codebase analysis and research findings, these are the gaps between AgentAugi's current state and what it needs for OpenClaw integration.

### 4.1 Algorithm Gaps

**No constraint enforcement on evolution.** AgentAugi's optimizers freely mutate prompts and workflows without guardrails. In production, evolved agents can drift into expensive, hallucinating, or unsafe behavior. The CEAO framework (Section 1.3) and DSPy's `Assert`/`Suggest` mechanisms provide solutions.

**No runtime planning.** All five algorithms optimize offline (before deployment). There's no mechanism for agents to plan or re-plan during execution. LATS (Section 2.3) provides this through runtime tree search. The gap means agents can't adapt to unexpected situations mid-task.

**No multi-objective optimization.** All optimizers maximize a single scalar metric. Real production agents need to balance accuracy, cost, latency, and safety simultaneously. MAP-Elites (Section 1.2) and cost-augmented MCTS (Section 3.4) address this.

**No meta-optimization.** The optimization algorithms themselves are fixed. metaTextGrad (Section 3.1) shows that optimizing the optimizer yields compounding returns. AgentAugi should evolve its own optimization strategies over time.

### 4.2 Workflow Gaps

**DAG-only workflows.** `WorkFlowGraph` enforces acyclicity, preventing iterative refinement patterns (try → evaluate → retry). LangGraph (Section 2.4) shows that cycles with max-iteration bounds are essential for production agents.

**Static workflow assignment.** One workflow per goal, regardless of input difficulty. DAAO (Section 3.1) shows that adaptive routing gives 11% accuracy improvement with 36% cost reduction. Simple inputs should use simple workflows.

**No streaming execution.** `WorkFlow.async_execute()` returns only final results. For OpenClaw's user experience, intermediate progress (which node is executing, current partial results) should stream to the frontend.

**Incomplete workflow generation.** `WorkFlowGenerator` has a commented-out `WorkFlowReviewer` (TODO in code). Generated workflows aren't validated or critiqued before use. Adding review (via Multi-Agent Reflexion patterns) would catch structural issues early.

### 4.3 Memory and Learning Gaps

**Fixed memory policies.** `LongTermMemory` uses hardcoded strategies for what to store and retrieve. AgeMem (Section 3.3) shows that learned memory policies significantly outperform fixed ones. Memory management should be an optimizable parameter.

**No failure learning.** When agents fail, the failure information is used only to compute a fitness score. Mistake Notebook (Section 3.3) and Reflexion (Section 2.2) show that extracting *why* agents fail and making that knowledge available to future attempts dramatically improves performance.

**No cross-session learning.** Each optimization run starts from scratch. ReasoningBank (Section 3.3) enables continuous improvement across runs by maintaining and retrieving successful reasoning traces.

**Memory scaling issues.** A FIXME in `rag/rag.py` notes "Computer's memory may increase dramatically" when building the KV store. This needs to be addressed for long-running optimization loops in OpenClaw.

### 4.4 Integration Gaps for OpenClaw

**No standardized tool protocols.** AgentAugi's 40+ tools use custom interfaces. MCP and A2A protocol support (Section 3.6) would make AgentAugi compatible with the broader agent tool ecosystem.

**No cost tracking.** There's no mechanism to track API call costs during optimization or execution. For OpenClaw, cost visibility and cost-constrained optimization are essential.

**No observability.** Beyond basic logging, AgentAugi lacks tracing, metrics, and monitoring infrastructure. OpenClaw needs to know: which optimization step is running, how much it's cost so far, what the score trajectory looks like, and whether convergence is stalling.

**Incomplete async support.** Several TODO comments indicate unfinished async implementations, particularly in agent initialization and tool serialization. OpenClaw will need fully async execution.

### 4.5 Missing Features Priority Matrix

| Feature Gap | Impact | Effort | Priority | Addresses |
|------------|--------|--------|----------|-----------|
| Evolution constraints (CEAO) | Critical | 2–4 weeks | P0 | Safety, cost control |
| Difficulty-aware routing | High | 2–3 weeks | P0 | Efficiency, cost |
| MAP-Elites (multi-objective) | High | 1–2 days | P0 | Optimization quality |
| Reflexion/episodic memory | High | 2–4 weeks | P0 | Learning from failures |
| Cost tracking & budgets | High | 1–2 weeks | P1 | Production readiness |
| Cycle support in workflows | High | 3–4 weeks | P1 | Iterative refinement |
| Runtime planning (LATS) | High | 3–4 weeks | P1 | Adaptive execution |
| Mistake notebook | High | 1 week | P1 | Lightweight learning |
| metaTextGrad | Medium | 2 weeks | P1 | Compounding optimization |
| Streaming execution | Medium | 1–2 weeks | P1 | User experience |
| MCP/A2A protocols | Medium | 2–3 weeks | P2 | Ecosystem compatibility |
| Learned memory policies | Medium | 3–4 weeks | P2 | Memory efficiency |
| Cross-session learning | Medium | 2–3 weeks | P2 | Continuous improvement |
| Observability/tracing | Medium | 2 weeks | P2 | Operability |
| Multi-agent reflection | Medium | 2 weeks | P2 | Richer feedback |

---

## 5. Prioritized Implementation Roadmap

### Phase 1: Critical Foundation (Weeks 1–14)

**Goal:** Make AgentAugi production-ready for OpenClaw with the highest-impact improvements.

**Week 1–2: Upstream Merges**
- Merge MAP-Elites optimizer from PR #222
- Merge concurrency bug fixes from PR #225
- Merge dependency refactor from PR #221
- Evaluate and potentially merge GEPA from feature/gepa branch

**Week 3–6: Core Algorithm Enhancements**
- Implement GEPA optimizer (from DSPy 3.x) as alternative to MIPRO
- Add evolution constraints (CEAO): cost budgets, hallucination filters, drift thresholds
- Integrate DSPy `Assert`/`Suggest` for programmatic output constraints

**Week 7–10: Learning and Memory**
- Implement Reflexion-based episodic memory (verbal self-reflection after execution)
- Add Mistake Notebook for lightweight failure pattern learning
- Add cost tracking across all LLM calls and optimization steps

**Week 11–14: Workflow Intelligence**
- Implement difficulty-aware routing (DAAO) for adaptive workflow selection
- Add cycle support to `WorkFlowGraph` with max-iteration bounds
- Add streaming execution for progress visibility

**Expected outcomes after Phase 1:**
- 15–25% accuracy improvement (GEPA + Reflexion + difficulty-aware routing)
- 30x+ evaluation efficiency (GEPA over MIPRO)
- Production safety through CEAO constraints
- Cost visibility and control

### Phase 2: Advanced Capabilities (Weeks 15–24)

**Goal:** Add sophisticated optimization and planning capabilities.

**Week 15–18: Meta-optimization and Planning**
- Implement metaTextGrad (optimize the optimizer's gradient strategy)
- Implement LATS for runtime tree-search planning
- Add cost-augmented MCTS to AFlow

**Week 19–22: Multi-Agent Evolution**
- Add JudgeAgent for internal evaluation (from MAE paper)
- Implement Multi-Agent Reflexion with diverse critic personas
- Add AgentEvolver-style curriculum generation for training data

**Week 23–24: Context Engineering**
- Implement ACE (context engineering) for holistic context optimization
- Add cross-session learning via ReasoningBank pattern

### Phase 3: Ecosystem and Scale (Ongoing)

**Goal:** Prepare AgentAugi for broad ecosystem integration.

- MCP server/client support for standardized tool access
- A2A protocol support for inter-agent communication
- Federated optimization for multi-organization deployments
- Learned memory management policies (AgeMem)
- Multimodal prompt evolution (vision-language)
- Observability infrastructure (tracing, metrics, dashboards)
- Complete HITL implementation (tool call review, multi-turn conversation)

### Resource Estimate

| Phase | Duration | Engineers | API Budget | Key Deliverables |
|-------|----------|-----------|------------|-----------------|
| Phase 1 | 14 weeks | 2–3 | $5–15k | Production-ready core |
| Phase 2 | 10 weeks | 2–3 | $10–20k | Advanced optimization |
| Phase 3 | Ongoing | 1–2 | Variable | Ecosystem integration |

---

## 6. Audit Findings: Critical Gaps

*Added March 29, 2026 — Post-audit addendum. These areas were identified as missing from the original report and must be addressed before or during implementation.*

### 6.1 Evaluation Strategy — No Benchmarks Defined

**Gap severity: CRITICAL.** The original report claims "15–25% accuracy improvement" but defines no benchmarks to measure it. Without a baseline and a fixed evaluation harness, claimed improvements cannot be validated.

#### Required Benchmarks

**TAU-Bench** — The most important benchmark for AgentAugi. Measures agent reliability over *k* attempts (pass^k metric). A system that succeeds 60% on attempt 1 but only 25% reliably over 8 attempts is worse than one at 50% that stays consistent. SOTA agents score <25% on pass^8. Maps directly to optimization quality.

**SWE-Bench (Pro)** — 1,865 real GitHub issues across 41 repos. Best model at time of writing: 23.3% (GPT-5). Long-horizon coding evaluation. Reveals gaps for complex multi-step tasks that AgentAugi's workflows are designed to handle.

**GAIA** — 466 real-world questions requiring reasoning, multimodality, and tool use. Human achieves 92%; GPT-4 achieves 15%. Tests the kind of general capability that self-evolving agents should improve toward.

**AgentBench** (ICLR 2024) — 8 environments, 29+ LLMs benchmarked. Multi-dimensional reasoning and decision-making. Establishes a standardized multi-environment baseline.

**MultiAgentBench** (arXiv:2503.01935) — Evaluates multi-agent collaboration and competition. Needed specifically when implementing MAE, Multi-Agent Reflexion, and JudgeAgent components.

#### Recommended Evaluation Protocol

Before implementing *any* improvement: establish baseline scores on TAU-Bench (pass^1 and pass^8), SWE-Bench, and AgentBench with the current five-algorithm system. Every Phase 1 and Phase 2 deliverable must report delta on these metrics. This is the only way to validate the roadmap's impact claims.

### 6.2 Process Reward Models — Entire Category Missing

**Gap severity: CRITICAL.** The report treats all evaluation as outcome-based (final score on a task). Process reward models evaluate each *step* of agent execution. The empirical gap is large: process supervision achieves 78.2% vs 34% for outcome-only on MATH for RAG agents (arXiv:2505.14069). This would dramatically improve TextGrad gradient quality and EvoPrompt fitness signals.

#### Key Systems

**AgentPRM** (arXiv:2511.08325) — Process Reward Models for agent tasks using step-wise "promise" (future potential) and "progress" (advancement made) signals. 3B models with AgentPRM outperform GPT-4o on ALFWorld. Directly applicable to AgentAugi's tool-use evaluation.

**ToolRM** (arXiv:2510.26167) — Specialized reward model for function-calling. 17.94% higher accuracy than frontier LLMs on tool-use tasks. Open-source. Structurally different from reasoning rewards—evaluates whether a tool call was the *right* action at that point, not just whether the final output is correct.

**Skywork-Reward-V2** — Open-source suite of 8 reward models (600M–8B parameters), 750K+ downloads, no license restrictions. Off-the-shelf process reward signals without training from scratch.

#### Integration with Existing Algorithms

- **TextGrad:** Replace single outcome-based loss with step-wise reward signal. Each tool call in a trajectory gets a reward from AgentPRM. TextGrad gradients propagate through the trajectory, not just from the final output.
- **EvoPrompt fitness:** Add per-step reward component alongside final score. Evolutionary fitness = α × outcome_score + β × mean(step_rewards).
- **AFlow MCTS:** Use step rewards as intermediate node values in tree search, enabling better expansion decisions rather than relying on rollouts to termination.

### 6.3 ADAS — Closest Competitor and Techniques Source

**Gap severity: CRITICAL.** ADAS (Automated Design of Agentic Systems, ICLR 2025) does essentially what AgentAugi does: automatically discovers novel agent designs. Its Meta Agent Search iteratively programs new agents in code, tests them, and archives discoveries into a growing library. It outperforms SOTA hand-designed agents across coding, science, and math tasks.

**Why this matters:** ADAS is both AgentAugi's closest competitor *and* a source of complementary techniques. The report not mentioning it is a significant omission—any investor, reviewer, or user familiar with the field will ask "how does this differ from ADAS?"

#### Key Differences and Integration Opportunities

| Dimension | ADAS | AgentAugi |
|-----------|------|-----------|
| Representation | Python code (agent implementations) | Prompts + workflow graphs |
| Search | Archive-based meta-search | TextGrad/EvoPrompt/AFlow |
| Evaluation | Task benchmarks | Task benchmarks + multi-objective |
| Memory | Growing archive of designs | LongTermMemory + planned Reflexion |

**Integration path:** ADAS's archiving mechanism (Pareto-optimal designs saved for reuse) should be incorporated into AFlow. Rather than discarding explored workflow variants after optimization, archive the Pareto-optimal ones. This creates a growing library of validated agent designs that future optimization runs can bootstrap from.

**Differentiation to establish:** AgentAugi's strength is continuous self-evolution with gradient-based optimization (TextGrad) combined with evolutionary search (EvoPrompt) and workflow structure search (AFlow). ADAS relies solely on code-level mutation and archive search. Hybrid: use ADAS-style archiving as the memory layer for AgentAugi's evolutionary process.

### 6.4 Letta/MemGPT — Production Memory Architecture

**Gap severity: CRITICAL.** The leading production agent memory system is completely absent from the report. This is especially significant given the documented `FIXME` in `rag/rag.py`: "Computer's memory may increase dramatically."

**Letta** (formerly MemGPT, github.com/letta-ai/letta) implements the "LLM as Operating System" paradigm: agents have in-context memory (RAM), external storage (disk), and explicit memory management instructions that the LLM executes. Letta Code recently ranked #1 on Terminal-Bench.

#### Key Concepts to Incorporate

**Paged memory model:** Rather than growing context windows indefinitely, treat in-context memory as a fixed-size page. When it fills, agent explicitly archives old content to long-term storage and retrieves relevant items. This directly solves the `rag.py` memory scaling issue.

**Cross-session persistence:** Letta's Conversations API maintains agent state across sessions. AgentAugi's `Environment` currently resets per optimization run. Letta's persistence model enables continuous improvement across runs without manual state management.

**Memory as first-class operations:** Letta agents execute explicit `recall_memory()`, `archive_memory()`, and `search_archival()` function calls. This makes memory management observable and optimizable—it can be tuned by the same evolutionary algorithms that optimize prompts.

#### Additional Memory Systems Identified

**A-Mem** (arXiv:2502.12110) — Zettelkasten-inspired memory with interconnected knowledge networks. Memories link to and strengthen each other over time, enabling emergent retrieval patterns.

**Hindsight Memory** (arXiv:2512.12818) — Four-layer architecture: world facts, experiences, entity summaries, evolving beliefs. Distinguishes fact from experience from abstraction—critical for agents that need to reason about their own history.

**Mem0** — Production graph-based memory layer with 50K+ developers. YC-backed. Option for managed memory infrastructure without building from scratch.

**Zep / Graphiti** — Bi-temporal memory modeling: when events occurred vs. when they were ingested. Critical for long-horizon optimization loops where ordering matters.

### 6.5 Cost Optimization Infrastructure

**Gap severity: HIGH.** The original report mentions "cost tracking" as a P1 item but misses the entire landscape of cost *reduction* techniques. These techniques together could reduce optimization loop costs by 80%+ and enable running far more evaluation rounds within the same budget.

#### Prompt Caching — Implement First

Provider-level prompt caching (Anthropic, OpenAI) caches KV states for repeated prompt prefixes. Cached tokens cost ~10% of regular tokens. Claude Code itself achieves 92% cache hit rate with 81% cost reduction. Optimization loops with a fixed system prompt and varying task inputs are ideal for this—the system prompt gets cached after the first call.

**Implementation:** Structure all LLM calls to place stable content (system prompt, optimization instructions, tool definitions) at the front of the prompt. Variable content (task instance, current candidate) comes last. Add a `CachingLLMWrapper` that manages TTL and reports cache hit rates.

#### Semantic Caching — Prevent Redundant Evaluations

Vector embedding similarity check before calling LLM. If a nearly-identical prompt was already evaluated this session (or a recent session), return the cached result. GPTCache provides a reference implementation. Particularly valuable in EvoPrompt where many candidate prompts differ by only a few tokens.

#### Model Routing / Cascading — FrugalGPT Pattern

Route simple evaluations to cheap models (GPT-4o-mini, Claude Haiku) and complex ones to expensive models. FrugalGPT demonstrates 50–98% cost reduction while matching GPT-4o quality. RouteLLM provides learned routing based on query difficulty.

**Integration:** DAAO's difficulty classifier (Section 3.1) can double as a routing signal. Simple tasks → cheap model. Complex tasks → expensive model. This is the same difficulty signal, applied to cost optimization.

#### Speculative Decoding — 4x Inference Speedup

Small draft model generates token predictions; large model verifies in parallel. Available in vLLM. Most valuable for inference-heavy components like EvoPrompt fitness evaluation where many candidates are evaluated sequentially.

#### Tool List Pruning — Ignored Token Cost

Agents tokenize the entire tool list on every call, even when most tools are irrelevant. Filtering tool definitions to only those relevant to the current task step reduces input token cost significantly on tool-heavy workflows. The existing `ToolSelector` could be enhanced to prune definitions, not just select tools.

### 6.6 Additional Gaps

#### Alignment Drift Detection (ATP)

**Alignment Tipping Process** (arXiv:2510.04860) shows empirically that self-evolving agents abandon alignment constraints *during deployment* through reinforcement loops—not through adversarial prompting but through normal optimization pressure. Static guardrails (CEAO's approach) are insufficient.

**Required addition:** Continuous monitoring of evolved agent behavior against alignment metrics throughout the evolution loop. Track: safety constraint violation rate across generations, semantic drift from baseline behavior, escalating cost patterns. Alert and halt evolution if drift thresholds are exceeded. This is separate from and complementary to CEAO's constraint filtering.

#### Parallel Tool Execution (GAP Framework)

**GAP (Graph-Based Planning)** (arXiv:2510.25320) learns explicit task dependency graphs that enable parallel tool invocation. Current AgentAugi execution is fully sequential (one tool call at a time). GAP shows 2–3x efficiency gains for multi-tool agents by identifying which tool calls are independent and can run concurrently.

**Integration:** Add a dependency analysis layer to `WorkFlowGraph` that identifies independent tool calls within a node and dispatches them concurrently via `asyncio.gather()`. The dependency graph can be learned (GAP) or statically declared in workflow definitions.

#### Tool Creation by Agents (LATM)

**LATM — LLMs as Tool Makers** (ICLR 2024) establishes a two-phase architecture: an expensive frontier model creates reusable Python tools for a task class; cheap models execute those tools. This enables agents to expand their own capabilities during the evolution loop—true self-improvement beyond prompt tuning.

**Integration:** Add a `ToolSynthesizer` component that, when the optimization loop encounters a task class that no existing tool handles well, generates and registers a new tool. Tools created this way are preserved in the `ToolRegistry` and available to future optimization runs.

#### Multi-Agent Debate Quality Controls

The original report covers Multi-Agent Reflexion (MAR) but misses critical quality controls:

- **Convergence detection** (NeurIPS 2025): Detect when debate agents have stabilized to prevent degeneration-of-thought—debates that continue past consensus produce worse outputs.
- **Controlled disagreement** (Liang et al., EMNLP 2024): Moderate, tunable disagreement beats maximal disagreement. Unstructured debate converges to polarization; add a `disagreement_temperature` parameter.
- **Heterogeneous roles** (A-HMAD): Specialized critic roles (logical reasoning, factual verification, strategic planning) beat identical-agent clones. Assign distinct personas to critic agents in MAR.
- **Agent-as-a-Judge** (arXiv:2410.10934, ICML 2025 + ICLR 2025): More validated than JudgeAgent from MAE. 90% human agreement vs 70% for LLM-as-Judge, with 97% cost/time reduction. Should replace or supplement external `Evaluator` objects.

### 6.7 Audit Gap Summary Matrix

| Gap Area | Severity | Original Coverage | Impact if Ignored |
|----------|----------|-------------------|-------------------|
| Evaluation benchmarks (TAU-Bench, SWE-Bench, GAIA) | **Critical** | None | Cannot validate 15-25% claim; no way to measure progress |
| Process reward models (AgentPRM, ToolRM) | **Critical** | None | TextGrad/EvoPrompt quality capped at outcome-only; 2x improvement left on table |
| ADAS competitive reference | **Critical** | None | Blind spot vs. closest competitor; duplicates work without knowing it |
| Letta/MemGPT memory architecture | **Critical** | None | Memory scaling FIXME remains; unsolved for production |
| Cost optimization (caching, routing, cascading) | **High** | Tracking only | 80%+ cost reduction unrealized; optimization loops remain expensive |
| Alignment drift detection (ATP) | **High** | Partial (CEAO only) | Static guardrails fail in deployment; alignment erodes silently |
| Parallel tool execution (GAP) | **High** | None | 2-3x efficiency loss on multi-tool workflows |
| Tool creation (LATM) | **Medium** | None | Tool use coverage but not self-expanding capability |
| Debate quality controls | **Medium** | Partial | Debate degrades without convergence detection and heterogeneous roles |

---

## Appendix: Full Paper Catalog

### Core Algorithm Extensions
1. TextGrad — arXiv:2406.07496 (June 2024, Nature)
2. metaTextGrad — arXiv:2505.18524 (May 2025)
3. FedTextGrad — arXiv:2502.19980 (February 2025)
4. AFlow — arXiv:2410.10762 (October 2024, ICLR 2025)
5. SEW — arXiv:2505.18646 (May 2025)
6. EvoPrompt — arXiv:2309.08532 (September 2023, ICLR 2024)
7. MIPRO — arXiv:2406.11695 (June 2024)

### Workflow Optimization
8. DAAO (Difficulty-Aware Orchestration) — arXiv:2509.11079 (September 2025)
9. BayesFlow — arXiv:2601.22305 (January 2026)
10. WorkflowLLM — arXiv:2411.05451 (November 2024)

### Self-Evolving Agents
11. Survey: Self-Evolving Agents — arXiv:2507.21046 (July 2025)
12. Multi-Agent Evolve (MAE) — arXiv:2510.23595 (October 2025)
13. SE-Agent — arXiv:2508.02085 (August 2025)
14. EvolveR — arXiv:2510.16079 (October 2025)

### Prompt Optimization
15. Evolutionary Multimodal Prompts — arXiv:2503.23503 (March 2025)
16. Promptomatix — arXiv:2507.14241 (July 2025)
17. DSPy Prompt-as-Code Study — arXiv:2507.03620 (July 2025)

### MCTS and Planning
18. MCTS for Heuristic Design — arXiv:2501.08603 (January 2025)
19. ToolTree — arXiv:2603.12740 (March 2026)
20. Cost-Augmented MCTS — arXiv:2505.14656 (May 2025)

### Memory and Learning
21. Memory in the Age of AI Agents (Survey) — arXiv:2512.13564 (December 2025)
22. AgeMem (Learned Memory) — arXiv:2601.01885 (January 2026)
23. Memoria — arXiv:2512.12686 (December 2025)
24. Memori — arXiv:2603.19935 (March 2026)
25. ReasoningBank — arXiv:2509.25140 (September 2025)
26. Mistake Notebook Learning — arXiv:2512.11485 (December 2025)

### Test-Time Scaling
27. Benchmark Test-Time Scaling — arXiv:2602.18998 (February 2026)
28. ARTIS (Risk-Aware Scaling) — arXiv:2602.01709 (February 2026)

### Context and Adaptation
29. ACE (Context Engineering) — arXiv:2510.04618 (October 2025)
30. Test-Time Adaptation — arXiv:2511.04847 (November 2025)

### Reflection and Introspection
31. Introspection of Thought — arXiv:2507.08664 (July 2025)
32. Multi-Agent Reflexion (MAR) — arXiv:2512.20845 (December 2025)
33. ReflectEvo — ACL 2025 Findings

### Reinforcement Learning
34. RE-PO (Robust Policy Optimization) — arXiv:2509.24159 (September 2025)
35. RLTR (Process Rewards) — arXiv:2508.19598 (August 2025)
36. Multi-Agent RL (Dec-POMDP) — arXiv:2512.24609 (December 2025)

### Tool Integration
37. Evolution of Tool Use (Survey) — arXiv:2603.22862 (March 2026)
38. Tool-to-Agent Retrieval — arXiv:2511.01854 (November 2025)
39. Tool Preference Reliability — arXiv:2505.18135 (May 2025)

### Federated Learning
40. pFedDC — arXiv:2506.21144 (June 2025)
41. Federated Multimodal Prompts — arXiv:2602.07081 (February 2026)

### Lifelong Learning
42. Lifelong Learning Roadmap — arXiv:2501.07278 (January 2025)

### EvoAgentX Publications
43. EvoAgentX Framework — arXiv:2507.03616 (July 2025)
44. Self-Evolving Agents Survey — arXiv:2508.07407 (August 2025)

---

### Audit Addendum: Additional Papers (March 2026)

**Benchmarks and Evaluation**
45. TAU-Bench (Tool-Augmented User interactions) — pass^k reliability metric, SOTA <25% pass^8
46. AgentBench — arXiv:2308.03688 (ICLR 2024), 8 environments, 29+ LLMs
47. MultiAgentBench — arXiv:2503.01935 (March 2025)
48. SWE-Bench Pro — 1,865 real GitHub issues, 41 repos

**Process Reward Models**
49. AgentPRM (Step-wise Process Rewards for Agents) — arXiv:2511.08325 (November 2025)
50. ToolRM (Function-Calling Reward Model) — arXiv:2510.26167 (October 2025)
51. Process vs Outcome Supervision for RAG Agents — arXiv:2505.14069 (May 2025)

**Automated Agent Design**
52. ADAS (Automated Design of Agentic Systems) — ICLR 2025, Meta Agent Search
53. Godel Agent (Self-Referential Self-Improvement) — ACL 2025

**Memory Architecture**
54. A-Mem (Zettelkasten Agent Memory) — arXiv:2502.12110 (February 2025)
55. Hindsight Memory (Four-Layer Architecture) — arXiv:2512.12818 (December 2025)

**Cost Optimization**
56. FrugalGPT (Model Cascading for Cost Reduction) — arXiv:2305.05176 (May 2023)
57. RouteLLM (Learned Routing) — arXiv:2406.18665 (June 2024)

**Alignment and Safety**
58. Alignment Tipping Process (ATP) — arXiv:2510.04860 (October 2025)
59. AgentDoG (Runtime Diagnostic Guardrails) — arXiv:2601.18491 (January 2026)

**Parallel Execution and Tool Creation**
60. GAP (Graph-Based Parallel Tool Execution) — arXiv:2510.25320 (October 2025)
61. LATM (LLMs as Tool Makers) — ICLR 2024, arXiv:2305.17126
62. ToolMaker (Paper-to-Tool Synthesis) — arXiv:2502.11705 (February 2025)

**Multi-Agent Debate Quality**
63. Agent-as-a-Judge — arXiv:2410.10934 (ICML 2025 + ICLR 2025)
64. Controlled Disagreement in Debate (Liang et al.) — EMNLP 2024
65. Tree-GRPO (Tree-Based Group Policy Optimization) — arXiv:2509.21240 (September 2025)

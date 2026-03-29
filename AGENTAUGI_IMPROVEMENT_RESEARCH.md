# AgentAugi — Improvement Research

> Pre-integration research for OpenClaw. Compiled: 2026-03-29.
> Covers: EvoAgentX upstream gaps, complementary projects, arXiv papers, prioritized build list.

---

## TL;DR

AgentAugi is missing **3 production-ready optimizers** from EvoAgentX branches that aren't merged yet
(GEPA, AlphaEvolve, MAP-Elites), plus a handful of incomplete features in its own codebase. The
highest-ROI improvements are: merge those 3 optimizers, implement `WorkflowReviewer`, integrate the
`UserIntentRouter`, and add the `Agent0`-style zero-data evolution loop. Total effort: ~3–4 weeks of
focused engineering.

---

## 1. EvoAgentX Upstream — What's Missing in AgentAugi

These are **finished or nearly-finished features in EvoAgentX branches** that haven't been merged
into main (and therefore aren't in AgentAugi). Each has concrete code already written.

### 1.1 GEPA Optimizer ⭐ Priority 1

- **Branch**: `EvoAgentX/EvoAgentX:feature/gepa` (last commit: 2025-11-09)
- **File**: `evoagentx/optimizers/gepa_optimizer.py`
- **Example**: `examples/optimization/gepa/math_gepa.py`
- **What it does**: Gradient-free Evolutionary Prompt Adapter — uses evolutionary search rather
  than LLM-gradient backprop to optimize prompts. Reportedly outperforms RL-based approaches
  (see EvoAgentX issue #219).
- **Why EvoAgentX hasn't merged it**: Unclear — community is asking the same question (#219).
  Main is only 25 commits ahead; gepa is only 1 commit ahead of main. Low merge risk.
- **Why AgentAugi needs it**: TextGrad and MIPRO both require differentiable-ish setups.
  GEPA is a cleaner black-box alternative for tasks where you don't have a formal gradient path.
- **How to add**: Cherry-pick or copy `gepa_optimizer.py` from the branch, add to
  `evoagentx/optimizers/__init__.py`. One file addition.

### 1.2 AlphaEvolve Optimizer ⭐ Priority 2

- **Branch**: `EvoAgentX/EvoAgentX:alphaevolve` (last commit: 2025-07-10)
- **Files**: `evoagentx/optimizers/alphaevolve/` (config, database, ensemble, sampler, template),
  `evoagentx/optimizers/alphaevolve_optimizer.py`,
  `evoagentx/evaluators/alphaevolve_evaluator.py`
- **Inspiration**: Google DeepMind's AlphaEvolve/FunSearch (2024) — evolve code/programs by
  keeping a population database and sampling promising variants. Paper: arxiv.org/abs/2506.13131
  (FunSearch follow-up) and the original FunSearch: arxiv.org/abs/2311.11600.
- **What it does**: Instead of mutating prompts, mutates **executable code** that agents run.
  The `ensemble.py` and `sampler.py` implement a population-based search over code programs
  with an LLM doing the mutations.
- **Why AgentAugi needs it**: Current optimizers mutate text. AlphaEvolve enables evolving
  **tool implementations and agent action code** — a qualitatively different capability.
- **How to add**: Copy the full `alphaevolve/` directory. Moderate integration effort (~1 day)
  since it has its own evaluator.

### 1.3 MAP-Elites Optimizer ⭐ Priority 3

- **PR**: EvoAgentX #222 (open, `feat/map-elites-optimizer`)
- **File**: `evoagentx/optimizers/map_elites_optimizer.py`
- **What it does**: Maintains a **2D grid archive** of diverse solutions, not just the single best.
  Maps solutions to feature dimensions (e.g., complexity vs. diversity) and keeps the best per cell.
  Based on Mouret & Clune (2015) MAP-Elites, adapted for LLM prompt search.
- **Why this matters for AgentAugi**: Current optimizers converge to one best prompt. For
  OpenClaw's heterogeneous task types (code tasks vs. writing tasks vs. analysis tasks), you want
  *different* prompts for different task profiles — MAP-Elites generates exactly that.
- **How to add**: PR is complete with tests and docs. Just merge or copy the single file.

### 1.4 Multi-Agent Debate — Already Present ✅

- **Status**: `evoagentx/frameworks/multi_agent_debate/` is in AgentAugi main.
- **What's there**: `debate.py`, `pruning.py`, `utils.py` + examples.
- **No action needed** — it's already merged.

### 1.5 DSPy MIPRO — Already Present ✅

- **Status**: `evoagentx/optimizers/mipro_optimizer.py` is in AgentAugi main.
- **No action needed**.

---

## 2. Internal AgentAugi Gaps

These are **incomplete or missing features inside the current codebase** (confirmed by code inspection).

### 2.1 WorkflowReviewer Not Implemented ⭐ Priority 1

- **Location**: `evoagentx/workflow/workflow_generator.py`, line ~54 — placeholder/stub.
- **What it should do**: After `WorkFlowGenerator` produces a workflow DAG, `WorkflowReviewer`
  should validate it: check for logical cycles, missing tool dependencies, incompatible
  input/output types between nodes, and hallucinated agent capabilities.
- **Impact**: Without it, generated workflows can silently produce garbage when the LLM
  makes structural errors. High impact for OpenClaw reliability.
- **Approach**: Use an LLM critic pattern (similar to how AFlow validates operator mutations).
  Pass the serialized workflow + a validation rubric, get back a pass/fail + diff of fixes.

### 2.2 UserIntentRouter Not Merged

- **Branch**: `EvoAgentX/EvoAgentX:dev_UserIntentRouter` (active)
- **What it does**: Classifies an incoming natural-language message and routes it to the
  correct workflow or agent automatically.
- **Why critical for OpenClaw**: The gateway layer (Nerve/kanban) needs this exact capability —
  receive a message, decide which workflow handles it, without hardcoded rules.
- **Status**: Branch is active in upstream. Monitor and cherry-pick when stable.

### 2.3 SEW Optimizer Incomplete

- **Location**: `evoagentx/optimizers/sew_optimizer.py` — in codebase but flagged as "in progress".
- **What it does**: Structural Workflow Evolution — rewrites the *topology* of a workflow DAG,
  not just prompts. Converts workflow to 5 representations (Python, YAML, BPMN, code, core)
  and mutates structurally.
- **Why it's the most powerful**: TextGrad improves prompts. SEW improves the *architecture*
  of what you're running — can merge nodes, split agents, reorder steps, add/remove tools.
- **Action**: Identify what's incomplete, finish the implementation. This is the single highest-
  leverage optimizer for OpenClaw's use case (whole-workflow improvement vs. prompt tweaking).

### 2.4 HITL Advanced Modes (3 TODOs)

- **review_edit** mode: let human edit agent outputs before they propagate downstream — not implemented.
- **tool_calls_review** mode: show agent tool calls to human for approval before execution — not implemented.
- **multi_turn_conversation**: extended back-and-forth HITL beyond single Q&A — not implemented.
- **Priority**: Medium. The basic approval workflow is complete and works for OpenClaw Week 4.

### 2.5 RAG Index Types Missing

- `TreeIndex` and `SummaryIndex` referenced but not implemented in `evoagentx/rag/`.
- **Priority**: Low — `VectorIndex` (FAISS/ChromaDB) covers 90% of use cases.

---

## 3. External Projects to Study/Integrate

### 3.1 Agent0 — Self-Evolving Without Demonstrations ⭐ Critical

- **Repo**: https://github.com/aiming-lab/Agent0 (1,107 stars)
- **Paper**: "Agent0 Series: Self-Evolving Agents from Zero Data" (2025)
- **Key insight**: Agents that evolve using **synthetic self-play and AI feedback**, without any
  human-labeled training data. The agent generates its own training signal via:
  1. Self-exploration (try diverse approaches to a task)
  2. Self-improvement (LLM scores its own attempts, keeps best)
  3. Self-distillation (compress learnings into updated model/prompts)
- **Why this matters**: AgentAugi's current evolution requires a `Benchmark` dataset (labeled
  examples). Agent0's approach works on **any new task without pre-collected data**.
- **Integration target**: Wrap Agent0's self-play loop as a new `SelfPlayOptimizer` that
  doesn't require a pre-defined benchmark — critical for OpenClaw's open-ended tasks.

### 3.2 AgentEvolver — In-Situ Evolution ⭐ High Priority

- **Repo**: https://github.com/modelscope/AgentEvolver (1,316 stars)
- **Key insight**: Evolves agents **while they run** (in-situ), not in a separate offline
  optimization loop. Uses a continual self-improvement mechanism triggered by task outcomes.
- **What AgentAugi currently does**: Batch optimization — collect data, then run optimizer
  offline (e.g., nightly TextGrad run).
- **What AgentEvolver adds**: Online/incremental evolution triggered on every task completion.
  Faster feedback loop, no need to wait for a batch.
- **Integration**: Study the `AgentEvolver` architecture and implement an `OnlineOptimizer`
  wrapper for AgentAugi that can incrementally update prompts after each task outcome.

### 3.3 PromptWizard — Task-Aware Prompt Optimization

- **Repo**: https://github.com/microsoft/PromptWizard (3,827 stars)
- **Paper**: "PromptWizard: Task-Aware Agent-Driven Prompt Optimization" (Microsoft Research, 2024)
- **Key differentiator vs. existing optimizers**: Uses chain-of-thought *reasoning about why*
  previous prompt attempts failed, not just gradient/score feedback. Generates task-specific
  critique, then synthesizes an improved prompt.
- **Comparison**:
  - TextGrad: LLM gradient on score
  - MIPRO: Bayesian + bootstrap few-shot
  - EvoPrompt: GA mutation/crossover
  - **PromptWizard**: Failure analysis + targeted rewrite
- **Integration**: Implement as `PromptWizardOptimizer` in AgentAugi using the paper's
  algorithm (it's Apache 2.0 licensed). Particularly good for instruction-following tasks.

### 3.4 DSPy GEPA (in DSPy itself, separate from EvoAgentX's version)

- **Repo**: https://github.com/stanfordnlp/dspy (33,244 stars)
- **Relevant optimizer**: `GEPA` in DSPy (Gradient-free Evolutionary Prompt Adapter, 2025)
- **Note**: EvoAgentX's `feature/gepa` branch appears to implement GEPA for EvoAgentX's
  optimizer engine. The original GEPA is in DSPy. AgentAugi already has DSPy as a dependency.
- **Action**: Instead of or in addition to porting EvoAgentX's GEPA branch, wrap DSPy's
  native GEPA optimizer directly (similar to how MIPRO is wrapped).

### 3.5 LATS — Language Agent Tree Search

- **Paper**: "Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models"
  (Zhou et al., 2023) — arxiv.org/abs/2310.04406
- **Repo**: Look for `princeton-nlp/tree-of-thought-llm` or `andyz24/lats` on GitHub.
- **What it does**: MCTS over reasoning+action sequences — the agent explores multiple
  reasoning paths in a tree, backtracking when paths fail.
- **Why for AgentAugi**: AFlow already does MCTS over *workflow operators*. LATS does MCTS
  over *agent reasoning steps within a node*. Combining them = evolution at every granularity.
- **Integration**: Add a `LATSAgent` action type that wraps any agent node with tree search.

### 3.6 Reflexion — Self-Reflection Memory

- **Paper**: "Reflexion: Language Agents with Verbal Reinforcement Learning" (Shinn et al., 2023)
  — arxiv.org/abs/2303.11366
- **What it does**: After each task failure, the agent generates a verbal "reflection"
  (episodic memory of what went wrong), which gets prepended to future attempts.
- **Why for AgentAugi**: HITL feedback + Reflexion = an agent that learns from human corrections
  *without* running a full optimizer. Lower cost, faster feedback loop than TextGrad.
- **Integration**: Add `ReflexionMemory` to `LongTermMemory` — on task failure, store a
  structured reflection alongside the failed output. Inject into next attempt's context.

### 3.7 MemSkill — Evolving Memory Skills

- **Repo**: https://github.com/ViktorAxelsen/MemSkill (370 stars)
- **Paper**: "MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents" (2025)
- **What it does**: Agents build a library of reusable "memory skills" (procedural patterns
  from past successes) that get retrieved and applied to new similar tasks.
- **Why for AgentAugi**: Complements `LongTermMemory` (episodic) with procedural memory
  (how-to patterns). Current `LongTermMemory` stores *what happened*; MemSkill stores
  *how to handle this type of task*.
- **Integration**: Add a `SkillLibrary` class to `evoagentx/memory/` that stores compressed
  successful task procedures and retrieves them during workflow generation.

---

## 4. arXiv Papers (Key References)

ArXiv rate-limiting prevented direct search, but these are the must-read papers for each
optimizer in AgentAugi + the improvements above:

| Paper | ArXiv ID | Relevance |
|-------|----------|-----------|
| TextGrad: Automatic "Differentiation" via Text | 2406.07496 | Already implemented |
| MIPRO (Optimizing Instructions and Demonstrations) | 2406.11695 | Already implemented |
| AFlow: Automating Agentic Workflow Generation | 2410.10762 | Already implemented |
| EvoPrompt: Language Evolution for LLM Prompts | 2309.08532 | Already implemented |
| GEPA: Gradient-free Evolutionary Prompt Adapter | DSPy docs/2025 | Need to port from branch |
| FunSearch (AlphaEvolve basis) | 2311.11600 | Reference for AlphaEvolve port |
| MAP-Elites | GECCO 2015 | Reference for PR #222 |
| LATS: Language Agent Tree Search | 2310.04406 | Should add |
| Reflexion: Verbal Reinforcement Learning | 2303.11366 | Should add |
| Agent0: Self-Evolving from Zero Data | aiming-lab 2025 | High priority integration |
| PromptWizard | Microsoft Research 2024 | High priority integration |

---

## 5. Prioritized Improvement List

### Tier 1 — Do Before OpenClaw Integration (highest ROI, low effort)

| # | Improvement | Effort | Impact | Source |
|---|-------------|--------|--------|--------|
| 1 | **Port GEPA optimizer** from `feature/gepa` branch | 0.5 days | High — new optimization algorithm, one file | EvoAgentX branch |
| 2 | **Implement WorkflowReviewer** | 1 day | High — prevents workflow quality failures | Internal gap |
| 3 | **Port MAP-Elites optimizer** from PR #222 | 0.5 days | Medium-High — diverse solution archive for multi-task | EvoAgentX PR |
| 4 | **Add Reflexion memory** to `LongTermMemory` | 1 day | High — free learning from HITL corrections | arxiv:2303.11366 |

### Tier 2 — Do During OpenClaw Integration (medium effort, high payoff)

| # | Improvement | Effort | Impact | Source |
|---|-------------|--------|--------|--------|
| 5 | **Port AlphaEvolve optimizer** from `alphaevolve` branch | 1.5 days | High — code-level evolution, new paradigm | EvoAgentX branch |
| 6 | **Complete SEW optimizer** | 3 days | Very High — structural workflow evolution | Internal gap |
| 7 | **Add online/in-situ evolution loop** (AgentEvolver pattern) | 2 days | High — removes batch-only constraint | AgentEvolver |
| 8 | **Cherry-pick UserIntentRouter** when stable | 1 day | Critical for OpenClaw gateway | EvoAgentX branch |
| 9 | **Add SkillLibrary (MemSkill pattern)** to memory | 2 days | High — procedural memory for task patterns | MemSkill |

### Tier 3 — Post-Integration (longer term, research-level)

| # | Improvement | Effort | Impact | Source |
|---|-------------|--------|--------|--------|
| 10 | **SelfPlayOptimizer** (Agent0 pattern) — no labels needed | 4 days | Very High — removes benchmark requirement | Agent0 |
| 11 | **PromptWizardOptimizer** — failure-analysis based | 2 days | Medium-High — better critique than TextGrad | PromptWizard |
| 12 | **LATS integration** — tree search within workflow nodes | 3 days | High — better reasoning per node | arxiv:2310.04406 |
| 13 | **HITL advanced modes** (review_edit, tool_calls_review) | 2 days | Medium — better human oversight | Internal gap |
| 14 | **Multimodal memory** (from `feature/multimodal_memory` branch) | 2 days | Medium — image/video in memory | EvoAgentX branch |

---

## 6. What NOT to Integrate

These are popular but **redundant** given what AgentAugi already has:

- **LangGraph**: Workflow DAG execution → AgentAugi's `WorkFlowGraph` already does this better
  for evolutionary purposes.
- **AutoGen/CrewAI**: Multi-agent coordination → `WorkFlowGenerator` + `multi_agent_debate`
  covers the main use cases. Adding these creates more seams to maintain.
- **LangChain**: AgentAugi's tool ecosystem is more integrated. Use ActionAgent wrapping
  if you need a specific LangChain tool, don't pull in the whole framework.
- **Dify/Flowise**: Visual workflow builders → these are user-facing products, not libraries.
  OpenClaw's Nerve cockpit will serve this role.

---

## 7. Immediate Next Steps

```
Week 1 of AgentAugi improvements (before OpenClaw integration):

Day 1:  Port GEPA optimizer (copy gepa_optimizer.py + add to __init__.py)
Day 1:  Port MAP-Elites optimizer (copy map_elites_optimizer.py from PR #222)
Day 2:  Implement WorkflowReviewer (LLM critic of generated workflow DAG)
Day 3:  Add Reflexion memory (ReflexionMemory class in evoagentx/memory/)
Day 4:  Port AlphaEvolve optimizer (copy alphaevolve/ directory)
Day 5:  Complete SEW optimizer (identify blockers, finish core structural mutation)
```

After these 5 days, AgentAugi will have **8 optimization algorithms** (vs. 5 today) and
two critical quality improvements (WorkflowReviewer + Reflexion) — at which point the
OpenClaw integration plan in `OPENCLAW_INTEGRATION_ANALYSIS.md` is ready to execute.

---

*Research sources: EvoAgentX GitHub (branches, PRs, issues), GitHub search (self-evolving agents,
prompt optimization), arXiv API (rate-limited; papers referenced from known IDs), direct codebase
inspection of AgentAugi main branch.*

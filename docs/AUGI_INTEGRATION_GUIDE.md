# AgentAugi — Augi Integration Guide

> **Audience:** Augi (the AI agent system). This document catalogs everything
> AgentAugi provides so that Augi's Claude Code sessions and orchestration layers
> can invoke it correctly.
>
> **Last updated:** 2026-03-29

---

## 1. What AgentAugi Is

AgentAugi is a **self-evolving agent framework** — a fork of
[EvoAgentX](https://github.com/EvoAgentX/EvoAgentX) with **7 major modules**
added on top:

| # | Module | Purpose |
|---|--------|---------|
| 1 | **MASTER Search** | Monte-Carlo tree search with LLM self-evaluation; replaces LATS (~11K tokens vs ~185K, +5% accuracy) |
| 2 | **HyperAgents** | Meta-optimization layer: evolves optimizer *strategies* themselves via LLM-guided mutation/crossover |
| 3 | **Plan Caching** | Semantic two-level cache for agent plans; target ~50% cost reduction |
| 4 | **PASTE** | Speculative execution — prefetches likely next steps |
| 5 | **EvoSkill** | Closed-loop failure → skill synthesis pipeline |
| 6 | **JitRL** | Just-in-time reinforcement learning via statistical action biasing (+19 pts WebArena vs Reflexion) |
| 7 | **Integrations** | OpenClaw bridge, AceForge connector, Nerve HITL |

AgentAugi sits in the Augi stack as the **evolution + optimization engine**:

```
Nerve (task UI / HITL)  ←→  OpenClaw Gateway  ←→  AgentAugi
                                                         ↕
                                                    AugiVector
                                                    (LLM routing)
                                                         ↕
                                                    AceForge
                                                    (skill synthesis)
```

---

## 2. Module Catalog

### 2.1 MASTER Search

**What it does:** Monte-Carlo tree search over candidate solutions, using
LLM confidence-weighted self-evaluation instead of expensive rollouts.

**Import path:** `evoagentx.core.master_search`

**Key classes:**

```python
from evoagentx.core.master_search import MASTERSearch, MASTERConfig

cfg = MASTERConfig(
    max_iterations=10,
    max_depth=5,
    exploration_weight=1.0,
    confidence_threshold=0.7,
    num_candidates=4,
    use_self_evaluation=True,
    token_budget=15000,
)
searcher = MASTERSearch(config=cfg, llm_fn=my_llm_callable)
result = searcher.search(problem="Optimize the retry logic for X")
# result is a LATSResult (drop-in replacement)
```

**Config file:** none — configured inline via `MASTERConfig`.

---

### 2.2 HyperAgents (HyperOptimizer)

**What it does:** Meta-optimization. Maintains a *population of optimizer
strategies* (`OptimizerGenome`), evolves them over generations using
LLM-guided mutation and crossover, and selects winners by fitness on a task
suite. The nightly evolution script drives this automatically.

**Import path:** `evoagentx.optimizers.hyper_optimizer`

**Key classes:**

```python
from evoagentx.optimizers.hyper_optimizer import (
    HyperOptimizer, HyperOptimizerConfig, OptimizerGenome, OptimizerPopulation
)

cfg = HyperOptimizerConfig(
    population_size=8,
    generations=20,
    mutation_rate=0.35,
    crossover_rate=0.25,
    tournament_size=3,
    evaluation_budget=50,
    meta_llm_fn=my_llm_callable,   # LiteLLM or Anthropic SDK callable
    fitness_metric="accuracy",
)

optimizer = HyperOptimizer(
    config=cfg,
    base_optimizers=[SEWOptimizer, MAPElitesOptimizer],  # seed population
    evaluation_fn=my_eval_fn,        # (genome, task_suite) -> float
)

gen_result = optimizer.evolve_generation(task_suite=[...])
best = optimizer.best_optimizer()   # OptimizerGenome
optimizer.save("data/evolution/population_state.json")
```

**`OptimizerGenome` fields:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Human-readable identifier |
| `source_code` | `str` | Natural-language algorithm description |
| `config` | `Dict[str, Any]` | Numeric hyperparameters |
| `fitness_history` | `List[float]` | Per-generation fitness scores |
| `generation` | `int` | Birth generation |
| `parent_name` | `Optional[str]` | Parent genome |
| `structural_hash` | `str` | SHA-256 dedup key |

**Seed optimizer classes** (resolved by name in config):

```python
from evoagentx.optimizers import (
    SEWOptimizer, MAPElitesOptimizer,
    TextGradOptimizer, GEPAOptimizer,
    AFlowOptimizer, MIPROOptimizer, EvoPromptOptimizer,
)
```

**Config file:** `configs/nightly_evolution.yaml` — see §5.

---

### 2.3 Plan Caching

**What it does:** Two-level semantic cache for agent plans. Avoids
re-generating plans for structurally similar tasks. Integrates with
`CostTracker` to measure savings.

**Import path:** `evoagentx.core.plan_cache`

```python
from evoagentx.core.plan_cache import PlanCache, PlanTemplate, PlanStep

cache = PlanCache(path="data/evolution/plan_cache.json")
cache.load()

# Store a plan
template = PlanTemplate(
    task_description="Summarize a PDF document",
    steps=[PlanStep(action_type="read", description="Read PDF", ...)],
    outcome="summary text",
    structural_signature="read→summarize",
)
cache.store(template)

# Retrieve (returns best match or None)
match = cache.lookup("Summarize this research paper")
cache.save()
```

**Config file:** path set in `configs/nightly_evolution.yaml` under
`paths.plan_cache`.

---

### 2.4 PASTE (Speculative Executor)

**What it does:** Prefetches likely next steps in a workflow while the
current step is executing, reducing perceived latency.

**Import path:** `evoagentx.core.speculative_executor`

```python
from evoagentx.core.speculative_executor import SpeculativeExecutor

exec = SpeculativeExecutor(llm_fn=my_llm_callable)
result = exec.run(current_step="parse input", context={...})
```

**Config file:** none — configured inline.

---

### 2.5 EvoSkill

**What it does:** Closed-loop pipeline that observes agent failures
(via `MistakeNotebook`), identifies recurring *skill gaps*, synthesizes
new tools to fill them (via `ToolSynthesizer`), and optionally deploys
them automatically.

**Import path:** `evoagentx.core.evoskill`

```python
from evoagentx.core.evoskill import (
    EvoSkillPipeline, EvoSkillConfig, SkillGap, SkillDiscovery
)

cfg = EvoSkillConfig(
    min_mistake_frequency=3,
    severity_threshold=0.5,
    auto_deploy=False,
    analysis_interval=300,      # seconds
    max_pending_gaps=20,
)
pipeline = EvoSkillPipeline(config=cfg, llm_fn=my_llm_callable)

# Run one analysis cycle
discoveries: list[SkillDiscovery] = pipeline.analyze()

# SkillGap fields: gap_type, description, frequency, severity, suggested_tool_spec
# SkillDiscovery fields: gap, synthesized_tool, validation_score, deployed
```

**Config file:** `configs/aceforge.yaml` controls how gaps/skills sync
with AceForge (see §3.3).

---

### 2.6 JitRL (Just-in-Time RL)

**What it does:** Learns statistical action-success rates from historical
trajectories and injects biasing nudges into agent prompts. No gradient
updates — pure statistical inference.

**Import path:** `evoagentx.memory.jitrl`

```python
from evoagentx.memory.jitrl import (
    JitRLMemory, JitRLConfig, JitRLAgent,
    TrajectoryStep, TrajectoryStatistics,
)

cfg = JitRLConfig()
mem = JitRLMemory(config=cfg, path="data/evolution/jitrl_memory.json")
mem.load()

# Wrap an existing agent
smart_agent = JitRLAgent(base_agent=my_agent, memory=mem)
result = smart_agent.run(task)

# Direct nudge injection
nudge = mem.nudge_prompt(context_hash="hash_of_task_context")
bias  = mem.get_action_bias(action_type="web_search")

mem.save()
```

**Config file:** path set in `configs/nightly_evolution.yaml` under
`paths.jitrl_state`.

---

### 2.7 Cost Tracker

**What it does:** Global singleton that intercepts every LLM call, maps
tokens → USD, enforces budget ceilings, and exposes session-scoped
context managers.

**Import path:** `evoagentx.core.cost_tracker`

```python
from evoagentx.core.cost_tracker import get_tracker, CostBudgetExceeded

tracker = get_tracker()
tracker.set_budget(max_usd=10.0, warn_fraction=0.75)

with tracker.session("nightly_run"):
    # all LLM calls within are tracked
    ...

print(tracker.total_usd())
```

---

## 3. Integration Points

### 3.1 AugiVector Routing

**Config file:** `configs/openclaw.yaml` → `llm_profiles` section

AugiVector runs at `http://localhost:3000/v1` (OpenAI-compatible proxy).
The `llm_profiles` map routes each OpenClaw agent type to an AugiVector
virtual model name:

| Agent type | AugiVector profile | Underlying model |
|------------|-------------------|-----------------|
| `general` | `auto` | AugiVector intelligent routing |
| `researcher` | `research` | Kimi K2.5 (256K context) |
| `analyst` | `research` | Kimi K2.5 (256K context) |
| `writer` | `auto` | AugiVector intelligent routing |
| `reviewer` | `auto` | AugiVector intelligent routing |

To use AugiVector directly:

```python
from evoagentx.models.litellm_model import LiteLLMModel

model = LiteLLMModel(
    model="openai/auto",            # virtual model name
    base_url="http://localhost:3000/v1",
    api_key="unused",               # AugiVector does not require auth locally
)
```

---

### 3.2 OpenClaw Bridge

**File:** `evoagentx/integrations/openclaw_bridge.py`

**What it does:** Receives tasks dispatched by the OpenClaw Gateway
(`127.0.0.1:18789`), translates them into AgentAugi `WorkFlowConfig`
objects, executes them via `WorkFlowGenerator` + `WorkFlow.run()`, and
returns structured results.

```python
from evoagentx.integrations import OpenClawBridge, OpenClawConfig, OpenClawTask
import yaml, asyncio

with open("configs/openclaw.yaml") as f:
    raw = yaml.safe_load(f)
cfg = OpenClawConfig(**raw["gateway"])

async def main():
    async with OpenClawBridge(cfg) as bridge:
        task = OpenClawTask(
            session_id="abc123",
            goal="Summarize the Q3 earnings report",
            agent_type="researcher",   # AgentType enum value
            context={},
        )
        result = await bridge.submit(task)
        # result.output — final answer
        # result.workflow_trace — step-by-step trace
        # result.metrics — cost, latency, token counts

        # Batch (concurrent, respects max_concurrent_tasks=8)
        results = await bridge.submit_batch([task1, task2, task3])

asyncio.run(main())
```

**Key data classes:**

| Class | Purpose |
|-------|---------|
| `OpenClawConfig` | Connection settings (host, port, token, timeouts, retries) |
| `OpenClawTask` | Inbound task (session_id, goal, agent_type, context) |
| `OpenClawResult` | Output (output, workflow_trace, metrics, status) |
| `WorkflowConfig` | Translated AgentAugi config passed to WorkFlowGenerator |
| `AgentType` | Enum: GENERAL, RESEARCHER, ANALYST, WRITER, REVIEWER |
| `TaskStatus` | Enum: PENDING, RUNNING, COMPLETED, FAILED, TIMEOUT |

---

### 3.3 AceForge Connector

**File:** `evoagentx/integrations/aceforge_connector.py`

**What it does:** Bidirectional sync between EvoSkill's gap/skill pipeline
and AceForge's tool-forge system. Reads AceForge's `patterns.jsonl` to
import skill gaps; writes `evoskill_gaps.jsonl` to export gaps to AceForge;
exports successful `SkillDiscovery` objects as `SKILL.md` proposals.

```python
from evoagentx.integrations import AceForgeConnector, AceForgeConnectorConfig
from evoagentx.core.evoskill import EvoSkillPipeline, EvoSkillConfig
import yaml

with open("configs/aceforge.yaml") as f:
    raw = yaml.safe_load(f)
c = raw["connector"]
cfg = AceForgeConnectorConfig(
    forge_dir=c["forge_dir"],
    skills_dir=c["skills_dir"],
    external_gaps_file=c["external_gaps_file"],
    evolution_trigger_file=c["evolution_trigger_file"],
    sync_interval_seconds=c["sync_interval_seconds"],
    dry_run=c["dry_run"],
)

pipeline = EvoSkillPipeline(config=EvoSkillConfig(), llm_fn=my_llm_fn)
connector = AceForgeConnector(pipeline=pipeline, config=cfg)

# One full bidirectional sync
sync_result = connector.run_sync()
# sync_result.imported_count, sync_result.exported_count, sync_result.errors

# Granular control
import_result = connector.pull_gaps_from_aceforge()
export_result = connector.push_gap_to_aceforge(gap)

# Check if AceForge wants AgentAugi to evolve specific tools
triggers = connector.poll_evolution_requests()   # List[str] — tool names
connector.request_aceforge_evolution(["web_search", "pdf_reader"])
```

**File exchange protocol:**

| File | Owner | Format | Purpose |
|------|-------|--------|---------|
| `~/.openclaw/workspace/.forge/patterns.jsonl` | AceForge writes | JSONL | Gap candidates for EvoSkill to import |
| `~/.openclaw/workspace/.forge/evoskill_gaps.jsonl` | AgentAugi writes | JSONL | Gaps exported to AceForge |
| `~/.openclaw/workspace/.forge/evoskill_trigger.json` | AceForge writes | JSON | Evolution request triggers |
| `~/.openclaw/workspace/skills/SKILL.md` | AgentAugi writes | Markdown+YAML | Synthesized skill proposals |

---

### 3.4 Nerve HITL

**File:** `evoagentx/integrations/nerve_hitl.py`

**What it does:** Routes `EvolutionProposal` objects to the Nerve kanban
(`http://localhost:3080`) for human review. Classifies proposals by
score delta (MINOR/MODERATE/MAJOR), auto-approves below threshold, and
polls for decisions on the rest.

```python
from evoagentx.integrations import NerveHITL, NerveConfig, EvolutionProposal
import yaml

with open("configs/nerve.yaml") as f:
    raw = yaml.safe_load(f)
n, t = raw["nerve"], raw["thresholds"]

config = NerveConfig(
    base_url=n["base_url"],
    api_prefix=n["api_prefix"],
    api_key=n.get("api_key", ""),
    default_labels=n["default_labels"],
    created_by=n["created_by"],
    max_retries=n["max_retries"],
    request_timeout=n["request_timeout"],
)

hitl = NerveHITL(config=config)

proposal = hitl.build_proposal(
    optimizer_name="SEWOptimizer-gen7",
    description="Increased mutation diversity by adding Gaussian noise",
    score_before=0.72,
    score_after=0.81,
    diff_summary="Changed mutation_rate from 0.2 → 0.35 and added Gaussian jitter",
)

# Returns ProposalStatus: APPROVED, REJECTED, AUTO_APPROVED, TIMED_OUT, ERROR
status = hitl.submit_and_wait(proposal, raise_on_timeout=False)

# Batch submission
statuses = hitl.submit_many([proposal1, proposal2])

# Report outcome back to Nerve task
hitl.report_metrics(proposal, deployed=(status == ProposalStatus.APPROVED))
```

**Classification thresholds** (from `configs/nerve.yaml`):

| Delta (relative) | Classification | Action |
|-----------------|----------------|--------|
| < 2% | MINOR | Auto-approved |
| 2%–10% | MODERATE | Auto-approved if `auto_approve_moderate: true`; else sent to Nerve |
| ≥ 10% | MAJOR | Always sent to Nerve for human review |

---

### 3.5 Nightly Evolution

**File:** `scripts/nightly_evolution.py`

**What it does:** Overnight HyperOptimizer runner. Loads population state,
runs N generations, enforces USD budget, records trajectories to JitRL,
caches the best strategy in PlanCache, writes a JSON + text report, and
sends a notification.

**Run manually:**

```bash
# Dry run (no LLM calls, no writes)
python scripts/nightly_evolution.py --dry-run

# Custom config
python scripts/nightly_evolution.py --config configs/nightly_evolution.yaml

# Override individual values
python scripts/nightly_evolution.py --set evolution.generations=5 budget.max_usd=2.0
```

**Install as cron (2 AM daily):**

```bash
bash scripts/install_cron.sh
```

**Programmatic use:**

```python
from scripts.nightly_evolution import run_evolution, load_config

cfg = load_config(
    "configs/nightly_evolution.yaml",
    overrides={"evolution.generations": 5, "budget.max_usd": 2.0},
)
exit_code = run_evolution(cfg)   # 0=success, 1=error, 2=budget_exceeded
```

**Output files** (under `data/evolution/`):

| File | Description |
|------|-------------|
| `population_state.json` | Serialized HyperOptimizer population |
| `jitrl_memory.json` | JitRL trajectory history |
| `plan_cache.json` | Best-strategy plan cache |
| `reports/YYYYMMDD_HHMMSS_<run_id>.json` | Full structured report |
| `reports/YYYYMMDD_HHMMSS_<run_id>.txt` | Human-readable report |
| `notifications.log` | One-line completion entries |

---

## 4. Workflow Examples

### 4.1 Optimize a Prompt Using MASTER Search

```python
from evoagentx.core.master_search import MASTERSearch, MASTERConfig
from evoagentx.models.litellm_model import LiteLLMModel

model = LiteLLMModel(model="anthropic/claude-sonnet-4-6")
llm_fn = lambda prompt, **kw: model.generate(prompt, **kw)

searcher = MASTERSearch(
    config=MASTERConfig(
        max_iterations=8,
        confidence_threshold=0.75,
        num_candidates=4,
        token_budget=20000,
    ),
    llm_fn=llm_fn,
)

result = searcher.search(
    problem="Write a system prompt for a customer support agent "
            "that handles refund requests accurately and empathetically"
)
print(result.best_solution)
print(f"Tokens used: {result.total_tokens}")
```

---

### 4.2 Discover and Fill Skill Gaps with EvoSkill + AceForge

```python
from evoagentx.core.evoskill import EvoSkillPipeline, EvoSkillConfig
from evoagentx.integrations import AceForgeConnector
import yaml

# 1. Set up EvoSkill
pipeline = EvoSkillPipeline(
    config=EvoSkillConfig(
        min_mistake_frequency=3,
        severity_threshold=0.4,
        auto_deploy=False,
    ),
    llm_fn=llm_fn,
)

# 2. Set up AceForge connector
with open("configs/aceforge.yaml") as f:
    raw = yaml.safe_load(f)
connector = AceForgeConnector(
    pipeline=pipeline,
    config=AceForgeConnectorConfig(**raw["connector"]),
)

# 3. Pull gaps AceForge has discovered, push ours, export new skills
sync = connector.run_sync()
print(f"Imported {sync.imported_count} gaps, exported {sync.exported_count}")

# 4. Run EvoSkill analysis to synthesize tools for pending gaps
discoveries = pipeline.analyze()
for d in discoveries:
    print(f"Gap: {d.gap.description}")
    print(f"Tool synthesized: {d.synthesized_tool}")
    print(f"Validation score: {d.validation_score:.2f}")
    print(f"Deployed: {d.deployed}")
```

---

### 4.3 Run Overnight Evolution and Review Results in Nerve

```bash
# 1. Trigger evolution manually
python scripts/nightly_evolution.py --config configs/nightly_evolution.yaml

# 2. Check the report
cat data/evolution/reports/$(ls -t data/evolution/reports/*.txt | head -1)

# 3. Open Nerve to see any pending HITL review tasks
open http://localhost:3080
```

Evolution proposals requiring human review appear in Nerve with labels
`evoagentx`, `hitl`, `auto-generated`. Approve or reject them in the
kanban; the script's `submit_and_wait()` call will unblock within
`poll_interval_seconds` (default 15s).

---

### 4.4 Route a Complex Task Through the Full Pipeline

```python
import asyncio, yaml
from evoagentx.integrations import OpenClawBridge, OpenClawConfig, OpenClawTask
from evoagentx.integrations import AgentType

async def route_task(goal: str, agent_type: AgentType = AgentType.RESEARCHER):
    with open("configs/openclaw.yaml") as f:
        raw = yaml.safe_load(f)
    cfg = OpenClawConfig(**raw["gateway"])

    async with OpenClawBridge(cfg) as bridge:
        # Health check first
        ok = await bridge._check_health()
        if not ok:
            raise RuntimeError("OpenClaw Gateway unreachable")

        task = OpenClawTask(
            session_id="my_session_001",
            goal=goal,
            agent_type=agent_type,
        )
        result = await bridge.submit(task)

    return result

result = asyncio.run(route_task(
    "Analyze the latest commit history and summarize risk areas",
    AgentType.ANALYST,
))
print(result.output)
print("Cost USD:", result.metrics.get("cost_usd"))
```

The bridge automatically:
1. Translates the task → `WorkflowConfig` with AugiVector LLM profile
2. Executes via `WorkFlowGenerator` → `WorkFlow.run()`
3. Routes LLM calls through `http://localhost:3000/v1` (AugiVector)
4. Returns structured `OpenClawResult` with trace + metrics

---

## 5. Config Reference

### `configs/openclaw.yaml`

Controls the OpenClaw Gateway bridge.

| Key | Default | Description |
|-----|---------|-------------|
| `gateway.host` | `127.0.0.1` | OpenClaw Gateway host |
| `gateway.port` | `18789` | OpenClaw Gateway port |
| `gateway.token` | env `OPENCLAW_TOKEN` | Bearer auth token |
| `gateway.timeout_connect` | `10.0` | Connect timeout (s) |
| `gateway.timeout_read` | `30.0` | Read timeout (s) |
| `gateway.timeout_total` | `300.0` | Total request timeout (s) |
| `gateway.max_retries` | `3` | Retry attempts on failure |
| `gateway.retry_backoff` | `2.0` | Exponential backoff base |
| `gateway.heartbeat_interval` | `30.0` | Background heartbeat (s) |
| `llm_profiles.*` | see file | agent type → AugiVector profile |
| `concurrency.max_concurrent_tasks` | `8` | Max parallel `submit_batch()` calls |

---

### `configs/aceforge.yaml`

Controls AceForge ↔ EvoSkill file exchange.

| Key | Default | Description |
|-----|---------|-------------|
| `connector.forge_dir` | `~/.openclaw/workspace/.forge` | AceForge working dir |
| `connector.skills_dir` | `~/.openclaw/workspace/skills` | Deployed SKILL.md dir |
| `connector.external_gaps_file` | `evoskill_gaps.jsonl` | EvoSkill → AceForge gaps |
| `connector.evolution_trigger_file` | `evoskill_trigger.json` | AceForge → EvoSkill trigger |
| `connector.sync_interval_seconds` | `30` | Sync loop interval (0 = manual) |
| `connector.dry_run` | `false` | Log writes without executing |
| `import_settings.max_severity_aceforge` | `100.0` | Normalization ceiling for AceForge scores |
| `import_settings.min_aceforge_severity` | `3.0` | Minimum severity to import |
| `export_settings.validation_threshold` | `0.6` | Min EvoSkill score to export |

---

### `configs/nerve.yaml`

Controls HITL submission to the Nerve kanban.

| Key | Default | Description |
|-----|---------|-------------|
| `nerve.base_url` | `http://localhost:3080` | Nerve instance URL |
| `nerve.api_prefix` | `/api/kanban` | Kanban endpoint prefix |
| `nerve.api_key` | `""` | Bearer token (leave blank for local) |
| `nerve.default_labels` | `[evoagentx, hitl, auto-generated]` | Labels added to every task |
| `nerve.created_by` | `evoagentx` | Creator field on tasks |
| `nerve.max_retries` | `3` | HTTP retry count |
| `nerve.request_timeout` | `10.0` | Per-request timeout (s) |
| `thresholds.minor_delta` | `0.02` | Below → auto-approve (MINOR) |
| `thresholds.major_delta` | `0.10` | Above → always send to human (MAJOR) |
| `thresholds.auto_approve_moderate` | `false` | Auto-approve MODERATE changes |
| `thresholds.poll_timeout_seconds` | `3600.0` | Give up waiting after this long |
| `thresholds.poll_interval_seconds` | `15.0` | Check Nerve every N seconds |

---

### `configs/nightly_evolution.yaml`

Controls the overnight HyperOptimizer run.

| Key | Default | Description |
|-----|---------|-------------|
| `evolution.generations` | `20` | Generations per nightly run |
| `evolution.population_size` | `8` | Genomes in population |
| `evolution.mutation_rate` | `0.35` | Mutation probability |
| `evolution.crossover_rate` | `0.25` | Crossover probability |
| `evolution.tournament_size` | `3` | Tournament selection size |
| `evolution.evaluation_budget` | `50` | Max task evaluations per genome |
| `evolution.fitness_metric` | `accuracy` | Key in eval_fn return dict |
| `meta_llm.model` | `anthropic/claude-haiku-4-5` | LLM for mutation/crossover |
| `meta_llm.temperature` | `0.7` | Generation temperature |
| `meta_llm.max_tokens` | `1024` | Token cap per meta-LLM call |
| `budget.max_usd` | `10.00` | Hard spend ceiling (USD) |
| `budget.warn_fraction` | `0.75` | Warn at this fraction of budget |
| `paths.population_state` | `data/evolution/population_state.json` | Population persistence |
| `paths.jitrl_state` | `data/evolution/jitrl_memory.json` | JitRL persistence |
| `paths.plan_cache` | `data/evolution/plan_cache.json` | PlanCache persistence |
| `paths.reports_dir` | `data/evolution/reports` | Report output dir |
| `paths.lock_file` | `/tmp/agentaugi_nightly_evolution.lock` | Concurrency lock |
| `seed_optimizers` | `[SEWOptimizer, MAPElitesOptimizer, ...]` | Initial population seeds |
| `notification.backend` | `file` | `slack`, `file`, or `none` |
| `schedule.cron` | `0 2 * * *` | Cron expression (2 AM daily) |
| `schedule.timeout_minutes` | `240` | SIGTERM after this many minutes |

---

## 6. Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════════╗
║                         AUGI STACK                                   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ┌──────────┐     ┌──────────────────┐     ┌──────────────────────┐ ║
║  │  Nerve   │◄────│  OpenClaw        │────►│     AgentAugi        │ ║
║  │  Kanban  │     │  Gateway         │     │                      │ ║
║  │  :3080   │     │  :18789          │     │  WorkFlowGenerator   │ ║
║  │          │     │  (ACP sessions)  │     │  WorkFlow.run()      │ ║
║  │  HITL    │◄─── │                  │ ◄── │  MASTER Search       │ ║
║  │  Review  │     └──────────────────┘     │  HyperOptimizer      │ ║
║  └──────────┘                              │  EvoSkill Pipeline   │ ║
║       ▲                                    │  JitRL Memory        │ ║
║       │  Evolution                         │  Plan Cache          │ ║
║       │  Proposals                         │  Cost Tracker        │ ║
║       │                                    │  PASTE Executor      │ ║
║       └────────────────────────────────────┤                      │ ║
║                                            │  integrations/       │ ║
║  ┌──────────────────┐                      │  ├ openclaw_bridge   │ ║
║  │   AugiVector     │◄─────────────────────│  ├ aceforge_conn.   │ ║
║  │   localhost:3000 │                      │  └ nerve_hitl       │ ║
║  │   (LLM router)   │                      └──────────────────────┘ ║
║  │                  │                               ▲               ║
║  │  auto → routing  │                               │               ║
║  │  research → Kimi │                   ┌───────────┴──────────┐    ║
║  │  code → DeepSeek │                   │     AceForge         │    ║
║  └──────────────────┘                   │  ~/.openclaw/...     │    ║
║                                         │  patterns.jsonl      │    ║
║                                         │  evoskill_gaps.jsonl │    ║
║                                         │  SKILL.md proposals  │    ║
║                                         └──────────────────────┘    ║
║                                                                      ║
║  ┌────────────────────────────────────────────────────────────────┐ ║
║  │              Nightly Evolution (cron 2 AM)                     │ ║
║  │  scripts/nightly_evolution.py                                  │ ║
║  │  HyperOptimizer → N generations → JitRL record → PlanCache    │ ║
║  │  → Nerve HITL proposals → report JSON/TXT → notification      │ ║
║  └────────────────────────────────────────────────────────────────┘ ║
╚══════════════════════════════════════════════════════════════════════╝
```

### Data Flow: Task Execution

```
OpenClaw Gateway
      │  OpenClawTask (session_id, goal, agent_type)
      ▼
OpenClawBridge.submit()
      │  translate → WorkflowConfig + LLM profile
      ▼
WorkFlowGenerator.generate(goal)
      │  task DAG
      ▼
WorkFlow.run()
      ├── LiteLLMModel → AugiVector :3000/v1
      ├── MASTER Search (planning nodes)
      ├── PlanCache (lookup before generating)
      ├── JitRL nudge injection
      └── CostTracker (enforces budget)
      │  result
      ▼
OpenClawResult (output, trace, metrics)
      │
      ▼
OpenClaw Gateway → caller
```

### Data Flow: Nightly Evolution

```
scripts/nightly_evolution.py
      │
      ├── Load: population_state.json, jitrl_memory.json, plan_cache.json
      │
      ├── For each generation (up to N=20):
      │     HyperOptimizer.evolve_generation(task_suite)
      │     ├── tournament selection of parents
      │     ├── LLM mutation / crossover → child genomes
      │     ├── evaluate each child (accuracy proxy)
      │     ├── CostTracker.check_budget() → abort if over $10
      │     └── SIGTERM handler → graceful early stop
      │
      ├── Record best trajectory → JitRLMemory
      ├── Cache best genome → PlanCache
      ├── Write reports/ JSON + TXT
      │
      ├── If MAJOR improvement:
      │     NerveHITL.submit_and_wait(proposal)
      │     └── poll localhost:3080 until approved/rejected/timeout
      │
      └── Notify (file append or Slack webhook)
```

### Data Flow: Skill Gap Sync

```
AceForge                              AgentAugi
  │                                        │
  │── patterns.jsonl ──────────────────►  AceForgeConnector.pull_gaps_from_aceforge()
  │                                        │  translate → SkillGap objects
  │                                        ▼
  │                                   EvoSkillPipeline.analyze()
  │                                        │  MistakeNotebook → ToolSynthesizer
  │                                        │  → SkillDiscovery (validated tool)
  │                                        │
  │◄─── evoskill_gaps.jsonl ─────────── push_gap_to_aceforge()
  │◄─── skills/SKILL.md ──────────────  export_discovery_as_skill_md()
  │                                        │
  │── evoskill_trigger.json ──────────►  poll_evolution_requests()
  │      (AceForge requests evolution)     └── triggers EvoSkill analysis cycle
```

---

## 7. Quick Reference

### Service Endpoints

| Service | URL | Notes |
|---------|-----|-------|
| OpenClaw Gateway | `http://127.0.0.1:18789` | ACP session manager |
| AugiVector | `http://localhost:3000/v1` | OpenAI-compatible LLM proxy |
| Nerve | `http://localhost:3080` | Kanban HITL UI |

### Environment Variables

| Variable | Used by | Purpose |
|----------|---------|---------|
| `OPENCLAW_TOKEN` | `openclaw.yaml` | Gateway auth token |
| `SLACK_WEBHOOK_URL` | `nightly_evolution.yaml` | Slack notification hook |
| `ACEFORGE_CONNECTOR_DRY_RUN` | `aceforge.yaml` | Override dry_run flag |

### Key File Locations

| Path | Purpose |
|------|---------|
| `configs/openclaw.yaml` | OpenClaw bridge config |
| `configs/aceforge.yaml` | AceForge connector config |
| `configs/nerve.yaml` | Nerve HITL config |
| `configs/nightly_evolution.yaml` | Overnight evolution config |
| `data/evolution/population_state.json` | HyperOptimizer population |
| `data/evolution/jitrl_memory.json` | JitRL trajectory history |
| `data/evolution/plan_cache.json` | Plan cache |
| `data/evolution/reports/` | Evolution run reports |
| `~/.openclaw/workspace/.forge/` | AceForge file exchange |

### Checking System Health

```python
import asyncio, yaml
from evoagentx.integrations import OpenClawBridge, OpenClawConfig, NerveHITL, NerveConfig

async def health():
    with open("configs/openclaw.yaml") as f:
        raw = yaml.safe_load(f)
    bridge = OpenClawBridge(OpenClawConfig(**raw["gateway"]))
    print("OpenClaw:", await bridge._check_health())

    with open("configs/nerve.yaml") as f:
        raw = yaml.safe_load(f)
    nerve = NerveHITL(NerveConfig(**raw["nerve"]))
    print("Nerve:", nerve.health_check())

asyncio.run(health())
```

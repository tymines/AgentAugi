# AgentAugi × OpenClaw — Integration Analysis

> Concrete capabilities map and build plan.
> Written: 2026-03-29 | Branch: main

---

## TL;DR

AgentAugi (EvoAgentX fork) gives OpenClaw three things it doesn't have today:
1. **Workflow auto-construction** — describe a task in plain English, get a working multi-agent DAG
2. **Self-evolution** — agent prompts and workflow structures improve automatically from outcomes (TextGrad, AFlow, SEW, MIPRO, EvoPrompt)
3. **Evaluation harness** — track whether Augi is actually getting better over time

The main integration surface is AgentAugi's **FastAPI app** (already built) + its **Python SDK**. AugiFlo is the execution layer; AgentAugi is the intelligence layer above it.

---

## 1. What AgentAugi Actually Does

### 1.1 Workflow Auto-Generation (`WorkFlowGenerator`)

```python
# What happens under the hood when you call:
wf = WorkFlowGenerator(llm=llm).generate_workflow(goal="Analyze competitor pricing and write a report")

# Step 1: TaskPlanner decomposes goal into subtask DAG
# Step 2: AgentGenerator assigns/creates an agent per subtask node
# Step 3: WorkFlowReviewer validates and optionally refines
# Step 4: Returns WorkFlowGraph with nodes, edges, agent specs
```

**Concrete output**: A JSON-serializable `WorkFlowGraph` with:
- Named nodes (e.g., `PriceDataCollector`, `ReportWriter`)
- Directed edges (dependency ordering)
- Per-node agent specs (name, description, system prompt, actions, tools)
- Input/output parameter contracts between nodes

This is what the `Wonderful_workflow_corpus/` directory demonstrates — 6 complete workflow examples including stock analysis, arxiv digest, travel planning, Fengshui analysis (yes, really), recipe generation, and Tetris game generation.

### 1.2 Self-Evolution Algorithms

| Algorithm | What it mutates | Input needed | Benchmark results |
|-----------|-----------------|--------------|-------------------|
| **TextGrad** | System prompts (gradient backprop via LLM) | Dataset of (input, expected_output) | +7.4% HotPotQA F1, +10% MATH |
| **AFlow** | Workflow operators (MCTS over Code/Test/Ensemble/Rephrase nodes) | Same dataset | +11% MBPP Pass@1 |
| **MIPRO** | System prompts (Bayesian black-box) | Same dataset | +5.6% HotPotQA, +6.3% MATH |
| **EvoPrompt** | Prompts (genetic algorithm: mutation + crossover) | Prompt variants + feedback | Not yet benchmarked in repo |
| **SEW** | Entire workflow structure (Python/YAML/BPMN representations) | Same dataset | In progress |

**Key point**: SEW is the most powerful — it can rewrite the *structure* of a workflow, not just prompt wording. It converts workflows to 5 representations (Python, YAML, BPMN, code, core) and mutates them structurally.

### 1.3 Human-in-the-Loop (HITL)

Full HITL system already built (`evoagentx/hitl/`):
- `HITLManager` — approval queue with persist/resolve lifecycle
- `HITLInterceptorAgent` — pauses workflow at any node, waits for human input
- `HITLConversationAgent` — interactive back-and-forth with user mid-workflow
- `HITLOutsideConversationAgent` — sends request to external channel, waits for response
- `workflow_editor.py` — edit workflow graph mid-execution
- GUI support (`hitl_gui.py`)

### 1.4 Long-Term Memory

`LongTermMemory` uses RAG (FAISS/other backends) for semantic retrieval:
```python
# Stores any Message (with metadata: agent, action, workflow, timestamp)
# Retrieves by semantic similarity to a Query object
memory.add(message)
results = memory.retrieve(query="previous competitor analysis tasks")
```

Key: retrieval is semantic, not recency-only. Past task outputs, agent decisions, and user feedback are all searchable.

### 1.5 Tool Ecosystem

Already built, ready to use:
- **Web**: DuckDuckGo, Google, SerpAPI, SerperAPI, Wikipedia search
- **Browser**: `browser_tool.py` / `browser_use.py` (Playwright-based)
- **Code**: Python interpreter (local + Docker sandbox)
- **Files**: Read/write/list (`file_tool.py`)
- **Databases**: FAISS vector store, MongoDB, PostgreSQL
- **Finance**: Stock prices, crypto, financial materials, Google Finance
- **Communication**: Gmail toolkit, Telegram
- **Data**: RSS feeds, Google Maps, ArXiv
- **MCP**: `mcp.py` — any MCP server can be exposed as an agent tool

The MCP integration is critical: **every AugiFlo MCP tool can be injected directly into AgentAugi agents** via `mcp.py`.

### 1.6 Evaluation & Benchmarks

Built-in datasets:
- HotPotQA (multi-hop QA)
- MBPP / HumanEval / LiveCodeBench (coding)
- MATH / GSM8K (reasoning)
- GAIA (general agentic)
- NQ, WorfBench, BigBench Hard, MMRAG

Custom evaluators via `Evaluator` base class. This is what enables the evolution algorithms — every optimizer needs a benchmark to score against.

### 1.7 FastAPI REST App

Complete at `evoagentx/app/`:
- `POST /agents` — create agent spec
- `GET/PUT /agents/{id}` — manage agents
- `POST /workflows` — create workflow
- `POST /workflows/{id}/execute` — run workflow (async via `BackgroundTasks`)
- `GET /executions/{id}` — poll execution status + results
- JWT auth with user roles
- MongoDB persistence (`app/db.py`)

---

## 2. What OpenClaw Needs (Current Gaps)

Based on the OpenClaw config structure (`openclaw.json`):
```
meta, env, wizard, auth, acp, models, agents, tools, messages, commands, channels, gateway, memory, plugins
```

Channels: WhatsApp, Telegram, Slack
Agent pool: file-based JSON agents in `/agents/`
AugiFlo: multi-agent orchestration (agent spawn/status/terminate, task claims, hive-mind)

**Gaps AgentAugi fills:**

| Gap | Current state | AgentAugi solution |
|-----|--------------|-------------------|
| Workflow creation | Manual agent config | Auto-generate from task description |
| Agent improvement | Static prompts | TextGrad/SEW evolution on outcomes |
| Performance tracking | None | Built-in benchmark + evaluator framework |
| Memory recall | File-based, recency | Semantic RAG over all past interactions |
| HITL in workflows | Not systematic | `HITLManager` + interceptor agents |
| New skill creation | Manual dev | Workflow generator + workflow corpus |
| Tool orchestration | Per-agent config | Declarative tool assignment in workflow nodes |

---

## 3. Concrete Integration Points

### 3.1 Kanban Task → Auto-Generated Workflow

**How it works:**
1. User creates kanban task: "Research our top 3 competitors' pricing pages and write a comparison table"
2. OpenClaw sends task description to AgentAugi `WorkFlowGenerator`
3. Generator produces: `PricePageFetcher → DataExtractor → TableFormatter → ReportWriter` (DAG)
4. Each node gets an agent with the right tool assignments
5. AugiFlo executes the workflow via its existing agent pool

**What needs building:**
- Kanban webhook → AgentAugi FastAPI `/workflows` endpoint (POST with goal text)
- Map AgentAugi's tool names to AugiFlo's MCP tool names (adapter layer)
- Return workflow execution ID to kanban task for status tracking

**Effort**: ~2 days. The REST API is already there.

### 3.2 Nerve (Cockpit) ↔ HITL Integration

**How it works:**
1. Workflow hits a `HITLInterceptorAgent` checkpoint
2. `HITLManager` emits approval request event
3. Nerve receives it, displays context + action options to human
4. Human approves/modifies/rejects via Nerve UI
5. `HITLManager.resolve()` resumes workflow

**What needs building:**
- WebSocket bridge: AgentAugi `HITLManager` → Nerve frontend
- Nerve UI component for HITL approval cards (show workflow state, agent outputs, options)
- `HITLOutsideConversationAgent` config pointing at Nerve's WebSocket endpoint

**Effort**: ~3 days. The `HITLOutsideConversationAgent` already handles external async comms.

### 3.3 Nightly Evolution Loop (Make Augi Smarter)

**How it works:**
1. Collect task outcomes from previous N days (success/fail + actual output vs expected)
2. Format as `Benchmark` dataset (AgentAugi format)
3. Run `TextGradOptimizer` against Augi's most-used agent prompts
4. Optionally run `SEWOptimizer` on frequently-executed workflow patterns
5. Write optimized prompts back to OpenClaw agent configs
6. Next day, Augi uses improved prompts

**What needs building:**
- Task outcome logger: records (input_task, agent_outputs, human_feedback_rating) in structured format
- Evolution scheduler: cron job calling optimizer with collected dataset
- Prompt update applier: writes optimizer output back to agent JSON files in `/agents/`

**Effort**: ~4 days. Key challenge: defining the "ground truth" evaluator for open-ended tasks (might need human rating or LLM-as-judge via `Evaluator`).

### 3.4 AugiFlo Agent Pool → AgentAugi `existing_agents`

**How it works:**
When generating a workflow, pass current AugiFlo agents as candidates:
```python
augiflo_agents = [load_agent(f) for f in openclaw_agents_dir]
wf = generator.generate_workflow(goal=task, existing_agents=augiflo_agents)
```
AgentAugi will preferentially assign existing agents to nodes rather than generating new ones. New agents only get created when no existing agent fits.

**What needs building:**
- `AgentAugiAdapter`: converts OpenClaw agent JSON → EvoAgentX `Agent` object
- Reverse adapter: EvoAgentX generated agent spec → OpenClaw agent JSON for persistence

**Effort**: ~1 day. Mostly data mapping.

### 3.5 Semantic Long-Term Memory for Augi

**How it works:**
Replace/supplement current file-based memory at `~/.claude/projects/` with AgentAugi's `LongTermMemory`:
```python
# Every task execution stores output + metadata
memory.add(Message(
    content=task_output,
    wf_goal=task_description,
    agent="Augi",
    action="task_execute"
))

# Augi recalls relevant past work on new tasks
past = memory.retrieve(Query(text="competitor analysis tasks I've done before"))
```

**What needs building:**
- Memory write hook: after each task completion, store to `LongTermMemory`
- Memory read injection: pre-task, query for relevant past context, inject into Augi's prompt
- Storage backend: FAISS (already in AgentAugi) or connect to existing ChromaDB at `~/.openclaw/chromadb`

**Effort**: ~2 days. ChromaDB is already running in OpenClaw — AgentAugi supports it as a backend.

### 3.6 Skill Auto-Generation Pipeline

**How it works:**
1. Augi or user says: "I need a skill that monitors my GitHub PRs and summarizes daily activity"
2. `WorkFlowGenerator` creates the workflow (with `browser_tool`, `request.py`, etc.)
3. Workflow is serialized to `Wonderful_workflow_corpus/` format (`workflow.json` + `tools.json`)
4. AugiFlo/n8n registers it as a callable skill

**What needs building:**
- Skill generation endpoint in AgentAugi FastAPI: `POST /skills/generate` with {description, available_tools}
- Workflow → n8n workflow converter (or just use AgentAugi's own execution engine)
- Skill registry integration: write to OpenClaw's plugin system

**Effort**: ~3 days. The workflow corpus format is already standardized.

---

## 4. What's Missing / Needs to Be Built

### Priority 1 — Foundation (do these first)

| Component | What it does | Where it lives |
|-----------|-------------|----------------|
| **OpenClaw-AgentAugi Bridge** | Kanban task → AgentAugi workflow goal + result back to task | New Python package `agentaugi_bridge/` in `/openclaw/integration/` |
| **Tool Name Mapper** | Maps AugiFlo MCP tool names ↔ AgentAugi tool class names | Config file + adapter class |
| **Agent JSON Adapter** | OpenClaw agent format ↔ EvoAgentX `Agent` pydantic model | ~100 lines Python |

### Priority 2 — Intelligence Layer

| Component | What it does | Dependencies |
|-----------|-------------|--------------|
| **Task Outcome Logger** | Records (task, output, rating) for evolution | Bridge |
| **Evolution Scheduler** | Nightly TextGrad run on collected outcomes | Outcome Logger |
| **Memory Write Hook** | Stores task outputs to `LongTermMemory` | Bridge |
| **Memory Read Injection** | Pre-task semantic recall into Augi context | Memory Write Hook |

### Priority 3 — User-Facing

| Component | What it does | Dependencies |
|-----------|-------------|--------------|
| **Nerve HITL Panel** | UI for workflow approval/modification checkpoints | AgentAugi HITL + Nerve frontend |
| **Skill Generator UI** | Natural language → registered skill | Workflow generator + Skill registry |
| **Evolution Dashboard** | Track agent performance over time | Evolution Scheduler + Evaluator |

---

## 5. What NOT to Build (Already Exists)

- **Don't rebuild** a workflow engine — AgentAugi's `WorkFlowGraph` + async execution is production-ready
- **Don't rebuild** memory — `LongTermMemory` with RAG backend is complete
- **Don't rebuild** evaluation — all major benchmarks are integrated
- **Don't rebuild** HITL — the full system with GUI support is there
- **Don't rebuild** tool integrations — browser, search, code execution, Gmail, Telegram are all done
- **Don't rebuild** the FastAPI layer — CRUD + execution + auth is complete

---

## 6. EvoAgentX Upstream Features to Watch

From the roadmap and recent PRs:

| Feature | Status | OpenClaw relevance |
|---------|--------|-------------------|
| **SEW optimizer** (structural workflow evolution) | In progress (branch: sew) | Direct — can evolve workflow topology |
| **G-Memory** (hierarchical multi-agent memory, talk 2025-09-28) | Research paper | Could replace current memory with graph-structured recall |
| **Agentic RL Policy Optimization** (talk 2025-10-30) | Research | Future evolution signal beyond text-gradient |
| **AlphaEvolve** (branch: `alphaevolve`) | In development | Code-level agent evolution |
| **MAD** (Multi-Agent Debate, branch: `dev_mad`) | In development | Consensus mechanism for multi-agent workflows |
| **UserIntentRouter** (branch: `dev_UserIntentRouter`) | In development | Auto-route incoming requests to the right workflow — directly useful for OpenClaw channel routing |
| **LongTermMemoryAgent** (branch: `dev_long_term_memory_agent`) | In development | Drop-in memory agent for existing workflows |
| **Visual Workflow Editor** | Roadmap | Would integrate with Nerve cockpit naturally |

**Most relevant upstream branch to track**: `dev_UserIntentRouter` — this is exactly what OpenClaw's gateway needs: classify an incoming message and route to the right workflow/agent automatically.

---

## 7. Quick Start Integration Plan

### Week 1: Proof of Concept
1. Install AgentAugi Python package in OpenClaw's Python env
2. Write `AgentAdapter` (100 lines): OpenClaw agent JSON ↔ EvoAgentX Agent
3. Write `BridgeClient` (200 lines): calls AgentAugi REST API from Node.js/OpenClaw
4. Test: take one kanban task, generate workflow, execute, return result

### Week 2: Memory + Tools
5. Wire AugiFlo MCP tools into AgentAugi via `mcp.py` tool wrapper
6. Connect `LongTermMemory` to existing ChromaDB instance
7. Add memory read injection to Augi's task context

### Week 3: Evolution
8. Build task outcome logger (write after each task)
9. Script nightly TextGrad run on last 7 days of outcomes
10. Verify prompt quality improves on held-out test tasks

### Week 4: HITL + Nerve
11. Wire `HITLOutsideConversationAgent` to Nerve WebSocket
12. Build minimal Nerve HITL approval card
13. Test end-to-end: complex task → HITL checkpoint → human approves → completion

---

## 8. AugiFlo vs AgentAugi: Complementary Roles

These are NOT competing systems:

| Concern | AugiFlo | AgentAugi |
|---------|---------|-----------|
| Agent lifecycle | Spawn, monitor, terminate, health | Generate specs, optimize prompts, evolve structure |
| Task distribution | Claims system, load balancing, hive-mind | Workflow DAG decomposition, node assignment |
| Memory | AgentDB (pattern store, semantic route) | LongTermMemory (RAG-backed, per-interaction) |
| Tool execution | MCP server dispatch | Tool-enabled agent nodes in workflows |
| Learning | Neural patterns, autopilot | TextGrad/AFlow/SEW optimization against benchmarks |
| Orchestration | Multi-agent coordination (consensus, topology) | Workflow graph with typed inputs/outputs per node |

**The integration**: AugiFlo spawns and manages the agents that AgentAugi's workflows specify. AgentAugi designs what to build; AugiFlo executes it.

---

*Analysis based on: full codebase read of `evoagentx/` (all modules), `Wonderful_workflow_corpus/` examples, `docs/tutorial/`, README, and OpenClaw `openclaw.json` + agent pool structure.*

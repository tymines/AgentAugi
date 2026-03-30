# AgentAugi — Architecture & Stack Integration Analysis

*Written: 2026-03-29 | Research-only snapshot*

---

## 1. Where AgentAugi Sits in the Stack

AgentAugi (the EvoAgentX fork at `~/AgentAugi/`) is the **evolution and optimization layer** of Augi's stack. Every other component either feeds raw material into it or consumes its outputs.

```
┌─────────────────────────────────────────────────────────────────┐
│                        NERVE (nerve.augiport.com)               │
│              Kanban task management / control UI                │
└───────────────────────────────┬─────────────────────────────────┘
                                │ task cards / HITL panels
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                  OPENCLAW GATEWAY (127.0.0.1:18789)             │
│   ACP session management · agent pool · plugin hooks · skills   │
│   Plugins: AceForge, OpenViking, security-sentinel, LangGraph   │
└──────────┬────────────────────┬────────────────────────────────-┘
           │ ACP dispatch       │ after_tool_call / llm_output hooks
           ▼                    ▼
┌──────────────────┐   ┌────────────────────────────────────────┐
│  AUGIFLO         │   │         AGENTAUGI (~/AgentAugi)        │
│  (~/ruflo-clean) │   │  WorkFlowGenerator · Optimizers        │
│  60+ swarm       │   │  Memory · HITL · Evaluators · RAG      │
│  agents          │   │  EvoSkill · LATS · ModelCascade        │
│  Claude Code     │   │  FastAPI REST app                      │
│  orchestration   │   └──────────────────┬─────────────────────┘
└──────────────────┘                      │ LiteLLM calls
                                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                  AUGIVECTOR (port 3000)                         │
│   DeepSeek Chat/Reasoner (code)  ·  Kimi K2.5 (research)       │
│   Ollama (embeddings/local)                                     │
└─────────────────────────────────────────────────────────────────┘
```

**In one sentence:** AgentAugi is the brain that takes a task, decomposes it into an executable multi-agent DAG, routes model calls through AugiVector, evolves its own prompts/workflows based on outcomes, and surfaces results through OpenClaw back into Nerve.

The critical distinction from the rest of the stack:
- **AugiFlo** executes agents (Claude Code, fixed capabilities, static workflows)
- **AgentAugi** *evolves* those workflows — it is measurement + optimization, not execution

---

## 2. Full Data Flow: Nerve Task → Deployed Result

```
1. TASK INTAKE
   Nerve kanban card created
       │
       ▼
2. OPENCLAW DISPATCH
   OpenClaw Gateway receives task via ACP
   Determines: "this needs multi-step workflow" → routes to AgentAugi API
       │
       ▼
3. WORKFLOW GENERATION (AgentAugi)
   WorkFlowGenerator.generate(task_description)
   → decomposes goal into subtask DAG
   → assigns agent specs to each node
   → returns WorkFlowGraph (JSON-serializable)
       │
       ▼
4. WORKFLOW EXECUTION (AgentAugi → AugiVector)
   WorkFlow.run(graph)
   Each agent node makes LLM calls via LiteLLMModel
   LiteLLM routes to AugiVector (port 3000):
     • Code tasks    → DeepSeek Chat/Reasoner
     • Research/QA   → Kimi K2.5
     • Embeddings    → Ollama
   Tool calls observed by AceForge hooks (after_tool_call)
       │
       ▼
5. EVALUATION (AgentAugi)
   Evaluator scores the result (task-specific metric)
   CostTracker records token spend
   AlignmentDriftDetector checks for drift
       │
       ▼
6. HITL GATE (AgentAugi ↔ OpenClaw ↔ Nerve)
   If confidence < threshold → approval_manager routes to Nerve HITL panel
   Human approves/rejects/edits in Nerve UI
       │
       ▼
7. OUTCOME LOGGING (AgentAugi → OpenViking cold-tier)
   Task outcome, score, cost logged to MistakeNotebook / ReasoningBank
   Summary pushed to OpenViking cold-tier storage (periodic batch)
       │
       ▼
8. EVOLUTION LOOP (AgentAugi, nightly)
   Optimizer runs on accumulated outcomes:
     • TextGrad / MIPRO on agent prompts
     • AFlow / SEW on workflow structure
   Improved prompt/workflow variants stored in DesignArchive
       │
       ▼
9. SKILL CRYSTALLIZATION (AceForge ← AgentAugi EvoSkill)
   Recurring tool-call patterns → AceForge crystallizes skills
   EvoSkill-generated tools → exported to OpenClaw skill registry
   New skills available to all future agents
       │
       ▼
10. RESULT DISPLAY (→ Nerve)
    Task complete, score, cost, workflow trace surfaced in Nerve kanban card
    Evolution improvements logged to dashboard (when built)
```

---

## 3. Integration Points — Detail

### 3a. AgentAugi ↔ AugiVector (Model Routing)

**What's needed:** A YAML or env config pointing AgentAugi's `LiteLLMModel` at `http://localhost:3000`.

**AugiVector routing strategy for AgentAugi:**

| AgentAugi Use Case | Route to | Rationale |
|---|---|---|
| `WorkFlowGenerator` (planning, decomposition) | Kimi K2.5 | Long-context research reasoning |
| `TextGrad / MIPRO` (prompt optimization) | DeepSeek Reasoner | Deep chain-of-thought reasoning |
| `ActionAgent` executing code tasks | DeepSeek Chat | Code specialist |
| `RAG embeddings` | Ollama (nomic-embed-text) | Local, fast, no API cost |
| `Evaluator` scoring | Kimi K2.5 | Research-grade judgment |
| `ModelCascade` (fast tier) | DeepSeek Chat | Cheapest capable model |
| `ModelCascade` (accurate tier) | Kimi K2.5 | Best quality fallback |
| `DifficultyRouter` (hard tasks) | Kimi K2.5 | Highest reasoning ceiling |

**What exists now:** `LiteLLMModel` in `evoagentx/models/litellm_model.py` supports custom base URLs. AugiVector is already registered in `openclaw.json` as a provider. **Config change only — no new code.**

**What's missing:** AgentAugi's example configs point to OpenAI by default. A `config_augivector.yaml` file needs to be written with provider mappings.

---

### 3b. AgentAugi ↔ OpenClaw Gateway (Session Management)

**What exists:**
- OpenClaw Gateway at `127.0.0.1:18789`, token-authenticated
- ACP protocol supports up to 8 concurrent sessions, 30-min TTL
- `openclaw.json` has `tools.sandbox` with `exec`, `web_search`, `web_fetch`, `read`, `write`, `apply_patch`
- AgentAugi has a full **FastAPI REST app** (`evoagentx/app/`) with JWT auth, CRUD, async workflow execution

**Integration path:**
- OpenClaw registers AgentAugi's FastAPI app as an HTTP tool endpoint or ACP plugin
- A thin `openclaw-agentaugi-bridge.py` translates OpenClaw's ACP task format → AgentAugi's `WorkFlowGenerator` API call
- Responses flow back as ACP tool results

**What's ready:** Both sides have HTTP APIs. OpenClaw's `tools` config supports custom endpoints. AgentAugi's `app/api.py` is functional.

**What needs building:** The bridge adapter (task schema translation) and OpenClaw plugin config entry for AgentAugi. Estimated ~1-2 days per OPENCLAW_INTEGRATION_ANALYSIS.md.

---

### 3c. AgentAugi ↔ AceForge (Skill Generation Pipeline)

**What AceForge does:** Watches `after_tool_call` hooks inside OpenClaw, detects recurring tool-call sequences, uses a dual-LLM pipeline (generator + reviewer) to crystallize them into reusable skills. Threshold: 3 recurrences + 70% success rate before crystallizing. Human approval required before deployment.

**What AgentAugi has:** `core/evoskill.py` (`EvoSkill`) and `core/tool_synthesizer.py` (`ToolSynthesizer`). These operate *inside* AgentAugi's workflow execution to generate new tools dynamically during a run.

**The relationship:**
- These are **complementary, not duplicate** — they operate at different layers
- AceForge = **OpenClaw-level** skill crystallization (observes all agents in the gateway)
- EvoSkill = **AgentAugi-internal** tool synthesis (generates tools mid-workflow for a specific task)

**Integration path:**
1. AgentAugi's `EvoSkill` generates a new tool → writes it to a staging file
2. AceForge's `ACEFORGE_CUSTOM_SKILLS_PATH` picks it up and runs it through the quality/review pipeline
3. Approved skills land in `~/.openclaw/workspace/skills/` and become available to all OpenClaw agents
4. AceForge-approved skills can be registered back into AgentAugi's `ToolRegistry`

**What's ready:** AceForge's file-watching and approval hooks exist. AgentAugi's ToolRegistry accepts dynamic registration.

**What needs building:** A file-drop contract (agreed schema between EvoSkill output and AceForge intake format) and a two-way sync script. ~2-3 days.

---

### 3d. AgentAugi ↔ OpenViking (Memory — Cold-Tier Only)

**Prior analysis conclusion:** OpenViking is a **Claude Code context engine** running on port 1933. It is tightly coupled to the Claude Code session lifecycle. AgentAugi is a Python library/service with its own multi-tier memory stack.

**Boundary:** OpenViking handles warm context for the active Claude Code session. AgentAugi handles its own short-term and long-term memory internally during workflow execution.

**The only valid integration point is cold-tier writes:**
- After a workflow completes, AgentAugi's `MistakeNotebook`, `ReasoningBank`, and `PagedMemory` cold-page dumps are written to a path OpenViking can index
- OpenViking makes those summaries available as context when a new Claude Code session starts a related task
- This is periodic/async — not in the hot path

**What's ready:** OpenViking's context engine indexes files from configured paths. AgentAugi's memory modules already serialize to disk.

**What needs building:** A configured export path that OpenViking watches, and a scheduled export job in AgentAugi. Low effort, low risk — correct boundary to enforce.

---

### 3e. AgentAugi ↔ Nerve (Task Tracking & Results Display)

**What Nerve is:** The kanban-based control UI at `nerve.augiport.com`, listed in `openclaw.json` as the `controlUI` for the OpenClaw gateway.

**Integration surfaces:**
1. **Task intake:** Nerve card → OpenClaw → AgentAugi bridge (covered in 3b above)
2. **HITL panel:** AgentAugi's `hitl/approval_manager.py` needs a UI endpoint. Nerve is the natural host. OpenClaw's `gateway.controlUI` is Nerve — the HITL approval queue should surface as a card state in Nerve.
3. **Results display:** Workflow trace, score, cost, and evolution metrics pushed back to the originating Nerve card via the OpenClaw gateway's reverse callback.
4. **Evolution dashboard:** Optimizer improvement history (TextGrad deltas, AFlow pass@1 changes) would live as a Nerve board view or sidebar panel.

**What's ready:** The HITL module exists in AgentAugi (`hitl/hitl_gui.py`, `approval_manager.py`). Nerve is the live UI. The bridge from OpenClaw back to Nerve already exists (controlUI integration).

**What needs building:** An AgentAugi-specific Nerve board template, webhook from `approval_manager` to Nerve card status, and a result-callback endpoint. The HITL GUI could be embedded or replaced with Nerve's native card UI.

---

## 4. What's Ready Now vs What Needs Wiring

### Ready Today (config/config + minimal glue)

| Integration | What exists | Effort |
|---|---|---|
| AgentAugi → AugiVector | `LiteLLMModel` supports base_url override; AugiVector is a registered OpenClaw provider | `config_augivector.yaml` + env vars — 2 hours |
| AgentAugi FastAPI up | `evoagentx/app/` is functional with JWT, CRUD, async execution | `uvicorn app.main:app` — minutes |
| OpenClaw calling AgentAugi | OpenClaw `tools.sandbox` supports HTTP endpoints | Plugin config entry — 1 day |
| OpenViking cold-tier writes | AgentAugi memory serializes to disk; OpenViking watches paths | Configure export path — 4 hours |
| AceForge observing AgentAugi tool calls | AceForge hooks `after_tool_call` at gateway level | Already fires if AgentAugi runs through OpenClaw |

### Needs New Code (wiring work)

| Integration | What's missing | Estimate |
|---|---|---|
| Nerve → AgentAugi task bridge | Schema adapter: Nerve card JSON → WorkFlowGenerator call | 2 days |
| AgentAugi result → Nerve callback | Reverse webhook from workflow completion to Nerve card | 1 day |
| HITL → Nerve card state | approval_manager webhook → Nerve card status change | 2 days |
| EvoSkill → AceForge file-drop | Agreed skill schema + file-drop contract | 2 days |
| AceForge skills → AgentAugi ToolRegistry | Reverse sync: approved skills registered in ToolRegistry | 1 day |
| Nightly evolution scheduler | Cron job: run optimizer on accumulated outcome logs | 1-2 days |
| Evolution dashboard in Nerve | New board view for optimizer metrics | 3 days |
| AugiVector routing config in AgentAugi | `config_augivector.yaml` with model-to-task mappings | 4 hours |

### Not Yet Started (Phase 0 prerequisites)

The `MASTER_IMPLEMENTATION_PLAN.md` Phase 0 items must land first before integrations deliver real value:
- Evaluation harness (TAU-bench, SWE-bench, GAIA, AgentBench) — without this, evolution has no objective signal
- Cost tracking layer — without this, AugiVector routing optimization is blind
- Async concurrency fixes — without this, multi-node workflow execution will have race conditions
- Fix `rag/rag.py` unbounded memory growth — required before any long-running workflow

---

## 5. Recommended Integration Order

### Step 1 — AugiVector (Days 1-2, maximum ROI)
**Do this first.** It unlocks cost-efficient model routing for every subsequent integration. Write `config_augivector.yaml`, set `LITELLM_BASE_URL=http://localhost:3000`, map task types to models. All AgentAugi LLM calls then go through the right models at local-first cost.

### Step 2 — Phase 0 infrastructure (Weeks 1-4, prerequisite unlock)
Evaluation harness + cost tracking + async fixes. Without these, the evolution loop has no signal and no safety. These are internal to AgentAugi — no external wiring needed.

### Step 3 — OpenClaw → AgentAugi task bridge (Week 2-3, core value unlock)
The bridge that lets Nerve tasks flow into AgentAugi's WorkFlowGenerator. This is the moment AgentAugi stops being a standalone library and becomes part of the live stack. Thin Python adapter, ~2 days of coding.

### Step 4 — AceForge ↔ EvoSkill skill pipeline (Week 3-4)
Once tasks are flowing through AgentAugi via OpenClaw, AceForge's hooks are already firing. Define the file-drop schema for EvoSkill → AceForge handoff. The skill crystallization loop starts producing reusable skills for the whole stack.

### Step 5 — Nerve HITL panel (Week 4-5)
Wire `approval_manager.py` to Nerve card state changes. This surfaces AgentAugi's human-oversight capability in the existing UI without building anything new. The existing Nerve control UI integration in OpenClaw is the hook.

### Step 6 — Nightly evolution scheduler (Week 5-6)
OpenClaw's `cron/` directory already has 21 scheduled task directories. Add an AgentAugi evolution job here — runs the optimizer (start with TextGrad/MIPRO, safest) on accumulated outcome logs from the day's tasks.

### Step 7 — OpenViking cold-tier memory (Week 6)
Configure AgentAugi's memory export path → OpenViking's indexed path. Lowest effort, lowest risk integration. Provides cross-session context seeding.

### Step 8 — Evolution dashboard in Nerve (Week 7+)
Nice-to-have visualization of optimizer improvement metrics. Build only after the evolution loop is producing data worth displaying.

---

## 6. Architectural Risks & Notes

**AugiFlo stays separate.** AugiFlo (`~/ruflo-clean`) is Claude Code–only orchestration. AgentAugi is Python library + FastAPI. They serve different layers — AugiFlo executes Claude Code agents, AgentAugi evolves workflows. Do not attempt to merge them. The connection point is OpenClaw routing tasks to *either* AugiFlo (for simple Claude Code tasks) *or* AgentAugi (for tasks needing workflow generation or evolution).

**Phase 0 before Phase N.** The MASTER_IMPLEMENTATION_PLAN.md is explicit: no optimizer integration delivers value without a measurement layer. Wiring AgentAugi to the live stack before Phase 0 is done means connecting an unvalidated engine to production traffic. The safe sequence is: measure first, evolve second, integrate third.

**AceForge and EvoSkill must not compete.** AceForge has human-approval gates and crystallization thresholds for good reason — auto-deployed skills are a security surface. EvoSkill should write to a *staging* path that goes through AceForge's review pipeline, not directly to the skill registry.

**OpenClaw concurrent session limit.** The gateway allows max 8 concurrent ACP sessions. AgentAugi's parallel executor can spawn many sub-agents during workflow execution. These sub-agents need to be accounted for in the session budget or run outside the ACP session pool (direct API calls through AugiVector instead).

**Cost accountability from day one.** `evoagentx/core/cost_tracker.py` exists. Wire it to every LLM call before any production use. AugiVector billing routes to paid providers (DeepSeek, Kimi) — unbounded evolution loops could generate unexpected costs without this gate.

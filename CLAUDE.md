# AgentAugi — Claude Code Session Context

AgentAugi is a **self-evolving agent framework** (EvoAgentX fork) that serves as
the evolution and optimization engine for the Augi stack.

## Stack at a glance

| Component | Location | Role |
|-----------|----------|------|
| AgentAugi | `~/AgentAugi/` | Evolution engine (this repo) |
| AugiFlo / ruflo | `~/ruflo-clean/` | Orchestration layer |
| AugiVector | `localhost:3000` | LLM routing proxy (OpenAI-compatible) |
| OpenClaw Gateway | `localhost:18789` | ACP session manager / task dispatcher |
| Nerve | `localhost:3080` | Kanban HITL UI |
| AceForge | `~/.openclaw/workspace/.forge/` | Tool/skill forge |

## 7 modules added to EvoAgentX

| Module | Import | Purpose |
|--------|--------|---------|
| MASTER Search | `evoagentx.core.master_search` | MCTS + LLM self-eval (vs LATS) |
| HyperAgents | `evoagentx.optimizers.hyper_optimizer` | Evolve optimizer strategies |
| Plan Cache | `evoagentx.core.plan_cache` | Semantic plan reuse |
| PASTE | `evoagentx.core.speculative_executor` | Speculative step prefetch |
| EvoSkill | `evoagentx.core.evoskill` | Failure → skill synthesis |
| JitRL | `evoagentx.memory.jitrl` | Statistical action biasing |
| Integrations | `evoagentx.integrations` | OpenClaw / AceForge / Nerve bridges |

## Integration modules

```python
from evoagentx.integrations import (
    OpenClawBridge, OpenClawConfig,          # configs/openclaw.yaml
    AceForgeConnector, AceForgeConnectorConfig,  # configs/aceforge.yaml
    NerveHITL, NerveConfig,                  # configs/nerve.yaml
)
```

- **OpenClawBridge** — receives tasks from Gateway, executes via WorkFlowGenerator,
  routes LLM calls through AugiVector
- **AceForgeConnector** — bidirectional JSONL file exchange for skill gaps/discoveries
- **NerveHITL** — submits evolution proposals to Nerve kanban, polls for approval

## Nightly evolution

```bash
python scripts/nightly_evolution.py             # uses configs/nightly_evolution.yaml
python scripts/nightly_evolution.py --dry-run   # simulate without LLM calls
bash scripts/install_cron.sh                    # schedule at 2 AM daily
```

Outputs to `data/evolution/` — reports, population state, JitRL memory, plan cache.

## Full integration guide

See `docs/AUGI_INTEGRATION_GUIDE.md` for:
- Complete module catalog with constructors and examples
- Config reference for all 4 YAML files
- Step-by-step workflow examples
- Architecture diagram and data-flow maps

## Key config files

| File | Controls |
|------|----------|
| `configs/openclaw.yaml` | Gateway connection, LLM profile routing |
| `configs/aceforge.yaml` | AceForge file exchange paths and thresholds |
| `configs/nerve.yaml` | Nerve connection, HITL approval thresholds |
| `configs/nightly_evolution.yaml` | Evolution params, budget, cron schedule |

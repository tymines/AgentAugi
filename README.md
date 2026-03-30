<div align="center">
  <h1>AgentAugi</h1>
  <p><strong>Self-evolving AI agents with MASTER search, HyperAgents meta-optimization, speculative execution, and JitRL learning</strong></p>
</div>

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-tymines%2FAgentAugi-blue?logo=github)](https://github.com/tymines/AgentAugi)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

</div>

<div align="center">

<a href="./README.md" style="text-decoration: underline;">English</a> | <a href="./README-zh.md">简体中文</a>

</div>

---

## What is AgentAugi

AgentAugi is a self-evolving AI agent framework built on top of [EvoAgentX](https://github.com/EvoAgentX/EvoAgentX) — forked and extended with seven new modules that push beyond the base framework. Where EvoAgentX provides the foundation (workflow generation, evaluation, optimization, memory, HITL), AgentAugi adds a second layer: agents that search better, learn faster, execute speculatively, and evolve their own optimization strategies.

AgentAugi is part of the broader Augi stack (OpenClaw, AugiFlo, AugiVector, Nerve), designed for production deployment of continuously improving agent systems.

### What AgentAugi Adds (7 New Modules)

| Module | What It Does |
|--------|-------------|
| **MASTER** | Monte-Carlo tree search with self-refinement for deeper, more accurate agent reasoning |
| **HyperAgents** | Meta-optimization layer — a `HyperOptimizer` that evolves optimizer strategies themselves via LLM-guided mutation and crossover |
| **Plan Caching** | Semantic cache for agent plans — reuses proven solutions, cuts redundant LLM calls |
| **PASTE** | Speculative execution — prefetches likely next steps in parallel to reduce latency |
| **EvoSkill** | Distills successful workflows into reusable, versioned tool skills |
| **JitRL** | Just-in-time reinforcement learning — agents improve from task outcomes in real time |
| **Tool-MAD** | Multi-agent debate for tool selection — agents vote on which tools to invoke |

### Foundation Features (from EvoAgentX)

- **Workflow Autoconstruction** — build structured multi-agent workflows from a single natural language goal
- **Built-in Evaluation** — score agent behavior automatically against task-specific criteria
- **Self-Evolution Algorithms** — TextGrad, AFlow, MIPRO, EvoPrompt and more, ready to optimize workflows
- **Plug-and-Play Models** — OpenAI, Claude, DeepSeek, Kimi, Qwen, Ollama, and any LiteLLM-supported model
- **Comprehensive Built-in Tools** — code execution, search, filesystem, databases, image tools, browser automation
- **Memory Module** — short-term and long-term memory for persistent agent state
- **Human-in-the-Loop (HITL)** — pause workflows for human approval at any step


## Installation

Install from source (recommended):

```bash
git clone https://github.com/tymines/AgentAugi.git
cd AgentAugi
pip install -e .
```

Or install directly via pip:

```bash
pip install git+https://github.com/tymines/AgentAugi.git
```

For local development with conda:

```bash
git clone https://github.com/tymines/AgentAugi.git
cd AgentAugi
conda create -n agentaugi python=3.11
conda activate agentaugi
pip install -e .
```


## LLM Configuration

### API Key Setup

```bash
# Linux/macOS
export OPENAI_API_KEY=<your-openai-api-key>

# Windows Command Prompt
set OPENAI_API_KEY=<your-openai-api-key>

# Windows PowerShell
$env:OPENAI_API_KEY="<your-openai-api-key>"
```

Or use a `.env` file:

```bash
OPENAI_API_KEY=<your-openai-api-key>
```

### Configure and Use the LLM

```python
from evoagentx.models import OpenAILLMConfig, OpenAILLM
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_config = OpenAILLMConfig(
    model="gpt-4o-mini",
    openai_key=OPENAI_API_KEY,
    stream=True,
    output_response=True
)

llm = OpenAILLM(config=openai_config)
response = llm.generate(prompt="What is Agentic Workflow?")
```


## Automatic Workflow Generation

```python
from evoagentx.workflow import WorkFlowGenerator, WorkFlowGraph, WorkFlow
from evoagentx.agents import AgentManager

goal = "Generate html code for the Tetris game"
workflow_graph = WorkFlowGenerator(llm=llm).generate_workflow(goal)

agent_manager = AgentManager()
agent_manager.add_agents_from_workflow(workflow_graph, llm_config=openai_config)

workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
output = workflow.execute()
print(output)
```

- Visualize the workflow: `workflow_graph.display()`
- Save/load workflows: `save_module()` / `from_file()`


## Built-in Tools

AgentAugi inherits a comprehensive tool suite for agent interactions with real-world environments:

<details>
<summary>Click to expand full tool table</summary>

<br>

| Toolkit | Description | Code |
|---------|-------------|------|
| **Code Interpreters** | | |
| PythonInterpreterToolkit | Sandboxed Python execution | [link](evoagentx/tools/interpreter_python.py) |
| DockerInterpreterToolkit | Isolated Docker container execution | [link](evoagentx/tools/interpreter_docker.py) |
| **Search & HTTP** | | |
| WikipediaSearchToolkit | Wikipedia search with full content | [link](evoagentx/tools/search_wiki.py) |
| GoogleSearchToolkit | Google Custom Search (API key required) | [link](evoagentx/tools/search_google.py) |
| GoogleFreeSearchToolkit | Google-style search without API credentials | [link](evoagentx/tools/search_google_f.py) |
| DDGSSearchToolkit | DuckDuckGo search | [link](evoagentx/tools/search_ddgs.py) |
| SerpAPIToolkit | Multi-engine search via SerpAPI | [link](evoagentx/tools/search_serpapi.py) |
| ArxivToolkit | arXiv paper search | [link](evoagentx/tools/request_arxiv.py) |
| RequestToolkit | General HTTP client (GET/POST/PUT/DELETE) | [link](evoagentx/tools/request.py) |
| RSSToolkit | RSS feed fetching with content extraction | [link](evoagentx/tools/rss_feed.py) |
| **Filesystem** | | |
| StorageToolkit | File I/O (save/read/append/delete/list) | [link](evoagentx/tools/storage_file.py) |
| CMDToolkit | Shell command execution | [link](evoagentx/tools/cmd_toolkit.py) |
| FileToolkit | File and directory management | [link](evoagentx/tools/file_tool.py) |
| **Databases** | | |
| MongoDBToolkit | MongoDB queries and aggregations | [link](evoagentx/tools/database_mongodb.py) |
| PostgreSQLToolkit | PostgreSQL operations | [link](evoagentx/tools/database_postgresql.py) |
| FaissToolkit | Vector similarity search (FAISS) | [link](evoagentx/tools/database_faiss.py) |
| **Image Tools** | | |
| ImageAnalysisToolkit | Vision analysis (GPT-4o) | [link](evoagentx/tools/OpenAI_Image_Generation.py) |
| OpenAIImageGenerationToolkit | Text-to-image via DALL-E | [link](evoagentx/tools/OpenAI_Image_Generation.py) |
| FluxImageGenerationToolkit | Text-to-image via Flux Kontext | [link](evoagentx/tools/flux_image_generation.py) |
| **Browser** | | |
| BrowserToolkit | Fine-grained browser automation | [link](evoagentx/tools/browser_tool.py) |
| BrowserUseToolkit | LLM-driven natural language browser control | [link](evoagentx/tools/browser_use.py) |

</details>

MCP tools are also supported — see the [MCP tutorial](docs/tutorial/mcp.md).


## Tool-Enabled Workflows

Provide tools directly to `WorkFlowGenerator` and agents will use them automatically:

```python
from evoagentx.tools import ArxivToolkit

arxiv_toolkit = ArxivToolkit()

wf_generator = WorkFlowGenerator(llm=llm, tools=[arxiv_toolkit])
workflow_graph = wf_generator.generate_workflow(
    goal="Find and summarize the latest research on AI in finance on arXiv"
)

agent_manager = AgentManager(tools=[arxiv_toolkit])
agent_manager.add_agents_from_workflow(workflow_graph, llm_config=openai_config)

workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm)
output = workflow.execute()
```


## Human-in-the-Loop (HITL)

```python
from evoagentx.hitl import HITLManager, HITLInterceptorAgent, HITLInteractionType, HITLMode

hitl_manager = HITLManager()
hitl_manager.activate()

interceptor = HITLInterceptorAgent(
    target_agent_name="DataSendingAgent",
    target_action_name="DummyEmailSendAction",
    interaction_type=HITLInteractionType.APPROVE_REJECT,
    mode=HITLMode.PRE_EXECUTION
)
hitl_manager.hitl_input_output_mapping = {"human_verified_data": "extracted_data"}

agent_manager.add_agent(interceptor)
workflow = WorkFlow(graph=workflow_graph, agent_manager=agent_manager, llm=llm, hitl_manager=hitl_manager)
```


## Evolution Algorithms

| Algorithm | Description | Paper |
|-----------|-------------|-------|
| **TextGrad** | Gradient-based prompt optimization | [Nature (2025)](https://www.nature.com/articles/s41586-025-08661-4) |
| **MIPRO** | Model-agnostic iterative prompt optimization | [arXiv:2406.11695](https://arxiv.org/abs/2406.11695) |
| **AFlow** | MCTS-based agent workflow evolution | [arXiv:2410.10762](https://arxiv.org/abs/2410.10762) |
| **EvoPrompt** | Feedback-driven evolutionary prompt refinement | [arXiv:2309.08532](https://arxiv.org/abs/2309.08532) |
| **HyperAgents** | Meta-optimizer that evolves optimizer strategies (AgentAugi) | [ARCHITECTURE_INTEGRATION.md](./ARCHITECTURE_INTEGRATION.md) |

### Benchmark Results (EvoAgentX baseline algorithms)

| Method | HotPotQA (F1%) | MBPP (Pass@1%) | MATH (Solve Rate%) |
|--------|----------------|----------------|---------------------|
| Original | 63.58 | 69.00 | 66.00 |
| TextGrad | 71.02 | 71.00 | 76.00 |
| AFlow | 65.09 | 79.00 | 71.00 |
| MIPRO | 69.16 | 68.00 | 72.30 |


## Tutorial and Use Cases

> New to the framework? Start with the [Quickstart Guide](./docs/quickstart.md).

| Cookbook | Description |
|:---------|:------------|
| **[Build Your First Agent](./docs/tutorial/first_agent.md)** | Create and manage agents with multi-action capabilities |
| **[Build Your First Workflow](./docs/tutorial/first_workflow.md)** | Build collaborative workflows with multiple agents |
| **[Working with Tools](./docs/tutorial/tools.md)** | Master the tool ecosystem for agent interactions |
| **[Automatic Workflow Generation](./docs/quickstart.md#automatic-workflow-generation-and-execution)** | Generate workflows from natural language goals |
| **[Benchmark and Evaluation](./docs/tutorial/benchmark_and_evaluation.md)** | Evaluate agent performance on benchmark datasets |
| **[TextGrad Optimizer](./docs/tutorial/textgrad_optimizer.md)** | Optimize prompts within multi-agent workflows |
| **[AFlow Optimizer](./docs/tutorial/aflow_optimizer.md)** | Optimize both prompts and workflow structure |
| **[Human-In-The-Loop](./docs/tutorial/hitl.md)** | Enable HITL in your workflows |


## Roadmap

- [ ] **MASTER integration** — wire Monte-Carlo tree search into WorkFlow execution
- [ ] **JitRL online training loop** — real-time RL from task outcome signals
- [ ] **EvoSkill registry** — persistent skill store with versioning and quality gating
- [ ] **HyperAgents production harness** — full meta-optimization pipeline with scheduler
- [ ] **PASTE latency benchmarks** — measure speculative execution gains on real workloads
- [ ] **OpenClaw bridge** — task intake and result callbacks via ACP plugin
- [ ] **Visual workflow editor** — display and edit workflow structure interactively
- [ ] **Modular evolution algorithms** — plug-and-play optimizer interface


## Contributing

Contributions are welcome. Please open an issue or pull request on [tymines/AgentAugi](https://github.com/tymines/AgentAugi).


## Acknowledgements

AgentAugi is forked from [EvoAgentX](https://github.com/EvoAgentX/EvoAgentX) by Wang et al. (2025). The seven AgentAugi modules are built on top of its workflow, evaluation, and optimization foundations. Additional open-source foundations include [AFlow](https://github.com/FoundationAgents/MetaGPT/tree/main/metagpt/ext/aflow), [TextGrad](https://github.com/zou-group/textgrad), [DSPy](https://github.com/stanfordnlp/dspy), and [EvoPrompt](https://github.com/beeevita/EvoPrompt).


## Citation

If you use AgentAugi in your work, please also cite the EvoAgentX foundation it builds on:

```bibtex
@article{wang2025evoagentx,
  title={EvoAgentX: An Automated Framework for Evolving Agentic Workflows},
  author={Wang, Yingxu and Liu, Siwei and Fang, Jinyuan and Meng, Zaiqiao},
  journal={arXiv preprint arXiv:2507.03616},
  year={2025}
}
```


## License

Source code in this repository is made available under the [MIT License](./LICENSE).

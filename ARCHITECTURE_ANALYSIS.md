# AgentAugi / EvoAgentX — Architecture Analysis

> Deep analysis of the AgentAugi codebase (forked from EvoAgentX).
> Generated: 2026-03-29

---

## Table of Contents

1. [Overview & Philosophy](#1-overview--philosophy)
2. [Repository Structure](#2-repository-structure)
3. [Foundational Architecture: Core Module System](#3-foundational-architecture-core-module-system)
4. [Agent Architecture](#4-agent-architecture)
5. [Action System](#5-action-system)
6. [Workflow System](#6-workflow-system)
7. [LLM & Model System](#7-llm--model-system)
8. [Memory System](#8-memory-system)
9. [Tool System](#9-tool-system)
10. [Optimization & Self-Evolution](#10-optimization--self-evolution)
11. [Evaluation & Benchmarking](#11-evaluation--benchmarking)
12. [Human-in-the-Loop (HITL)](#12-human-in-the-loop-hitl)
13. [RAG System](#13-rag-system)
14. [FastAPI Application](#14-fastapi-application)
15. [Key Design Patterns](#15-key-design-patterns)
16. [End-to-End Data Flow](#16-end-to-end-data-flow)
17. [Class Hierarchy Summary](#17-class-hierarchy-summary)
18. [Concurrency & Threading](#18-concurrency--threading)
19. [Configuration Management](#19-configuration-management)
20. [Key Takeaways & Extension Points](#20-key-takeaways--extension-points)

---

## 1. Overview & Philosophy

EvoAgentX (and by extension AgentAugi) is an open-source framework for building, evaluating, and **automatically evolving** LLM-based agents and multi-agent workflows. The design philosophy rests on four pillars:

- **Self-Evolution**: Agents and workflows improve continuously through iterative optimization loops, using multiple published algorithms (TextGrad, MIPRO, AFlow, EvoPrompt, SEW).
- **Modularity**: Every component—agents, actions, tools, optimizers, memory backends—is a plug-and-play `BaseModule` subclass.
- **Automation**: Natural-language goals are automatically decomposed into multi-agent workflows with assigned agents.
- **LLM-Agnosticism**: Multiple LLM providers are supported (OpenAI, Azure, LiteLLM, SiliconFlow, OpenRouter) with a single interface.

---

## 2. Repository Structure

```
evoagentx/
├── core/               # Foundation: BaseModule, Message, registries
├── agents/             # Agent classes, agent manager
├── actions/            # Action definitions, input/output schemas
├── workflow/           # WorkflowGraph, WorkflowManager, WorkflowGenerator
├── models/             # LLM configs and implementations
├── memory/             # Short-term and long-term memory
├── tools/              # 40+ tool toolkits
├── optimizers/         # TextGrad, MIPRO, AFlow, EvoPrompt, SEW
├── benchmark/          # Datasets: HotPotQA, MBPP, MATH, GSM8K, etc.
├── evaluators/         # Evaluator class, metric functions
├── hitl/               # Human-in-the-loop interceptors
├── rag/                # Retrieval-augmented generation engine
├── storages/           # Storage backends (MongoDB, PostgreSQL, file)
├── app/                # FastAPI REST API
└── utils/              # Logging, IO, path utilities
```

---

## 3. Foundational Architecture: Core Module System

### 3.1 `BaseModule` (`core/module.py`)

The root base class for **every** object in EvoAgentX. It subclasses Pydantic's `BaseModel` and uses a custom `MetaModule` metaclass that automatically registers every subclass in a global `MODULE_REGISTRY`, enabling transparent serialization and deserialization by class name.

```python
class BaseModule(BaseModel, metaclass=MetaModule):
    class_name: str = None
    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow",
        ...
    }

    def init_module(self):
        """Override in subclasses for custom initialization logic."""
        pass

    # Serialization
    def to_dict(self) -> dict: ...
    def to_json(self) -> str: ...
    @classmethod
    def from_dict(cls, data: dict): ...
    @classmethod
    def from_json(cls, s: str): ...
    @classmethod
    def from_file(cls, path: str): ...
    def save_module(self, path: str): ...
```

**Key capabilities:**
- Deep copy with fallback to shallow copy for non-copyable attributes.
- Exception buffering via a callback manager during `__init__`.
- All subclasses auto-registered, so `BaseModule.from_dict({"class_name": "OpenAILLM", ...})` reconstructs the correct type at runtime.

### 3.2 `BaseConfig` (`core/base_config.py`)

Parent class for all configuration objects. Adds `save()`, `get_config_params()`, and `get_set_params()` (returns only explicitly set fields, not defaults).

### 3.3 `Message` (`core/message.py`)

The universal data-passing currency throughout the system.

```python
class Message(BaseModule):
    content: Any
    agent: Optional[str]
    action: Optional[str]
    prompt: Optional[str]
    next_actions: Optional[List[str]]
    msg_type: MessageType   # REQUEST, RESPONSE, COMMAND, ERROR, INPUT, UNKNOWN
    wf_goal: Optional[str]
    wf_task: Optional[str]
    message_id: str
    timestamp: float
```

Static helpers: `sort_by_timestamp()`, `sort()`, `merge()`.

### 3.4 Registries (`core/registry.py`)

Four global registries:

| Registry | Purpose |
|---|---|
| `ModuleRegistry` | Maps class name → class for all `BaseModule` subclasses |
| `ModelRegistry` | Maps LLM type string → (LLM class, config class) |
| `ParseFunctionRegistry` | Maps name → custom parse function |
| `ActionFunctionRegistry` | Maps name → callable for `ActionAgent` serialization |

This registry system is what makes the entire framework dynamically extensible—load any component from JSON/YAML without knowing the concrete type at call time.

---

## 4. Agent Architecture

### 4.1 `Agent` (`agents/agent.py`)

```python
class Agent(BaseModule):
    name: str                               # Unique identifier
    description: str
    llm_config: Optional[LLMConfig]
    llm: Optional[BaseLLM]
    agent_id: Optional[str]
    system_prompt: Optional[str]
    short_term_memory: Optional[ShortTermMemory]
    use_long_term_memory: bool = False
    long_term_memory: Optional[LongTermMemory]
    actions: List[Action]
    n: int                                  # # of recent messages for context
    is_human: bool = False
    version: int = 0
```

**Execution lifecycle:**

```python
def execute(
    self,
    action_name: str,
    msgs: Optional[List[Message]] = None,
    action_input_data: Optional[dict] = None,
    **kwargs
) -> Message:
    # 1. _prepare_execution(): add msgs to short-term memory,
    #    extract action inputs from memory context
    # 2. Execute the named Action
    # 3. _create_output_message(): wrap result, update memories
```

Both sync (`execute()`) and async (`async_execute()`) variants are provided throughout.

### 4.2 `ActionAgent` (`agents/action_agent.py`)

Executes a Python callable directly—no LLM involved. Useful for deterministic steps in a workflow.

```python
ActionAgent(
    name="data_loader",
    description="Loads data from CSV",
    inputs=[{"name": "path", "type": "str", "description": "...", "required": True}],
    outputs=[{"name": "data", "type": "dict", "description": "..."}],
    execute_func=my_load_function,
    async_execute_func=my_async_load_function,  # optional
)
```

Dynamically creates typed Pydantic `ActionInput`/`ActionOutput` models at construction time using `create_model()`.

### 4.3 `CustomizeAgent` (`agents/customize_agent.py`)

Loads agent configuration from dictionaries or persisted storage. Useful for runtime-configured agents without code changes.

### 4.4 Specialized Agents for Workflow Generation

| Agent | Role |
|---|---|
| `TaskPlanner` | Breaks a goal into ordered subtasks |
| `AgentGenerator` | Assigns/creates agents for each subtask |
| `WorkFlowReviewer` | Reviews and critiques workflow structure |

These three agents work in concert inside `WorkFlowGenerator` (see §6.3).

### 4.5 `AgentManager` (`agents/agent_manager.py`)

Thread-safe lifecycle manager for a fleet of agents.

```python
class AgentManager(BaseModule):
    agents: List[Agent]
    agent_states: Dict[str, AgentState]   # AVAILABLE or RUNNING
    storage_handler: Optional[StorageHandler]
    tools: Optional[List[Toolkit]]
```

**Responsibilities:**
- Create, add, remove, retrieve agents by name.
- Track per-agent state to prevent concurrent reuse.
- Distribute registered tools to agent actions.
- Persist/load agents via `StorageHandler`.
- Uses `@atomic_method` (locking decorator) for thread safety.

---

## 5. Action System

### 5.1 `Action` (`actions/action.py`)

The atomic unit of agent behavior.

```python
class Action(BaseModule):
    name: str
    description: str
    prompt: Optional[str]
    prompt_template: Optional[PromptTemplate]
    tools: Optional[List[Toolkit]]
    inputs_format: Optional[Type[ActionInput]]    # Pydantic model class
    outputs_format: Optional[Type[Parser]]        # Pydantic model class
```

### 5.2 `ActionInput` / `ActionOutput`

Both inherit from `LLMOutputParser` (a Pydantic model with parsing capabilities).

`ActionInput` provides:
- `get_input_specification()` → JSON spec used in prompts
- `get_required_input_names()` → list of required fields

`ActionOutput`/`Parser` provides:
- Structured output parsing in multiple modes: `"str"`, `"json"`, `"xml"`, `"title"`, `"custom"`
- Methods: `_parse_json_content()`, `_parse_xml_content()`, `_parse_title_content()`

The parse mode controls how the LLM's raw text is converted to structured fields—a key mechanism for reliable extraction.

---

## 6. Workflow System

### 6.1 `WorkFlowGraph` (`workflow/workflow_graph.py`)

The static structure of a workflow: a directed acyclic graph (DAG).

```python
class WorkFlowNode(BaseModule):
    name: str
    description: str
    inputs: List[Parameter]
    outputs: List[Parameter]
    reason: Optional[str]
    agents: Optional[List[Union[str, dict, Agent]]]
    action_graph: Optional[ActionGraph]
    status: WorkFlowNodeState   # PENDING, RUNNING, COMPLETED, FAILED

class WorkFlowEdge(BaseModule):
    source: str
    target: str
    condition: Optional[str]

class WorkFlowGraph(BaseModule):
    goal: str
    nodes: List[WorkFlowNode]
    edges: List[WorkFlowEdge]
    entry_point: Optional[str]
    exit_point: Optional[str]
```

**Key graph methods:**
- `add_node()`, `remove_node()`, `get_node()`
- `add_edge()`, `remove_edge()`
- `find_initial_nodes()` — nodes with no incoming edges
- `find_unvisited_nodes()` — PENDING nodes whose dependencies are all COMPLETED
- `find_blocked_nodes()` — nodes with a FAILED dependency
- `_validate_workflow_structure()` — checks DAG validity, connectivity, I/O consistency
- `display()` — ASCII/rich visualization
- `get_workflow_description()` — text summary for LLM prompts

### 6.2 `ActionGraph` (`workflow/action_graph.py`)

A sub-graph within a single workflow node. Where `WorkFlowGraph` coordinates agents, `ActionGraph` coordinates the internal multi-step logic of one node.

```python
class ActionGraph(BaseModule):
    name: str
    description: str
    llm_config: LLMConfig

    def execute(self, *args, **kwargs) -> dict: ...
    async def async_execute(self, *args, **kwargs) -> dict: ...
```

**`QAActionGraph`** is a concrete example: it uses `AnswerGenerate` + `QAScEnsemble` operators to implement self-consistency decoding (multiple LLM samples → majority vote).

### 6.3 `WorkFlowGenerator` (`workflow/workflow_generator.py`)

Converts a natural-language goal into a complete `WorkFlowGraph`:

```python
class WorkFlowGenerator(BaseModule):
    llm: Optional[BaseLLM]
    task_planner: Optional[TaskPlanner]
    agent_generator: Optional[AgentGenerator]
    workflow_reviewer: Optional[WorkFlowReviewer]
    num_turns: Optional[PositiveInt]
    tools: Optional[List[Toolkit]]

def generate_workflow(
    self,
    goal: str,
    existing_agents: Optional[List[Agent]] = None,
    retry: int = 1,
    **kwargs
) -> WorkFlowGraph: ...
```

**Generation pipeline:**

```
Goal (str)
  ↓ TaskPlanner
Plan (subtasks + dependencies)
  ↓ build_workflow_from_plan()
WorkFlowGraph (structure only)
  ↓ AgentGenerator
WorkFlowGraph (agents assigned)
  ↓ [num_turns iterations]
  WorkFlowReviewer → feedback
  Incorporate feedback
  ↓
Final WorkFlowGraph
```

### 6.4 `WorkFlow` (`workflow/workflow.py`)

The runtime executor of a `WorkFlowGraph`.

```python
class WorkFlow(BaseModule):
    graph: WorkFlowGraph
    llm: Optional[BaseLLM]
    agent_manager: AgentManager
    workflow_manager: WorkFlowManager
    environment: Environment
    workflow_id: str
    max_execution_steps: int
    hitl_manager: Optional[HITLManager]
```

**Execution flow:**

```python
async def async_execute(self, inputs: dict = {}, **kwargs) -> str:
    # 1. Merge inputs into environment
    # 2. Validate workflow structure
    # 3. Initialize environment with workflow goal
    # 4. Loop (up to max_execution_steps):
    #    a. workflow_manager.schedule_next_task() → picks next PENDING node
    #    b. execute_task(task_node)
    #       - schedule_next_action()
    #       - action_graph.execute() OR agent.execute()
    #       - update environment with outputs
    #       - check HITL intercepts
    #    c. mark node COMPLETED
    #    d. check graph.is_complete
    # 5. workflow_manager.extract_output() → final string result
```

### 6.5 `WorkFlowManager` (`workflow/workflow_manager.py`)

Uses an LLM to make dynamic scheduling decisions:
- `schedule_next_task()` — selects which pending node to execute next (considering dependencies)
- `schedule_next_action()` — selects which action to run within a node
- `extract_output()` — synthesizes the final output from the environment

---

## 7. LLM & Model System

### 7.1 Config Hierarchy

```
BaseConfig
└── LLMConfig
    ├── OpenAILLMConfig       (OpenAI direct)
    ├── AzureOpenAIConfig     (Azure OpenAI)
    ├── LiteLLMConfig         (multi-provider via LiteLLM)
    ├── SiliconFlowConfig     (SiliconFlow API)
    └── OpenRouterConfig      (OpenRouter API)
```

Common fields across configs:
- `llm_type`, `model`, `output_response`
- Generation: `temperature`, `max_tokens`, `top_p`, `stream`
- Tool calling: `tools`, `tool_choice`, `parallel_tool_calls`
- Advanced: `reasoning_effort`, `logprobs`, `prediction`

### 7.2 `BaseLLM` (`models/base_model.py`)

```python
class BaseLLM(BaseModule):
    config: LLMConfig

    def generate(
        self,
        prompt: Union[str, List[dict]],
        system_prompt: Optional[str] = None,
        parse_mode: str = "str",
        return_prompt: bool = False,
        **kwargs
    ) -> Union[str, Parser]: ...

    async def async_generate(...) -> Union[str, Parser]: ...
```

### 7.3 `LLMOutputParser`

Base class for all structured outputs. Wraps an LLM string response and parses it according to `parse_mode`:

| Mode | Mechanism |
|---|---|
| `"str"` | Raw string passthrough |
| `"json"` | Extract JSON block from markdown fences or raw JSON |
| `"xml"` | Extract XML elements by tag |
| `"title"` | Extract sections by `## Title` headings |
| `"custom"` | User-registered parse function |

### 7.4 Implementations

- `OpenAILLM` — direct OpenAI API
- `LiteLLM` — wraps LiteLLM for 100+ providers
- `SiliconFlowLLM`
- `OpenRouterLLM`

All support streaming and tool/function calling.

---

## 8. Memory System

### 8.1 `BaseMemory` (`memory/memory.py`)

```python
class BaseMemory(BaseModule):
    messages: List[Message]
    memory_id: str
    capacity: Optional[PositiveInt]

    def add_message(self, message: Message): ...
    def add_messages(self, messages: List[Message]): ...
    def remove_message(self, message: Message): ...
    def clear(self): ...
```

### 8.2 `ShortTermMemory`

- Ephemeral, per-execution session memory.
- Stores the recent `n` messages for action context extraction.
- Wiped after each workflow execution completes.

### 8.3 `LongTermMemory` (`memory/long_term_memory.py`)

- Persistent across workflow executions and sessions.
- Backed by a storage handler (MongoDB, PostgreSQL).
- Uses vector embeddings for semantic retrieval.

```python
def add(self, messages: List[Message], **kwargs): ...
def search(self, query: str, top_k: int = 5) -> List[Message]: ...
def delete(self, message_ids: List[str]): ...
def update(self, message_id: str, new_content: Any): ...
```

### 8.4 `MemoryManager` (`memory/memory_manager.py`)

Coordinates short-term and long-term memory. When an action needs context, `MemoryManager` blends recent short-term messages with semantically relevant long-term memories.

---

## 9. Tool System

### 9.1 Base Classes

```python
class Tool(BaseModule):
    name: str
    description: str
    inputs: Dict[str, Dict[str, Any]]   # {param_name: {type, description}}
    required: Optional[List[str]]

    def __call__(self, **kwargs): ...
    def get_tool_schema(self) -> Dict: ...  # OpenAI function-call schema

class Toolkit(BaseModule):
    name: str
    tools: List[Tool]

    def get_tool_names(self) -> List[str]: ...
    def get_tool_schemas(self) -> List[Dict]: ...
    def add_tool(self, tool: Tool): ...
    def remove_tool(self, name: str): ...
```

### 9.2 Available Toolkits (40+)

| Category | Toolkits |
|---|---|
| **Code Execution** | `PythonInterpreterToolkit`, `DockerInterpreterToolkit` |
| **Web Search** | `WikipediaSearchToolkit`, `GoogleSearchToolkit`, `ArxivToolkit`, `DDGSSearchToolkit` |
| **HTTP** | `RequestToolkit`, `RSSToolkit`, `GoogleMapsToolkit` |
| **File/Storage** | `StorageToolkit`, `CMDToolkit`, `FileToolkit` |
| **Databases** | `MongoDBToolkit`, `PostgreSQLToolkit`, `FaissToolkit` |
| **Images** | `ImageAnalysisToolkit`, `OpenAIImageGenerationToolkit`, `FluxImageGenerationToolkit` |
| **Browser** | `BrowserToolkit`, `BrowserUseToolkit` |
| **Communication** | `GmailToolkit`, `TelegramToolkit` |
| **Finance/Crypto** | `CryptoToolkit`, `FinanceToolkit` |

Tools are attached to agents via `AgentManager.tools` or directly on `Action.tools`.

---

## 10. Optimization & Self-Evolution

This is the defining differentiator of EvoAgentX: multiple published optimization algorithms to automatically improve workflows and agents.

### 10.1 `Optimizer` Base (`optimizers/optimizer.py`)

```python
class Optimizer(BaseModule):
    graph: Union[WorkFlowGraph, ActionGraph]
    evaluator: Evaluator
    llm: BaseLLM
    max_steps: int
    eval_every_n_steps: int
    convergence_threshold: int

    def optimize(self, dataset: Benchmark, **kwargs): ...
    def step(self, **kwargs): ...
    def evaluate(self, dataset, graph=None, **kwargs) -> dict: ...
    def convergence_check(self) -> bool: ...
```

### 10.2 Implemented Algorithms

#### TextGrad (`textgrad_optimizer.py`)
- **Approach**: Treats LLM outputs as differentiable "tensors." Runs a backward pass using an LLM critic to generate "gradients" (natural language feedback) and propagates them to update prompts.
- **Target**: Prompts within `ActionGraph` nodes.
- **Paper**: *Nature* (2025).

#### MIPRO (`mipro_optimizer.py`)
- **Approach**: Model-Agnostic Iterative Prompt Optimization. Uses black-box evaluations to score candidate prompts, then an LLM proposer generates new candidates via adaptive reranking.
- **Target**: Individual action prompts.
- **Paper**: arXiv:2406.11695.

#### AFlow (`aflow_optimizer.py`)
- **Approach**: Reinforcement-learning-inspired workflow evolution. Uses Monte Carlo Tree Search (MCTS) to explore the space of possible workflow structures and prompt edits.
- **Target**: Both workflow structure (graph topology) and action prompts.
- **Paper**: arXiv:2410.10762.

#### EvoPrompt (`evoprompt_optimizer.py`)
- **Approach**: Evolutionary algorithm over prompts. Maintains a population of candidate prompts; uses LLM-based crossover and mutation operators. Selects survivors based on benchmark performance.
- **Target**: Action prompts.
- **Paper**: arXiv:2309.08532.

#### SEW (`sew_optimizer.py`)
- **Approach**: Self-Evolving Workflows. Combines structure optimization (topology) and prompt optimization in a joint loop, using feedback from the evaluator to guide both.
- **Target**: Full `WorkFlowGraph`.

### 10.3 Evolution Loop (Typical)

```
Initialize workflow/graph
  ↓
[Repeat max_steps times]
  step():
    - Sample or mutate current configuration
    - Execute on train split of benchmark
    - Collect scores

  [Every eval_every_n_steps]
  evaluate():
    - Run on dev/test split
    - Log metrics

  convergence_check():
    - If no improvement for convergence_threshold steps → stop
  ↓
Return best graph/workflow
```

---

## 11. Evaluation & Benchmarking

### 11.1 `Evaluator` (`evaluators/evaluator.py`)

```python
class Evaluator:
    llm: BaseLLM
    num_workers: int
    agent_manager: Optional[AgentManager]
    collate_func: Callable       # formats dataset sample for workflow
    output_postprocess_func: Callable   # cleans workflow output

    def evaluate(
        self,
        graph: Union[WorkFlowGraph, ActionGraph],
        benchmark: Benchmark,
        eval_mode: str = "test",
        **kwargs
    ) -> dict: ...
```

Uses `ThreadPoolExecutor` for parallel evaluation across dataset samples.

### 11.2 Benchmarks (`benchmark/`)

| Benchmark | Task |
|---|---|
| HotPotQA | Multi-hop open-domain QA |
| MBPP | Python code generation |
| MATH | Competition-level math |
| GSM8K | Grade-school math (reasoning) |
| LLMEval | LLM-as-judge evaluations |
| LiveCodeBench | Real-world code challenges |
| BigBenchHard | Diverse complex reasoning |

All benchmarks support train/dev/test splits and standard metrics (exact match, F1, pass@k).

---

## 12. Human-in-the-Loop (HITL)

### 12.1 Core Types (`hitl/hitl.py`)

```python
class HITLInteractionType(Enum):
    APPROVE_REJECT           # Human approves or rejects an action
    COLLECT_USER_INPUT       # Pause to collect input from user
    REVIEW_EDIT_STATE        # Human reviews/edits workflow state
    REVIEW_TOOL_CALLS        # Human reviews pending tool calls
    MULTI_TURN_CONVERSATION  # Open-ended back-and-forth

class HITLMode(Enum):
    PRE_EXECUTION    # Intercept before action runs
    POST_EXECUTION   # Intercept after action completes

class HITLDecision(Enum):
    APPROVE
    REJECT
    MODIFY
    CONTINUE
```

### 12.2 Data Flow

```python
class HITLContext:
    task_name: str
    agent_name: str
    action_name: str
    action_inputs: Dict[str, Any]
    execution_result: Optional[Any]

class HITLRequest:
    request_id: str
    interaction_type: HITLInteractionType
    mode: HITLMode
    context: HITLContext
    prompt_message: str

class HITLResponse:
    request_id: str
    decision: HITLDecision
    modified_content: Optional[Any]
    feedback: Optional[str]
```

### 12.3 `HITLManager`

Central coordinator. `WorkFlow.execute_task()` calls `hitl_manager.intercept()` at configured points, pauses execution, and routes requests to registered handlers (CLI, web UI, API, etc.). Decisions map back to workflow inputs.

### 12.4 Special HITL Agents (`hitl/special_hitl_agent.py`)

- `HITLInterceptorAgent` — gates workflow steps with approval/rejection
- `HITLUserInputCollectorAgent` — pauses and waits for user-provided data

---

## 13. RAG System

### 13.1 `RAGEngine` (`rag/rag.py`)

```python
class RAGEngine:
    config: RAGConfig
    storage_handler: StorageHandler
    llm: Optional[BaseLLM]

    def read(self, file_paths, corpus_id=None) -> Corpus: ...
    def build_index(self, corpus_id: str) -> None: ...
    def retrieve(self, query: str, corpus_id: str, top_k: int = 10) -> RagResult: ...
```

### 13.2 Pipeline Components

```
Documents (PDF, DOCX, HTML, ...)
  ↓ Readers (LLamaIndexReader, MultimodalReader)
Raw text/content
  ↓ ChunkFactory
Chunks (Corpus)
  ↓ EmbeddingFactory
Embeddings
  ↓ IndexFactory (FAISS, LanceDB, ...)
Vector Index
  ↓ [At query time]
  Query → Retriever → top-k Chunks
  ↓ PostProcessors (reranking, filtering)
RagResult
```

**Key data models:**
- `Corpus` — collection of chunks
- `Query` — search query with metadata
- `RagResult` — retrieved chunks with similarity scores
- `ChunkMetadata` — source document, position, custom fields

All components use the factory pattern (`ChunkFactory`, `EmbeddingFactory`, `IndexFactory`, `RetrieverFactory`) so strategies are swappable by config.

---

## 14. FastAPI Application

### 14.1 Routers (`evoagentx/app/`)

The app is a production-ready FastAPI service with JWT authentication, MongoDB persistence, and async execution.

| Router | Endpoints |
|---|---|
| `auth_router` | POST `/auth/register`, POST `/auth/token` |
| `agents_router` | POST `/agents/create`, GET `/agents/{id}`, DELETE `/agents/{id}` |
| `workflows_router` | POST `/workflows/generate`, POST `/workflows/{id}/execute`, GET `/workflows/{id}` |
| `executions_router` | GET `/executions/{id}`, GET `/executions/` |
| `system_router` | GET `/` (health), GET `/metrics` |

### 14.2 Key Files

| File | Responsibility |
|---|---|
| `app/main.py` | App factory, middleware, router registration |
| `app/db.py` | Async MongoDB client, database initialization |
| `app/security.py` | JWT token creation/validation, password hashing (bcrypt) |
| `app/schemas.py` | Pydantic request/response models for all endpoints |
| `app/services.py` | Business logic layer (workflow generation, execution, agent CRUD) |

---

## 15. Key Design Patterns

### Registry Pattern
`MetaModule` metaclass auto-registers every `BaseModule` subclass. Combined with `class_name` in serialized dicts, this enables dynamic instantiation of the correct concrete type from JSON without explicit import chains. This is the backbone of the entire serialization/deserialization system.

### Factory Pattern
`ChunkFactory`, `EmbeddingFactory`, `IndexFactory`, `RetrieverFactory` in the RAG system create strategy implementations at runtime based on config strings. Same pattern for LLM provider selection.

### Strategy Pattern
- Parsing modes (`"json"`, `"xml"`, `"str"`, `"title"`, `"custom"`) selectable per action.
- Chunking and retrieval strategies swappable via config.
- Optimization algorithms as interchangeable `Optimizer` subclasses.

### Adapter Pattern
`BaseLLM` provides a unified interface over heterogeneous providers (OpenAI, Azure, LiteLLM, etc.). `StorageHandler` adapts MongoDB/PostgreSQL/file backends.

### Template Method Pattern
- `BaseModule.init_module()` — subclasses override for custom init logic.
- `Optimizer.optimize()` calls abstract `step()` — subclasses implement the algorithm.
- `Agent.execute()` calls `_prepare_execution()` and `_create_output_message()` — subclasses can override hooks.

### Dependency Injection
Agents receive `llm`, `memory`, and `tools` at construction time. Workflows receive `agent_manager`, `llm`, and `hitl_manager`. Optimizers receive `evaluator` and `llm`. Nothing is hardcoded — everything is injected.

### Observer/Callback Pattern
`CallbackManager` for exception buffering during init. HITL intercepts are effectively observer hooks injected into the execution loop. Workflow node state changes (PENDING → RUNNING → COMPLETED) drive the scheduling loop.

---

## 16. End-to-End Data Flow

### Workflow Generation

```
User goal (str)
  → WorkFlowGenerator.generate_workflow()
    → TaskPlanner.execute() → TaskPlanningOutput (subtask list + deps)
    → build_workflow_from_plan() → WorkFlowGraph (nodes + edges)
    → AgentGenerator.execute() → agents assigned to nodes
    → [num_turns review loop]
      → WorkFlowReviewer.execute() → feedback
      → incorporate feedback → updated WorkFlowGraph
  → WorkFlowGraph
```

### Workflow Execution

```
WorkFlowGraph + inputs dict
  → WorkFlow.execute()
    → Validate structure
    → Initialize Environment
    → [Loop]
        WorkFlowManager.schedule_next_task()
        execute_task(node):
          WorkFlowManager.schedule_next_action()
          [HITL pre-execution intercept?]
          ActionGraph.execute() OR Agent.execute()
          [HITL post-execution intercept?]
          Environment.update(outputs)
          node.status = COMPLETED
        [graph.is_complete?]
    → WorkFlowManager.extract_output()
  → str result
```

### Agent Action Execution

```
inputs dict + message context
  → Agent._prepare_execution()
    → ShortTermMemory.add_messages()
    → extract action inputs from memory + direct inputs
  → Action.execute()
    → ActionInput parsing (validate + type-check)
    → LLM.generate(prompt, parse_mode=...)
    → ActionOutput.parse(raw_text)
  → Agent._create_output_message()
    → ShortTermMemory.add_message(result)
    → LongTermMemory.add(result) [if enabled]
  → Message(result)
```

---

## 17. Class Hierarchy Summary

```
BaseModule (metaclass=MetaModule)
├── BaseConfig
│   ├── LLMConfig
│   │   ├── OpenAILLMConfig
│   │   ├── AzureOpenAIConfig
│   │   ├── LiteLLMConfig
│   │   ├── SiliconFlowConfig
│   │   └── OpenRouterConfig
│   └── Parameter
├── Message
├── Agent
│   ├── ActionAgent
│   ├── CustomizeAgent
│   ├── TaskPlanner
│   ├── AgentGenerator
│   ├── WorkFlowReviewer
│   ├── HITLInterceptorAgent
│   └── HITLUserInputCollectorAgent
├── AgentManager
├── Action
│   ├── TaskPlanningAction
│   └── AgentGenerationAction
├── BaseLLM
│   ├── OpenAILLM
│   ├── LiteLLM
│   ├── SiliconFlowLLM
│   └── OpenRouterLLM
├── BaseMemory
│   ├── ShortTermMemory
│   └── LongTermMemory
├── Tool
├── Toolkit
│   ├── PythonInterpreterToolkit
│   ├── BrowserToolkit
│   ├── GmailToolkit
│   └── ... (40+ total)
├── WorkFlowGraph
│   ├── WorkFlowNode
│   └── WorkFlowEdge
├── ActionGraph
│   └── QAActionGraph
├── WorkFlow
├── WorkFlowGenerator
├── WorkFlowManager
└── Optimizer
    ├── TextGradOptimizer
    ├── MIPROOptimizer
    ├── AFlowOptimizer
    ├── EvoPromptOptimizer
    └── SEWOptimizer
```

---

## 18. Concurrency & Threading

- **`@atomic_method` decorator**: Wraps `AgentManager` mutations with a `threading.Lock`, preventing race conditions when multiple workflow steps try to acquire the same agent.
- **`AgentState`**: Per-agent AVAILABLE/RUNNING state; prevents the same agent being used in two tasks simultaneously.
- **Async throughout**: Every major execution path has both `execute()` (sync wrapper creating an event loop) and `async_execute()` (native async) variants.
- **Parallel evaluation**: `Evaluator` uses `ThreadPoolExecutor(max_workers=num_workers)` to evaluate multiple benchmark samples concurrently.
- **Async web API**: FastAPI app uses fully async MongoDB client and async workflow execution.

---

## 19. Configuration Management

- **All configs are Pydantic models** inheriting from `BaseConfig`.
- **`get_set_params()`** returns only explicitly set fields (not defaults), preventing unintentional config inheritance.
- **`save(path)`** serializes to YAML or JSON (detected by extension).
- **`from_file(path)`** reconstructs from YAML or JSON, dispatching to the correct subclass via `MODULE_REGISTRY`.
- **Provider-specific validation**: each `LLMConfig` subclass uses Pydantic validators to enforce required API keys, model name formats, etc.

---

## 20. Key Takeaways & Extension Points

### What makes this framework distinctive

1. **Automatic workflow construction from a goal string** — no manual DAG design required.
2. **Five different optimization algorithms** for improving agents/workflows — swap them or combine them.
3. **The `BaseModule` + `MetaModule` registry pattern** — add a new component (agent, tool, optimizer) and it's instantly serializable/deserializable.
4. **Unified async execution** — all components can run in an async event loop, making it suitable for production API deployment.

### Extension Points

| Area | How to extend |
|---|---|
| New LLM provider | Subclass `BaseLLM` + `LLMConfig`, register in `ModelRegistry` |
| New tool | Subclass `Tool` or `Toolkit`, register with `AgentManager` |
| New agent type | Subclass `Agent`, override `execute()` / `async_execute()` |
| New optimizer | Subclass `Optimizer`, implement `step()` |
| New benchmark | Subclass benchmark base class, implement train/dev/test splits |
| New parse mode | Register via `ParseFunctionRegistry` |
| New action function | Register via `ActionFunctionRegistry` (for `ActionAgent`) |
| New storage backend | Implement `StorageHandler` interface |
| New RAG strategy | Implement chunker/embedder/retriever factory interfaces |

### Potential Areas of Complexity

- **`WorkFlowManager` scheduling via LLM** — scheduling decisions are made by an LLM, introducing latency and potential non-determinism.
- **Circular dependency risk in graph validation** — `_validate_workflow_structure()` needs to catch cycles before execution begins.
- **Memory extraction for action inputs** — the logic that pulls relevant inputs from short-term memory for each action is implicit and can be hard to debug.
- **Optimizer convergence** — all five algorithms have hyperparameters (`max_steps`, `convergence_threshold`) that require tuning per task.

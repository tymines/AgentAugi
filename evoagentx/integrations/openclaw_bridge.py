"""
OpenClaw Task Bridge — Phase 2 integration.

Connects AgentAugi to the OpenClaw Gateway (127.0.0.1:18789).  Receives task
assignments from OpenClaw sessions, translates them into AgentAugi
WorkFlowGenerator calls, runs the workflow, and returns structured results
back to the caller.

OpenClaw Gateway API (REST):
  GET  /health           — liveness probe; returns {"status": "live", ...}
  POST /execute          — submit a single task; body: OpenClawTask JSON
  POST /batch            — submit a list of tasks; body: list[OpenClawTask]
  GET  /status           — runtime status / session summary

Auth: Bearer token passed in the Authorization header.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AgentType(str, Enum):
    """Agent capability profiles supported by the OpenClaw agent pool."""
    GENERAL = "general"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    WRITER = "writer"
    REVIEWER = "reviewer"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class OpenClawConfig(BaseModel):
    """Connection settings for the OpenClaw Gateway."""
    host: str = Field(default="127.0.0.1", description="Gateway hostname or IP")
    port: int = Field(default=18789, description="Gateway port")
    token: str = Field(..., description="Bearer auth token")
    timeout_connect: float = Field(default=10.0, description="TCP connect timeout (s)")
    timeout_read: float = Field(default=30.0, description="Per-request read timeout (s)")
    timeout_total: float = Field(default=300.0, description="Total request timeout (s)")
    max_retries: int = Field(default=3, description="Max retry attempts on transient errors")
    retry_backoff: float = Field(default=2.0, description="Exponential backoff base (s)")
    heartbeat_interval: float = Field(default=30.0, description="Health-check interval (s)")

    @field_validator("port")
    @classmethod
    def _valid_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(f"port must be 1–65535, got {v}")
        return v

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class OpenClawTask(BaseModel):
    """Inbound task assignment from an OpenClaw session."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task: str = Field(..., description="Natural-language task description")
    agent_type: AgentType = Field(default=AgentType.GENERAL)
    session_id: Optional[str] = Field(default=None, description="Originating session ID")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


class WorkflowConfig(BaseModel):
    """Translated AgentAugi workflow configuration derived from an OpenClawTask."""
    goal: str
    agent_type: AgentType
    llm_profile: str  # maps to AugiVector routing key (code / research / auto)
    max_agents: int = 3
    num_refinement_turns: int = 1
    tools_enabled: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class OpenClawResult(BaseModel):
    """Result payload returned to OpenClaw after workflow execution."""
    task_id: str
    session_id: Optional[str]
    status: TaskStatus
    output: Optional[str] = None
    workflow_trace: List[Dict[str, Any]] = Field(default_factory=list)
    score: Optional[float] = None
    cost_usd: Optional[float] = None
    duration_s: Optional[float] = None
    error: Optional[str] = None
    completed_at: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Task translator
# ---------------------------------------------------------------------------

# Maps OpenClaw agent types to AugiVector routing profiles (openclaw.json)
_AGENT_TYPE_TO_LLM_PROFILE: Dict[AgentType, str] = {
    AgentType.GENERAL: "auto",
    AgentType.RESEARCHER: "research",
    AgentType.ANALYST: "research",
    AgentType.WRITER: "auto",
    AgentType.REVIEWER: "auto",
}

# Tools enabled per agent type
_AGENT_TYPE_TO_TOOLS: Dict[AgentType, List[str]] = {
    AgentType.GENERAL: ["web_search", "web_fetch", "read", "write"],
    AgentType.RESEARCHER: ["web_search", "web_fetch", "read"],
    AgentType.ANALYST: ["read", "write"],
    AgentType.WRITER: ["read", "write"],
    AgentType.REVIEWER: ["read"],
}


def translate_task(task: OpenClawTask) -> WorkflowConfig:
    """
    Convert an OpenClawTask into an AgentAugi WorkflowConfig.

    The mapping rules follow the AugiVector routing table documented in
    ARCHITECTURE_INTEGRATION.md §3a.
    """
    llm_profile = _AGENT_TYPE_TO_LLM_PROFILE.get(task.agent_type, "auto")
    tools = _AGENT_TYPE_TO_TOOLS.get(task.agent_type, [])

    # Research tasks use more agents and an extra refinement pass
    is_research = task.agent_type in (AgentType.RESEARCHER, AgentType.ANALYST)

    return WorkflowConfig(
        goal=task.task,
        agent_type=task.agent_type,
        llm_profile=llm_profile,
        max_agents=5 if is_research else 3,
        num_refinement_turns=1 if is_research else 0,
        tools_enabled=tools,
        metadata={**task.metadata, "session_id": task.session_id},
    )


# ---------------------------------------------------------------------------
# HTTP client helpers
# ---------------------------------------------------------------------------

def _build_http_client(cfg: OpenClawConfig) -> httpx.AsyncClient:
    timeout = httpx.Timeout(
        connect=cfg.timeout_connect,
        read=cfg.timeout_read,
        write=cfg.timeout_read,
        pool=cfg.timeout_total,
    )
    headers = {
        "Authorization": f"Bearer {cfg.token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    return httpx.AsyncClient(
        base_url=cfg.base_url,
        headers=headers,
        timeout=timeout,
        follow_redirects=False,
    )


async def _request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    path: str,
    max_retries: int,
    backoff: float,
    **kwargs: Any,
) -> httpx.Response:
    """Issue an HTTP request, retrying on transient (5xx / network) errors."""
    attempt = 0
    last_exc: Optional[Exception] = None

    while attempt <= max_retries:
        try:
            response = await client.request(method, path, **kwargs)
            if response.status_code < 500:
                return response
            logger.warning(
                "OpenClaw gateway returned %s on %s %s (attempt %d/%d)",
                response.status_code, method, path, attempt + 1, max_retries + 1,
            )
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as exc:
            logger.warning(
                "OpenClaw request error on %s %s: %s (attempt %d/%d)",
                method, path, exc, attempt + 1, max_retries + 1,
            )
            last_exc = exc

        attempt += 1
        if attempt <= max_retries:
            delay = backoff * (2 ** (attempt - 1))
            await asyncio.sleep(delay)

    if last_exc:
        raise last_exc
    raise httpx.HTTPStatusError(
        f"Gateway returned 5xx after {max_retries + 1} attempts",
        request=None,  # type: ignore[arg-type]
        response=None,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Workflow execution shim
# ---------------------------------------------------------------------------

async def _run_workflow(config: WorkflowConfig) -> Dict[str, Any]:
    """
    Execute an AgentAugi workflow for the given config.

    This function is the AgentAugi execution boundary.  It imports
    WorkFlowGenerator lazily so the bridge module can be imported in
    environments where the full AgentAugi stack is not initialised (e.g.
    during unit tests with mocked dependencies).

    Returns a dict with keys: output, workflow_trace, score, cost_usd.
    """
    try:
        from evoagentx.workflow.workflow_generator import WorkFlowGenerator  # noqa: PLC0415
        from evoagentx.workflow.workflow import WorkFlow  # noqa: PLC0415
    except ImportError as exc:
        logger.error("AgentAugi workflow imports unavailable: %s", exc)
        raise

    # Build a minimal LLM config pointing at AugiVector (config_augivector.yaml
    # sets LITELLM_BASE_URL=http://localhost:3000 and routes by llm_profile).
    llm_kwargs: Dict[str, Any] = {
        "model": f"augivector/{config.llm_profile}",
        "temperature": 0.3 if config.agent_type == AgentType.ANALYST else 0.7,
    }

    try:
        from evoagentx.models.litellm_model import LiteLLMModel  # noqa: PLC0415
        llm = LiteLLMModel(**llm_kwargs)
    except ImportError:
        llm = None

    generator = WorkFlowGenerator(
        llm=llm,
        num_turns=config.num_refinement_turns,
    )

    t0 = time.monotonic()
    workflow_graph = generator.generate(goal=config.goal)

    workflow = WorkFlow(graph=workflow_graph, llm=llm)
    result = workflow.run()
    duration = time.monotonic() - t0

    # Extract a plain-text summary from the result
    output: str = ""
    if isinstance(result, str):
        output = result
    elif hasattr(result, "content"):
        output = str(result.content)
    elif isinstance(result, dict):
        output = result.get("output", json.dumps(result))

    # Collect a lightweight trace from the workflow graph
    trace: List[Dict[str, Any]] = []
    if hasattr(workflow_graph, "nodes"):
        for node in workflow_graph.nodes:
            trace.append({
                "node_id": getattr(node, "id", None),
                "name": getattr(node, "name", None),
                "agent": getattr(node, "agent", None),
            })

    return {
        "output": output,
        "workflow_trace": trace,
        "score": None,   # Evaluator integration wired in Phase 3
        "cost_usd": None,  # CostTracker integration wired in Phase 3
        "duration_s": round(duration, 3),
    }


# ---------------------------------------------------------------------------
# OpenClawBridge
# ---------------------------------------------------------------------------

class OpenClawBridge:
    """
    Bridge between the OpenClaw Gateway and AgentAugi's workflow engine.

    Lifecycle::

        bridge = OpenClawBridge(cfg)
        await bridge.connect()          # verify gateway reachable
        result = await bridge.submit(task)
        await bridge.disconnect()

    The bridge can also run in long-lived mode with a background heartbeat::

        async with bridge:
            result = await bridge.submit(task)
    """

    def __init__(self, config: OpenClawConfig) -> None:
        self._cfg = config
        self._client: Optional[httpx.AsyncClient] = None
        self._heartbeat_task: Optional[asyncio.Task] = None  # type: ignore[type-arg]
        self._connected: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the HTTP client and verify the gateway is reachable."""
        if self._connected:
            return
        self._client = _build_http_client(self._cfg)
        await self._check_health()
        self._connected = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("OpenClawBridge connected to %s", self._cfg.base_url)

    async def disconnect(self) -> None:
        """Stop the heartbeat and close the HTTP client."""
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.aclose()
            self._client = None
        self._connected = False
        logger.info("OpenClawBridge disconnected")

    async def __aenter__(self) -> "OpenClawBridge":
        await self.connect()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # Health / heartbeat
    # ------------------------------------------------------------------

    async def _check_health(self) -> Dict[str, Any]:
        """
        Probe GET /health on the gateway.

        The gateway returns JSON on the actual /health path; the web-UI SPA
        catches all other paths and returns HTML.  We accept any 2xx status
        because early versions may return 200 with HTML when the production
        REST API is not yet mounted — the bridge degrades gracefully.
        """
        assert self._client is not None
        resp = await _request_with_retry(
            self._client,
            "GET",
            "/health",
            max_retries=self._cfg.max_retries,
            backoff=self._cfg.retry_backoff,
        )
        if resp.status_code == 200:
            try:
                return resp.json()
            except Exception:
                # HTML response from the SPA — gateway is up but REST not mounted
                return {"status": "live", "note": "html_response"}
        resp.raise_for_status()
        return {}

    async def _heartbeat_loop(self) -> None:
        """Periodically ping /health so we detect gateway restarts early."""
        while True:
            await asyncio.sleep(self._cfg.heartbeat_interval)
            try:
                await self._check_health()
                logger.debug("OpenClaw heartbeat OK")
            except Exception as exc:
                logger.warning("OpenClaw heartbeat failed: %s", exc)

    # ------------------------------------------------------------------
    # Task submission
    # ------------------------------------------------------------------

    async def submit(self, task: OpenClawTask) -> OpenClawResult:
        """
        Translate an OpenClawTask, run it through AgentAugi, and return a
        structured OpenClawResult.

        This is the primary integration point:
          1. Translate the OpenClaw task format → AgentAugi WorkflowConfig
          2. Execute the AgentAugi workflow
          3. Package the result back into OpenClaw's result schema
        """
        if not self._connected:
            await self.connect()

        t_start = time.monotonic()
        logger.info(
            "Bridge: submitting task %s (type=%s)",
            task.task_id, task.agent_type.value,
        )

        workflow_cfg = translate_task(task)

        try:
            run_result = await _run_workflow(workflow_cfg)
            status = TaskStatus.COMPLETED
            error: Optional[str] = None
        except asyncio.TimeoutError:
            run_result = {}
            status = TaskStatus.TIMEOUT
            error = "Workflow execution timed out"
        except Exception as exc:
            run_result = {}
            status = TaskStatus.FAILED
            error = str(exc)
            logger.exception("Workflow execution failed for task %s", task.task_id)

        duration = time.monotonic() - t_start

        result = OpenClawResult(
            task_id=task.task_id,
            session_id=task.session_id,
            status=status,
            output=run_result.get("output"),
            workflow_trace=run_result.get("workflow_trace", []),
            score=run_result.get("score"),
            cost_usd=run_result.get("cost_usd"),
            duration_s=run_result.get("duration_s", round(duration, 3)),
            error=error,
        )

        logger.info(
            "Bridge: task %s finished — status=%s duration=%.2fs",
            task.task_id, status.value, duration,
        )
        return result

    async def submit_batch(self, tasks: List[OpenClawTask]) -> List[OpenClawResult]:
        """Submit a list of tasks concurrently (respects OpenClaw's 8-session cap)."""
        # Limit concurrency to stay within the gateway's maxConcurrentSessions=8
        semaphore = asyncio.Semaphore(8)

        async def _guarded(t: OpenClawTask) -> OpenClawResult:
            async with semaphore:
                return await self.submit(t)

        return list(await asyncio.gather(*[_guarded(t) for t in tasks]))

    # ------------------------------------------------------------------
    # Session helpers (thin wrappers around gateway tools)
    # ------------------------------------------------------------------

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """Fetch active sessions from the OpenClaw gateway (sessions_list)."""
        if not self._connected:
            await self.connect()
        assert self._client is not None
        resp = await _request_with_retry(
            self._client, "GET", "/gateway/sessions",
            max_retries=self._cfg.max_retries,
            backoff=self._cfg.retry_backoff,
        )
        if resp.status_code == 200:
            try:
                return resp.json()
            except Exception:
                return []
        return []

    async def spawn_session(self, agent_id: str, task: str) -> Dict[str, Any]:
        """Request the gateway to spawn a new agent session (sessions_spawn)."""
        if not self._connected:
            await self.connect()
        assert self._client is not None
        payload = {"agent": agent_id, "task": task}
        resp = await _request_with_retry(
            self._client, "POST", "/gateway/sessions/spawn",
            max_retries=self._cfg.max_retries,
            backoff=self._cfg.retry_backoff,
            json=payload,
        )
        if resp.status_code == 200:
            try:
                return resp.json()
            except Exception:
                return {"raw": resp.text}
        return {"status_code": resp.status_code}

    async def send_to_session(self, session_id: str, message: str) -> Dict[str, Any]:
        """Send a follow-up message to an active session (sessions_send)."""
        if not self._connected:
            await self.connect()
        assert self._client is not None
        payload = {"session_id": session_id, "message": message}
        resp = await _request_with_retry(
            self._client, "POST", "/gateway/sessions/send",
            max_retries=self._cfg.max_retries,
            backoff=self._cfg.retry_backoff,
            json=payload,
        )
        if resp.status_code == 200:
            try:
                return resp.json()
            except Exception:
                return {"raw": resp.text}
        return {"status_code": resp.status_code}

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    def __repr__(self) -> str:
        state = "connected" if self._connected else "disconnected"
        return f"OpenClawBridge({self._cfg.base_url}, {state})"

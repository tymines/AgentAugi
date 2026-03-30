"""
Tests for evoagentx/integrations/openclaw_bridge.py

Covers:
  - Data model validation (OpenClawConfig, OpenClawTask, OpenClawResult)
  - Task translation (translate_task)
  - HTTP retry logic (_request_with_retry)
  - Bridge connect / disconnect / heartbeat lifecycle
  - submit() happy path and error path
  - submit_batch() concurrency cap
  - Session helpers (list_sessions, spawn_session, send_to_session)

All tests mock outbound HTTP — no live gateway required.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from evoagentx.integrations.openclaw_bridge import (
    AgentType,
    OpenClawBridge,
    OpenClawConfig,
    OpenClawResult,
    OpenClawTask,
    TaskStatus,
    WorkflowConfig,
    _build_http_client,
    _request_with_retry,
    translate_task,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg() -> OpenClawConfig:
    return OpenClawConfig(
        host="127.0.0.1",
        port=18789,
        token="test-token-abc",
        timeout_connect=5.0,
        timeout_read=10.0,
        timeout_total=60.0,
        max_retries=2,
        retry_backoff=0.01,   # near-zero so tests run fast
        heartbeat_interval=999.0,  # disable during tests
    )


@pytest.fixture
def sample_task() -> OpenClawTask:
    return OpenClawTask(
        task_id="task-001",
        task="Summarise the latest AI research papers on multi-agent systems",
        agent_type=AgentType.RESEARCHER,
        session_id="session-xyz",
        metadata={"source": "nerve"},
    )


@pytest.fixture
def bridge(cfg: OpenClawConfig) -> OpenClawBridge:
    return OpenClawBridge(cfg)


# ---------------------------------------------------------------------------
# OpenClawConfig
# ---------------------------------------------------------------------------

class TestOpenClawConfig:
    def test_base_url(self, cfg: OpenClawConfig) -> None:
        assert cfg.base_url == "http://127.0.0.1:18789"

    def test_invalid_port_raises(self) -> None:
        with pytest.raises(Exception):
            OpenClawConfig(token="t", port=0)

    def test_invalid_port_too_large(self) -> None:
        with pytest.raises(Exception):
            OpenClawConfig(token="t", port=99999)

    def test_defaults(self) -> None:
        cfg = OpenClawConfig(token="tok")
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 18789
        assert cfg.max_retries == 3
        assert cfg.heartbeat_interval == 30.0


# ---------------------------------------------------------------------------
# OpenClawTask
# ---------------------------------------------------------------------------

class TestOpenClawTask:
    def test_auto_task_id(self) -> None:
        t = OpenClawTask(task="do something")
        assert len(t.task_id) == 36  # UUID4

    def test_default_agent_type(self) -> None:
        t = OpenClawTask(task="do something")
        assert t.agent_type == AgentType.GENERAL

    def test_all_agent_types_valid(self) -> None:
        for at in AgentType:
            t = OpenClawTask(task="x", agent_type=at)
            assert t.agent_type == at

    def test_metadata_default_empty(self) -> None:
        t = OpenClawTask(task="x")
        assert t.metadata == {}

    def test_created_at_is_recent(self) -> None:
        before = time.time()
        t = OpenClawTask(task="x")
        after = time.time()
        assert before <= t.created_at <= after


# ---------------------------------------------------------------------------
# translate_task
# ---------------------------------------------------------------------------

class TestTranslateTask:
    def test_researcher_gets_research_profile(self, sample_task: OpenClawTask) -> None:
        cfg = translate_task(sample_task)
        assert cfg.llm_profile == "research"

    def test_researcher_gets_more_agents(self, sample_task: OpenClawTask) -> None:
        cfg = translate_task(sample_task)
        assert cfg.max_agents == 5

    def test_researcher_gets_refinement_turn(self, sample_task: OpenClawTask) -> None:
        cfg = translate_task(sample_task)
        assert cfg.num_refinement_turns == 1

    def test_general_gets_auto_profile(self) -> None:
        t = OpenClawTask(task="x", agent_type=AgentType.GENERAL)
        cfg = translate_task(t)
        assert cfg.llm_profile == "auto"

    def test_general_fewer_agents(self) -> None:
        t = OpenClawTask(task="x", agent_type=AgentType.GENERAL)
        cfg = translate_task(t)
        assert cfg.max_agents == 3

    def test_general_no_refinement(self) -> None:
        t = OpenClawTask(task="x", agent_type=AgentType.GENERAL)
        cfg = translate_task(t)
        assert cfg.num_refinement_turns == 0

    def test_analyst_gets_research_profile(self) -> None:
        t = OpenClawTask(task="x", agent_type=AgentType.ANALYST)
        cfg = translate_task(t)
        assert cfg.llm_profile == "research"

    def test_goal_matches_task_description(self, sample_task: OpenClawTask) -> None:
        cfg = translate_task(sample_task)
        assert cfg.goal == sample_task.task

    def test_session_id_in_metadata(self, sample_task: OpenClawTask) -> None:
        cfg = translate_task(sample_task)
        assert cfg.metadata["session_id"] == sample_task.session_id

    def test_tools_nonempty_for_all_types(self) -> None:
        for at in AgentType:
            t = OpenClawTask(task="x", agent_type=at)
            cfg = translate_task(t)
            assert len(cfg.tools_enabled) > 0

    def test_researcher_has_web_search(self) -> None:
        t = OpenClawTask(task="x", agent_type=AgentType.RESEARCHER)
        cfg = translate_task(t)
        assert "web_search" in cfg.tools_enabled

    def test_reviewer_readonly_tools_only(self) -> None:
        t = OpenClawTask(task="x", agent_type=AgentType.REVIEWER)
        cfg = translate_task(t)
        assert "write" not in cfg.tools_enabled

    def test_result_is_workflow_config(self, sample_task: OpenClawTask) -> None:
        cfg = translate_task(sample_task)
        assert isinstance(cfg, WorkflowConfig)


# ---------------------------------------------------------------------------
# _build_http_client
# ---------------------------------------------------------------------------

class TestBuildHttpClient:
    def test_base_url_set(self, cfg: OpenClawConfig) -> None:
        client = _build_http_client(cfg)
        assert "127.0.0.1" in str(client.base_url)
        asyncio.get_event_loop().run_until_complete(client.aclose())

    def test_auth_header_present(self, cfg: OpenClawConfig) -> None:
        client = _build_http_client(cfg)
        assert client.headers["Authorization"] == "Bearer test-token-abc"
        asyncio.get_event_loop().run_until_complete(client.aclose())


# ---------------------------------------------------------------------------
# _request_with_retry
# ---------------------------------------------------------------------------

class TestRequestWithRetry:
    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self, cfg: OpenClawConfig) -> None:
        mock_resp = MagicMock(spec=httpx.Response)
        mock_resp.status_code = 200

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.request.return_value = mock_resp

        resp = await _request_with_retry(
            mock_client, "GET", "/health",
            max_retries=cfg.max_retries,
            backoff=cfg.retry_backoff,
        )
        assert resp.status_code == 200
        mock_client.request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retries_on_500(self, cfg: OpenClawConfig) -> None:
        bad_resp = MagicMock(spec=httpx.Response)
        bad_resp.status_code = 500
        good_resp = MagicMock(spec=httpx.Response)
        good_resp.status_code = 200

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.request.side_effect = [bad_resp, good_resp]

        resp = await _request_with_retry(
            mock_client, "GET", "/health",
            max_retries=cfg.max_retries,
            backoff=cfg.retry_backoff,
        )
        assert resp.status_code == 200
        assert mock_client.request.await_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_all_retries_exhausted(
        self, cfg: OpenClawConfig
    ) -> None:
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.request.side_effect = httpx.ConnectError("refused")

        with pytest.raises(httpx.ConnectError):
            await _request_with_retry(
                mock_client, "GET", "/health",
                max_retries=1,
                backoff=0.001,
            )
        assert mock_client.request.await_count == 2  # 1 attempt + 1 retry

    @pytest.mark.asyncio
    async def test_4xx_not_retried(self, cfg: OpenClawConfig) -> None:
        not_found = MagicMock(spec=httpx.Response)
        not_found.status_code = 404

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.request.return_value = not_found

        resp = await _request_with_retry(
            mock_client, "GET", "/nonexistent",
            max_retries=cfg.max_retries,
            backoff=cfg.retry_backoff,
        )
        assert resp.status_code == 404
        mock_client.request.assert_awaited_once()


# ---------------------------------------------------------------------------
# OpenClawBridge — connect / disconnect
# ---------------------------------------------------------------------------

def _make_health_response(status_code: int = 200, body: Dict = None) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = body or {"status": "live"}
    resp.text = json.dumps(body or {"status": "live"})
    return resp


class TestBridgeLifecycle:
    @pytest.mark.asyncio
    async def test_connect_calls_health_check(self, bridge: OpenClawBridge) -> None:
        health_resp = _make_health_response()

        with patch(
            "evoagentx.integrations.openclaw_bridge._request_with_retry",
            new_callable=AsyncMock,
            return_value=health_resp,
        ):
            await bridge.connect()
            assert bridge.is_connected
            await bridge.disconnect()

    @pytest.mark.asyncio
    async def test_double_connect_is_idempotent(self, bridge: OpenClawBridge) -> None:
        health_resp = _make_health_response()

        with patch(
            "evoagentx.integrations.openclaw_bridge._request_with_retry",
            new_callable=AsyncMock,
            return_value=health_resp,
        ) as mock_req:
            await bridge.connect()
            call_count_after_first = mock_req.await_count
            await bridge.connect()  # second call — should no-op
            assert mock_req.await_count == call_count_after_first
            await bridge.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_sets_not_connected(
        self, bridge: OpenClawBridge
    ) -> None:
        health_resp = _make_health_response()

        with patch(
            "evoagentx.integrations.openclaw_bridge._request_with_retry",
            new_callable=AsyncMock,
            return_value=health_resp,
        ):
            await bridge.connect()
            await bridge.disconnect()
            assert not bridge.is_connected

    @pytest.mark.asyncio
    async def test_context_manager(self, bridge: OpenClawBridge) -> None:
        health_resp = _make_health_response()

        with patch(
            "evoagentx.integrations.openclaw_bridge._request_with_retry",
            new_callable=AsyncMock,
            return_value=health_resp,
        ):
            async with bridge:
                assert bridge.is_connected
            assert not bridge.is_connected

    @pytest.mark.asyncio
    async def test_connect_fails_on_unreachable_gateway(
        self, bridge: OpenClawBridge
    ) -> None:
        with patch(
            "evoagentx.integrations.openclaw_bridge._request_with_retry",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("refused"),
        ):
            with pytest.raises(httpx.ConnectError):
                await bridge.connect()
            assert not bridge.is_connected


# ---------------------------------------------------------------------------
# OpenClawBridge — submit
# ---------------------------------------------------------------------------

async def _fake_run_workflow(config: WorkflowConfig) -> Dict[str, Any]:
    """Minimal stand-in for _run_workflow used in submit tests."""
    return {
        "output": f"Result for: {config.goal}",
        "workflow_trace": [{"node_id": "n1", "name": "planner", "agent": "TaskPlanner"}],
        "score": 0.92,
        "cost_usd": 0.0012,
        "duration_s": 1.23,
    }


class TestBridgeSubmit:
    @pytest.mark.asyncio
    async def test_submit_returns_completed_result(
        self, bridge: OpenClawBridge, sample_task: OpenClawTask
    ) -> None:
        health_resp = _make_health_response()

        with patch(
            "evoagentx.integrations.openclaw_bridge._request_with_retry",
            new_callable=AsyncMock,
            return_value=health_resp,
        ), patch(
            "evoagentx.integrations.openclaw_bridge._run_workflow",
            new_callable=AsyncMock,
            side_effect=_fake_run_workflow,
        ):
            await bridge.connect()
            result = await bridge.submit(sample_task)
            await bridge.disconnect()

        assert isinstance(result, OpenClawResult)
        assert result.status == TaskStatus.COMPLETED
        assert result.task_id == sample_task.task_id
        assert result.session_id == sample_task.session_id
        assert result.output is not None
        assert len(result.workflow_trace) > 0

    @pytest.mark.asyncio
    async def test_submit_sets_scores_and_cost(
        self, bridge: OpenClawBridge, sample_task: OpenClawTask
    ) -> None:
        health_resp = _make_health_response()

        with patch(
            "evoagentx.integrations.openclaw_bridge._request_with_retry",
            new_callable=AsyncMock,
            return_value=health_resp,
        ), patch(
            "evoagentx.integrations.openclaw_bridge._run_workflow",
            new_callable=AsyncMock,
            side_effect=_fake_run_workflow,
        ):
            await bridge.connect()
            result = await bridge.submit(sample_task)
            await bridge.disconnect()

        assert result.score == 0.92
        assert result.cost_usd == 0.0012

    @pytest.mark.asyncio
    async def test_submit_handles_workflow_exception(
        self, bridge: OpenClawBridge, sample_task: OpenClawTask
    ) -> None:
        health_resp = _make_health_response()

        async def _failing(_config: WorkflowConfig) -> Dict[str, Any]:
            raise RuntimeError("LLM API unavailable")

        with patch(
            "evoagentx.integrations.openclaw_bridge._request_with_retry",
            new_callable=AsyncMock,
            return_value=health_resp,
        ), patch(
            "evoagentx.integrations.openclaw_bridge._run_workflow",
            new_callable=AsyncMock,
            side_effect=_failing,
        ):
            await bridge.connect()
            result = await bridge.submit(sample_task)
            await bridge.disconnect()

        assert result.status == TaskStatus.FAILED
        assert "LLM API unavailable" in result.error

    @pytest.mark.asyncio
    async def test_submit_handles_timeout(
        self, bridge: OpenClawBridge, sample_task: OpenClawTask
    ) -> None:
        health_resp = _make_health_response()

        async def _timeout(_config: WorkflowConfig) -> Dict[str, Any]:
            raise asyncio.TimeoutError()

        with patch(
            "evoagentx.integrations.openclaw_bridge._request_with_retry",
            new_callable=AsyncMock,
            return_value=health_resp,
        ), patch(
            "evoagentx.integrations.openclaw_bridge._run_workflow",
            new_callable=AsyncMock,
            side_effect=_timeout,
        ):
            await bridge.connect()
            result = await bridge.submit(sample_task)
            await bridge.disconnect()

        assert result.status == TaskStatus.TIMEOUT

    @pytest.mark.asyncio
    async def test_submit_auto_connects_if_not_connected(
        self, bridge: OpenClawBridge, sample_task: OpenClawTask
    ) -> None:
        """submit() should call connect() implicitly when not yet connected."""
        health_resp = _make_health_response()
        assert not bridge.is_connected

        with patch(
            "evoagentx.integrations.openclaw_bridge._request_with_retry",
            new_callable=AsyncMock,
            return_value=health_resp,
        ), patch(
            "evoagentx.integrations.openclaw_bridge._run_workflow",
            new_callable=AsyncMock,
            side_effect=_fake_run_workflow,
        ):
            result = await bridge.submit(sample_task)
            await bridge.disconnect()

        assert result.status == TaskStatus.COMPLETED


# ---------------------------------------------------------------------------
# OpenClawBridge — submit_batch
# ---------------------------------------------------------------------------

class TestBridgeSubmitBatch:
    @pytest.mark.asyncio
    async def test_batch_returns_all_results(
        self, bridge: OpenClawBridge
    ) -> None:
        tasks = [
            OpenClawTask(task=f"task {i}", agent_type=AgentType.GENERAL)
            for i in range(4)
        ]
        health_resp = _make_health_response()

        with patch(
            "evoagentx.integrations.openclaw_bridge._request_with_retry",
            new_callable=AsyncMock,
            return_value=health_resp,
        ), patch(
            "evoagentx.integrations.openclaw_bridge._run_workflow",
            new_callable=AsyncMock,
            side_effect=_fake_run_workflow,
        ):
            await bridge.connect()
            results = await bridge.submit_batch(tasks)
            await bridge.disconnect()

        assert len(results) == 4
        assert all(r.status == TaskStatus.COMPLETED for r in results)

    @pytest.mark.asyncio
    async def test_batch_task_ids_preserved(
        self, bridge: OpenClawBridge
    ) -> None:
        ids = [str(uuid.uuid4()) for _ in range(3)]
        tasks = [
            OpenClawTask(task_id=tid, task=f"task {i}")
            for i, tid in enumerate(ids)
        ]
        health_resp = _make_health_response()

        with patch(
            "evoagentx.integrations.openclaw_bridge._request_with_retry",
            new_callable=AsyncMock,
            return_value=health_resp,
        ), patch(
            "evoagentx.integrations.openclaw_bridge._run_workflow",
            new_callable=AsyncMock,
            side_effect=_fake_run_workflow,
        ):
            await bridge.connect()
            results = await bridge.submit_batch(tasks)
            await bridge.disconnect()

        result_ids = {r.task_id for r in results}
        assert result_ids == set(ids)

    @pytest.mark.asyncio
    async def test_batch_respects_concurrency_cap(
        self, bridge: OpenClawBridge
    ) -> None:
        """Verify the semaphore keeps active tasks ≤ 8."""
        active_count = 0
        max_observed = 0

        async def _counting_workflow(_config: WorkflowConfig) -> Dict[str, Any]:
            nonlocal active_count, max_observed
            active_count += 1
            max_observed = max(max_observed, active_count)
            await asyncio.sleep(0.01)  # simulate work
            active_count -= 1
            return await _fake_run_workflow(_config)

        tasks = [
            OpenClawTask(task=f"task {i}", agent_type=AgentType.GENERAL)
            for i in range(16)
        ]
        health_resp = _make_health_response()

        with patch(
            "evoagentx.integrations.openclaw_bridge._request_with_retry",
            new_callable=AsyncMock,
            return_value=health_resp,
        ), patch(
            "evoagentx.integrations.openclaw_bridge._run_workflow",
            new_callable=AsyncMock,
            side_effect=_counting_workflow,
        ):
            await bridge.connect()
            await bridge.submit_batch(tasks)
            await bridge.disconnect()

        assert max_observed <= 8


# ---------------------------------------------------------------------------
# OpenClawBridge — session helpers
# ---------------------------------------------------------------------------

class TestSessionHelpers:
    @pytest.mark.asyncio
    async def test_list_sessions_returns_list(
        self, bridge: OpenClawBridge
    ) -> None:
        health_resp = _make_health_response()
        sessions_resp = MagicMock(spec=httpx.Response)
        sessions_resp.status_code = 200
        sessions_resp.json.return_value = [
            {"id": "s1", "agent": "main", "status": "running"},
            {"id": "s2", "agent": "dev", "status": "idle"},
        ]

        with patch(
            "evoagentx.integrations.openclaw_bridge._request_with_retry",
            new_callable=AsyncMock,
            side_effect=[health_resp, sessions_resp],
        ):
            await bridge.connect()
            sessions = await bridge.list_sessions()
            await bridge.disconnect()

        assert len(sessions) == 2
        assert sessions[0]["id"] == "s1"

    @pytest.mark.asyncio
    async def test_list_sessions_returns_empty_on_non_200(
        self, bridge: OpenClawBridge
    ) -> None:
        health_resp = _make_health_response()
        bad_resp = MagicMock(spec=httpx.Response)
        bad_resp.status_code = 404

        with patch(
            "evoagentx.integrations.openclaw_bridge._request_with_retry",
            new_callable=AsyncMock,
            side_effect=[health_resp, bad_resp],
        ):
            await bridge.connect()
            sessions = await bridge.list_sessions()
            await bridge.disconnect()

        assert sessions == []

    @pytest.mark.asyncio
    async def test_spawn_session_returns_dict(
        self, bridge: OpenClawBridge
    ) -> None:
        health_resp = _make_health_response()
        spawn_resp = MagicMock(spec=httpx.Response)
        spawn_resp.status_code = 200
        spawn_resp.json.return_value = {"session_id": "new-sess-1", "status": "spawned"}

        with patch(
            "evoagentx.integrations.openclaw_bridge._request_with_retry",
            new_callable=AsyncMock,
            side_effect=[health_resp, spawn_resp],
        ):
            await bridge.connect()
            resp = await bridge.spawn_session("main", "analyse logs")
            await bridge.disconnect()

        assert resp["session_id"] == "new-sess-1"

    @pytest.mark.asyncio
    async def test_send_to_session_returns_dict(
        self, bridge: OpenClawBridge
    ) -> None:
        health_resp = _make_health_response()
        send_resp = MagicMock(spec=httpx.Response)
        send_resp.status_code = 200
        send_resp.json.return_value = {"ack": True, "session_id": "sess-99"}

        with patch(
            "evoagentx.integrations.openclaw_bridge._request_with_retry",
            new_callable=AsyncMock,
            side_effect=[health_resp, send_resp],
        ):
            await bridge.connect()
            resp = await bridge.send_to_session("sess-99", "follow-up question")
            await bridge.disconnect()

        assert resp["ack"] is True


# ---------------------------------------------------------------------------
# OpenClawResult
# ---------------------------------------------------------------------------

class TestOpenClawResult:
    def test_failed_result_has_error(self) -> None:
        r = OpenClawResult(
            task_id="t1",
            session_id=None,
            status=TaskStatus.FAILED,
            error="something broke",
        )
        assert r.error == "something broke"
        assert r.output is None

    def test_completed_result_serialises_to_dict(self) -> None:
        r = OpenClawResult(
            task_id="t2",
            session_id="s1",
            status=TaskStatus.COMPLETED,
            output="Here is the answer",
            score=0.88,
            cost_usd=0.005,
            duration_s=3.14,
        )
        d = r.model_dump()
        assert d["status"] == "completed"
        assert d["output"] == "Here is the answer"
        assert d["score"] == 0.88

    def test_completed_at_is_set(self) -> None:
        before = time.time()
        r = OpenClawResult(task_id="t3", session_id=None, status=TaskStatus.COMPLETED)
        after = time.time()
        assert before <= r.completed_at <= after


# ---------------------------------------------------------------------------
# Repr
# ---------------------------------------------------------------------------

class TestBridgeRepr:
    def test_repr_disconnected(self, bridge: OpenClawBridge) -> None:
        assert "disconnected" in repr(bridge)
        assert "127.0.0.1:18789" in repr(bridge)

    @pytest.mark.asyncio
    async def test_repr_connected(self, bridge: OpenClawBridge) -> None:
        health_resp = _make_health_response()

        with patch(
            "evoagentx.integrations.openclaw_bridge._request_with_retry",
            new_callable=AsyncMock,
            return_value=health_resp,
        ):
            await bridge.connect()
            assert "connected" in repr(bridge)
            await bridge.disconnect()

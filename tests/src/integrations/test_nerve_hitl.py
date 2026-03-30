"""
Unit tests for evoagentx.integrations.nerve_hitl.

All tests are fully offline — no real Nerve server required.
Network calls are intercepted via unittest.mock.patch.
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch, call
from typing import Any, Dict

import requests

from evoagentx.integrations.nerve_hitl import (
    ApprovalThresholds,
    ApprovalTimeoutError,
    ChangeClassification,
    EvolutionProposal,
    NerveAPIError,
    NerveConfig,
    NerveHITL,
    NerveTaskPriority,
    ProposalStatus,
    _PRIORITY_MAP,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hitl(**kwargs) -> NerveHITL:
    cfg = NerveConfig(
        base_url="http://nerve.test",
        thresholds=ApprovalThresholds(
            minor_delta=0.02,
            major_delta=0.10,
            auto_approve_moderate=False,
            poll_timeout_seconds=0.5,   # short for tests
            poll_interval_seconds=0.05,
        ),
        **kwargs,
    )
    # Patch requests.Session so no real HTTP happens during construction
    with patch("evoagentx.integrations.nerve_hitl.requests.Session") as MockSession:
        MockSession.return_value = MagicMock()
        hitl = NerveHITL(cfg)
    return hitl


def _fake_response(status_code: int = 200, json_data: Any = None) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(response=resp)
    return resp


# ---------------------------------------------------------------------------
# Classification tests
# ---------------------------------------------------------------------------

class TestChangeClassification(unittest.TestCase):

    def setUp(self):
        self.hitl = _make_hitl()

    def _proposal(self, before: float, after: float) -> EvolutionProposal:
        return self.hitl.build_proposal(
            optimizer_name="TestOpt",
            change_summary="test",
            score_before=before,
            score_after=after,
        )

    def test_minor_below_threshold(self):
        p = self._proposal(0.80, 0.81)          # +1.25% < 2%
        self.assertEqual(p.classification, ChangeClassification.MINOR)

    def test_moderate_between_thresholds(self):
        p = self._proposal(0.80, 0.85)          # +6.25% between 2% and 10%
        self.assertEqual(p.classification, ChangeClassification.MODERATE)

    def test_major_above_threshold(self):
        p = self._proposal(0.70, 0.80)          # +14.3% >= 10%
        self.assertEqual(p.classification, ChangeClassification.MAJOR)

    def test_negative_delta_minor(self):
        # Slight regression is still minor (absolute value used)
        p = self._proposal(0.80, 0.79)          # -1.25%
        self.assertEqual(p.classification, ChangeClassification.MINOR)

    def test_zero_score_before_no_crash(self):
        p = self._proposal(0.0, 0.5)
        # delta_pct is 0 when score_before == 0 (guarded against ZeroDivisionError)
        self.assertIsNotNone(p.classification)

    def test_exact_minor_boundary_is_minor(self):
        # delta_pct = 0.019... < 0.02 → MINOR
        p = self._proposal(1.0, 1.019)
        self.assertEqual(p.classification, ChangeClassification.MINOR)

    def test_exact_major_boundary_is_major(self):
        # delta_pct = 0.10 → MAJOR
        p = self._proposal(1.0, 1.10)
        self.assertEqual(p.classification, ChangeClassification.MAJOR)


# ---------------------------------------------------------------------------
# Auto-approval logic
# ---------------------------------------------------------------------------

class TestAutoApproval(unittest.TestCase):

    def test_minor_always_auto_approved(self):
        hitl = _make_hitl()
        p = hitl.build_proposal("Opt", "x", 0.8, 0.81)
        self.assertTrue(hitl.should_auto_approve(p))

    def test_moderate_not_auto_approved_by_default(self):
        hitl = _make_hitl()
        p = hitl.build_proposal("Opt", "x", 0.8, 0.85)
        self.assertFalse(hitl.should_auto_approve(p))

    def test_moderate_auto_approved_when_flag_set(self):
        hitl = _make_hitl()
        hitl._cfg.thresholds.auto_approve_moderate = True
        p = hitl.build_proposal("Opt", "x", 0.8, 0.85)
        self.assertTrue(hitl.should_auto_approve(p))

    def test_major_never_auto_approved(self):
        hitl = _make_hitl()
        hitl._cfg.thresholds.auto_approve_moderate = True   # even with this True
        p = hitl.build_proposal("Opt", "x", 0.7, 0.80)
        self.assertFalse(hitl.should_auto_approve(p))


# ---------------------------------------------------------------------------
# Task payload construction
# ---------------------------------------------------------------------------

class TestTaskPayload(unittest.TestCase):

    def setUp(self):
        self.hitl = _make_hitl()

    def test_payload_contains_title(self):
        p = self.hitl.build_proposal("HyperOpt", "switched strategy", 0.70, 0.80,
                                     affected_params=["lr", "strategy"])
        payload = self.hitl._build_task_payload(p)
        self.assertIn("HyperOpt", payload["title"])
        self.assertIn("MAJOR", payload["title"])

    def test_payload_description_has_scores(self):
        p = self.hitl.build_proposal("HyperOpt", "changed prompt", 0.60, 0.62)
        payload = self.hitl._build_task_payload(p)
        self.assertIn("0.6000", payload["description"])
        self.assertIn("0.6200", payload["description"])

    def test_payload_includes_classification_label(self):
        p = self.hitl.build_proposal("Opt", "x", 0.8, 0.85)
        payload = self.hitl._build_task_payload(p)
        self.assertIn("moderate", payload["labels"])

    def test_priority_maps_correctly(self):
        for cls, expected_priority in _PRIORITY_MAP.items():
            p = self.hitl.build_proposal("Opt", "x", 0.5, 0.5)
            p.classification = cls
            payload = self.hitl._build_task_payload(p)
            self.assertEqual(payload["priority"], expected_priority.value)

    def test_metadata_appears_in_description(self):
        p = self.hitl.build_proposal("Opt", "x", 0.5, 0.55,
                                     metadata={"cost_delta": "-12%", "strategy": "gaussian"})
        payload = self.hitl._build_task_payload(p)
        self.assertIn("cost_delta", payload["description"])
        self.assertIn("gaussian", payload["description"])

    def test_proposal_id_in_description(self):
        p = self.hitl.build_proposal("Opt", "x", 0.5, 0.55)
        payload = self.hitl._build_task_payload(p)
        self.assertIn(p.proposal_id, payload["description"])


# ---------------------------------------------------------------------------
# Nerve API — create_task
# ---------------------------------------------------------------------------

class TestCreateTask(unittest.TestCase):

    def _hitl_with_session_mock(self) -> tuple[NerveHITL, MagicMock]:
        hitl = _make_hitl()
        mock_session = MagicMock()
        hitl._session = mock_session
        return hitl, mock_session

    def test_create_task_sets_nerve_task_id(self):
        hitl, mock_session = self._hitl_with_session_mock()
        mock_session.request.return_value = _fake_response(
            200, {"id": "task-abc-123"}
        )
        p = hitl.build_proposal("Opt", "change", 0.5, 0.60)
        task_id = hitl.create_task(p)
        self.assertEqual(task_id, "task-abc-123")
        self.assertEqual(p.nerve_task_id, "task-abc-123")
        self.assertEqual(p.status, ProposalStatus.SUBMITTED)

    def test_create_task_posts_to_correct_endpoint(self):
        hitl, mock_session = self._hitl_with_session_mock()
        mock_session.request.return_value = _fake_response(200, {"id": "t1"})
        p = hitl.build_proposal("Opt", "change", 0.5, 0.60)
        hitl.create_task(p)
        args, kwargs = mock_session.request.call_args
        self.assertEqual(args[0], "POST")
        self.assertIn("/api/kanban/tasks", args[1])

    def test_create_task_raises_on_http_error(self):
        hitl, mock_session = self._hitl_with_session_mock()
        mock_session.request.return_value = _fake_response(500)
        p = hitl.build_proposal("Opt", "change", 0.5, 0.60)
        with self.assertRaises(NerveAPIError):
            hitl.create_task(p)


# ---------------------------------------------------------------------------
# Polling logic
# ---------------------------------------------------------------------------

class TestPollForDecision(unittest.TestCase):

    def _hitl_with_get_task(self, statuses) -> NerveHITL:
        """Return a NerveHITL whose get_task() returns successive statuses."""
        hitl = _make_hitl()
        responses = iter(statuses)
        hitl.get_task = MagicMock(side_effect=lambda tid: {"status": next(responses), "description": "", "version": 1})
        return hitl

    def _submitted_proposal(self) -> EvolutionProposal:
        p = EvolutionProposal(optimizer_name="Opt", change_summary="x",
                               score_before=0.5, score_after=0.65)
        p.nerve_task_id = "task-1"
        p.status = ProposalStatus.SUBMITTED
        p.classification = ChangeClassification.MAJOR
        return p

    def test_poll_returns_approved_on_done(self):
        hitl = self._hitl_with_get_task(["in-review", "in-review", "done"])
        p = self._submitted_proposal()
        result = hitl.poll_for_decision(p)
        self.assertEqual(result, ProposalStatus.APPROVED)
        self.assertEqual(p.status, ProposalStatus.APPROVED)

    def test_poll_returns_rejected_on_cancelled(self):
        hitl = self._hitl_with_get_task(["in-review", "cancelled"])
        p = self._submitted_proposal()
        result = hitl.poll_for_decision(p)
        self.assertEqual(result, ProposalStatus.REJECTED)

    def test_poll_raises_timeout_error(self):
        # Always return "in-review" → will time out (poll_timeout=0.5s)
        hitl = _make_hitl()
        hitl.get_task = MagicMock(return_value={"status": "in-review", "description": "", "version": 1})
        p = self._submitted_proposal()
        with self.assertRaises(ApprovalTimeoutError) as ctx:
            hitl.poll_for_decision(p)
        self.assertEqual(ctx.exception.proposal_id, p.proposal_id)
        self.assertEqual(p.status, ProposalStatus.TIMED_OUT)

    def test_poll_raises_if_no_task_id(self):
        hitl = _make_hitl()
        p = EvolutionProposal()
        with self.assertRaises(ValueError):
            hitl.poll_for_decision(p)

    def test_poll_continues_on_api_error(self):
        """A transient Nerve error should not immediately fail polling."""
        hitl = _make_hitl()
        call_count = [0]

        def _get_task(tid):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise NerveAPIError("transient")
            return {"status": "done", "description": "", "version": 1}

        hitl.get_task = _get_task
        p = self._submitted_proposal()
        result = hitl.poll_for_decision(p)
        self.assertEqual(result, ProposalStatus.APPROVED)


# ---------------------------------------------------------------------------
# submit_and_wait — integration of classify + create + poll
# ---------------------------------------------------------------------------

class TestSubmitAndWait(unittest.TestCase):

    def test_minor_auto_approved_no_nerve_call(self):
        hitl = _make_hitl()
        hitl.create_task = MagicMock()
        p = hitl.build_proposal("Opt", "tiny tweak", 0.80, 0.81)
        result = hitl.submit_and_wait(p)
        self.assertEqual(result, ProposalStatus.AUTO_APPROVED)
        hitl.create_task.assert_not_called()

    def test_major_creates_nerve_task(self):
        hitl = _make_hitl()
        hitl.create_task = MagicMock(return_value="task-99")
        hitl.poll_for_decision = MagicMock(return_value=ProposalStatus.APPROVED)
        p = hitl.build_proposal("Opt", "big jump", 0.50, 0.70)
        result = hitl.submit_and_wait(p)
        self.assertEqual(result, ProposalStatus.APPROVED)
        hitl.create_task.assert_called_once_with(p)
        hitl.poll_for_decision.assert_called_once_with(p)

    def test_error_status_on_create_task_failure(self):
        hitl = _make_hitl()
        hitl.create_task = MagicMock(side_effect=NerveAPIError("boom"))
        p = hitl.build_proposal("Opt", "big jump", 0.50, 0.70)
        result = hitl.submit_and_wait(p)
        self.assertEqual(result, ProposalStatus.ERROR)
        self.assertEqual(p.status, ProposalStatus.ERROR)

    def test_timeout_returns_timed_out_by_default(self):
        hitl = _make_hitl()
        hitl.create_task = MagicMock(return_value="task-x")
        hitl.poll_for_decision = MagicMock(
            side_effect=ApprovalTimeoutError("pid", 0.5)
        )
        p = hitl.build_proposal("Opt", "big jump", 0.50, 0.70)
        result = hitl.submit_and_wait(p)
        self.assertEqual(result, ProposalStatus.TIMED_OUT)

    def test_timeout_raises_when_flag_set(self):
        hitl = _make_hitl()
        hitl.create_task = MagicMock(return_value="task-x")
        hitl.poll_for_decision = MagicMock(
            side_effect=ApprovalTimeoutError("pid", 0.5)
        )
        p = hitl.build_proposal("Opt", "big jump", 0.50, 0.70)
        with self.assertRaises(ApprovalTimeoutError):
            hitl.submit_and_wait(p, raise_on_timeout=True)

    def test_reclassifies_before_submit(self):
        hitl = _make_hitl()
        # Force MINOR on the proposal, but then change scores to MAJOR before calling
        p = hitl.build_proposal("Opt", "x", 0.80, 0.81)
        self.assertEqual(p.classification, ChangeClassification.MINOR)

        # Simulate optimizer updating scores post-build
        p.score_after = 0.95
        hitl.create_task = MagicMock(return_value="t")
        hitl.poll_for_decision = MagicMock(return_value=ProposalStatus.APPROVED)
        hitl.submit_and_wait(p)
        # Should have been reclassified to MAJOR and routed to Nerve
        hitl.create_task.assert_called_once()


# ---------------------------------------------------------------------------
# submit_many
# ---------------------------------------------------------------------------

class TestSubmitMany(unittest.TestCase):

    def test_returns_dict_keyed_by_proposal_id(self):
        hitl = _make_hitl()
        proposals = [
            hitl.build_proposal("Opt", f"change {i}", 0.80, 0.81)   # all minor
            for i in range(3)
        ]
        results = hitl.submit_many(proposals)
        self.assertEqual(len(results), 3)
        for p in proposals:
            self.assertEqual(results[p.proposal_id], ProposalStatus.AUTO_APPROVED)


# ---------------------------------------------------------------------------
# report_metrics
# ---------------------------------------------------------------------------

class TestReportMetrics(unittest.TestCase):

    def _setup(self) -> tuple[NerveHITL, EvolutionProposal]:
        hitl = _make_hitl()
        p = EvolutionProposal(
            optimizer_name="Opt",
            change_summary="x",
            score_before=0.7,
            score_after=0.8,
        )
        p.nerve_task_id = "task-123"
        p.status = ProposalStatus.APPROVED
        p.classification = ChangeClassification.MAJOR
        return hitl, p

    def test_no_task_id_does_nothing(self):
        hitl = _make_hitl()
        p = EvolutionProposal()
        hitl.get_task = MagicMock()
        hitl.report_metrics(p, deployed=True)
        hitl.get_task.assert_not_called()

    def test_appends_outcome_to_description(self):
        hitl, p = self._setup()
        hitl.get_task = MagicMock(return_value={
            "description": "original description", "version": 2
        })
        hitl.update_task = MagicMock(return_value={})
        hitl.report_metrics(p, deployed=True, extra={"cost": "-5%"})
        call_args = hitl.update_task.call_args
        updated = call_args[0][1]["description"]
        self.assertIn("Deployment Outcome", updated)
        self.assertIn("Yes", updated)
        self.assertIn("cost", updated)

    def test_get_task_failure_is_silent(self):
        hitl, p = self._setup()
        hitl.get_task = MagicMock(side_effect=NerveAPIError("oops"))
        # Should not raise
        hitl.report_metrics(p, deployed=False)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class TestHealthCheck(unittest.TestCase):

    def test_health_ok_when_200(self):
        hitl = _make_hitl()
        hitl._session = MagicMock()
        hitl._session.request.return_value = _fake_response(200, {"board": "ok"})
        self.assertTrue(hitl.health_check())

    def test_health_false_on_api_error(self):
        hitl = _make_hitl()
        hitl._request = MagicMock(side_effect=NerveAPIError("down"))
        self.assertFalse(hitl.health_check())


# ---------------------------------------------------------------------------
# NerveConfig defaults
# ---------------------------------------------------------------------------

class TestNerveConfig(unittest.TestCase):

    def test_default_base_url(self):
        cfg = NerveConfig()
        self.assertEqual(cfg.base_url, "http://localhost:3080")

    def test_default_labels(self):
        cfg = NerveConfig()
        self.assertIn("evoagentx", cfg.default_labels)
        self.assertIn("hitl", cfg.default_labels)

    def test_thresholds_defaults(self):
        t = ApprovalThresholds()
        self.assertEqual(t.minor_delta, 0.02)
        self.assertEqual(t.major_delta, 0.10)
        self.assertFalse(t.auto_approve_moderate)


# ---------------------------------------------------------------------------
# EvolutionProposal
# ---------------------------------------------------------------------------

class TestEvolutionProposal(unittest.TestCase):

    def test_score_delta(self):
        p = EvolutionProposal(score_before=0.6, score_after=0.75)
        self.assertAlmostEqual(p.score_delta, 0.15, places=10)

    def test_score_delta_pct(self):
        p = EvolutionProposal(score_before=0.60, score_after=0.66)
        self.assertAlmostEqual(p.score_delta_pct, 0.10, places=5)

    def test_unique_proposal_ids(self):
        ids = {EvolutionProposal().proposal_id for _ in range(20)}
        self.assertEqual(len(ids), 20)

    def test_default_status_pending(self):
        p = EvolutionProposal()
        self.assertEqual(p.status, ProposalStatus.PENDING)


if __name__ == "__main__":
    unittest.main()

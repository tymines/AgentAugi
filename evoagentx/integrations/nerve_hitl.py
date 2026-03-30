"""
Nerve HITL Integration — Phase 4: Human-in-the-Loop via Nerve Kanban.

Surfaces HyperOptimizer evolution decisions to the Nerve kanban board for
human review. Minor changes are auto-approved; significant changes require
a human to approve or reject the task on the board before deployment proceeds.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & constants
# ---------------------------------------------------------------------------

class ChangeClassification(str, Enum):
    """How significant an optimizer evolution step is."""
    MINOR = "minor"       # score delta below minor_threshold  → auto-approve
    MODERATE = "moderate" # between minor and major thresholds → configurable
    MAJOR = "major"       # above major_threshold              → always human review


class ProposalStatus(str, Enum):
    """Lifecycle state of an evolution proposal."""
    PENDING = "pending"           # created locally, not yet sent to Nerve
    SUBMITTED = "submitted"       # task created in Nerve, awaiting decision
    APPROVED = "approved"         # human (or auto) approved
    REJECTED = "rejected"         # human rejected
    AUTO_APPROVED = "auto_approved"
    TIMED_OUT = "timed_out"       # poll window expired without decision
    ERROR = "error"               # Nerve API failure


class NerveTaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Map change classification to default Nerve priority
_PRIORITY_MAP: Dict[ChangeClassification, NerveTaskPriority] = {
    ChangeClassification.MINOR: NerveTaskPriority.LOW,
    ChangeClassification.MODERATE: NerveTaskPriority.MEDIUM,
    ChangeClassification.MAJOR: NerveTaskPriority.HIGH,
}

# Nerve kanban task statuses (as returned by the API)
_NERVE_STATUS_APPROVED = {"done"}
_NERVE_STATUS_REJECTED = {"cancelled"}
_NERVE_STATUS_PENDING  = {"todo", "in-review", "in-progress"}


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class NerveAPIError(RuntimeError):
    """Raised when Nerve REST calls fail after retries."""
    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ApprovalTimeoutError(TimeoutError):
    """Raised when a proposal is not decided within the configured window."""
    def __init__(self, proposal_id: str, timeout_seconds: float) -> None:
        super().__init__(
            f"Proposal {proposal_id} was not approved/rejected within "
            f"{timeout_seconds}s"
        )
        self.proposal_id = proposal_id
        self.timeout_seconds = timeout_seconds


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ApprovalThresholds:
    """Score-delta thresholds that govern auto-approval behaviour.

    If ``score_delta`` (absolute improvement) is:
      - < minor_delta  → MINOR  (auto-approved regardless of other settings)
      - < major_delta  → MODERATE (auto-approved only if auto_approve_moderate)
      - >= major_delta → MAJOR   (always routed to human review)
    """
    minor_delta: float = 0.02    # 2 % improvement → minor
    major_delta: float = 0.10    # 10 % improvement → major
    auto_approve_moderate: bool = False

    # How long (seconds) to poll Nerve before giving up
    poll_timeout_seconds: float = 3600.0   # 1 hour
    poll_interval_seconds: float = 15.0


@dataclass
class NerveConfig:
    """Connection settings for the Nerve kanban REST API."""
    base_url: str = "http://localhost:3080"
    api_prefix: str = "/api/kanban"

    # Optional bearer token / password for remote Nerve instances
    api_key: Optional[str] = None
    password: Optional[str] = None

    # Default task metadata
    default_labels: List[str] = field(default_factory=lambda: ["evoagentx", "hitl"])
    created_by: str = "evoagentx"

    # Retry settings
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0

    # Request timeout (seconds)
    request_timeout: float = 10.0

    thresholds: ApprovalThresholds = field(default_factory=ApprovalThresholds)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class EvolutionProposal:
    """Captures a single optimizer evolution step for HITL review."""

    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    optimizer_name: str = ""
    change_summary: str = ""

    # Performance metrics
    score_before: float = 0.0
    score_after: float = 0.0

    # Parameters changed by this evolution step
    affected_params: List[str] = field(default_factory=list)

    # Extra context (e.g. prompt diffs, strategy names, cost delta)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Filled in after classification
    classification: ChangeClassification = ChangeClassification.MINOR
    status: ProposalStatus = ProposalStatus.PENDING

    # Nerve task ID once submitted
    nerve_task_id: Optional[str] = None

    # ISO-8601 timestamps
    created_at: float = field(default_factory=time.time)
    decided_at: Optional[float] = None

    # Human feedback text (from Nerve task description on reject/approve)
    human_feedback: Optional[str] = None

    @property
    def score_delta(self) -> float:
        return self.score_after - self.score_before

    @property
    def score_delta_pct(self) -> float:
        if self.score_before == 0:
            return 0.0
        return self.score_delta / abs(self.score_before)


# ---------------------------------------------------------------------------
# Core integration class
# ---------------------------------------------------------------------------

class NerveHITL:
    """Human-in-the-Loop bridge between AgentAugi optimizers and Nerve kanban.

    Usage::

        cfg = NerveConfig(base_url="http://nerve.augiport.com")
        hitl = NerveHITL(cfg)

        proposal = hitl.build_proposal(
            optimizer_name="HyperOptimizer",
            change_summary="Switched mutation strategy from uniform to Gaussian",
            score_before=0.72,
            score_after=0.85,
            affected_params=["mutation_rate", "strategy"],
        )

        decision = hitl.submit_and_wait(proposal)
        if decision == ProposalStatus.APPROVED:
            deploy_evolved_agent()
    """

    def __init__(self, config: Optional[NerveConfig] = None) -> None:
        self._cfg = config or NerveConfig()
        self._session = self._build_session()

    # ------------------------------------------------------------------
    # Session / HTTP helpers
    # ------------------------------------------------------------------

    def _build_session(self) -> requests.Session:
        s = requests.Session()
        s.headers.update({"Content-Type": "application/json", "Accept": "application/json"})
        if self._cfg.api_key:
            s.headers["Authorization"] = f"Bearer {self._cfg.api_key}"
        elif self._cfg.password:
            # Nerve uses signed-cookie auth; POST /api/auth/login to obtain it
            self._authenticate(s)
        return s

    def _authenticate(self, session: requests.Session) -> None:
        """Exchange password for a session cookie (Nerve password-auth flow)."""
        login_url = urljoin(self._cfg.base_url, "/api/auth/login")
        try:
            resp = session.post(
                login_url,
                json={"password": self._cfg.password},
                timeout=self._cfg.request_timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise NerveAPIError(f"Nerve authentication failed: {exc}") from exc

    def _url(self, path: str) -> str:
        return urljoin(self._cfg.base_url, self._cfg.api_prefix + path)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict] = None,
        raise_on_error: bool = True,
    ) -> requests.Response:
        url = self._url(path)
        last_exc: Optional[Exception] = None
        for attempt in range(self._cfg.max_retries):
            try:
                resp = self._session.request(
                    method, url, json=json, timeout=self._cfg.request_timeout
                )
                if raise_on_error:
                    resp.raise_for_status()
                return resp
            except requests.HTTPError as exc:
                raise NerveAPIError(
                    f"Nerve API {method} {url} failed: {exc}",
                    status_code=exc.response.status_code if exc.response else None,
                ) from exc
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self._cfg.max_retries - 1:
                    wait = self._cfg.retry_backoff_seconds * (2 ** attempt)
                    logger.warning(
                        "Nerve request failed (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, self._cfg.max_retries, wait, exc,
                    )
                    time.sleep(wait)
        raise NerveAPIError(f"Nerve API {method} {url} failed after retries: {last_exc}") from last_exc

    # ------------------------------------------------------------------
    # Proposal lifecycle
    # ------------------------------------------------------------------

    def classify(self, proposal: EvolutionProposal) -> ChangeClassification:
        """Assign a ChangeClassification based on score delta."""
        delta = abs(proposal.score_delta_pct)
        thresholds = self._cfg.thresholds
        if delta < thresholds.minor_delta:
            return ChangeClassification.MINOR
        if delta < thresholds.major_delta:
            return ChangeClassification.MODERATE
        return ChangeClassification.MAJOR

    def build_proposal(
        self,
        optimizer_name: str,
        change_summary: str,
        score_before: float,
        score_after: float,
        affected_params: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvolutionProposal:
        """Construct and classify an EvolutionProposal."""
        proposal = EvolutionProposal(
            optimizer_name=optimizer_name,
            change_summary=change_summary,
            score_before=score_before,
            score_after=score_after,
            affected_params=affected_params or [],
            metadata=metadata or {},
        )
        proposal.classification = self.classify(proposal)
        return proposal

    def should_auto_approve(self, proposal: EvolutionProposal) -> bool:
        """Return True if the proposal should be approved without human review."""
        if proposal.classification == ChangeClassification.MINOR:
            return True
        if proposal.classification == ChangeClassification.MODERATE:
            return self._cfg.thresholds.auto_approve_moderate
        return False

    # ------------------------------------------------------------------
    # Nerve task operations
    # ------------------------------------------------------------------

    def _build_task_payload(self, proposal: EvolutionProposal) -> Dict[str, Any]:
        priority = _PRIORITY_MAP.get(proposal.classification, NerveTaskPriority.MEDIUM)
        delta_sign = "+" if proposal.score_delta >= 0 else ""
        description_lines = [
            f"**Optimizer:** {proposal.optimizer_name}",
            f"**Change:** {proposal.change_summary}",
            "",
            f"**Score before:** {proposal.score_before:.4f}",
            f"**Score after:**  {proposal.score_after:.4f}",
            f"**Delta:**        {delta_sign}{proposal.score_delta:.4f} "
            f"({delta_sign}{proposal.score_delta_pct * 100:.1f}%)",
            "",
            f"**Classification:** {proposal.classification.value}",
            f"**Affected params:** {', '.join(proposal.affected_params) or 'N/A'}",
        ]
        if proposal.metadata:
            description_lines += ["", "**Extra metadata:**"]
            for k, v in proposal.metadata.items():
                description_lines.append(f"- {k}: {v}")
        description_lines += [
            "",
            "---",
            "**To approve:** move to Done.  **To reject:** move to Cancelled.",
            f"*proposal_id: {proposal.proposal_id}*",
        ]

        return {
            "title": (
                f"[EvoAgentX] {proposal.optimizer_name} — "
                f"{proposal.classification.value.upper()} change "
                f"(Δ{delta_sign}{proposal.score_delta_pct * 100:.1f}%)"
            ),
            "description": "\n".join(description_lines),
            "status": "in-review",
            "priority": priority.value,
            "createdBy": self._cfg.created_by,
            "labels": self._cfg.default_labels + [proposal.classification.value],
        }

    def create_task(self, proposal: EvolutionProposal) -> str:
        """Create a Nerve kanban task for the proposal. Returns the task ID."""
        payload = self._build_task_payload(proposal)
        resp = self._request("POST", "/tasks", json=payload)
        task_id: str = resp.json()["id"]
        proposal.nerve_task_id = task_id
        proposal.status = ProposalStatus.SUBMITTED
        logger.info(
            "Nerve task created: id=%s  proposal=%s  classification=%s",
            task_id, proposal.proposal_id, proposal.classification.value,
        )
        return task_id

    def get_task(self, task_id: str) -> Dict[str, Any]:
        """Fetch current task state from Nerve."""
        resp = self._request("GET", f"/tasks/{task_id}", raise_on_error=False)
        if resp.status_code == 404:
            raise NerveAPIError(f"Nerve task {task_id} not found", status_code=404)
        resp.raise_for_status()
        return resp.json()

    def update_task(self, task_id: str, fields: Dict[str, Any]) -> Dict[str, Any]:
        """PATCH an existing Nerve task (versioned CAS — version must be included)."""
        resp = self._request("PATCH", f"/tasks/{task_id}", json=fields)
        return resp.json()

    # ------------------------------------------------------------------
    # Polling for human decision
    # ------------------------------------------------------------------

    def poll_for_decision(self, proposal: EvolutionProposal) -> ProposalStatus:
        """Block until Nerve task is approved/rejected or the timeout elapses.

        Nerve encodes the human decision by moving the task to:
          - ``done``      → approved
          - ``cancelled`` → rejected

        Returns the final ProposalStatus.
        """
        if proposal.nerve_task_id is None:
            raise ValueError("Proposal has no Nerve task ID; call create_task() first.")

        thresholds = self._cfg.thresholds
        deadline = time.monotonic() + thresholds.poll_timeout_seconds

        logger.info(
            "Polling Nerve task %s for proposal %s (timeout=%.0fs, interval=%.0fs)",
            proposal.nerve_task_id, proposal.proposal_id,
            thresholds.poll_timeout_seconds, thresholds.poll_interval_seconds,
        )

        while time.monotonic() < deadline:
            try:
                task = self.get_task(proposal.nerve_task_id)
            except NerveAPIError as exc:
                logger.warning("Failed to fetch Nerve task: %s", exc)
                time.sleep(thresholds.poll_interval_seconds)
                continue

            status = task.get("status", "")
            if status in _NERVE_STATUS_APPROVED:
                proposal.status = ProposalStatus.APPROVED
                proposal.decided_at = time.time()
                logger.info("Proposal %s APPROVED via Nerve.", proposal.proposal_id)
                return ProposalStatus.APPROVED

            if status in _NERVE_STATUS_REJECTED:
                proposal.status = ProposalStatus.REJECTED
                proposal.decided_at = time.time()
                proposal.human_feedback = task.get("description", "")
                logger.info("Proposal %s REJECTED via Nerve.", proposal.proposal_id)
                return ProposalStatus.REJECTED

            remaining = deadline - time.monotonic()
            logger.debug(
                "Task %s still %s — %.0fs remaining.",
                proposal.nerve_task_id, status, remaining,
            )
            time.sleep(thresholds.poll_interval_seconds)

        proposal.status = ProposalStatus.TIMED_OUT
        proposal.decided_at = time.time()
        logger.warning(
            "Proposal %s timed out after %.0fs.",
            proposal.proposal_id, thresholds.poll_timeout_seconds,
        )
        raise ApprovalTimeoutError(proposal.proposal_id, thresholds.poll_timeout_seconds)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report_metrics(
        self,
        proposal: EvolutionProposal,
        deployed: bool,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append an outcome comment to the Nerve task after deployment.

        Uses PATCH to update the task description with the final result.
        Does nothing if the task ID is unknown.
        """
        if proposal.nerve_task_id is None:
            return

        try:
            task = self.get_task(proposal.nerve_task_id)
        except NerveAPIError as exc:
            logger.warning("Could not fetch task for metric report: %s", exc)
            return

        outcome_lines = [
            "",
            "---",
            "## Deployment Outcome",
            f"- **Deployed:** {'Yes' if deployed else 'No'}",
            f"- **Final status:** {proposal.status.value}",
            f"- **Score delta:** {proposal.score_delta:+.4f}",
        ]
        if extra:
            for k, v in extra.items():
                outcome_lines.append(f"- **{k}:** {v}")

        updated_description = (task.get("description") or "") + "\n".join(outcome_lines)
        version = task.get("version", 1)

        try:
            self.update_task(
                proposal.nerve_task_id,
                {"description": updated_description, "version": version},
            )
        except NerveAPIError as exc:
            logger.warning("Could not update task with metrics: %s", exc)

    # ------------------------------------------------------------------
    # High-level entry point
    # ------------------------------------------------------------------

    def submit_and_wait(
        self,
        proposal: EvolutionProposal,
        *,
        raise_on_timeout: bool = False,
    ) -> ProposalStatus:
        """Full HITL flow: classify → (auto-approve or submit+poll).

        Args:
            proposal: The evolution proposal to process.
            raise_on_timeout: If True, raise ApprovalTimeoutError on timeout;
                              otherwise return ProposalStatus.TIMED_OUT.

        Returns:
            Final ProposalStatus after the decision is made.
        """
        # Re-classify in case scores were updated after build_proposal
        proposal.classification = self.classify(proposal)

        if self.should_auto_approve(proposal):
            proposal.status = ProposalStatus.AUTO_APPROVED
            proposal.decided_at = time.time()
            logger.info(
                "Proposal %s AUTO-APPROVED (%s, delta=%.4f).",
                proposal.proposal_id, proposal.classification.value, proposal.score_delta,
            )
            return ProposalStatus.AUTO_APPROVED

        # Route to Nerve for human decision
        try:
            self.create_task(proposal)
        except NerveAPIError as exc:
            proposal.status = ProposalStatus.ERROR
            logger.error("Failed to create Nerve task for proposal %s: %s", proposal.proposal_id, exc)
            return ProposalStatus.ERROR

        try:
            return self.poll_for_decision(proposal)
        except ApprovalTimeoutError:
            if raise_on_timeout:
                raise
            return ProposalStatus.TIMED_OUT

    def submit_many(
        self,
        proposals: List[EvolutionProposal],
        *,
        raise_on_timeout: bool = False,
    ) -> Dict[str, ProposalStatus]:
        """Process multiple proposals sequentially.

        Returns a mapping of proposal_id → final ProposalStatus.
        """
        return {
            p.proposal_id: self.submit_and_wait(p, raise_on_timeout=raise_on_timeout)
            for p in proposals
        }

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Return True if the Nerve API is reachable."""
        try:
            resp = self._request("GET", "/../config", raise_on_error=False)
            return resp.status_code < 500
        except NerveAPIError:
            return False

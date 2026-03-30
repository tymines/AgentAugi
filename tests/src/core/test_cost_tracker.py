"""Tests for the CostTracker (Phase 0)."""
from __future__ import annotations

import pytest


class TestCostTracker:

    def setup_method(self):
        from evoagentx.core.cost_tracker import CostTracker
        self.tracker = CostTracker()

    def test_record_basic(self):
        rec = self.tracker.record("openai", "gpt-4o", input_tokens=100, output_tokens=50)
        assert rec.input_tokens == 100
        assert rec.output_tokens == 50
        assert rec.cost_usd > 0

    def test_total_cost_accumulates(self):
        self.tracker.record("openai", "gpt-4o", 100, 50)
        self.tracker.record("openai", "gpt-4o", 100, 50)
        assert self.tracker.total_cost() > 0
        single_cost = self.tracker.records()[0].cost_usd
        assert abs(self.tracker.total_cost() - 2 * single_cost) < 1e-9

    def test_session_isolation(self):
        with self.tracker.session("session-A"):
            self.tracker.record("openai", "gpt-4o-mini", 1000, 200)
        with self.tracker.session("session-B"):
            self.tracker.record("openai", "gpt-4o-mini", 500, 100)

        cost_a = self.tracker.total_cost(session_id="session-A")
        cost_b = self.tracker.total_cost(session_id="session-B")
        assert cost_a > 0
        assert cost_b > 0
        assert abs(cost_a - cost_b * 2) < 1e-9

    def test_session_cost_within_context(self):
        with self.tracker.session("my-run"):
            self.tracker.record("anthropic", "claude-haiku-4-5", 200, 100)
            cost = self.tracker.session_cost()
        assert cost > 0

    def test_budget_exceeded(self):
        from evoagentx.core.cost_tracker import CostBudgetExceeded
        self.tracker.set_budget(max_usd=0.000001)
        with pytest.raises(CostBudgetExceeded):
            self.tracker.record("openai", "gpt-4o", input_tokens=10000, output_tokens=5000)

    def test_per_session_budget(self):
        from evoagentx.core.cost_tracker import CostBudgetExceeded
        with self.tracker.session("limited"):
            self.tracker.set_budget(max_usd=0.000001)
            with pytest.raises(CostBudgetExceeded):
                self.tracker.record("openai", "gpt-4o", 100000, 50000)

    def test_custom_pricing(self):
        from evoagentx.core.cost_tracker import ModelPricing
        self.tracker.register_pricing("myco", "my-model", ModelPricing(1.0, 2.0))
        rec = self.tracker.record("myco", "my-model", 1000, 500)
        # 1000 input @ $1/1k + 500 output @ $2/1k = $1.00 + $1.00 = $2.00
        assert abs(rec.cost_usd - 2.0) < 1e-9

    def test_summary_structure(self):
        self.tracker.record("openai", "gpt-4o", 100, 50)
        self.tracker.record("anthropic", "claude-haiku-4-5", 200, 80)
        s = self.tracker.summary()
        assert s.total_calls == 2
        assert "openai/gpt-4o" in s.per_model
        assert "anthropic/claude-haiku-4-5" in s.per_model
        assert str(s)

    def test_reset(self):
        self.tracker.record("openai", "gpt-4o", 100, 50)
        assert len(self.tracker.records()) == 1
        self.tracker.reset()
        assert len(self.tracker.records()) == 0

    def test_reset_session(self):
        with self.tracker.session("keep"):
            self.tracker.record("openai", "gpt-4o", 100, 50)
        with self.tracker.session("drop"):
            self.tracker.record("openai", "gpt-4o", 100, 50)
        self.tracker.reset(session_id="drop")
        assert len(self.tracker.records(session_id="keep")) == 1
        assert len(self.tracker.records(session_id="drop")) == 0

    def test_get_tracker_singleton(self):
        from evoagentx.core.cost_tracker import get_tracker
        t1 = get_tracker()
        t2 = get_tracker()
        assert t1 is t2

    def test_estimate_tokens(self):
        from evoagentx.core.cost_tracker import estimate_tokens
        assert estimate_tokens("hello world") > 0
        assert estimate_tokens("") == 0

    def test_estimate_messages_tokens(self):
        from evoagentx.core.cost_tracker import estimate_messages_tokens
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        tokens = estimate_messages_tokens(msgs)
        assert tokens > 0

    def test_unknown_model_uses_fallback(self):
        rec = self.tracker.record("unknown_provider", "unknown_model", 100, 100)
        assert rec.cost_usd > 0

"""Unit tests for evoagentx.workflow.cyclic_workflow."""

import unittest

from evoagentx.core.base_config import Parameter
from evoagentx.workflow.workflow_graph import WorkFlowEdge, WorkFlowNode
from evoagentx.workflow.cyclic_workflow import (
    CycleBreakReason,
    CycleConfig,
    CycleState,
    CyclicWorkFlowGraph,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node(name: str, in_name: str = "input", out_name: str = "output") -> WorkFlowNode:
    return WorkFlowNode(
        name=name,
        description=f"Node {name}",
        inputs=[Parameter(name=in_name,  type="string", description="in")],
        outputs=[Parameter(name=out_name, type="string", description="out")],
    )


# ---------------------------------------------------------------------------
# CycleConfig tests
# ---------------------------------------------------------------------------

class TestCycleConfig(unittest.TestCase):

    def test_strategy_no_escalation(self):
        cfg = CycleConfig()
        self.assertEqual(cfg.strategy_for_iteration(0), "default")
        self.assertEqual(cfg.strategy_for_iteration(99), "default")

    def test_strategy_with_escalation(self):
        cfg = CycleConfig(strategy_escalation={0: "simple", 2: "medium", 4: "hard"})
        self.assertEqual(cfg.strategy_for_iteration(0), "simple")
        self.assertEqual(cfg.strategy_for_iteration(1), "simple")
        self.assertEqual(cfg.strategy_for_iteration(2), "medium")
        self.assertEqual(cfg.strategy_for_iteration(4), "hard")
        self.assertEqual(cfg.strategy_for_iteration(100), "hard")

    def test_strategy_no_applicable_key(self):
        cfg = CycleConfig(strategy_escalation={5: "heavy"})
        self.assertEqual(cfg.strategy_for_iteration(0), "default")


# ---------------------------------------------------------------------------
# CyclicWorkFlowGraph tests
# ---------------------------------------------------------------------------

class TestCyclicWorkFlowGraph(unittest.TestCase):

    def _make_cycle_graph(self):
        """
        Build a graph where Start has in-degree 0, then a back-edge creates
        a cycle:

            Start → Loop → Refine → Loop  (back-edge: Refine → Loop)
                                  ↓
                                Done
        """
        node_start  = _make_node("Start",  in_name="task",    out_name="draft")
        node_loop   = _make_node("Loop",   in_name="draft",   out_name="improved")
        node_refine = _make_node("Refine", in_name="improved", out_name="draft")
        node_done   = _make_node("Done",   in_name="improved", out_name="result")
        return CyclicWorkFlowGraph(
            goal="cyclic test",
            nodes=[node_start, node_loop, node_refine, node_done],
            edges=[
                WorkFlowEdge(source="Start",  target="Loop"),
                WorkFlowEdge(source="Loop",   target="Refine"),
                WorkFlowEdge(source="Refine", target="Loop"),    # back-edge → cycle
                WorkFlowEdge(source="Loop",   target="Done"),
            ],
        )

    # --- construction ---

    def test_cycle_detected(self):
        self.assertTrue(self._make_cycle_graph().is_loop_start("Loop"))

    def test_register_valid_cycle(self):
        graph = self._make_cycle_graph()
        graph.register_cycle("Loop", CycleConfig(max_iterations=3))
        self.assertIn("Loop", graph.cycle_configs)

    def test_register_unknown_node_raises(self):
        with self.assertRaises(KeyError):
            self._make_cycle_graph().register_cycle("MISSING", CycleConfig())

    def test_register_non_loop_node_raises(self):
        with self.assertRaises(ValueError):
            self._make_cycle_graph().register_cycle("Done", CycleConfig())

    # --- begin / step / break ---

    def test_begin_returns_fresh_state(self):
        state = self._make_cycle_graph().begin_cycle("Loop")
        self.assertEqual(state.iteration, 0)
        self.assertFalse(state.is_broken)

    def test_step_increments_iteration(self):
        graph = self._make_cycle_graph()
        graph.begin_cycle("Loop")
        state = graph.step_cycle("Loop")
        self.assertEqual(state.iteration, 1)

    def test_max_iterations_stops_cycle(self):
        graph = self._make_cycle_graph()
        graph.register_cycle("Loop", CycleConfig(max_iterations=3))
        graph.begin_cycle("Loop")
        for _ in range(3):
            state = graph.step_cycle("Loop")
        self.assertTrue(state.is_broken)
        self.assertEqual(state.break_reason, CycleBreakReason.MAX_ITERATIONS)

    def test_quality_threshold_stops_cycle(self):
        graph = self._make_cycle_graph()
        graph.register_cycle("Loop", CycleConfig(max_iterations=10, quality_threshold=0.9))
        graph.begin_cycle("Loop")
        state = graph.step_cycle("Loop", quality_score=0.95)
        self.assertTrue(state.is_broken)
        self.assertEqual(state.break_reason, CycleBreakReason.QUALITY_MET)

    def test_below_threshold_does_not_stop(self):
        graph = self._make_cycle_graph()
        graph.register_cycle("Loop", CycleConfig(max_iterations=10, quality_threshold=0.9))
        graph.begin_cycle("Loop")
        state = graph.step_cycle("Loop", quality_score=0.5)
        self.assertFalse(state.is_broken)

    def test_convergence_stops_cycle(self):
        graph = self._make_cycle_graph()
        graph.register_cycle("Loop", CycleConfig(
            max_iterations=20, convergence_window=2, convergence_delta=0.01,
        ))
        graph.begin_cycle("Loop")
        graph.step_cycle("Loop", quality_score=0.80)
        graph.step_cycle("Loop", quality_score=0.801)
        state = graph.step_cycle("Loop", quality_score=0.802)
        self.assertTrue(state.is_broken)
        self.assertEqual(state.break_reason, CycleBreakReason.CONVERGENCE)

    def test_external_break(self):
        graph = self._make_cycle_graph()
        graph.begin_cycle("Loop")
        graph.break_cycle("Loop")
        self.assertTrue(graph.should_break_cycle("Loop"))
        self.assertEqual(graph.cycle_state("Loop").break_reason, CycleBreakReason.EXTERNAL)

    def test_should_break_false_before_begin(self):
        self.assertFalse(self._make_cycle_graph().should_break_cycle("Loop"))

    # --- strategy escalation ---

    def test_current_strategy_default(self):
        graph = self._make_cycle_graph()
        graph.begin_cycle("Loop")
        self.assertEqual(graph.current_strategy("Loop"), "default")

    def test_current_strategy_escalates(self):
        graph = self._make_cycle_graph()
        graph.register_cycle("Loop", CycleConfig(
            max_iterations=10,
            strategy_escalation={0: "simple", 2: "medium"},
        ))
        graph.begin_cycle("Loop")
        self.assertEqual(graph.current_strategy("Loop"), "simple")
        graph.step_cycle("Loop")
        graph.step_cycle("Loop")
        self.assertEqual(graph.current_strategy("Loop"), "medium")

    # --- status snapshot ---

    def test_cycle_status_snapshot(self):
        graph = self._make_cycle_graph()
        graph.begin_cycle("Loop")
        graph.step_cycle("Loop", quality_score=0.7)
        status = graph.cycle_status()
        self.assertIn("Loop", status)
        self.assertEqual(status["Loop"]["iteration"], 1)
        self.assertIn(0.7, status["Loop"]["quality_history"])

    # --- reset ---

    def test_reset_clears_state(self):
        graph = self._make_cycle_graph()
        graph.begin_cycle("Loop")
        graph.step_cycle("Loop")
        graph.reset_cycle("Loop")
        self.assertIsNone(graph.cycle_state("Loop"))
        self.assertFalse(graph.should_break_cycle("Loop"))


if __name__ == "__main__":
    unittest.main()

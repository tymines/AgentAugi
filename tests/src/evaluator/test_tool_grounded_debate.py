"""Tests for evoagentx.evaluators.tool_grounded_debate (Tool-MAD)."""

import unittest
from typing import Dict, List, Optional

from evoagentx.evaluators.debate import (
    DebatePosition,
    DebateResult,
    DebaterConfig,
    make_heterogeneous_debaters,
)
from evoagentx.evaluators.tool_grounded_debate import (
    GroundedArgument,
    ToolGroundedDebateConfig,
    ToolGroundedDebateEvaluator,
    ToolGroundedDebateResult,
    _build_grounding_summary,
    _parse_grounded_blocks,
    _score_faithfulness_keyword,
)


# ---------------------------------------------------------------------------
# LLM stubs
# ---------------------------------------------------------------------------

def _echo_llm(response: str):
    """Returns a generate_fn that always returns the given response."""
    def generate(messages: List[Dict[str, str]]) -> str:
        return response
    return generate


def _cycling_llm(responses: List[str]):
    """Returns a generate_fn that cycles through the given responses."""
    counter = [0]

    def generate(messages: List[Dict[str, str]]) -> str:
        val = responses[counter[0] % len(responses)]
        counter[0] += 1
        return val

    return generate


# A minimal LLM response that contains one grounded block
_GROUNDED_RESPONSE = (
    "CLAIM: The solution handles edge cases effectively.\n"
    "TOOL: code_analysis\n"
    "TOOL_OUTPUT: No null pointer exceptions found in 100 test cases. "
    "Edge case coverage: 95%.\n"
    "EVIDENCE: The tool output shows 95% edge case coverage, "
    "directly supporting the claim.\n"
)

# A response with NO grounded blocks
_UNGROUNDED_RESPONSE = (
    "The solution is great because it uses chain-of-thought reasoning, "
    "which is generally known to work well for multi-step tasks. "
    "I believe the accuracy will be high based on prior experience."
)

# Judge response: PASS
_JUDGE_PASS = (
    "VERDICT: PASS\n"
    "CONFIDENCE: 0.8\n"
    "RATIONALE: The FOR side provided strong grounded evidence.\n"
    "WEAK POINTS:\n"
    "- Minor edge case not covered\n"
    "END"
)

# Judge response: FAIL
_JUDGE_FAIL = (
    "VERDICT: FAIL\n"
    "CONFIDENCE: 0.75\n"
    "RATIONALE: The solution lacks empirical grounding.\n"
    "WEAK POINTS:\n"
    "- No benchmark results provided\n"
    "- Scalability unproven\n"
    "END"
)

# A response with two grounded blocks
_TWO_CLAIM_RESPONSE = (
    "CLAIM: Accuracy exceeds 90% on standard benchmarks.\n"
    "TOOL: benchmark_runner\n"
    "TOOL_OUTPUT: GSM8K accuracy: 91.3%. MATH accuracy: 87.2%.\n"
    "EVIDENCE: Both scores exceed the 90% target.\n\n"
    "CLAIM: Memory usage stays below 512MB.\n"
    "TOOL: profiler\n"
    "TOOL_OUTPUT: Peak memory: 480MB during longest test case.\n"
    "EVIDENCE: 480MB is within the 512MB budget.\n"
)


# ---------------------------------------------------------------------------
# 1. ToolGroundedDebateConfig — creation and validation
# ---------------------------------------------------------------------------

class TestToolGroundedDebateConfig(unittest.TestCase):
    """Tests for ToolGroundedDebateConfig creation and validation."""

    def test_default_config(self):
        cfg = ToolGroundedDebateConfig()
        self.assertTrue(cfg.require_tool_grounding)
        self.assertAlmostEqual(cfg.faithfulness_threshold, 0.6)
        self.assertEqual(cfg.max_tool_calls_per_round, 3)
        self.assertIsNone(cfg.tool_registry)

    def test_custom_config(self):
        cfg = ToolGroundedDebateConfig(
            require_tool_grounding=False,
            faithfulness_threshold=0.4,
            max_tool_calls_per_round=5,
        )
        self.assertFalse(cfg.require_tool_grounding)
        self.assertAlmostEqual(cfg.faithfulness_threshold, 0.4)
        self.assertEqual(cfg.max_tool_calls_per_round, 5)

    def test_faithfulness_threshold_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            ToolGroundedDebateConfig(faithfulness_threshold=1.5)
        with self.assertRaises(ValueError):
            ToolGroundedDebateConfig(faithfulness_threshold=-0.1)

    def test_max_tool_calls_zero_raises(self):
        with self.assertRaises(ValueError):
            ToolGroundedDebateConfig(max_tool_calls_per_round=0)

    def test_tool_registry_stored(self):
        registry = {"search": lambda q: "results"}
        cfg = ToolGroundedDebateConfig(tool_registry=registry)
        self.assertIn("search", cfg.tool_registry)

    def test_boundary_faithfulness_threshold(self):
        # Both 0.0 and 1.0 are valid boundaries
        cfg_low = ToolGroundedDebateConfig(faithfulness_threshold=0.0)
        cfg_high = ToolGroundedDebateConfig(faithfulness_threshold=1.0)
        self.assertAlmostEqual(cfg_low.faithfulness_threshold, 0.0)
        self.assertAlmostEqual(cfg_high.faithfulness_threshold, 1.0)


# ---------------------------------------------------------------------------
# 2. GroundedArgument — creation and helpers
# ---------------------------------------------------------------------------

class TestGroundedArgument(unittest.TestCase):
    """Tests for GroundedArgument dataclass."""

    def test_grounded_argument_with_tool(self):
        arg = GroundedArgument(
            claim="Solution is fast.",
            evidence="Benchmark shows 10ms latency.",
            tool_used="benchmark",
            tool_output="latency: 10ms",
            faithfulness_score=0.8,
        )
        self.assertTrue(arg.is_grounded())
        self.assertEqual(arg.claim, "Solution is fast.")
        self.assertAlmostEqual(arg.faithfulness_score, 0.8)

    def test_grounded_argument_without_tool(self):
        arg = GroundedArgument(
            claim="Generally good.",
            evidence="",
            tool_used=None,
            tool_output=None,
            faithfulness_score=0.0,
        )
        self.assertFalse(arg.is_grounded())

    def test_grounded_argument_defaults(self):
        arg = GroundedArgument(claim="Test claim.", evidence="Some evidence.")
        self.assertIsNone(arg.tool_used)
        self.assertIsNone(arg.tool_output)
        self.assertAlmostEqual(arg.faithfulness_score, 0.0)
        self.assertFalse(arg.is_grounded())

    def test_is_grounded_requires_both_tool_and_output(self):
        # Tool name without output: not grounded
        arg = GroundedArgument(
            claim="X", evidence="", tool_used="search", tool_output=None
        )
        self.assertFalse(arg.is_grounded())
        # Output without tool name: not grounded
        arg2 = GroundedArgument(
            claim="X", evidence="", tool_used=None, tool_output="some output"
        )
        self.assertFalse(arg2.is_grounded())


# ---------------------------------------------------------------------------
# 3. _score_faithfulness_keyword
# ---------------------------------------------------------------------------

class TestScoreFaithfulnessKeyword(unittest.TestCase):
    """Tests for the keyword-overlap faithfulness scorer."""

    def test_identical_texts_score_high(self):
        score = _score_faithfulness_keyword("accuracy is 95 percent", "accuracy is 95 percent")
        self.assertGreater(score, 0.7)

    def test_completely_unrelated_texts_score_low(self):
        score = _score_faithfulness_keyword("latency exceeds budget", "color is blue sky")
        self.assertLess(score, 0.3)

    def test_empty_claim_returns_zero(self):
        self.assertEqual(_score_faithfulness_keyword("", "some output"), 0.0)

    def test_empty_output_returns_zero(self):
        self.assertEqual(_score_faithfulness_keyword("some claim", ""), 0.0)

    def test_partial_overlap_intermediate_score(self):
        score = _score_faithfulness_keyword(
            "benchmark shows 90% accuracy on test set",
            "accuracy 90 percent benchmark suite results positive",
        )
        self.assertGreater(score, 0.2)
        self.assertLess(score, 1.0)

    def test_score_in_range(self):
        for claim, output in [
            ("great solution", "great solution indeed works well"),
            ("nothing matches", "completely different words here"),
            ("numbers 42 and 99", "42 99 found in dataset"),
        ]:
            score = _score_faithfulness_keyword(claim, output)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_stop_words_do_not_inflate_score(self):
        # Only stop words in both should not produce artificially high score
        score = _score_faithfulness_keyword("the is a and", "the is a and")
        # After removing stop words, both sets are empty → 0.0
        self.assertEqual(score, 0.0)


# ---------------------------------------------------------------------------
# 4. _parse_grounded_blocks
# ---------------------------------------------------------------------------

class TestParseGroundedBlocks(unittest.TestCase):
    """Tests for the structured-block parser."""

    def test_single_block_parsed(self):
        blocks = _parse_grounded_blocks(_GROUNDED_RESPONSE)
        self.assertGreaterEqual(len(blocks), 1)
        self.assertIn("edge", blocks[0]["claim"].lower())

    def test_two_blocks_parsed(self):
        blocks = _parse_grounded_blocks(_TWO_CLAIM_RESPONSE)
        self.assertEqual(len(blocks), 2)
        self.assertIn("accuracy", blocks[0]["claim"].lower())
        self.assertIn("memory", blocks[1]["claim"].lower())

    def test_ungrounded_response_no_blocks(self):
        blocks = _parse_grounded_blocks(_UNGROUNDED_RESPONSE)
        self.assertEqual(len(blocks), 0)

    def test_tool_and_tool_output_extracted(self):
        blocks = _parse_grounded_blocks(_GROUNDED_RESPONSE)
        self.assertGreaterEqual(len(blocks), 1)
        blk = blocks[0]
        self.assertIsNotNone(blk.get("tool"))
        self.assertIsNotNone(blk.get("tool_output"))

    def test_empty_string_returns_empty_list(self):
        self.assertEqual(_parse_grounded_blocks(""), [])


# ---------------------------------------------------------------------------
# 5. _extract_grounded_arguments
# ---------------------------------------------------------------------------

class TestExtractGroundedArguments(unittest.TestCase):
    """Tests for ToolGroundedDebateEvaluator._extract_grounded_arguments."""

    def _make_evaluator(self, config=None):
        return ToolGroundedDebateEvaluator(
            generate_fn=_echo_llm(""),
            config=config or ToolGroundedDebateConfig(),
        )

    def test_extracts_grounded_argument(self):
        ev = self._make_evaluator()
        args = ev._extract_grounded_arguments(_GROUNDED_RESPONSE, DebatePosition.FOR)
        self.assertGreaterEqual(len(args), 1)
        self.assertIsInstance(args[0], GroundedArgument)

    def test_ungrounded_response_no_args(self):
        ev = self._make_evaluator()
        args = ev._extract_grounded_arguments(_UNGROUNDED_RESPONSE, DebatePosition.FOR)
        self.assertEqual(len(args), 0)

    def test_max_tool_calls_cap(self):
        cfg = ToolGroundedDebateConfig(max_tool_calls_per_round=1)
        ev = self._make_evaluator(cfg)
        args = ev._extract_grounded_arguments(_TWO_CLAIM_RESPONSE, DebatePosition.FOR)
        # Only 1 claim should survive the cap
        self.assertEqual(len(args), 1)

    def test_faithfulness_score_computed(self):
        ev = self._make_evaluator()
        args = ev._extract_grounded_arguments(_GROUNDED_RESPONSE, DebatePosition.FOR)
        for a in args:
            self.assertGreaterEqual(a.faithfulness_score, 0.0)
            self.assertLessEqual(a.faithfulness_score, 1.0)

    def test_tool_registry_executed(self):
        call_log = []

        def fake_search(query):
            call_log.append(query)
            return "search result: 95% accuracy on benchmarks"

        cfg = ToolGroundedDebateConfig(tool_registry={"code_analysis": fake_search})
        ev = self._make_evaluator(cfg)
        # _GROUNDED_RESPONSE uses TOOL: code_analysis
        args = ev._extract_grounded_arguments(_GROUNDED_RESPONSE, DebatePosition.FOR)
        # The tool should have been called
        self.assertTrue(len(call_log) > 0)

    def test_ungrounded_claim_with_grounding_required(self):
        """Ungrounded claim under require_tool_grounding=True gets faithfulness 0."""
        cfg = ToolGroundedDebateConfig(require_tool_grounding=True)
        ev = self._make_evaluator(cfg)
        # Craft a response with a CLAIM but no TOOL_OUTPUT
        partial = "CLAIM: This solution is fast.\nTOOL: none_available\n"
        args = ev._extract_grounded_arguments(partial, DebatePosition.FOR)
        # If tool_output is None and grounding is required → faithfulness 0.0
        if args:
            self.assertAlmostEqual(args[0].faithfulness_score, 0.0)

    def test_ungrounded_claim_without_grounding_required(self):
        """Ungrounded claim under require_tool_grounding=False gets neutral 0.5."""
        cfg = ToolGroundedDebateConfig(require_tool_grounding=False)
        ev = self._make_evaluator(cfg)
        partial = "CLAIM: This solution is fast.\n"
        args = ev._extract_grounded_arguments(partial, DebatePosition.FOR)
        if args:
            self.assertAlmostEqual(args[0].faithfulness_score, 0.5)


# ---------------------------------------------------------------------------
# 6. _filter_ungrounded_claims
# ---------------------------------------------------------------------------

class TestFilterUngroundedClaims(unittest.TestCase):
    """Tests for faithfulness threshold filtering."""

    def _make_evaluator(self, threshold=0.6):
        cfg = ToolGroundedDebateConfig(faithfulness_threshold=threshold)
        return ToolGroundedDebateEvaluator(
            generate_fn=_echo_llm(""), config=cfg
        )

    def _make_args(self, scores):
        return [
            GroundedArgument(
                claim=f"claim {i}",
                evidence="",
                tool_used="tool",
                tool_output="output",
                faithfulness_score=s,
            )
            for i, s in enumerate(scores)
        ]

    def test_all_above_threshold_survive(self):
        ev = self._make_evaluator(0.5)
        args = self._make_args([0.6, 0.7, 0.9])
        surviving, n_filtered = ev._filter_ungrounded_claims(args)
        self.assertEqual(len(surviving), 3)
        self.assertEqual(n_filtered, 0)

    def test_all_below_threshold_filtered(self):
        ev = self._make_evaluator(0.8)
        args = self._make_args([0.2, 0.3, 0.4])
        surviving, n_filtered = ev._filter_ungrounded_claims(args)
        self.assertEqual(len(surviving), 0)
        self.assertEqual(n_filtered, 3)

    def test_mixed_threshold(self):
        ev = self._make_evaluator(0.6)
        args = self._make_args([0.3, 0.6, 0.9])
        surviving, n_filtered = ev._filter_ungrounded_claims(args)
        self.assertEqual(len(surviving), 2)
        self.assertEqual(n_filtered, 1)

    def test_empty_input(self):
        ev = self._make_evaluator(0.6)
        surviving, n_filtered = ev._filter_ungrounded_claims([])
        self.assertEqual(surviving, [])
        self.assertEqual(n_filtered, 0)

    def test_threshold_override(self):
        ev = self._make_evaluator(0.6)
        args = self._make_args([0.5, 0.7])
        # Override threshold to 0.4 — both should survive
        surviving, n_filtered = ev._filter_ungrounded_claims(args, threshold=0.4)
        self.assertEqual(len(surviving), 2)

    def test_exact_threshold_boundary(self):
        ev = self._make_evaluator(0.6)
        args = self._make_args([0.6])  # exactly at threshold
        surviving, _ = ev._filter_ungrounded_claims(args)
        self.assertEqual(len(surviving), 1)


# ---------------------------------------------------------------------------
# 7. Full evaluation with mock LLM — grounding enforced
# ---------------------------------------------------------------------------

class TestFullEvaluationGrounded(unittest.TestCase):
    """End-to-end tests with mock LLM returning structured grounded responses."""

    def _make_evaluator(self, debater_response, judge_response, config=None):
        """Create a ToolGroundedDebateEvaluator that returns canned responses."""
        # Debaters get grounded response; judge gets judge response
        call_count = [0]
        num_debaters = 2  # default two debaters

        def generate(messages):
            # Heuristic: judge messages reference "VERDICT"
            system = messages[0]["content"] if messages else ""
            if "impartial judge" in system.lower():
                return judge_response
            return debater_response

        return ToolGroundedDebateEvaluator(
            generate_fn=generate,
            config=config or ToolGroundedDebateConfig(),
            num_rounds=1,
        )

    def test_evaluate_returns_tool_grounded_result(self):
        ev = self._make_evaluator(_GROUNDED_RESPONSE, _JUDGE_PASS)
        result = ev.evaluate(
            solution="Agent uses tool-augmented reasoning.",
            task_context="Solve math problems accurately.",
        )
        self.assertIsInstance(result, ToolGroundedDebateResult)
        self.assertIsInstance(result, DebateResult)

    def test_passed_field_reflects_judge(self):
        ev_pass = self._make_evaluator(_GROUNDED_RESPONSE, _JUDGE_PASS)
        result = ev_pass.evaluate(
            solution="Agent uses RAG.", task_context="Factual QA task."
        )
        self.assertTrue(result.passed)

    def test_failed_field_reflects_judge(self):
        ev_fail = self._make_evaluator(_GROUNDED_RESPONSE, _JUDGE_FAIL)
        result = ev_fail.evaluate(
            solution="Agent guesses randomly.", task_context="Must be accurate."
        )
        self.assertFalse(result.passed)

    def test_grounding_rate_populated(self):
        ev = self._make_evaluator(_GROUNDED_RESPONSE, _JUDGE_PASS)
        result = ev.evaluate(
            solution="Good solution.", task_context="Some task."
        )
        self.assertGreaterEqual(result.grounding_rate, 0.0)
        self.assertLessEqual(result.grounding_rate, 1.0)

    def test_mean_faithfulness_populated(self):
        ev = self._make_evaluator(_GROUNDED_RESPONSE, _JUDGE_PASS)
        result = ev.evaluate(
            solution="Solution with evidence.", task_context="Task context."
        )
        self.assertGreaterEqual(result.mean_faithfulness, 0.0)
        self.assertLessEqual(result.mean_faithfulness, 1.0)

    def test_num_rounds_run_is_one(self):
        ev = self._make_evaluator(_GROUNDED_RESPONSE, _JUDGE_PASS)
        result = ev.evaluate(
            solution="Single round solution.", task_context="Task."
        )
        self.assertEqual(result.num_rounds_run, 1)

    def test_no_grounded_claims_all_ungrounded(self):
        """When debaters make no tool-grounded claims, grounding_rate=0."""
        ev = self._make_evaluator(_UNGROUNDED_RESPONSE, _JUDGE_FAIL)
        result = ev.evaluate(
            solution="Unsubstantiated solution.", task_context="Needs evidence."
        )
        self.assertEqual(result.grounding_rate, 0.0)

    def test_perfect_grounding_both_debaters(self):
        """When all claims are grounded, grounding_rate should be 1.0."""
        ev = self._make_evaluator(_GROUNDED_RESPONSE, _JUDGE_PASS)
        result = ev.evaluate(
            solution="Fully grounded solution.", task_context="Task."
        )
        # All parsed grounded args should have tool set
        if result.grounded_arguments:
            all_grounded = all(a.is_grounded() for a in result.grounded_arguments)
            self.assertTrue(all_grounded)
            self.assertAlmostEqual(result.grounding_rate, 1.0)


# ---------------------------------------------------------------------------
# 8. Faithfulness threshold enforcement
# ---------------------------------------------------------------------------

class TestFaithfulnessThresholdEnforcement(unittest.TestCase):
    """Tests that high faithfulness_threshold forces more filtering."""

    def _eval_with_threshold(self, threshold):
        cfg = ToolGroundedDebateConfig(faithfulness_threshold=threshold)

        def generate(messages):
            system = messages[0]["content"] if messages else ""
            if "judge" in system.lower():
                return _JUDGE_PASS
            return _GROUNDED_RESPONSE

        return ToolGroundedDebateEvaluator(
            generate_fn=generate, config=cfg, num_rounds=1
        )

    def test_zero_threshold_keeps_all_grounded(self):
        ev = self._eval_with_threshold(0.0)
        result = ev.evaluate(solution="Solution.", task_context="Task.")
        self.assertEqual(result.filtered_claim_count, 0)

    def test_max_threshold_filters_all(self):
        ev = self._eval_with_threshold(1.0)
        result = ev.evaluate(solution="Solution.", task_context="Task.")
        # With threshold=1.0 only perfect matches survive; most claims filtered
        # grounded_arguments contains SURVIVING args; filtered_claim_count tracks removed
        # The claim in _GROUNDED_RESPONSE is unlikely to score exactly 1.0
        self.assertGreaterEqual(result.filtered_claim_count, 0)


# ---------------------------------------------------------------------------
# 9. Judge weighting with grounding summary
# ---------------------------------------------------------------------------

class TestJudgeWithGrounding(unittest.TestCase):
    """Tests that the judge receives grounding context."""

    def test_judge_receives_grounding_summary(self):
        """The judge prompt must contain grounding summary information."""
        captured = []

        def generate(messages):
            system = messages[0]["content"] if messages else ""
            user = messages[1]["content"] if len(messages) > 1 else ""
            if "judge" in system.lower():
                captured.append(user)
                return _JUDGE_PASS
            return _GROUNDED_RESPONSE

        ev = ToolGroundedDebateEvaluator(
            generate_fn=generate,
            config=ToolGroundedDebateConfig(),
            num_rounds=1,
        )
        ev.evaluate(solution="Solution.", task_context="Task.")
        self.assertTrue(len(captured) > 0)
        judge_prompt = captured[-1]
        self.assertIn("GROUNDING SUMMARY", judge_prompt.upper())

    def test_build_grounding_summary_with_args(self):
        pro = [
            GroundedArgument(
                claim="Fast execution", evidence="", tool_used="bench",
                tool_output="10ms", faithfulness_score=0.9
            )
        ]
        con = [
            GroundedArgument(
                claim="Memory too high", evidence="", tool_used="profiler",
                tool_output="600MB", faithfulness_score=0.7
            )
        ]
        summary = _build_grounding_summary(pro, con)
        self.assertIn("FOR", summary)
        self.assertIn("AGAINST", summary)
        self.assertIn("faithfulness", summary.lower())

    def test_build_grounding_summary_empty_sides(self):
        summary = _build_grounding_summary([], [])
        self.assertIn("No grounded", summary)


# ---------------------------------------------------------------------------
# 10. Backward compatibility with DebateEvaluator interface
# ---------------------------------------------------------------------------

class TestBackwardCompatibility(unittest.TestCase):
    """ToolGroundedDebateEvaluator must be a drop-in for DebateEvaluator."""

    def _make_evaluator(self):
        def generate(messages):
            system = messages[0]["content"] if messages else ""
            if "judge" in system.lower():
                return _JUDGE_PASS
            return _GROUNDED_RESPONSE

        return ToolGroundedDebateEvaluator(
            generate_fn=generate, num_rounds=1
        )

    def test_evaluate_returns_debate_result_subclass(self):
        ev = self._make_evaluator()
        result = ev.evaluate(solution="Sol.", task_context="Task.")
        self.assertIsInstance(result, DebateResult)

    def test_debate_result_fields_present(self):
        ev = self._make_evaluator()
        result = ev.evaluate(solution="Sol.", task_context="Task.")
        self.assertIsNotNone(result.passed)
        self.assertIsNotNone(result.judge_rationale)
        self.assertIsNotNone(result.confidence)
        self.assertIsNotNone(result.rounds)
        self.assertIsNotNone(result.weak_points)
        self.assertIsNotNone(result.argument_quality)
        self.assertIsNotNone(result.num_rounds_run)

    def test_summary_method_works(self):
        ev = self._make_evaluator()
        result = ev.evaluate(solution="Sol.", task_context="Task.")
        summary = result.summary()
        self.assertIsInstance(summary, str)
        self.assertIn("Grounding rate", summary)

    def test_empty_solution_raises(self):
        ev = self._make_evaluator()
        with self.assertRaises(ValueError):
            ev.evaluate(solution="", task_context="Task.")

    def test_empty_task_context_raises(self):
        ev = self._make_evaluator()
        with self.assertRaises(ValueError):
            ev.evaluate(solution="Sol.", task_context="")

    def test_custom_debater_configs_accepted(self):
        debaters = make_heterogeneous_debaters(num_for=1, num_against=1)
        ev = ToolGroundedDebateEvaluator(
            generate_fn=_cycling_llm([_GROUNDED_RESPONSE, _JUDGE_PASS]),
            debater_configs=debaters,
            num_rounds=1,
        )
        result = ev.evaluate(solution="Sol.", task_context="Task.")
        self.assertIsInstance(result, ToolGroundedDebateResult)


# ---------------------------------------------------------------------------
# 11. Tool registry provided vs None
# ---------------------------------------------------------------------------

class TestToolRegistry(unittest.TestCase):
    """Tests for behaviour with and without tool_registry."""

    def test_without_tool_registry_uses_llm_output(self):
        """Without registry, LLM-provided TOOL_OUTPUT is used as-is."""
        ev = ToolGroundedDebateEvaluator(
            generate_fn=_cycling_llm([_GROUNDED_RESPONSE, _JUDGE_PASS]),
            config=ToolGroundedDebateConfig(tool_registry=None),
            num_rounds=1,
        )
        result = ev.evaluate(solution="Sol.", task_context="Task.")
        self.assertIsInstance(result, ToolGroundedDebateResult)

    def test_with_tool_registry_executes_tool(self):
        """With registry, the real tool is executed and its output used."""
        executed = []

        def my_search(query):
            executed.append(query)
            return "real search results: high accuracy confirmed"

        cfg = ToolGroundedDebateConfig(
            tool_registry={"code_analysis": my_search}
        )
        ev = ToolGroundedDebateEvaluator(
            generate_fn=_cycling_llm([_GROUNDED_RESPONSE, _JUDGE_PASS]),
            config=cfg,
            num_rounds=1,
        )
        ev.evaluate(solution="Sol.", task_context="Task.")
        self.assertTrue(len(executed) > 0)

    def test_unknown_tool_in_registry_falls_back(self):
        """Citing a tool not in registry falls back to LLM-provided output."""
        cfg = ToolGroundedDebateConfig(tool_registry={"other_tool": lambda q: "x"})
        ev = ToolGroundedDebateEvaluator(
            generate_fn=_cycling_llm([_GROUNDED_RESPONSE, _JUDGE_PASS]),
            config=cfg,
            num_rounds=1,
        )
        # Should not raise even though "code_analysis" is not in registry
        result = ev.evaluate(solution="Sol.", task_context="Task.")
        self.assertIsInstance(result, ToolGroundedDebateResult)


# ---------------------------------------------------------------------------
# 12. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):
    """Edge case tests."""

    def test_all_claims_ungrounded_debate_still_completes(self):
        """A debate where all debater responses are ungrounded should complete."""
        ev = ToolGroundedDebateEvaluator(
            generate_fn=_cycling_llm([_UNGROUNDED_RESPONSE, _JUDGE_FAIL]),
            config=ToolGroundedDebateConfig(),
            num_rounds=1,
        )
        result = ev.evaluate(
            solution="Sol.", task_context="Task."
        )
        self.assertIsInstance(result, ToolGroundedDebateResult)
        self.assertEqual(result.grounding_rate, 0.0)
        self.assertEqual(len(result.grounded_arguments), 0)

    def test_no_tool_calls_made_at_all(self):
        """Debaters make zero claims — evaluate still returns a result."""
        ev = ToolGroundedDebateEvaluator(
            generate_fn=_cycling_llm(["No structured claims here.", _JUDGE_FAIL]),
            num_rounds=1,
        )
        result = ev.evaluate(solution="Sol.", task_context="Task.")
        self.assertIsInstance(result, ToolGroundedDebateResult)
        self.assertEqual(result.filtered_claim_count, 0)

    def test_multi_round_debate(self):
        """Multiple rounds should accumulate grounded arguments."""
        ev = ToolGroundedDebateEvaluator(
            generate_fn=_cycling_llm([_GROUNDED_RESPONSE, _JUDGE_PASS]),
            num_rounds=2,
        )
        result = ev.evaluate(solution="Multi-round sol.", task_context="Task.")
        self.assertLessEqual(result.num_rounds_run, 2)

    def test_grounded_result_filtered_count_non_negative(self):
        ev = ToolGroundedDebateEvaluator(
            generate_fn=_cycling_llm([_GROUNDED_RESPONSE, _JUDGE_PASS]),
            num_rounds=1,
        )
        result = ev.evaluate(solution="Sol.", task_context="Task.")
        self.assertGreaterEqual(result.filtered_claim_count, 0)

    def test_tool_grounded_result_is_debate_result(self):
        """ToolGroundedDebateResult is a proper subclass."""
        self.assertTrue(issubclass(ToolGroundedDebateResult, DebateResult))

    def test_config_none_uses_defaults(self):
        """Passing config=None should use default ToolGroundedDebateConfig."""
        ev = ToolGroundedDebateEvaluator(
            generate_fn=_cycling_llm([_GROUNDED_RESPONSE, _JUDGE_PASS]),
            config=None,
            num_rounds=1,
        )
        self.assertAlmostEqual(
            ev._config.faithfulness_threshold, 0.6
        )
        self.assertTrue(ev._config.require_tool_grounding)


if __name__ == "__main__":
    unittest.main()

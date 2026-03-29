"""Unit tests for evoagentx.evaluators.debate."""

import unittest
from typing import Any, Dict, List, Optional

from evoagentx.evaluators.debate import (
    DebateArgument,
    DebaterConfig,
    DebateEvaluator,
    DebatePosition,
    DebateResult,
    DebateRound,
    make_heterogeneous_debaters,
    _estimate_argument_quality,
    _word_overlap_similarity,
    _cosine_similarity,
    _extract_weak_points,
    _parse_judge_verdict,
    _heuristic_confidence,
    _extract_confidence,
    _format_transcript,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_echo_llm(response: str = "default response"):
    """Returns a generate_fn that always returns the given response."""
    def generate(messages: List[Dict[str, str]]) -> str:
        return response
    return generate


def _make_turn_taking_llm(responses: List[str]):
    """Returns a generate_fn that cycles through the given responses."""
    call_count = [0]

    def generate(messages: List[Dict[str, str]]) -> str:
        idx = call_count[0] % len(responses)
        call_count[0] += 1
        return responses[idx]

    return generate


def _make_embed(text: str) -> List[float]:
    """Deterministic fake embedding: character frequency fingerprint."""
    counts = [0.0] * 26
    for ch in text.lower():
        if "a" <= ch <= "z":
            counts[ord(ch) - ord("a")] += 1.0
    total = max(1.0, sum(counts))
    return [c / total for c in counts]


# ---------------------------------------------------------------------------
# _estimate_argument_quality
# ---------------------------------------------------------------------------

class TestEstimateArgumentQuality(unittest.TestCase):

    def test_empty_string(self):
        self.assertEqual(_estimate_argument_quality(""), 0.0)

    def test_very_short_string(self):
        score = _estimate_argument_quality("yes")
        self.assertLessEqual(score, 0.15)

    def test_medium_quality_argument(self):
        text = (
            "The solution fails because it lacks error handling. "
            "For example, when the API returns a 500 error, the agent will "
            "silently discard the result rather than retrying. "
            "Therefore this is a critical gap."
        )
        score = _estimate_argument_quality(text)
        self.assertGreater(score, 0.3)
        self.assertLessEqual(score, 1.0)

    def test_heavily_hedged_argument_penalised(self):
        text = (
            "Maybe possibly perhaps the solution might or might not work, "
            "it seems like it could be unclear whether this is valid or not. "
            "Perhaps it might possibly be correct, perhaps not."
        )
        score = _estimate_argument_quality(text)
        # Hedging should pull the score down
        normal_text = "This is a clear and specific argument with evidence. Numbers: 42%."
        normal_score = _estimate_argument_quality(normal_text)
        self.assertLessEqual(score, normal_score)

    def test_score_in_range(self):
        for text in ["short", "a " * 50, "reason because therefore for example 1 2 3"]:
            s = _estimate_argument_quality(text)
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)


# ---------------------------------------------------------------------------
# _word_overlap_similarity
# ---------------------------------------------------------------------------

class TestWordOverlapSimilarity(unittest.TestCase):

    def test_identical_texts(self):
        t = "the quick brown fox"
        self.assertAlmostEqual(_word_overlap_similarity(t, t), 1.0)

    def test_completely_different_texts(self):
        sim = _word_overlap_similarity("alpha beta gamma", "delta epsilon zeta")
        self.assertAlmostEqual(sim, 0.0)

    def test_partial_overlap(self):
        sim = _word_overlap_similarity("cat dog bird", "cat fish snake")
        # 1 common out of 5 unique → Jaccard = 1/5
        self.assertAlmostEqual(sim, 1.0 / 5.0)

    def test_empty_strings(self):
        self.assertAlmostEqual(_word_overlap_similarity("", ""), 1.0)


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity(unittest.TestCase):

    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(_cosine_similarity(v, v), 1.0)

    def test_orthogonal_vectors(self):
        sim = _cosine_similarity([1.0, 0.0], [0.0, 1.0])
        self.assertAlmostEqual(sim, 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            _cosine_similarity([1.0, 2.0], [1.0])

    def test_zero_vector_returns_zero(self):
        self.assertAlmostEqual(_cosine_similarity([0.0, 0.0], [1.0, 0.0]), 0.0)

    def test_clamped_to_zero(self):
        # Anti-parallel should be clamped to 0 (not -1)
        result = _cosine_similarity([-1.0, 0.0], [1.0, 0.0])
        self.assertGreaterEqual(result, 0.0)


# ---------------------------------------------------------------------------
# _extract_confidence
# ---------------------------------------------------------------------------

class TestExtractConfidence(unittest.TestCase):

    def test_numeric_decimal(self):
        result = _extract_confidence("CONFIDENCE: 0.82")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.82, places=3)

    def test_numeric_percentage(self):
        result = _extract_confidence("confidence score: 85%")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.85, places=3)

    def test_qualitative_high(self):
        result = _extract_confidence("confidence: high")
        self.assertIsNotNone(result)
        self.assertGreater(result, 0.7)

    def test_qualitative_low(self):
        result = _extract_confidence("confidence: low")
        self.assertIsNotNone(result)
        self.assertLess(result, 0.5)

    def test_none_when_absent(self):
        result = _extract_confidence("No confidence signal here at all.")
        self.assertIsNone(result)

    def test_clamped_to_unit_interval(self):
        result = _extract_confidence("confidence: 150")
        self.assertIsNotNone(result)
        self.assertLessEqual(result, 1.0)


# ---------------------------------------------------------------------------
# _extract_weak_points
# ---------------------------------------------------------------------------

class TestExtractWeakPoints(unittest.TestCase):

    def test_numbered_list(self):
        text = "RATIONALE: ...\nWEAK POINTS:\n1. Missing error handling\n2. No tests"
        points = _extract_weak_points(text)
        self.assertIn("Missing error handling", points)
        self.assertIn("No tests", points)

    def test_bulleted_list(self):
        text = "Issues:\n- Slow performance\n- Memory leak\n* Edge case failure"
        points = _extract_weak_points(text)
        self.assertGreaterEqual(len(points), 2)

    def test_weakness_keyword(self):
        text = "weakness: The agent does not handle retries."
        points = _extract_weak_points(text)
        self.assertTrue(any("retries" in p for p in points))

    def test_empty_text(self):
        self.assertEqual(_extract_weak_points(""), [])

    def test_max_ten_returned(self):
        items = "\n".join(f"- Point {i}" for i in range(20))
        points = _extract_weak_points(items)
        self.assertLessEqual(len(points), 10)


# ---------------------------------------------------------------------------
# _parse_judge_verdict
# ---------------------------------------------------------------------------

class TestParseJudgeVerdict(unittest.TestCase):

    def test_explicit_pass(self):
        text = "VERDICT: PASS\nCONFIDENCE: 0.9\nRATIONALE: Solution is solid.\nWEAK POINTS:\nEND"
        passed, pos, conf, rationale = _parse_judge_verdict(text)
        self.assertTrue(passed)
        self.assertEqual(pos, DebatePosition.FOR)
        self.assertGreater(conf, 0.5)

    def test_explicit_fail(self):
        text = "VERDICT: FAIL\nCONFIDENCE: 0.75\nRATIONALE: Too many flaws.\nWEAK POINTS:\nEND"
        passed, pos, conf, rationale = _parse_judge_verdict(text)
        self.assertFalse(passed)
        self.assertEqual(pos, DebatePosition.AGAINST)

    def test_fallback_positive_keywords(self):
        text = "The solution is sound and acceptable. It clearly demonstrates correctness."
        passed, _, _, _ = _parse_judge_verdict(text)
        self.assertTrue(passed)

    def test_fallback_negative_keywords(self):
        text = "The solution is flawed and weak. It is insufficient and invalid."
        passed, _, _, _ = _parse_judge_verdict(text)
        self.assertFalse(passed)

    def test_rationale_extracted(self):
        text = "VERDICT: PASS\nCONFIDENCE: 0.8\nRATIONALE: Well reasoned.\nWEAK POINTS:\n- Minor issue\nEND"
        _, _, _, rationale = _parse_judge_verdict(text)
        self.assertIn("reasoned", rationale)


# ---------------------------------------------------------------------------
# DebaterConfig
# ---------------------------------------------------------------------------

class TestDebaterConfig(unittest.TestCase):

    def test_valid_config(self):
        cfg = DebaterConfig(name="Alice", position=DebatePosition.FOR)
        self.assertEqual(cfg.name, "Alice")
        self.assertEqual(cfg.position, DebatePosition.FOR)

    def test_empty_name_raises(self):
        with self.assertRaises(ValueError):
            DebaterConfig(name="", position=DebatePosition.FOR)

    def test_persona_optional(self):
        cfg = DebaterConfig(name="Bob", position=DebatePosition.AGAINST)
        self.assertEqual(cfg.persona, "")


# ---------------------------------------------------------------------------
# DebateResult
# ---------------------------------------------------------------------------

class TestDebateResult(unittest.TestCase):

    def _make_result(self, passed=True, confidence=0.8, weak_points=None):
        return DebateResult(
            passed=passed,
            winning_position=DebatePosition.FOR if passed else DebatePosition.AGAINST,
            judge_rationale="Some rationale here.",
            confidence=confidence,
            rounds=[],
            weak_points=weak_points or [],
            argument_quality=0.6,
            num_rounds_run=2,
        )

    def test_summary_contains_pass(self):
        r = self._make_result(passed=True)
        self.assertIn("PASSED", r.summary())

    def test_summary_contains_fail(self):
        r = self._make_result(passed=False)
        self.assertIn("FAILED", r.summary())

    def test_summary_lists_weak_points(self):
        r = self._make_result(weak_points=["Missing retries", "No logging"])
        summary = r.summary()
        self.assertIn("Missing retries", summary)
        self.assertIn("No logging", summary)


# ---------------------------------------------------------------------------
# _format_transcript
# ---------------------------------------------------------------------------

class TestFormatTranscript(unittest.TestCase):

    def test_empty_rounds(self):
        self.assertEqual(_format_transcript([]), "")

    def test_single_round_single_argument(self):
        arg = DebateArgument(
            debater_name="Alice",
            position=DebatePosition.FOR,
            round_number=1,
            argument="This solution works well.",
        )
        rnd = DebateRound(round_number=1, arguments=[arg])
        transcript = _format_transcript([rnd])
        self.assertIn("Round 1", transcript)
        self.assertIn("Alice", transcript)
        self.assertIn("FOR", transcript)
        self.assertIn("This solution works well.", transcript)


# ---------------------------------------------------------------------------
# make_heterogeneous_debaters
# ---------------------------------------------------------------------------

class TestMakeHeterogeneousDebaters(unittest.TestCase):

    def test_default_1_for_2_against(self):
        configs = make_heterogeneous_debaters(num_for=1, num_against=2)
        self.assertEqual(len(configs), 3)
        for_count = sum(1 for c in configs if c.position == DebatePosition.FOR)
        against_count = sum(1 for c in configs if c.position == DebatePosition.AGAINST)
        self.assertEqual(for_count, 1)
        self.assertEqual(against_count, 2)

    def test_all_have_names_and_personas(self):
        configs = make_heterogeneous_debaters(2, 2)
        for c in configs:
            self.assertTrue(c.name)
            self.assertTrue(c.persona)

    def test_invalid_zero_raises(self):
        with self.assertRaises(ValueError):
            make_heterogeneous_debaters(num_for=0, num_against=1)
        with self.assertRaises(ValueError):
            make_heterogeneous_debaters(num_for=1, num_against=0)


# ---------------------------------------------------------------------------
# DebateEvaluator — construction
# ---------------------------------------------------------------------------

class TestDebateEvaluatorConstruction(unittest.TestCase):

    def test_default_construction(self):
        ev = DebateEvaluator(generate_fn=_make_echo_llm())
        self.assertEqual(ev.num_rounds, 3)

    def test_invalid_num_rounds(self):
        with self.assertRaises(ValueError):
            DebateEvaluator(generate_fn=_make_echo_llm(), num_rounds=0)

    def test_invalid_convergence_threshold(self):
        with self.assertRaises(ValueError):
            DebateEvaluator(generate_fn=_make_echo_llm(), convergence_threshold=0.0)
        with self.assertRaises(ValueError):
            DebateEvaluator(generate_fn=_make_echo_llm(), convergence_threshold=1.5)

    def test_invalid_pass_threshold(self):
        with self.assertRaises(ValueError):
            DebateEvaluator(generate_fn=_make_echo_llm(), pass_threshold=-0.1)

    def test_missing_for_debater_raises(self):
        configs = [DebaterConfig("C", DebatePosition.AGAINST)]
        with self.assertRaises(ValueError):
            DebateEvaluator(generate_fn=_make_echo_llm(), debater_configs=configs)

    def test_missing_against_debater_raises(self):
        configs = [DebaterConfig("A", DebatePosition.FOR)]
        with self.assertRaises(ValueError):
            DebateEvaluator(generate_fn=_make_echo_llm(), debater_configs=configs)

    def test_empty_debaters_raises(self):
        with self.assertRaises(ValueError):
            DebateEvaluator(generate_fn=_make_echo_llm(), debater_configs=[])


# ---------------------------------------------------------------------------
# DebateEvaluator — evaluate() happy path
# ---------------------------------------------------------------------------

class TestDebateEvaluatorEvaluate(unittest.TestCase):

    SOLUTION = "Use chain-of-thought prompting to solve math problems."
    TASK = "Solve GSM8K arithmetic tasks with 85% accuracy."

    # Number of debaters used by default DebateEvaluator configs
    _N_DEFAULT_DEBATERS = 2

    def _make_pass_evaluator(self, num_rounds=1):
        """Evaluator whose judge always returns PASS."""
        # Debater calls: n_debaters * num_rounds; judge call comes last.
        debater_response = (
            "This solution is effective because it forces explicit reasoning steps. "
            "Evidence: chain-of-thought improves GPT-4 accuracy by 25 percentage points."
        )
        judge_response = (
            "VERDICT: PASS\n"
            "CONFIDENCE: 0.85\n"
            "RATIONALE: The chain-of-thought approach is well supported.\n"
            "WEAK POINTS:\n"
            "- May be slow on simple arithmetic.\n"
            "END"
        )
        n_debater_calls = self._N_DEFAULT_DEBATERS * num_rounds
        responses = [debater_response] * n_debater_calls + [judge_response]
        return DebateEvaluator(
            generate_fn=_make_turn_taking_llm(responses),
            num_rounds=num_rounds,
        )

    def _make_fail_evaluator(self):
        """Evaluator whose judge always returns FAIL."""
        debater_response = "This is a weak argument without specific evidence."
        judge_response = (
            "VERDICT: FAIL\n"
            "CONFIDENCE: 0.78\n"
            "RATIONALE: The solution lacks concrete error handling.\n"
            "WEAK POINTS:\n"
            "- No retry logic.\n"
            "- No fallback mechanism.\n"
            "END"
        )
        n_debater_calls = self._N_DEFAULT_DEBATERS * 1  # num_rounds=1
        responses = [debater_response] * n_debater_calls + [judge_response]
        return DebateEvaluator(
            generate_fn=_make_turn_taking_llm(responses),
            num_rounds=1,
        )

    def test_evaluate_returns_debate_result(self):
        ev = self._make_pass_evaluator()
        result = ev.evaluate(self.SOLUTION, self.TASK)
        self.assertIsInstance(result, DebateResult)

    def test_pass_verdict(self):
        ev = self._make_pass_evaluator()
        result = ev.evaluate(self.SOLUTION, self.TASK)
        self.assertTrue(result.passed)

    def test_fail_verdict(self):
        ev = self._make_fail_evaluator()
        result = ev.evaluate(self.SOLUTION, self.TASK)
        self.assertFalse(result.passed)

    def test_rounds_populated(self):
        ev = self._make_pass_evaluator(num_rounds=2)
        result = ev.evaluate(self.SOLUTION, self.TASK)
        self.assertGreaterEqual(result.num_rounds_run, 1)
        self.assertLessEqual(result.num_rounds_run, 2)

    def test_each_round_has_both_positions(self):
        ev = self._make_pass_evaluator(num_rounds=1)
        result = ev.evaluate(self.SOLUTION, self.TASK)
        first_round = result.rounds[0]
        positions = {arg.position for arg in first_round.arguments}
        self.assertIn(DebatePosition.FOR, positions)
        self.assertIn(DebatePosition.AGAINST, positions)

    def test_argument_quality_in_range(self):
        ev = self._make_pass_evaluator()
        result = ev.evaluate(self.SOLUTION, self.TASK)
        self.assertGreaterEqual(result.argument_quality, 0.0)
        self.assertLessEqual(result.argument_quality, 1.0)

    def test_weak_points_extracted(self):
        ev = self._make_fail_evaluator()
        result = ev.evaluate(self.SOLUTION, self.TASK)
        self.assertGreater(len(result.weak_points), 0)

    def test_empty_solution_raises(self):
        ev = self._make_pass_evaluator()
        with self.assertRaises(ValueError):
            ev.evaluate("", self.TASK)

    def test_empty_task_raises(self):
        ev = self._make_pass_evaluator()
        with self.assertRaises(ValueError):
            ev.evaluate(self.SOLUTION, "")

    def test_confidence_below_pass_threshold_fails(self):
        """A PASS with low confidence should be overridden to FAIL."""
        debater_response = "Good because it is structured and clear."
        # Low confidence PASS
        judge_response = (
            "VERDICT: PASS\n"
            "CONFIDENCE: 0.30\n"
            "RATIONALE: Somewhat acceptable but not sure.\n"
            "END"
        )
        responses = [debater_response] * 2 + [judge_response]  # 2 debaters × 1 round
        ev = DebateEvaluator(
            generate_fn=_make_turn_taking_llm(responses),
            num_rounds=1,
            pass_threshold=0.60,  # Requires confidence >= 0.6
        )
        result = ev.evaluate(self.SOLUTION, self.TASK)
        self.assertFalse(result.passed)

    def test_summary_is_string(self):
        ev = self._make_pass_evaluator()
        result = ev.evaluate(self.SOLUTION, self.TASK)
        summary = result.summary()
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 10)


# ---------------------------------------------------------------------------
# DebateEvaluator — convergence detection
# ---------------------------------------------------------------------------

class TestDebateConvergence(unittest.TestCase):

    SOLUTION = "Use RAG to answer knowledge-intensive questions."
    TASK = "Answer NaturalQuestions with 70% accuracy."

    def test_convergence_stops_debate_early(self):
        """If every round produces identical text, convergence triggers early."""
        same_response = (
            "This approach works well because retrieval augments generation. "
            "Evidence demonstrates strong performance on QA benchmarks."
        )
        judge_response = "VERDICT: PASS\nCONFIDENCE: 0.9\nRATIONALE: Good.\nEND"
        responses = [same_response] * 20 + [judge_response]

        ev = DebateEvaluator(
            generate_fn=_make_turn_taking_llm(responses),
            num_rounds=5,
            convergence_threshold=0.85,
            embed_fn=_make_embed,  # Use our fake embedder
        )
        result = ev.evaluate(self.SOLUTION, self.TASK)
        # Should converge before all 5 rounds complete
        self.assertLess(result.num_rounds_run, 5)

    def test_no_convergence_when_responses_differ(self):
        """Diverse responses should not trigger early convergence."""
        responses_pool = [
            "The solution has merit because of its retrieval precision and recall trade-off.",
            "This approach fails because retrieval errors cascade into generation errors, specifically hallucinations.",
            "Historical evidence shows RAG consistently improves accuracy on factual QA.",
            "The weakness is latency: every query requires a retrieval step adding 300ms.",
            "Evidence from REALM, RAG, and Atlas papers supports this design choice.",
            "Failure mode: when the retrieved context is off-topic, the model still uses it.",
        ]
        judge_response = "VERDICT: PASS\nCONFIDENCE: 0.85\nRATIONALE: Good.\nEND"
        all_responses = responses_pool * 3 + [judge_response]

        ev = DebateEvaluator(
            generate_fn=_make_turn_taking_llm(all_responses),
            num_rounds=3,
            convergence_threshold=0.99,  # Very tight — won't trigger easily
        )
        result = ev.evaluate(self.SOLUTION, self.TASK)
        self.assertEqual(result.num_rounds_run, 3)


# ---------------------------------------------------------------------------
# DebateEvaluator — debater generation failure
# ---------------------------------------------------------------------------

class TestDebateGenerationFailure(unittest.TestCase):

    def test_failing_generate_fn_does_not_crash(self):
        """A generate_fn that raises should not crash the evaluator."""
        call_count = [0]
        judge_response = "VERDICT: PASS\nCONFIDENCE: 0.9\nRATIONALE: OK.\nEND"

        def fragile_generate(messages):
            call_count[0] += 1
            # Fail on first call; succeed (with judge response) thereafter
            if call_count[0] == 1:
                raise RuntimeError("API timeout")
            return judge_response

        ev = DebateEvaluator(
            generate_fn=fragile_generate,
            num_rounds=1,
        )
        # Should not raise even though one debater call failed
        result = ev.evaluate(
            "A proposed solution.",
            "A task context.",
        )
        self.assertIsInstance(result, DebateResult)


# ---------------------------------------------------------------------------
# DebateEvaluator — alignment drift integration (mocked)
# ---------------------------------------------------------------------------

class TestDebateAlignmentIntegration(unittest.TestCase):

    def test_drift_report_attached_when_no_agent_fn(self):
        """Without solution_agent_fn the drift_report stays None."""
        ev = DebateEvaluator(generate_fn=_make_echo_llm("VERDICT: PASS\nCONFIDENCE: 0.9\nRATIONALE: ok\nEND"))
        result = ev.evaluate("solution", "task context")
        # No agent fn → no drift
        self.assertIsNone(result.drift_report)

    def test_drift_override_forces_fail(self):
        """A drift-exceeding report should override a PASS verdict to FAIL."""

        class _FakeReport:
            any_exceeded = True

            def summary(self):
                return "Drift exceeded."

        class _FakeDetector:
            baseline_captured = True

            def check(self, agent_fn=None, **kwargs):
                return _FakeReport()

        judge_response = (
            "VERDICT: PASS\nCONFIDENCE: 0.95\nRATIONALE: Very good solution.\nEND"
        )
        responses = ["Good argument because of evidence."] * 10 + [judge_response]

        ev = DebateEvaluator(
            generate_fn=_make_turn_taking_llm(responses),
            num_rounds=1,
            alignment_detector=_FakeDetector(),
        )

        def dummy_agent(example):
            return "ok"

        result = ev.evaluate(
            "solution text here",
            "task context",
            solution_agent_fn=dummy_agent,
            evaluation_inputs=[{"q": "x"}],
        )
        # Even though judge said PASS, drift detector overrides to FAIL
        self.assertFalse(result.passed)
        self.assertTrue(
            any("alignment drift" in wp.lower() for wp in result.weak_points)
        )


if __name__ == "__main__":
    unittest.main()

"""Unit tests for evoagentx.core.tool_synthesizer."""

import json
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from evoagentx.core.tool_synthesizer import (
    SynthesizedTool,
    ToolRegistry,
    ToolSynthesisError,
    ToolSynthesizer,
    ToolValidationError,
    ValidationResult,
    _check_for_forbidden_patterns,
    _compile_function,
    get_tool_registry,
)


# ---------------------------------------------------------------------------
# Safe code fixture
# ---------------------------------------------------------------------------

_DOUBLE_FN_SOURCE = """\
def double_number(n):
    \"\"\"Return n * 2.\"\"\"
    return n * 2
"""

_DOUBLE_SPEC = {
    "tool_name": "double_number",
    "description": "Returns the double of a number.",
    "parameter_schema": {"n": {"type": "integer", "description": "Input number"}},
    "required_params": ["n"],
    "source_code": _DOUBLE_FN_SOURCE,
}


def _spec_json(spec: dict) -> str:
    return json.dumps(spec)


def _make_llm(content: str) -> MagicMock:
    llm = MagicMock()
    resp = MagicMock()
    resp.content = content
    llm.generate.return_value = resp
    return llm


# ---------------------------------------------------------------------------
# _check_for_forbidden_patterns
# ---------------------------------------------------------------------------

class TestForbiddenPatterns(unittest.TestCase):

    def test_safe_code_returns_none(self):
        self.assertIsNone(_check_for_forbidden_patterns(_DOUBLE_FN_SOURCE))

    def test_import_detected(self):
        self.assertIsNotNone(_check_for_forbidden_patterns("__import__('os')"))

    def test_open_detected(self):
        self.assertIsNotNone(_check_for_forbidden_patterns("open('/etc/passwd')"))

    def test_eval_detected(self):
        self.assertIsNotNone(_check_for_forbidden_patterns("eval(x)"))

    def test_subprocess_detected(self):
        self.assertIsNotNone(_check_for_forbidden_patterns("subprocess.run(['ls'])"))


# ---------------------------------------------------------------------------
# _compile_function
# ---------------------------------------------------------------------------

class TestCompileFunction(unittest.TestCase):

    def test_compiles_valid_function(self):
        fn = _compile_function(_DOUBLE_FN_SOURCE, "double_number")
        self.assertIsNotNone(fn)
        self.assertEqual(fn(5), 10)

    def test_returns_none_on_syntax_error(self):
        bad_source = "def broken(:\n    return 1"
        fn = _compile_function(bad_source, "broken")
        self.assertIsNone(fn)

    def test_returns_none_when_name_not_found(self):
        fn = _compile_function(_DOUBLE_FN_SOURCE, "nonexistent_fn")
        self.assertIsNone(fn)

    def test_restricted_scope_blocks_open(self):
        source = "def evil():\n    return open('/etc/passwd').read()"
        fn = _compile_function(source, "evil")
        # Compilation may succeed (no syntax error) but calling must fail
        # because 'open' is not in the restricted builtins.
        if fn is not None:
            with self.assertRaises((NameError, TypeError, Exception)):
                fn()

    def test_arithmetic_works(self):
        source = "def add(a, b):\n    return a + b"
        fn = _compile_function(source, "add")
        self.assertIsNotNone(fn)
        self.assertEqual(fn(3, 4), 7)


# ---------------------------------------------------------------------------
# SynthesizedTool
# ---------------------------------------------------------------------------

class TestSynthesizedTool(unittest.TestCase):

    def _make_tool(self, with_fn: bool = True) -> SynthesizedTool:
        fn = _compile_function(_DOUBLE_FN_SOURCE, "double_number") if with_fn else None
        return SynthesizedTool(
            tool_name="double_number",
            description="Doubles a number",
            parameter_schema={"n": {"type": "integer", "description": "Input"}},
            required_params=["n"],
            source_code=_DOUBLE_FN_SOURCE,
            callable_fn=fn,
            validation_passed=with_fn,
        )

    def test_call_valid_tool(self):
        tool = self._make_tool()
        self.assertEqual(tool.call(n=6), 12)

    def test_call_without_fn_raises(self):
        tool = self._make_tool(with_fn=False)
        with self.assertRaises(ToolSynthesisError):
            tool.call(n=5)

    def test_to_registry_entry_has_required_keys(self):
        tool = self._make_tool()
        entry = tool.to_registry_entry()
        for key in ("tool_name", "description", "parameter_schema",
                    "required_params", "source_code", "version",
                    "created_at", "validation_passed"):
            self.assertIn(key, entry)

    def test_version_default_one(self):
        tool = self._make_tool()
        self.assertEqual(tool.version, 1)


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class TestToolRegistry(unittest.TestCase):

    def setUp(self):
        self.registry = ToolRegistry()

    def _make_tool(self, name="double_number", version=1) -> SynthesizedTool:
        source = f"def {name}(n):\n    \"\"\"Returns n * 2.\"\"\"\n    return n * 2\n"
        fn = _compile_function(source, name)
        return SynthesizedTool(
            tool_name=name,
            description="Test tool",
            parameter_schema={},
            required_params=[],
            source_code=source,
            callable_fn=fn,
            version=version,
            validation_passed=True,
        )

    def test_register_and_get(self):
        tool = self._make_tool()
        self.registry.register(tool)
        self.assertIs(self.registry.get("double_number"), tool)

    def test_has_returns_true_when_registered(self):
        self.registry.register(self._make_tool())
        self.assertTrue(self.registry.has("double_number"))

    def test_has_returns_false_when_not_registered(self):
        self.assertFalse(self.registry.has("nonexistent"))

    def test_list_names(self):
        self.registry.register(self._make_tool("tool_alpha"))
        self.registry.register(self._make_tool("tool_beta"))
        names = self.registry.list_names()
        self.assertIn("tool_alpha", names)
        self.assertIn("tool_beta", names)

    def test_newer_version_replaces_older(self):
        v1 = self._make_tool(version=1)
        v2 = self._make_tool(version=2)
        self.registry.register(v1)
        self.registry.register(v2)
        self.assertEqual(self.registry.get("double_number").version, 2)

    def test_older_version_does_not_replace_newer(self):
        v2 = self._make_tool(version=2)
        v1 = self._make_tool(version=1)
        self.registry.register(v2)
        self.registry.register(v1)
        self.assertEqual(self.registry.get("double_number").version, 2)

    def test_descriptions_returns_dict(self):
        self.registry.register(self._make_tool())
        descs = self.registry.descriptions()
        self.assertIn("double_number", descs)
        self.assertIsInstance(descs["double_number"], str)

    def test_save_and_load(self):
        import tempfile

        tool = self._make_tool()
        self.registry.register(tool)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            self.registry.save(path)
            new_registry = ToolRegistry()
            new_registry.load(path)
            loaded = new_registry.get("double_number")
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.tool_name, "double_number")
            self.assertIsNotNone(loaded.callable_fn)
        finally:
            os.unlink(path)

    def test_load_skips_dangerous_code(self):
        import tempfile
        import json as json_mod

        data = {
            "evil_tool": {
                "tool_name": "evil_tool",
                "description": "dangerous",
                "parameter_schema": {},
                "required_params": [],
                "source_code": "def evil_tool(): return __import__('os').getcwd()",
                "version": 1,
                "created_at": 0.0,
                "validation_passed": True,
            }
        }
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            json_mod.dump(data, f)
            path = f.name

        try:
            reg = ToolRegistry()
            reg.load(path)
            self.assertFalse(reg.has("evil_tool"))
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

class TestValidationResult(unittest.TestCase):

    def test_passed_true_when_no_failures(self):
        r = ValidationResult(passed=True)
        self.assertTrue(r.passed)
        self.assertEqual(r.failed_cases, [])
        self.assertEqual(r.error_messages, [])

    def test_passed_false_with_failures(self):
        r = ValidationResult(passed=False, failed_cases=[0, 2], error_messages=["e1", "e2"])
        self.assertFalse(r.passed)
        self.assertEqual(r.failed_cases, [0, 2])


# ---------------------------------------------------------------------------
# ToolSynthesizer
# ---------------------------------------------------------------------------

class TestToolSynthesizer(unittest.TestCase):

    def setUp(self):
        # Use a fresh registry for each test to avoid cross-test pollution
        self.registry = ToolRegistry()
        self.llm = _make_llm(_spec_json(_DOUBLE_SPEC))
        self.synth = ToolSynthesizer(
            llm=self.llm,
            registry=self.registry,
            max_retries=1,
        )

    # ------------------------------------------------------------------
    # detect_tool_gap
    # ------------------------------------------------------------------

    def test_gap_detected_when_registry_empty(self):
        self.assertTrue(
            self.synth.detect_tool_gap("compute the square root of a number")
        )

    def test_gap_filled_by_keyword_match(self):
        tool = SynthesizedTool(
            tool_name="sqrt_tool",
            description="Computes square root",
            parameter_schema={},
            required_params=[],
            source_code="def sqrt_tool(x): return x ** 0.5",
            callable_fn=lambda x: x ** 0.5,
            validation_passed=True,
        )
        self.registry.register(tool)
        gap = self.synth.detect_tool_gap(
            "square root", similarity_keywords=["sqrt"]
        )
        self.assertFalse(gap)

    def test_gap_filled_by_word_overlap(self):
        tool = SynthesizedTool(
            tool_name="reverse_string",
            description="Reverses a string input",
            parameter_schema={},
            required_params=[],
            source_code="def reverse_string(s): return s[::-1]",
            callable_fn=lambda s: s[::-1],
            validation_passed=True,
        )
        self.registry.register(tool)
        # "reverse" and "string" both appear in name/description
        gap = self.synth.detect_tool_gap("reverse a string input")
        self.assertFalse(gap)

    # ------------------------------------------------------------------
    # synthesize (happy path — no test cases)
    # ------------------------------------------------------------------

    def test_synthesize_no_examples(self):
        tool = self.synth.synthesize(
            task_description="Return double of a number",
        )
        self.assertEqual(tool.tool_name, "double_number")
        self.assertTrue(tool.validation_passed)
        self.assertTrue(self.registry.has("double_number"))

    # ------------------------------------------------------------------
    # synthesize with test cases
    # ------------------------------------------------------------------

    def test_synthesize_with_passing_examples(self):
        examples = [
            {"args": {"n": 3}, "expected": 6},
            {"args": {"n": 0}, "expected": 0},
        ]
        tool = self.synth.synthesize(
            task_description="Double a number",
            examples=examples,
        )
        self.assertTrue(tool.validation_passed)

    def test_synthesize_with_failing_examples_raises_after_retries(self):
        """LLM always returns a tool that returns wrong values."""
        wrong_source = """\
def double_number(n):
    return n + 1  # intentionally wrong
"""
        bad_spec = dict(_DOUBLE_SPEC)
        bad_spec["source_code"] = wrong_source
        llm = _make_llm(_spec_json(bad_spec))
        synth = ToolSynthesizer(llm=llm, registry=self.registry, max_retries=0)

        examples = [{"args": {"n": 5}, "expected": 10}]
        with self.assertRaises(ToolSynthesisError):
            synth.synthesize("Double a number", examples=examples)

    # ------------------------------------------------------------------
    # synthesize — parse failure path
    # ------------------------------------------------------------------

    def test_raises_on_unparseable_llm_response(self):
        llm = _make_llm("not valid json at all")
        synth = ToolSynthesizer(llm=llm, registry=self.registry, max_retries=0)
        with self.assertRaises(ToolSynthesisError):
            synth.synthesize("do something")

    def test_raises_on_forbidden_code(self):
        evil_spec = dict(_DOUBLE_SPEC)
        evil_spec["source_code"] = (
            "def double_number(n):\n"
            "    return __import__('os').getcwd()"
        )
        llm = _make_llm(_spec_json(evil_spec))
        synth = ToolSynthesizer(llm=llm, registry=self.registry, max_retries=0)
        with self.assertRaises(ToolSynthesisError):
            synth.synthesize("do something dangerous")

    def test_raises_on_missing_spec_keys(self):
        incomplete = {"tool_name": "foo", "description": "bar"}
        llm = _make_llm(json.dumps(incomplete))
        synth = ToolSynthesizer(llm=llm, registry=self.registry, max_retries=0)
        with self.assertRaises(ToolSynthesisError):
            synth.synthesize("incomplete spec task")

    # ------------------------------------------------------------------
    # improve
    # ------------------------------------------------------------------

    def test_improve_bumps_version(self):
        # First synthesis
        tool = self.synth.synthesize("Double a number")
        self.assertEqual(tool.version, 1)

        # Improve — LLM still returns same spec (good enough)
        improved = self.synth.improve(
            tool_name="double_number",
            failure_description="returned wrong value for negative inputs",
        )
        self.assertGreater(improved.version, 1)

    def test_improve_raises_when_tool_not_registered(self):
        with self.assertRaises(ToolSynthesisError):
            self.synth.improve("nonexistent_tool", "some failure")

    # ------------------------------------------------------------------
    # cost_tracker integration
    # ------------------------------------------------------------------

    def test_cost_tracker_called_during_synthesis(self):
        tracker = MagicMock()
        synth = ToolSynthesizer(
            llm=_make_llm(_spec_json(_DOUBLE_SPEC)),
            registry=self.registry,
            cost_tracker=tracker,
            provider="test",
            model_name="test-model",
            max_retries=0,
        )
        synth.synthesize("Double a number")
        self.assertTrue(tracker.record.called)

    # ------------------------------------------------------------------
    # _validate internal
    # ------------------------------------------------------------------

    def test_validate_all_pass(self):
        fn = _compile_function(_DOUBLE_FN_SOURCE, "double_number")
        tool = SynthesizedTool(
            tool_name="double_number",
            description="test",
            parameter_schema={},
            required_params=[],
            source_code=_DOUBLE_FN_SOURCE,
            callable_fn=fn,
        )
        result = self.synth._validate(
            tool,
            [{"args": {"n": 4}, "expected": 8}, {"args": {"n": -1}, "expected": -2}],
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.failed_cases, [])

    def test_validate_one_fail(self):
        fn = _compile_function(_DOUBLE_FN_SOURCE, "double_number")
        tool = SynthesizedTool(
            tool_name="double_number",
            description="test",
            parameter_schema={},
            required_params=[],
            source_code=_DOUBLE_FN_SOURCE,
            callable_fn=fn,
        )
        result = self.synth._validate(
            tool,
            [{"args": {"n": 3}, "expected": 999}],  # wrong expected
        )
        self.assertFalse(result.passed)
        self.assertIn(0, result.failed_cases)

    def test_validate_exception_in_tool(self):
        def broken(**kwargs):
            raise RuntimeError("boom")

        tool = SynthesizedTool(
            tool_name="broken",
            description="",
            parameter_schema={},
            required_params=[],
            source_code="",
            callable_fn=broken,
        )
        result = self.synth._validate(tool, [{"args": {}, "expected": 1}])
        self.assertFalse(result.passed)
        self.assertTrue(any("boom" in msg for msg in result.error_messages))


if __name__ == "__main__":
    unittest.main()

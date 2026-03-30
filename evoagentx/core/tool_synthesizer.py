"""ToolSynthesizer — agents can create their own tools at runtime.

When an agent fails a task because no existing tool is adequate, the
ToolSynthesizer asks an LLM to write a new Python tool, validates it in a
restricted sandbox, and registers it so all future agents can use it.

Conceptually inspired by LATM (LLMs as Tool Makers, ICLR 2024) but
implemented from scratch for the EvoAgentX architecture.

Lifecycle
---------
1. ``detect_tool_gap`` — scan the tool registry; return True when nothing
   handles the requested capability.
2. ``synthesize`` — ask the LLM for: tool name, description, parameter
   schema, and the Python implementation body.
3. ``validate`` — execute the generated function inside a restricted
   ``exec`` scope on provided test cases; check outputs and that no unsafe
   side effects occurred (no ``open``, ``__import__``, network calls, etc.).
4. ``register`` — on successful validation, add the tool to the registry and
   persist it to disk for future sessions.

Safety
------
Generated code runs in a restricted ``exec`` environment:

- Only a safe subset of builtins is available (arithmetic, string, list,
  dict operations; no ``open``, ``exec``, ``eval``, ``__import__``).
- Any attempt to use file-system or network primitives raises a
  ``ToolSynthesisError`` at validation time.
- The registry entry records the source code, so the user can audit it.

Usage
-----
    >>> from evoagentx.core.tool_synthesizer import ToolSynthesizer
    >>> from unittest.mock import MagicMock
    >>> llm = MagicMock()
    >>> synth = ToolSynthesizer(llm=llm)
    >>> result = synth.synthesize(
    ...     task_description="Compute the nth Fibonacci number",
    ...     examples=[{"args": {"n": 10}, "expected": 55}],
    ... )
    >>> print(result.tool_name)
"""

from __future__ import annotations

import json
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .logging import logger


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ToolSynthesisError(Exception):
    """Raised when tool generation or validation fails unrecoverably."""


class ToolValidationError(ToolSynthesisError):
    """Raised when generated code fails validation against test cases."""


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SynthesizedTool:
    """A tool produced by the ToolSynthesizer.

    Attributes
    ----------
    tool_name:
        Snake-case identifier; must be a valid Python function name.
    description:
        Human-readable description of what the tool does.
    parameter_schema:
        OpenAI-style JSON schema for the tool's parameters:
        ``{"param_name": {"type": "...", "description": "..."}}``.
    required_params:
        List of parameter names that are mandatory.
    source_code:
        Full Python source of the tool function (for auditing).
    callable_fn:
        The compiled callable, populated after successful validation.
    version:
        Monotonically increasing integer; incremented when the tool is
        improved via ``improve()``.
    created_at:
        Unix timestamp when the tool was first synthesized.
    validation_passed:
        Whether the tool passed all provided test cases.
    """

    tool_name: str
    description: str
    parameter_schema: Dict[str, Dict[str, str]]
    required_params: List[str]
    source_code: str
    callable_fn: Optional[Callable] = field(default=None, repr=False)
    version: int = 1
    created_at: float = field(default_factory=time.time)
    validation_passed: bool = False

    def call(self, **kwargs: Any) -> Any:
        """Execute the synthesized tool with the given keyword arguments.

        Raises
        ------
        ToolSynthesisError
            If the tool has not been validated (``callable_fn`` is None).
        """
        if self.callable_fn is None:
            raise ToolSynthesisError(
                f"Tool '{self.tool_name}' has not been validated; cannot call."
            )
        return self.callable_fn(**kwargs)

    def to_registry_entry(self) -> Dict[str, Any]:
        """Serialise the tool to a JSON-compatible registry entry."""
        return {
            "tool_name": self.tool_name,
            "description": self.description,
            "parameter_schema": self.parameter_schema,
            "required_params": self.required_params,
            "source_code": self.source_code,
            "version": self.version,
            "created_at": self.created_at,
            "validation_passed": self.validation_passed,
        }


@dataclass
class ValidationResult:
    """Outcome of running a synthesized tool against test cases.

    Attributes
    ----------
    passed:
        True when all test cases produced the expected output.
    failed_cases:
        Indices of test cases that failed.
    error_messages:
        One error message per failed case (same order as ``failed_cases``).
    """

    passed: bool
    failed_cases: List[int] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Safe builtins whitelist
# ---------------------------------------------------------------------------

_SAFE_BUILTINS: Dict[str, Any] = {
    name: getattr(__builtins__ if isinstance(__builtins__, dict)
                  else __builtins__, name, None)
    for name in (
        "abs", "all", "any", "bin", "bool", "chr", "dict", "divmod",
        "enumerate", "filter", "float", "format", "frozenset", "hash",
        "hex", "int", "isinstance", "issubclass", "iter", "len", "list",
        "map", "max", "min", "next", "oct", "ord", "pow", "print",
        "range", "repr", "reversed", "round", "set", "slice", "sorted",
        "str", "sum", "tuple", "type", "zip",
    )
    if getattr(__builtins__ if isinstance(__builtins__, dict)
               else __builtins__, name, None) is not None
}
_SAFE_BUILTINS["None"] = None
_SAFE_BUILTINS["True"] = True
_SAFE_BUILTINS["False"] = False

# Patterns that indicate dangerous code
_FORBIDDEN_PATTERNS: List[str] = [
    "__import__", "importlib", "open(", "os.path", "subprocess",
    "socket.", "urllib", "requests.", "httpx.", "aiohttp.", "eval(",
    "exec(", "compile(", "globals(", "locals(", "__builtins__",
]


def _check_for_forbidden_patterns(source: str) -> Optional[str]:
    """Return the first forbidden pattern found in ``source``, or None."""
    for pattern in _FORBIDDEN_PATTERNS:
        if pattern in source:
            return pattern
    return None


# ---------------------------------------------------------------------------
# ToolRegistry (lightweight in-memory store)
# ---------------------------------------------------------------------------

class ToolRegistry:
    """In-process registry for synthesized tools.

    A single process-level instance is provided via
    :func:`get_tool_registry`.

    Tools can also be persisted to and loaded from a JSON file using
    :meth:`save` and :meth:`load`.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, SynthesizedTool] = {}

    # ------------------------------------------------------------------
    # Registration / retrieval
    # ------------------------------------------------------------------

    def register(self, tool: SynthesizedTool) -> None:
        """Add or replace a tool in the registry.

        If a tool with the same ``tool_name`` already exists its version is
        bumped if the new one carries a higher version number.
        """
        existing = self._tools.get(tool.tool_name)
        if existing is not None and tool.version <= existing.version:
            logger.warning(
                f"ToolRegistry | skipping older/equal version "
                f"v{tool.version} of '{tool.tool_name}' "
                f"(current: v{existing.version})"
            )
            return
        self._tools[tool.tool_name] = tool
        logger.info(
            f"ToolRegistry | registered '{tool.tool_name}' v{tool.version}"
        )

    def get(self, tool_name: str) -> Optional[SynthesizedTool]:
        """Return the tool with the given name, or None if not found."""
        return self._tools.get(tool_name)

    def list_names(self) -> List[str]:
        """Return names of all registered tools."""
        return list(self._tools.keys())

    def has(self, tool_name: str) -> bool:
        """Return True when a tool with ``tool_name`` is registered."""
        return tool_name in self._tools

    def descriptions(self) -> Dict[str, str]:
        """Return a mapping of tool name → description."""
        return {name: t.description for name, t in self._tools.items()}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the registry to a JSON file at ``path``."""
        import os

        data = {name: t.to_registry_entry() for name, t in self._tools.items()}
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        logger.info(f"ToolRegistry | saved {len(data)} tools to '{path}'")

    def load(self, path: str) -> None:
        """Load and compile tools from a JSON file at ``path``.

        Tools whose source code fails the safety check are skipped with a
        warning (they may have been tampered with on disk).
        """
        import os

        if not os.path.exists(path):
            logger.warning(f"ToolRegistry | file not found: '{path}'")
            return

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        loaded = 0
        for name, entry in data.items():
            source = entry.get("source_code", "")
            forbidden = _check_for_forbidden_patterns(source)
            if forbidden:
                logger.warning(
                    f"ToolRegistry | skipping '{name}': "
                    f"forbidden pattern '{forbidden}' found in loaded source"
                )
                continue

            fn = _compile_function(source, entry["tool_name"])
            if fn is None:
                logger.warning(f"ToolRegistry | skipping '{name}': compile failed")
                continue

            tool = SynthesizedTool(
                tool_name=entry["tool_name"],
                description=entry.get("description", ""),
                parameter_schema=entry.get("parameter_schema", {}),
                required_params=entry.get("required_params", []),
                source_code=source,
                callable_fn=fn,
                version=entry.get("version", 1),
                created_at=entry.get("created_at", 0.0),
                validation_passed=entry.get("validation_passed", False),
            )
            self._tools[name] = tool
            loaded += 1

        logger.info(f"ToolRegistry | loaded {loaded} tools from '{path}'")


# Process-level singleton
_GLOBAL_REGISTRY: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Return the process-wide :class:`ToolRegistry` singleton."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = ToolRegistry()
    return _GLOBAL_REGISTRY


# ---------------------------------------------------------------------------
# Compile helper
# ---------------------------------------------------------------------------

def _compile_function(source: str, fn_name: str) -> Optional[Callable]:
    """Compile ``source`` in a restricted scope and return the function.

    Parameters
    ----------
    source:
        Python source code defining a function named ``fn_name``.
    fn_name:
        The name of the function to extract from the executed scope.

    Returns
    -------
    Callable or None
        The compiled function, or None if compilation failed.
    """
    scope: Dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
    try:
        exec(textwrap.dedent(source), scope)  # noqa: S102
    except Exception as exc:
        logger.warning(f"ToolSynthesizer | compile error for '{fn_name}': {exc}")
        return None

    fn = scope.get(fn_name)
    if callable(fn):
        return fn
    logger.warning(
        f"ToolSynthesizer | function '{fn_name}' not found after exec"
    )
    return None


# ---------------------------------------------------------------------------
# ToolSynthesizer
# ---------------------------------------------------------------------------

class ToolSynthesizer:
    """Generate, validate, and register new tools on demand.

    Parameters
    ----------
    llm:
        A ``BaseLLM`` instance (or any object with a ``generate(prompt)``
        method returning an object with a ``.content`` string attribute).
    registry:
        Tool registry to check for gaps and to register new tools into.
        Defaults to the process-wide singleton.
    cost_tracker:
        Optional Phase 0 ``CostTracker`` for recording synthesis LLM calls.
    provider:
        Provider string for cost tracking.
    model_name:
        Model name for cost tracking.
    max_retries:
        Number of synthesis-and-validate attempts before giving up.
    """

    def __init__(
        self,
        llm: Any,
        registry: Optional[ToolRegistry] = None,
        cost_tracker: Optional[Any] = None,
        provider: str = "unknown",
        model_name: str = "unknown",
        max_retries: int = 2,
    ) -> None:
        self._llm = llm
        self._registry = registry or get_tool_registry()
        self._cost_tracker = cost_tracker
        self._provider = provider
        self._model_name = model_name
        self._max_retries = max_retries

    # ------------------------------------------------------------------
    # Gap detection
    # ------------------------------------------------------------------

    def detect_tool_gap(
        self,
        capability_description: str,
        similarity_keywords: Optional[List[str]] = None,
    ) -> bool:
        """Check whether the registry already covers a given capability.

        Parameters
        ----------
        capability_description:
            Natural-language description of the capability needed.
        similarity_keywords:
            Optional list of keywords; if any registered tool's name or
            description contains one, the gap is considered filled.

        Returns
        -------
        bool
            True when no adequate tool is found (gap detected).
        """
        keywords = [w.lower() for w in (similarity_keywords or [])]
        cap_words = [w.lower() for w in capability_description.split()]

        for name, desc in self._registry.descriptions().items():
            combined = f"{name} {desc}".lower()
            if any(kw in combined for kw in keywords):
                logger.debug(
                    f"ToolSynthesizer | gap check: '{name}' covers capability "
                    f"via keyword match"
                )
                return False
            # Lightweight overlap: if >30% of cap words appear in description
            if cap_words:
                overlap = sum(1 for w in cap_words if w in combined)
                if overlap / len(cap_words) > 0.3:
                    logger.debug(
                        f"ToolSynthesizer | gap check: '{name}' covers capability "
                        f"via word overlap ({overlap}/{len(cap_words)})"
                    )
                    return False

        logger.info(
            f"ToolSynthesizer | gap detected for: '{capability_description[:60]}'"
        )
        return True

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def synthesize(
        self,
        task_description: str,
        examples: Optional[List[Dict[str, Any]]] = None,
        force: bool = False,
    ) -> SynthesizedTool:
        """Synthesize, validate, and register a new tool for ``task_description``.

        Parameters
        ----------
        task_description:
            Natural-language description of what the tool should do.
        examples:
            Optional list of test cases, each a dict with keys:
            ``"args"`` (dict of argument name → value) and
            ``"expected"`` (the expected return value).
        force:
            When True, synthesize even if an existing tool covers the task.

        Returns
        -------
        SynthesizedTool
            The validated and registered tool.

        Raises
        ------
        ToolSynthesisError
            If synthesis or validation fails after ``max_retries`` attempts.
        """
        examples = examples or []
        errors: List[str] = []

        for attempt in range(1, self._max_retries + 2):
            logger.info(
                f"ToolSynthesizer | synthesis attempt {attempt} "
                f"for '{task_description[:50]}'"
            )
            prompt = self._build_synthesis_prompt(
                task_description, examples, errors
            )
            raw = self._call_llm(prompt, purpose="synthesis")
            spec = self._parse_tool_spec(raw)

            if spec is None:
                errors.append("Failed to parse tool specification JSON.")
                continue

            # Safety check before compile
            forbidden = _check_for_forbidden_patterns(spec["source_code"])
            if forbidden:
                errors.append(
                    f"Generated code contains forbidden pattern: '{forbidden}'"
                )
                continue

            fn = _compile_function(spec["source_code"], spec["tool_name"])
            if fn is None:
                errors.append(
                    f"Could not compile function '{spec['tool_name']}'."
                )
                continue

            tool = SynthesizedTool(
                tool_name=spec["tool_name"],
                description=spec["description"],
                parameter_schema=spec["parameter_schema"],
                required_params=spec.get("required_params", []),
                source_code=spec["source_code"],
                callable_fn=fn,
            )

            if examples:
                result = self._validate(tool, examples)
                if not result.passed:
                    errors.append(
                        "Validation failed on cases "
                        + str(result.failed_cases)
                        + ": "
                        + "; ".join(result.error_messages)
                    )
                    continue

            tool.validation_passed = True
            self._registry.register(tool)
            logger.info(
                f"ToolSynthesizer | registered '{tool.tool_name}' "
                f"on attempt {attempt}"
            )
            return tool

        raise ToolSynthesisError(
            f"Could not synthesize tool for '{task_description}' "
            f"after {self._max_retries + 1} attempts. "
            f"Last errors: {errors}"
        )

    def _build_synthesis_prompt(
        self,
        task_description: str,
        examples: List[Dict[str, Any]],
        prior_errors: List[str],
    ) -> str:
        """Construct the LLM prompt for tool synthesis."""
        examples_text = ""
        if examples:
            lines = []
            for i, ex in enumerate(examples[:3]):
                lines.append(
                    f"  Example {i + 1}: args={ex.get('args', {})}, "
                    f"expected={ex.get('expected', 'N/A')}"
                )
            examples_text = "Test cases:\n" + "\n".join(lines) + "\n\n"

        error_text = ""
        if prior_errors:
            error_text = (
                "Previous attempts failed with these errors:\n"
                + "\n".join(f"  - {e}" for e in prior_errors[-3:])
                + "\n\nPlease fix these issues in your response.\n\n"
            )

        return (
            f"You are a Python expert. Write a standalone Python function that "
            f"accomplishes the following task:\n\n"
            f"Task: {task_description}\n\n"
            f"{examples_text}"
            f"{error_text}"
            f"Requirements:\n"
            f"1. The function must be self-contained (no imports, no global state).\n"
            f"2. Use only built-in Python operations (no file I/O, no network calls, "
            f"   no subprocess).\n"
            f"3. Include a clear docstring.\n"
            f"4. Handle edge cases gracefully.\n\n"
            f"Return ONLY a JSON object with these exact keys:\n"
            f'{{\n'
            f'  "tool_name": "<snake_case_name>",\n'
            f'  "description": "<one-sentence description>",\n'
            f'  "parameter_schema": {{"<param>": {{"type": "<type>", "description": "<desc>"}}}},\n'
            f'  "required_params": ["<param1>"],\n'
            f'  "source_code": "<complete Python function source as a string>"\n'
            f'}}\n\n'
            f"The source_code must define a function named exactly tool_name."
        )

    def _parse_tool_spec(
        self, raw: str
    ) -> Optional[Dict[str, Any]]:
        """Extract tool specification dict from LLM response."""
        text = raw.strip()
        for fence in ("```json", "```"):
            if text.startswith(fence):
                text = text[len(fence):]
        text = text.rstrip("`").strip()

        try:
            spec = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                f"ToolSynthesizer | could not parse spec JSON: "
                f"'{raw[:80]}'"
            )
            return None

        required_keys = {"tool_name", "description", "parameter_schema", "source_code"}
        missing = required_keys - set(spec.keys())
        if missing:
            logger.warning(
                f"ToolSynthesizer | spec missing keys: {missing}"
            )
            return None

        # Validate tool_name is a valid identifier
        name = spec.get("tool_name", "")
        if not name or not str(name).replace("_", "").isalnum():
            logger.warning(
                f"ToolSynthesizer | invalid tool_name: '{name}'"
            )
            return None

        return spec

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(
        self,
        tool: SynthesizedTool,
        examples: List[Dict[str, Any]],
    ) -> ValidationResult:
        """Run all test cases against the compiled tool function.

        Parameters
        ----------
        tool:
            The tool to validate (must have ``callable_fn`` set).
        examples:
            List of ``{"args": {...}, "expected": ...}`` dicts.

        Returns
        -------
        ValidationResult
        """
        failed: List[int] = []
        errors: List[str] = []

        for idx, ex in enumerate(examples):
            args = ex.get("args", {})
            expected = ex.get("expected")
            try:
                actual = tool.callable_fn(**args)
                if expected is not None and actual != expected:
                    failed.append(idx)
                    errors.append(
                        f"args={args}: expected {expected!r}, got {actual!r}"
                    )
            except Exception as exc:
                failed.append(idx)
                errors.append(f"args={args}: raised {type(exc).__name__}: {exc}")

        passed = len(failed) == 0
        if passed:
            logger.info(
                f"ToolSynthesizer | '{tool.tool_name}' passed all "
                f"{len(examples)} test case(s)"
            )
        else:
            logger.warning(
                f"ToolSynthesizer | '{tool.tool_name}' failed "
                f"{len(failed)}/{len(examples)} test case(s)"
            )

        return ValidationResult(passed=passed, failed_cases=failed, error_messages=errors)

    # ------------------------------------------------------------------
    # Tool improvement
    # ------------------------------------------------------------------

    def improve(
        self,
        tool_name: str,
        failure_description: str,
        new_examples: Optional[List[Dict[str, Any]]] = None,
    ) -> SynthesizedTool:
        """Re-synthesize a registered tool with additional failure context.

        Parameters
        ----------
        tool_name:
            Name of the tool to improve.
        failure_description:
            Description of the failure mode that motivates the improvement.
        new_examples:
            Additional test cases that the improved tool must pass.

        Returns
        -------
        SynthesizedTool
            The improved tool (version incremented).

        Raises
        ------
        ToolSynthesisError
            If the tool is not registered or improvement synthesis fails.
        """
        existing = self._registry.get(tool_name)
        if existing is None:
            raise ToolSynthesisError(
                f"Tool '{tool_name}' is not in the registry; cannot improve."
            )

        task_desc = (
            f"{existing.description}\n\n"
            f"Current implementation failed: {failure_description}\n"
            f"Existing source for reference:\n{existing.source_code}"
        )
        improved = self.synthesize(
            task_description=task_desc,
            examples=new_examples or [],
        )
        # Bump version
        improved.version = existing.version + 1
        self._registry.register(improved)
        logger.info(
            f"ToolSynthesizer | improved '{tool_name}' → v{improved.version}"
        )
        return improved

    # ------------------------------------------------------------------
    # LLM call helper
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str, purpose: str = "") -> str:
        """Call the LLM and record cost if a tracker is configured."""
        try:
            response = self._llm.generate(prompt)
            content = getattr(response, "content", str(response))

            if self._cost_tracker is not None:
                # Inline estimate: ~4 chars per token, no external dependency
                in_tok = max(1, len(prompt) // 4)
                out_tok = max(1, len(content) // 4) if content else 1
                try:
                    self._cost_tracker.record(
                        self._provider,
                        self._model_name,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        metadata={"purpose": f"tool_synthesizer/{purpose}"},
                    )
                except Exception as cost_exc:
                    logger.warning(
                        f"ToolSynthesizer | cost tracking failed: {cost_exc}"
                    )

            return content
        except Exception as exc:
            logger.error(f"ToolSynthesizer | LLM call failed ({purpose}): {exc}")
            return ""

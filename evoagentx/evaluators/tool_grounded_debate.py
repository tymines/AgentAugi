"""Tool-MAD: Multi-Agent Debate with Tool Grounding.

Extends the base DebateEvaluator with mandatory tool grounding — debaters must
cite actual tool outputs as evidence for their claims.  This prevents hallucinated
arguments and adds a faithfulness scoring layer before the judge renders a verdict.

Architecture
------------
Each debater turn is prompted to produce structured responses:

    CLAIM: <statement>
    TOOL: <tool_name>
    TOOL_OUTPUT: <output from tool>
    EVIDENCE: <how the tool output supports the claim>

The evaluator:

1. Parses debater responses into ``GroundedArgument`` objects.
2. Scores each claim's faithfulness to its cited tool output using keyword overlap
   (or an optional LLM judge function).
3. Filters out claims below ``faithfulness_threshold``.
4. Passes only well-grounded arguments to the judge, along with a grounding summary
   that tells the judge how well each side supported their claims with evidence.

The ``evaluate()`` method is drop-in compatible with ``DebateEvaluator``:
it accepts the same arguments and returns a ``ToolGroundedDebateResult`` (a subclass
of ``DebateResult``) so existing code consuming ``DebateResult`` continues to work.

Usage
-----
    >>> from evoagentx.evaluators.tool_grounded_debate import (
    ...     ToolGroundedDebateEvaluator, ToolGroundedDebateConfig
    ... )
    >>> def my_llm(messages): return call_your_llm(messages)
    >>> def search_tool(query): return "search results..."
    >>> config = ToolGroundedDebateConfig(
    ...     require_tool_grounding=True,
    ...     faithfulness_threshold=0.6,
    ...     tool_registry={"search": search_tool},
    ... )
    >>> evaluator = ToolGroundedDebateEvaluator(
    ...     generate_fn=my_llm, config=config, num_rounds=2
    ... )
    >>> result = evaluator.evaluate(
    ...     solution="Agent uses RAG to answer factual queries.",
    ...     task_context="Must answer factual questions with verifiable sources.",
    ... )
    >>> print(result.grounding_rate, result.mean_faithfulness)
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.logging import logger
from .debate import (
    DebateArgument,
    DebaterConfig,
    DebateEvaluator,
    DebatePosition,
    DebateResult,
    DebateRound,
    _estimate_argument_quality,
    _extract_weak_points,
    _format_transcript,
    _parse_judge_verdict,
    _word_overlap_similarity,
    make_heterogeneous_debaters,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ToolGroundedDebateConfig:
    """Configuration for the tool-grounded debate evaluator.

    Attributes:
        require_tool_grounding: When True debaters are instructed to cite at
            least one tool output per argument and ungrounded claims are penalised.
        faithfulness_threshold: Minimum faithfulness score [0, 1] for a claim to
            survive filtering.  Claims with lower scores are stripped before the
            judge reads the transcript.
        max_tool_calls_per_round: Soft cap on the number of tool citations a
            debater may include in a single round.  Excess citations are ignored
            after parsing.
        tool_registry: Optional dict mapping tool names to callables that the
            evaluator may invoke during a debate turn.  When provided, the
            evaluator can actually execute tools and inject their real outputs.
            When None, debaters are expected to fabricate plausible tool outputs
            (useful in pure-LLM mock settings).
    """

    require_tool_grounding: bool = True
    faithfulness_threshold: float = 0.6
    max_tool_calls_per_round: int = 3
    tool_registry: Optional[Dict[str, Callable[..., str]]] = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.faithfulness_threshold <= 1.0):
            raise ValueError("faithfulness_threshold must be in [0, 1].")
        if self.max_tool_calls_per_round < 1:
            raise ValueError("max_tool_calls_per_round must be >= 1.")


# ---------------------------------------------------------------------------
# Grounded argument data structures
# ---------------------------------------------------------------------------

@dataclass
class GroundedArgument:
    """A single tool-grounded claim produced by a debater.

    Attributes:
        claim: The actual statement the debater is making.
        evidence: The debater's explanation of how the tool output supports the claim.
        tool_used: Name of the tool that was cited.  None if no tool was cited.
        tool_output: The raw output from the cited tool.  None if no tool cited.
        faithfulness_score: How faithfully the claim is supported by the tool output,
            in [0, 1].  Claims with no tool output receive 0.0 by default and are
            raised to 0.5 when grounding is not required.
    """

    claim: str
    evidence: str
    tool_used: Optional[str] = None
    tool_output: Optional[str] = None
    faithfulness_score: float = 0.0

    def is_grounded(self) -> bool:
        """Return True when this argument cites an actual tool output."""
        return self.tool_used is not None and self.tool_output is not None


@dataclass
class ToolGroundedDebateResult(DebateResult):
    """Debate result extended with grounding quality metrics.

    Inherits all fields from ``DebateResult`` and adds:

    Attributes:
        grounded_arguments: All parsed ``GroundedArgument`` objects from the full
            debate (across all rounds and debaters).
        grounding_rate: Fraction of arguments that cited a tool, in [0, 1].
        mean_faithfulness: Mean faithfulness score across all grounded arguments.
            0.0 when no grounded arguments exist.
        filtered_claim_count: Number of claims removed because their faithfulness
            score fell below ``faithfulness_threshold``.
    """

    grounded_arguments: List[GroundedArgument] = field(default_factory=list)
    grounding_rate: float = 0.0
    mean_faithfulness: float = 0.0
    filtered_claim_count: int = 0

    def summary(self) -> str:
        base = super().summary()
        lines = [
            base,
            f"  Grounding rate : {self.grounding_rate:.2f}",
            f"  Mean faithfulness: {self.mean_faithfulness:.2f}",
            f"  Filtered claims: {self.filtered_claim_count}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt templates for tool-grounded debaters
# ---------------------------------------------------------------------------

_GROUNDED_DEBATER_SYSTEM = (
    "You are {name}, participating in a structured debate about a proposed solution. "
    "Your assigned position is {position_label}.\n"
    "{persona_line}"
    "IMPORTANT: You MUST ground every claim in actual tool output. "
    "For each point you make, cite a tool and its output using this exact format:\n\n"
    "CLAIM: <your specific assertion>\n"
    "TOOL: <tool_name>\n"
    "TOOL_OUTPUT: <the relevant output from the tool>\n"
    "EVIDENCE: <how this output supports your claim>\n\n"
    "You may make up to {max_tool_calls} such grounded arguments per response. "
    "Ungrounded assertions will be ignored by the judge. "
    "Available tools: {available_tools}."
)

_GROUNDED_DEBATER_FIRST_ROUND = (
    "TASK CONTEXT:\n{task_context}\n\n"
    "PROPOSED SOLUTION:\n{solution}\n\n"
    "This is round {round_number} of {total_rounds}. "
    "Present your opening argument {position_instruction} using the grounded format above."
)

_GROUNDED_DEBATER_SUBSEQUENT_ROUND = (
    "TASK CONTEXT:\n{task_context}\n\n"
    "PROPOSED SOLUTION:\n{solution}\n\n"
    "DEBATE TRANSCRIPT SO FAR:\n{transcript}\n\n"
    "This is round {round_number} of {total_rounds}. "
    "Continue arguing {position_instruction}, addressing prior points with grounded evidence."
)

_GROUNDED_JUDGE_SYSTEM = (
    "You are an impartial judge evaluating a proposed solution. "
    "The debate you are reading used tool-grounded arguments — claims backed by real tool outputs. "
    "Grounded arguments should be weighted more heavily than ungrounded assertions. "
    "Consider both the quality of the reasoning and the strength of the evidence."
)

_GROUNDED_JUDGE_PROMPT = (
    "TASK CONTEXT:\n{task_context}\n\n"
    "PROPOSED SOLUTION:\n{solution}\n\n"
    "DEBATE TRANSCRIPT (grounded arguments only):\n{transcript}\n\n"
    "GROUNDING SUMMARY:\n{grounding_summary}\n\n"
    "Based on the debate above, provide your verdict:\n"
    "VERDICT: PASS or FAIL\n"
    "CONFIDENCE: a value between 0.0 and 1.0\n"
    "RATIONALE: Your reasoning (note which grounded arguments were most convincing)\n"
    "WEAK POINTS:\n"
    "- List each significant weakness, one per line\n"
    "END"
)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# Matches a single CLAIM/TOOL/TOOL_OUTPUT/EVIDENCE block (fields in any order,
# only CLAIM is strictly required).
_GROUNDED_BLOCK_RE = re.compile(
    r"CLAIM\s*:\s*(?P<claim>.+?)(?=TOOL\s*:|EVIDENCE\s*:|CLAIM\s*:|$)"
    r"(?:.*?TOOL\s*:\s*(?P<tool>[^\n]+))?"
    r"(?:.*?TOOL_OUTPUT\s*:\s*(?P<tool_output>.+?))?"
    r"(?:.*?EVIDENCE\s*:\s*(?P<evidence>.+?))?(?=CLAIM\s*:|$)",
    re.DOTALL | re.IGNORECASE,
)


def _parse_grounded_blocks(text: str) -> List[Dict[str, Optional[str]]]:
    """Extract all CLAIM/TOOL/TOOL_OUTPUT/EVIDENCE blocks from debater text.

    Args:
        text: Raw debater response text.

    Returns:
        List of dicts with keys 'claim', 'tool', 'tool_output', 'evidence'.
        All values may be None except 'claim'.
    """
    blocks: List[Dict[str, Optional[str]]] = []
    for match in _GROUNDED_BLOCK_RE.finditer(text):
        claim = (match.group("claim") or "").strip()
        if not claim:
            continue
        tool = (match.group("tool") or "").strip() or None
        tool_output = (match.group("tool_output") or "").strip() or None
        evidence = (match.group("evidence") or "").strip() or None
        blocks.append({
            "claim": claim,
            "tool": tool,
            "tool_output": tool_output,
            "evidence": evidence or "",
        })
    return blocks


# ---------------------------------------------------------------------------
# Faithfulness scoring
# ---------------------------------------------------------------------------

def _score_faithfulness_keyword(claim: str, tool_output: str) -> float:
    """Score how faithfully a claim is supported by a tool output.

    Uses word overlap between the claim and tool output as a proxy for
    faithfulness.  This is a fast, local heuristic that works without a
    secondary LLM call.

    Args:
        claim: The debater's assertion.
        tool_output: Raw output from the cited tool.

    Returns:
        Faithfulness score in [0, 1].  Higher = claim is more directly
        supported by the tool output.
    """
    if not claim.strip() or not tool_output.strip():
        return 0.0

    # Normalise: lower-case, strip punctuation
    def _tokens(t: str) -> set:
        t = t.lower()
        t = re.sub(r"[^\w\s]", " ", t)
        return set(t.split())

    claim_tokens = _tokens(claim)
    output_tokens = _tokens(tool_output)

    # Remove very common stop words to reduce noise
    _STOP = frozenset([
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "must", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "this", "that", "it", "its", "and", "or", "not", "but",
    ])
    claim_tokens -= _STOP
    output_tokens -= _STOP

    if not claim_tokens:
        return 0.0

    overlap = claim_tokens & output_tokens
    # Precision: how much of the claim appears in the output
    precision = len(overlap) / len(claim_tokens)
    # Recall: how much of the output is referenced by the claim
    recall = len(overlap) / max(1, len(output_tokens))
    # F1-style blend, weighted toward precision (claim must be grounded)
    if precision + recall == 0.0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    # Weight: 70% precision, 30% recall-aware F1
    score = 0.7 * precision + 0.3 * f1
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Grounding summary builder
# ---------------------------------------------------------------------------

def _build_grounding_summary(
    pro_args: List[GroundedArgument],
    con_args: List[GroundedArgument],
) -> str:
    """Build a plain-text grounding summary for the judge prompt.

    Args:
        pro_args: Filtered FOR-side grounded arguments.
        con_args: Filtered AGAINST-side grounded arguments.

    Returns:
        Multi-line summary string.
    """
    def _side_summary(label: str, args: List[GroundedArgument]) -> str:
        if not args:
            return f"{label}: No grounded arguments survived filtering."
        scores = [a.faithfulness_score for a in args]
        mean_f = statistics.mean(scores)
        grounded = sum(1 for a in args if a.is_grounded())
        lines = [
            f"{label}: {len(args)} grounded claim(s), "
            f"mean faithfulness={mean_f:.2f}, "
            f"{grounded}/{len(args)} cited tool outputs."
        ]
        for i, arg in enumerate(args, 1):
            tool_tag = f"[{arg.tool_used}]" if arg.tool_used else "[no tool]"
            lines.append(
                f"  {i}. {tool_tag} (faith={arg.faithfulness_score:.2f}) "
                f"{arg.claim[:80]}..."
            )
        return "\n".join(lines)

    return "\n".join([
        _side_summary("FOR", pro_args),
        _side_summary("AGAINST", con_args),
    ])


# ---------------------------------------------------------------------------
# Main evaluator class
# ---------------------------------------------------------------------------

class ToolGroundedDebateEvaluator:
    """Multi-agent debate evaluator that requires tool-grounded arguments.

    Drop-in replacement for ``DebateEvaluator`` — accepts the same ``evaluate()``
    signature and returns a ``ToolGroundedDebateResult`` (subclass of
    ``DebateResult``).

    Args:
        generate_fn: Callable ``(messages: List[dict]) -> str`` for LLM generation.
        config: ``ToolGroundedDebateConfig`` instance.  If None, uses defaults.
        num_rounds: Number of debate rounds (default 3).
        debater_configs: Explicit debater configurations.  If None, uses two
            default debaters from ``debate.py``.
        convergence_threshold: Similarity threshold for early stopping (default 0.85).
        embed_fn: Optional embedding function for convergence detection.
        alignment_detector: Optional ``AlignmentDriftDetector`` instance.
        pass_threshold: Minimum judge confidence for a genuine PASS (default 0.5).
        faithfulness_fn: Optional ``(claim: str, tool_output: str) -> float``
            override for faithfulness scoring.  When None, uses keyword overlap.

    Raises:
        ValueError: On invalid configuration.
    """

    def __init__(
        self,
        generate_fn: Callable[[List[Dict[str, str]]], str],
        config: Optional[ToolGroundedDebateConfig] = None,
        num_rounds: int = 3,
        debater_configs: Optional[List[DebaterConfig]] = None,
        convergence_threshold: float = 0.85,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        alignment_detector: Optional[Any] = None,
        pass_threshold: float = 0.5,
        faithfulness_fn: Optional[Callable[[str, str], float]] = None,
    ) -> None:
        self._config = config or ToolGroundedDebateConfig()
        self._faithfulness_fn = faithfulness_fn or _score_faithfulness_keyword

        # Delegate the core debate machinery to DebateEvaluator
        self._base_evaluator = DebateEvaluator(
            generate_fn=generate_fn,
            num_rounds=num_rounds,
            debater_configs=debater_configs,
            convergence_threshold=convergence_threshold,
            embed_fn=embed_fn,
            alignment_detector=alignment_detector,
            pass_threshold=pass_threshold,
        )

        # Store for use in custom prompts
        self._generate_fn = generate_fn
        self.num_rounds = num_rounds
        self.convergence_threshold = convergence_threshold
        self._embed_fn = embed_fn
        self.pass_threshold = pass_threshold
        self._debater_configs = self._base_evaluator._debater_configs

        logger.debug(
            "ToolGroundedDebateEvaluator: %d debaters, %d rounds, "
            "faithfulness_threshold=%.2f, require_grounding=%s",
            len(self._debater_configs),
            self.num_rounds,
            self._config.faithfulness_threshold,
            self._config.require_tool_grounding,
        )

    # ------------------------------------------------------------------
    # Faithfulness scoring (public for testing)
    # ------------------------------------------------------------------

    def _score_faithfulness(self, claim: str, tool_output: str) -> float:
        """Score claim faithfulness against tool output.

        Args:
            claim: Debater's assertion.
            tool_output: Raw tool output text.

        Returns:
            Faithfulness score in [0, 1].
        """
        return self._faithfulness_fn(claim, tool_output)

    # ------------------------------------------------------------------
    # Argument extraction
    # ------------------------------------------------------------------

    def _extract_grounded_arguments(
        self, response: str, position: DebatePosition
    ) -> List[GroundedArgument]:
        """Parse a debater response into ``GroundedArgument`` objects.

        Args:
            response: Raw text from the debater LLM.
            position: The debater's assigned position (used for logging only).

        Returns:
            List of ``GroundedArgument`` objects (may be empty if no blocks found).
        """
        blocks = _parse_grounded_blocks(response)
        # Cap at max_tool_calls_per_round
        blocks = blocks[: self._config.max_tool_calls_per_round]

        arguments: List[GroundedArgument] = []
        for blk in blocks:
            claim = blk["claim"]
            tool_name = blk.get("tool")
            tool_output = blk.get("tool_output")
            evidence = blk.get("evidence") or ""

            # If a tool_registry is available and a tool name was cited,
            # try to actually execute the tool.
            real_output: Optional[str] = tool_output
            if (
                tool_name
                and self._config.tool_registry
                and tool_name in self._config.tool_registry
            ):
                try:
                    real_output = str(
                        self._config.tool_registry[tool_name](claim)
                    )
                    logger.debug(
                        "ToolGroundedDebate: executed tool '%s' for claim '%s...'",
                        tool_name,
                        claim[:40],
                    )
                except Exception as exc:
                    logger.warning(
                        "ToolGroundedDebate: tool '%s' execution failed — %s",
                        tool_name,
                        exc,
                    )
                    real_output = tool_output  # fall back to LLM-provided output

            # Score faithfulness
            if real_output:
                faith = self._score_faithfulness(claim, real_output)
            elif not self._config.require_tool_grounding:
                # Grounding not required: give ungrounded claims a neutral score
                faith = 0.5
            else:
                faith = 0.0

            arguments.append(
                GroundedArgument(
                    claim=claim,
                    evidence=evidence,
                    tool_used=tool_name,
                    tool_output=real_output,
                    faithfulness_score=faith,
                )
            )

        return arguments

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _filter_ungrounded_claims(
        self,
        arguments: List[GroundedArgument],
        threshold: Optional[float] = None,
    ) -> Tuple[List[GroundedArgument], int]:
        """Remove arguments below the faithfulness threshold.

        Args:
            arguments: All parsed grounded arguments.
            threshold: Override threshold.  Falls back to config value if None.

        Returns:
            Tuple of (surviving_arguments, num_filtered).
        """
        t = threshold if threshold is not None else self._config.faithfulness_threshold
        surviving = [a for a in arguments if a.faithfulness_score >= t]
        filtered = len(arguments) - len(surviving)
        if filtered:
            logger.debug(
                "ToolGroundedDebate: filtered %d claim(s) below faithfulness %.2f",
                filtered,
                t,
            )
        return surviving, filtered

    # ------------------------------------------------------------------
    # Grounded debater turn
    # ------------------------------------------------------------------

    def _run_grounded_debater_turn(
        self,
        debater: DebaterConfig,
        solution: str,
        task_context: str,
        round_number: int,
        total_rounds: int,
        previous_rounds: List[DebateRound],
    ) -> Tuple[DebateArgument, List[GroundedArgument], int]:
        """Run one debater turn with tool grounding.

        Prompts the debater to use the structured grounding format, parses the
        response, scores faithfulness, and filters low-quality claims.

        Args:
            debater: Debater configuration.
            solution: Solution being debated.
            task_context: Task description.
            round_number: Current round (1-based).
            total_rounds: Total planned rounds.
            previous_rounds: All completed rounds.

        Returns:
            Tuple of (DebateArgument for transcript, grounded_args, num_filtered).
        """
        transcript = _format_transcript(previous_rounds)

        position_label = (
            "FOR the solution (advocate its strengths)"
            if debater.position == DebatePosition.FOR
            else "AGAINST the solution (critique its weaknesses)"
        )
        position_instruction = (
            "for the solution"
            if debater.position == DebatePosition.FOR
            else "against the solution"
        )
        persona_line = (
            f"Your persona: {debater.persona}\n" if debater.persona else ""
        )

        available_tools = (
            ", ".join(self._config.tool_registry.keys())
            if self._config.tool_registry
            else "any relevant tool (web_search, calculator, code_runner, etc.)"
        )

        system_content = _GROUNDED_DEBATER_SYSTEM.format(
            name=debater.name,
            position_label=position_label,
            persona_line=persona_line,
            max_tool_calls=self._config.max_tool_calls_per_round,
            available_tools=available_tools,
        )

        if round_number == 1 or not transcript.strip():
            user_content = _GROUNDED_DEBATER_FIRST_ROUND.format(
                task_context=task_context,
                solution=solution,
                round_number=round_number,
                total_rounds=total_rounds,
                position_instruction=position_instruction,
            )
        else:
            user_content = _GROUNDED_DEBATER_SUBSEQUENT_ROUND.format(
                task_context=task_context,
                solution=solution,
                transcript=transcript,
                round_number=round_number,
                total_rounds=total_rounds,
                position_instruction=position_instruction,
            )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        try:
            response_text = self._generate_fn(messages)
        except Exception as exc:
            logger.warning(
                "ToolGroundedDebate: debater '%s' generation failed — %s",
                debater.name,
                exc,
            )
            response_text = f"[Generation error: {exc}]"

        # Parse and filter grounded arguments
        grounded_args = self._extract_grounded_arguments(
            response_text, debater.position
        )
        surviving_args, num_filtered = self._filter_ungrounded_claims(grounded_args)

        # Build a debate argument from surviving claims for the transcript
        if surviving_args:
            combined_text = "\n".join(
                f"CLAIM: {a.claim}\nEVIDENCE: {a.evidence}"
                for a in surviving_args
            )
        else:
            combined_text = response_text  # fall back to raw text

        quality = _estimate_argument_quality(combined_text)
        debate_arg = DebateArgument(
            debater_name=debater.name,
            position=debater.position,
            round_number=round_number,
            argument=combined_text,
            quality_score=quality,
        )

        logger.debug(
            "ToolGroundedDebate: round %d, %s (%s), "
            "grounded=%d, filtered=%d, quality=%.2f",
            round_number,
            debater.name,
            debater.position.value,
            len(surviving_args),
            num_filtered,
            quality,
        )

        return debate_arg, surviving_args, num_filtered

    # ------------------------------------------------------------------
    # Grounded judge turn
    # ------------------------------------------------------------------

    def _judge_with_grounding(
        self,
        solution: str,
        task_context: str,
        rounds: List[DebateRound],
        pro_args: List[GroundedArgument],
        con_args: List[GroundedArgument],
    ) -> Tuple[bool, Optional[DebatePosition], float, str, List[str]]:
        """Run the judge with grounding context.

        Args:
            solution: The solution being evaluated.
            task_context: Task description.
            rounds: Completed debate rounds.
            pro_args: Surviving FOR-side grounded arguments.
            con_args: Surviving AGAINST-side grounded arguments.

        Returns:
            Tuple of (passed, winning_position, confidence, rationale, weak_points).
        """
        transcript = _format_transcript(rounds)
        grounding_summary = _build_grounding_summary(pro_args, con_args)

        user_content = _GROUNDED_JUDGE_PROMPT.format(
            task_context=task_context,
            solution=solution,
            transcript=transcript,
            grounding_summary=grounding_summary,
        )
        messages = [
            {"role": "system", "content": _GROUNDED_JUDGE_SYSTEM},
            {"role": "user", "content": user_content},
        ]

        try:
            judge_text = self._generate_fn(messages)
        except Exception as exc:
            logger.error(
                "ToolGroundedDebate: judge generation failed — %s", exc
            )
            return False, DebatePosition.AGAINST, 0.0, f"Judge error: {exc}", []

        passed, winning_pos, confidence, rationale = _parse_judge_verdict(judge_text)
        weak_points = _extract_weak_points(judge_text)

        logger.debug(
            "ToolGroundedDebate: judge verdict=%s confidence=%.2f",
            "PASS" if passed else "FAIL",
            confidence,
        )
        return passed, winning_pos, confidence, rationale, weak_points

    # ------------------------------------------------------------------
    # Convergence measurement (delegates to base evaluator)
    # ------------------------------------------------------------------

    def _measure_convergence(
        self, round_a: DebateRound, round_b: DebateRound
    ) -> float:
        """Delegate convergence measurement to the base evaluator."""
        return self._base_evaluator._measure_convergence(round_a, round_b)

    # ------------------------------------------------------------------
    # Main evaluate entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        solution: str,
        task_context: str,
        solution_agent_fn: Optional[Callable] = None,
        probe_examples: Optional[List[Dict[str, Any]]] = None,
        probe_evaluator: Optional[Callable[[str, Dict[str, Any]], float]] = None,
        evaluation_inputs: Optional[List[Dict[str, Any]]] = None,
    ) -> ToolGroundedDebateResult:
        """Run the tool-grounded multi-agent debate and return a quality verdict.

        Identical call signature to ``DebateEvaluator.evaluate()``.

        Args:
            solution: Text description of the proposed solution.
            task_context: Description of the target task and requirements.
            solution_agent_fn: Optional agent callable for drift detection.
            probe_examples: Safety probe examples for drift detection.
            probe_evaluator: Scorer for probe examples.
            evaluation_inputs: Inputs for semantic/behavioral drift.

        Returns:
            ``ToolGroundedDebateResult`` (subclass of ``DebateResult``).
        """
        if not solution or not solution.strip():
            raise ValueError("solution must not be empty.")
        if not task_context or not task_context.strip():
            raise ValueError("task_context must not be empty.")

        logger.info(
            "ToolGroundedDebateEvaluator: starting debate "
            "(%d debaters, up to %d rounds, require_grounding=%s)",
            len(self._debater_configs),
            self.num_rounds,
            self._config.require_tool_grounding,
        )

        rounds: List[DebateRound] = []
        all_grounded: List[GroundedArgument] = []
        total_filtered = 0

        for round_idx in range(1, self.num_rounds + 1):
            current_round = DebateRound(round_number=round_idx)

            for debater in self._debater_configs:
                debate_arg, grounded_args, n_filtered = (
                    self._run_grounded_debater_turn(
                        debater=debater,
                        solution=solution,
                        task_context=task_context,
                        round_number=round_idx,
                        total_rounds=self.num_rounds,
                        previous_rounds=rounds,
                    )
                )
                current_round.arguments.append(debate_arg)
                all_grounded.extend(grounded_args)
                total_filtered += n_filtered

            # Convergence check
            if rounds:
                conv_score = self._measure_convergence(rounds[-1], current_round)
                current_round.convergence_score = conv_score
                logger.debug(
                    "ToolGroundedDebate: round %d convergence=%.3f (threshold=%.2f)",
                    round_idx,
                    conv_score,
                    self.convergence_threshold,
                )
                if conv_score >= self.convergence_threshold:
                    rounds.append(current_round)
                    logger.info(
                        "ToolGroundedDebate: early stop at round %d — "
                        "convergence score %.3f >= %.2f",
                        round_idx,
                        conv_score,
                        self.convergence_threshold,
                    )
                    break

            rounds.append(current_round)

        # Separate pro/con grounded args for judge context
        pro_args = [
            a for a in all_grounded
            # We need position info — stored in rounds; re-collect from rounds
        ]
        # Re-split by position using the rounds structure
        pro_grounded: List[GroundedArgument] = []
        con_grounded: List[GroundedArgument] = []
        for rnd in rounds:
            for arg in rnd.arguments:
                # Find matching grounded args by cross-referencing claim text
                # (We stored grounded args in all_grounded in order)
                pass
        # Simpler: re-classify via debater_configs positions
        # all_grounded were collected in debater order per round; re-derive position
        # by re-running extraction from the debate argument text.
        # Actually the cleanest approach: store position alongside GroundedArgument
        # during collection.  We'll re-derive it here from the transcript.
        pro_grounded, con_grounded = self._split_grounded_by_position(
            rounds, all_grounded
        )

        # Judge evaluation with grounding
        passed, winning_pos, confidence, rationale, weak_points = (
            self._judge_with_grounding(
                solution=solution,
                task_context=task_context,
                rounds=rounds,
                pro_args=pro_grounded,
                con_args=con_grounded,
            )
        )

        # Apply pass_threshold
        if passed and confidence < self.pass_threshold:
            logger.info(
                "ToolGroundedDebate: PASS overridden to FAIL — "
                "confidence %.2f < pass_threshold %.2f",
                confidence,
                self.pass_threshold,
            )
            passed = False
            winning_pos = DebatePosition.AGAINST

        # Argument quality mean
        all_args = [arg for rnd in rounds for arg in rnd.arguments]
        arg_quality = (
            statistics.mean(a.quality_score for a in all_args)
            if all_args else 0.0
        )

        # Grounding rate and mean faithfulness
        if all_grounded:
            grounding_rate = sum(1 for a in all_grounded if a.is_grounded()) / len(all_grounded)
            mean_faithfulness = statistics.mean(
                a.faithfulness_score for a in all_grounded
            )
        else:
            grounding_rate = 0.0
            mean_faithfulness = 0.0

        # Optional alignment drift check
        drift_report = self._base_evaluator._run_drift_check(
            solution_agent_fn=solution_agent_fn,
            probe_examples=probe_examples,
            probe_evaluator=probe_evaluator,
            evaluation_inputs=evaluation_inputs,
        )

        if (
            drift_report is not None
            and hasattr(drift_report, "any_exceeded")
            and drift_report.any_exceeded
        ):
            logger.warning(
                "ToolGroundedDebate: result overridden to FAIL due to alignment drift."
            )
            passed = False
            winning_pos = DebatePosition.AGAINST
            weak_points.insert(
                0, "Alignment drift detected — behavioral regression from baseline."
            )

        result = ToolGroundedDebateResult(
            passed=passed,
            winning_position=winning_pos,
            judge_rationale=rationale,
            confidence=confidence,
            rounds=rounds,
            weak_points=weak_points,
            argument_quality=arg_quality,
            num_rounds_run=len(rounds),
            drift_report=drift_report,
            grounded_arguments=all_grounded,
            grounding_rate=grounding_rate,
            mean_faithfulness=mean_faithfulness,
            filtered_claim_count=total_filtered,
        )

        logger.info(
            "ToolGroundedDebate: verdict=%s confidence=%.2f rounds=%d "
            "grounding_rate=%.2f mean_faithfulness=%.2f filtered=%d",
            "PASS" if result.passed else "FAIL",
            result.confidence,
            result.num_rounds_run,
            result.grounding_rate,
            result.mean_faithfulness,
            result.filtered_claim_count,
        )
        return result

    # ------------------------------------------------------------------
    # Internal: split grounded args by position
    # ------------------------------------------------------------------

    def _split_grounded_by_position(
        self,
        rounds: List[DebateRound],
        all_grounded: List[GroundedArgument],
    ) -> Tuple[List[GroundedArgument], List[GroundedArgument]]:
        """Split accumulated grounded arguments into FOR and AGAINST buckets.

        We re-derive position by correlating grounded argument claim text with
        the argument text stored in each round's DebateArgument (which carries
        position information).

        Args:
            rounds: Completed debate rounds.
            all_grounded: All grounded arguments collected during the debate.

        Returns:
            Tuple of (for_args, against_args).
        """
        pro: List[GroundedArgument] = []
        con: List[GroundedArgument] = []

        # Build a mapping from debater name to position
        position_map: Dict[str, DebatePosition] = {
            cfg.name: cfg.position for cfg in self._debater_configs
        }

        # Reconstruct the interleaved grounded argument stream per debater
        # We collected grounded args in debater order, round by round.
        # Replay the same traversal order and assign position from position_map.
        grounded_iter = iter(all_grounded)
        debater_grounded_counts: Dict[str, int] = {}

        # We need to know how many grounded args each debater produced per round.
        # We can infer this from the debate argument text: count CLAIM: markers.
        for rnd in rounds:
            for debate_arg in rnd.arguments:
                debater_name = debate_arg.debater_name
                position = position_map.get(debater_name, DebatePosition.FOR)
                # Count how many claims this debater's argument contains
                claim_count = debate_arg.argument.upper().count("CLAIM:")
                claim_count = max(1, claim_count)  # at least 1 arg consumed
                # Consume that many from the iterator
                for _ in range(claim_count):
                    try:
                        grounded_arg = next(grounded_iter)
                        if position == DebatePosition.FOR:
                            pro.append(grounded_arg)
                        else:
                            con.append(grounded_arg)
                    except StopIteration:
                        break

        # Any remaining (shouldn't happen, but be safe)
        for grounded_arg in grounded_iter:
            pro.append(grounded_arg)

        return pro, con

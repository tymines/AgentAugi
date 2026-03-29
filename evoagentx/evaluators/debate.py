"""Multi-agent debate quality controls for evolved agent verification.

Inspired by:
- Liang et al. EMNLP 2024: controlled disagreement in multi-agent debate
- A-HMAD: heterogeneous role assignment for richer critique
- Agent-as-a-Judge (arXiv:2410.10934): 90% human agreement vs 70% for single-LLM judge
- NeurIPS 2025 convergence detection: stop debate when rounds degenerate

Architecture
------------
``DebateEvaluator`` orchestrates a structured debate over a proposed solution:

1. **Position assignment** — debaters are assigned FOR (advocate) or AGAINST (critic)
   positions, ensuring genuine adversarial coverage even when using the same model.

2. **Round structure** — each round, every debater generates an argument.  The
   FOR side builds the case; the AGAINST side attacks it.  Arguments reference the
   previous round's transcript so reasoning chains accumulate across rounds.

3. **Convergence detection** — semantic similarity between consecutive rounds is
   measured.  If similarity exceeds ``convergence_threshold`` the debate ends early
   (both sides are just rephrasing, signalling degeneration-of-thought).

4. **Judge evaluation** — after all rounds, a judge LLM reads the full transcript
   and returns a structured verdict: pass/fail + rationale + identified weak points.

5. **Alignment integration** — optionally accepts an ``AlignmentDriftDetector``
   instance.  If a ``solution_agent_fn`` is provided, the detector is run as an
   additional verification step and its ``DriftReport`` is attached to the result.

6. **Quality gate** — ``DebateResult.passed`` can be used as a Boolean gate before
   deploying or accepting an evolved agent configuration.

Usage
-----
    >>> from evoagentx.evaluators.debate import DebateEvaluator
    >>> def my_llm(messages):   # List[dict] -> str
    ...     return call_your_llm(messages)
    >>> evaluator = DebateEvaluator(generate_fn=my_llm, num_rounds=3)
    >>> result = evaluator.evaluate(
    ...     solution="Agent uses chain-of-thought to solve math problems.",
    ...     task_context="The agent must solve GSM8K-level arithmetic with >85% accuracy.",
    ... )
    >>> if result.passed:
    ...     deploy(agent)
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.logging import logger

# ---------------------------------------------------------------------------
# Optional alignment drift integration
# ---------------------------------------------------------------------------
try:
    from .alignment_drift import AlignmentDriftDetector, DriftReport  # type: ignore
    _DRIFT_AVAILABLE = True
except ImportError:
    _DRIFT_AVAILABLE = False
    AlignmentDriftDetector = None  # type: ignore
    DriftReport = None  # type: ignore


# ---------------------------------------------------------------------------
# Enumerations and configuration
# ---------------------------------------------------------------------------

class DebatePosition(Enum):
    """Assigned position for a debater in the evaluation debate.

    FOR debaters advocate for the solution; AGAINST debaters critique it.
    Having both sides ensures adversarial coverage regardless of which model
    is used.
    """

    FOR = "for"
    AGAINST = "against"


@dataclass
class DebaterConfig:
    """Configuration for a single debater agent.

    Attributes:
        name: Display name used in the transcript.
        position: Which side the debater argues.
        persona: Role description included in the system prompt, guiding
            the debater's reasoning style (e.g., "LogicalReasoningCritic").
    """

    name: str
    position: DebatePosition
    persona: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("DebaterConfig.name must not be empty.")


# ---------------------------------------------------------------------------
# Debate transcript data structures
# ---------------------------------------------------------------------------

@dataclass
class DebateArgument:
    """A single argument made by one debater in one round.

    Attributes:
        debater_name: Name of the debater who produced this argument.
        position: The debater's assigned position.
        round_number: 1-based round index.
        argument: The argument text.
        quality_score: Heuristic quality estimate in [0, 1].
            Higher = more substantive (longer, reasoned, specific).
    """

    debater_name: str
    position: DebatePosition
    round_number: int
    argument: str
    quality_score: float = 0.0


@dataclass
class DebateRound:
    """All arguments produced in a single debate round.

    Attributes:
        round_number: 1-based index.
        arguments: All debater arguments for this round.
        convergence_score: Semantic similarity to the previous round [0, 1].
            A score near 1.0 indicates the debate has degenerated (debaters are
            just repeating themselves).  ``None`` for the first round.
    """

    round_number: int
    arguments: List[DebateArgument] = field(default_factory=list)
    convergence_score: Optional[float] = None


@dataclass
class DebateResult:
    """Outcome of a complete multi-agent debate evaluation.

    Attributes:
        passed: True when the judge finds the solution acceptable.
        winning_position: The position that convinced the judge, or None if
            the judge declared a draw.
        judge_rationale: Full rationale text from the judge.
        confidence: Judge's self-reported or heuristically estimated confidence
            in [0, 1].
        rounds: All debate rounds in order.
        weak_points: Specific weaknesses identified during the debate.
        argument_quality: Mean quality score across all arguments.
        num_rounds_run: Actual rounds completed (may be less than ``num_rounds``
            if convergence was detected early).
        drift_report: Optional alignment drift report, populated when an
            ``AlignmentDriftDetector`` and ``solution_agent_fn`` are supplied.
    """

    passed: bool
    winning_position: Optional[DebatePosition]
    judge_rationale: str
    confidence: float
    rounds: List[DebateRound]
    weak_points: List[str]
    argument_quality: float
    num_rounds_run: int
    drift_report: Optional[Any] = None  # DriftReport when available

    def summary(self) -> str:
        """Return a human-readable summary of the debate outcome."""
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            "=== Debate Evaluation Result ===",
            f"  Status       : {status}",
            f"  Confidence   : {self.confidence:.2f}",
            f"  Rounds run   : {self.num_rounds_run}",
            f"  Arg quality  : {self.argument_quality:.2f}",
        ]
        if self.winning_position is not None:
            lines.append(f"  Winner       : {self.winning_position.value.upper()}")
        if self.weak_points:
            lines.append("  Weak points  :")
            for wp in self.weak_points:
                lines.append(f"    - {wp}")
        lines.append(f"  Judge rationale (excerpt): {self.judge_rationale[:200]}")
        if self.drift_report is not None and hasattr(self.drift_report, "summary"):
            lines.append("")
            lines.append(self.drift_report.summary())
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_HEDGING_WORDS = frozenset([
    "maybe", "perhaps", "possibly", "might", "could", "uncertain",
    "unclear", "probably", "appears", "seems", "suggests",
])

_REASONING_MARKERS = frozenset([
    "because", "therefore", "thus", "hence", "since", "given that",
    "as a result", "consequently", "specifically", "for example",
    "evidence", "demonstrates", "shows", "proves", "implies",
])


def _estimate_argument_quality(text: str) -> float:
    """Heuristic quality estimate for a debate argument in [0, 1].

    Penalises:
    - Very short arguments (< 20 words).
    - Excessive hedging without substance.
    - Boilerplate agreement/disagreement openers.

    Rewards:
    - Presence of reasoning markers.
    - Moderate length (20–300 words is the sweet spot).
    - Specific claims (numbers, named entities, quoted text).

    Args:
        text: Argument text.

    Returns:
        Quality estimate in [0, 1].
    """
    if not text or not text.strip():
        return 0.0

    words = text.split()
    word_count = len(words)

    if word_count < 5:
        return 0.05

    # Length score: peaks at ~80 words, tails off gently after 300
    if word_count < 20:
        length_score = word_count / 20.0 * 0.5
    elif word_count <= 300:
        length_score = 0.5 + (word_count - 20) / 280.0 * 0.5
    else:
        # Very long outputs may be padding
        length_score = max(0.5, 1.0 - (word_count - 300) / 1000.0)

    # Reasoning score
    lower_text = text.lower()
    reasoning_hits = sum(1 for m in _REASONING_MARKERS if m in lower_text)
    reasoning_score = min(1.0, reasoning_hits / 3.0)

    # Hedging penalty
    hedge_count = sum(1 for w in words if w.lower().rstrip(".,!?") in _HEDGING_WORDS)
    hedge_penalty = min(0.3, hedge_count / max(1, word_count) * 5.0)

    # Specificity bonus: numbers, percentages, quoted text
    has_numbers = bool(re.search(r"\d+(?:\.\d+)?%?", text))
    has_quotes = '"' in text or "'" in text
    specificity_bonus = 0.1 * (int(has_numbers) + int(has_quotes))

    raw = (0.5 * length_score + 0.3 * reasoning_score + specificity_bonus
           - hedge_penalty)
    return max(0.0, min(1.0, raw))


def _word_overlap_similarity(text_a: str, text_b: str) -> float:
    """Jaccard word-overlap similarity between two texts, used when no embed_fn.

    Args:
        text_a: First text.
        text_b: Second text.

    Returns:
        Jaccard similarity in [0, 1].
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a and not words_b:
        return 1.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Cosine similarity between two vectors in [−1, 1].

    Args:
        vec_a: First embedding.
        vec_b: Second embedding.

    Returns:
        Cosine similarity, clamped to [0, 1] for comparison purposes.
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Embedding length mismatch: {len(vec_a)} vs {len(vec_b)}"
        )
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return max(0.0, dot / (norm_a * norm_b))


def _extract_confidence(text: str) -> Optional[float]:
    """Attempt to parse an explicit confidence value from judge output.

    Looks for patterns like:
    - "confidence: 0.85"
    - "confidence score: 85%"
    - "(confidence: high)" → 0.9
    - "(confidence: medium)" → 0.6
    - "(confidence: low)" → 0.3

    Args:
        text: Judge response text.

    Returns:
        Parsed confidence in [0, 1] or None if not found.
    """
    lower = text.lower()

    # Numeric: "confidence: 0.85", "confidence score: 85%", etc.
    num_match = re.search(
        r"confidence\b[^0-9]*([0-9]+(?:\.[0-9]+)?)(%?)", lower
    )
    if num_match:
        value = float(num_match.group(1))
        if num_match.group(2) == "%":
            value /= 100.0
        return max(0.0, min(1.0, value))

    # Qualitative
    if re.search(r"confidence[:\s]+(very\s+)?high", lower):
        return 0.9
    if re.search(r"confidence[:\s]+medium", lower):
        return 0.6
    if re.search(r"confidence[:\s]+(very\s+)?low", lower):
        return 0.3

    return None


def _heuristic_confidence(text: str) -> float:
    """Fall-back heuristic confidence for judge output when no explicit value.

    Estimates confidence from text clarity: fewer hedging words, stronger
    declarative tone, specific references → higher confidence.

    Args:
        text: Judge response text.

    Returns:
        Heuristic confidence in [0, 1].
    """
    words = text.split()
    if not words:
        return 0.5

    hedge_count = sum(
        1 for w in words if w.lower().rstrip(".,!?") in _HEDGING_WORDS
    )
    hedge_rate = hedge_count / len(words)

    # Strong declarative markers
    strong_markers = ["clearly", "definitely", "certainly", "undeniably",
                      "obviously", "without doubt", "demonstrates", "proves"]
    strong_hits = sum(1 for m in strong_markers if m in text.lower())

    base = 0.65
    adjusted = base - hedge_rate * 2.0 + min(0.2, strong_hits * 0.05)
    return max(0.1, min(0.95, adjusted))


def _extract_weak_points(judge_text: str) -> List[str]:
    """Extract bullet-pointed or numbered weak points from judge text.

    Looks for lines following "weakness", "concern", "flaw", "issue", or
    bullet/numbered list items in the lower half of the judge response.

    Args:
        judge_text: Full judge response text.

    Returns:
        List of weak point strings (may be empty).
    """
    weak_points: List[str] = []

    # Pattern 1: numbered or bulleted lists
    list_pattern = re.compile(r"^\s*(?:\d+[.)]\s*|[-*•]\s+)(.+)$", re.MULTILINE)
    for match in list_pattern.finditer(judge_text):
        item = match.group(1).strip()
        if item:
            weak_points.append(item)

    # Pattern 2: lines following "weakness:", "concern:", "flaw:", "issue:"
    keyword_pattern = re.compile(
        r"(?:weakness|concern|flaw|issue|problem|limitation)[:\s]+(.+?)(?:\n|$)",
        re.IGNORECASE,
    )
    for match in keyword_pattern.finditer(judge_text):
        item = match.group(1).strip()
        if item and item not in weak_points:
            weak_points.append(item)

    return weak_points[:10]  # cap at 10 to avoid noise


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_DEBATER_SYSTEM = (
    "You are {name}, participating in a structured debate about a proposed solution. "
    "Your assigned position is {position_label}.\n"
    "{persona_line}"
    "Be specific, cite concrete evidence or reasoning, and directly engage with "
    "counterarguments from previous rounds. Avoid vague hedging."
)

_DEBATER_FIRST_ROUND = (
    "TASK CONTEXT:\n{task_context}\n\n"
    "PROPOSED SOLUTION:\n{solution}\n\n"
    "This is round {round_number} of {total_rounds}. "
    "Present your opening argument {position_instruction}."
)

_DEBATER_SUBSEQUENT_ROUND = (
    "TASK CONTEXT:\n{task_context}\n\n"
    "PROPOSED SOLUTION:\n{solution}\n\n"
    "DEBATE TRANSCRIPT SO FAR:\n{transcript}\n\n"
    "This is round {round_number} of {total_rounds}. "
    "Continue arguing {position_instruction}, addressing points raised in the transcript."
)

_JUDGE_SYSTEM = (
    "You are an impartial judge evaluating whether a proposed solution is acceptable "
    "for a given task. You have watched a structured debate between advocates and critics. "
    "Your role is to give a definitive verdict based on the strength of arguments presented."
)

_JUDGE_PROMPT = (
    "TASK CONTEXT:\n{task_context}\n\n"
    "PROPOSED SOLUTION:\n{solution}\n\n"
    "DEBATE TRANSCRIPT:\n{transcript}\n\n"
    "Based on the debate above, provide your verdict in the following format:\n"
    "VERDICT: PASS or FAIL\n"
    "CONFIDENCE: a value between 0.0 and 1.0\n"
    "RATIONALE: Your reasoning (be specific about what convinced you)\n"
    "WEAK POINTS:\n"
    "- List each significant weakness identified, one per line\n"
    "END"
)


def _build_debater_messages(
    debater: DebaterConfig,
    solution: str,
    task_context: str,
    round_number: int,
    total_rounds: int,
    transcript: str,
) -> List[Dict[str, str]]:
    """Construct the message list for a single debater turn.

    Args:
        debater: Debater configuration.
        solution: The solution text being debated.
        task_context: Description of the task/requirements.
        round_number: Current round (1-based).
        total_rounds: Total planned rounds.
        transcript: Formatted transcript of all previous arguments.

    Returns:
        List of message dicts with "role" and "content" keys.
    """
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
    system = _DEBATER_SYSTEM.format(
        name=debater.name,
        position_label=position_label,
        persona_line=persona_line,
    )

    if round_number == 1 or not transcript.strip():
        user = _DEBATER_FIRST_ROUND.format(
            task_context=task_context,
            solution=solution,
            round_number=round_number,
            total_rounds=total_rounds,
            position_instruction=position_instruction,
        )
    else:
        user = _DEBATER_SUBSEQUENT_ROUND.format(
            task_context=task_context,
            solution=solution,
            transcript=transcript,
            round_number=round_number,
            total_rounds=total_rounds,
            position_instruction=position_instruction,
        )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _build_judge_messages(
    solution: str,
    task_context: str,
    transcript: str,
) -> List[Dict[str, str]]:
    """Construct the message list for the judge evaluation.

    Args:
        solution: The solution being evaluated.
        task_context: Task description.
        transcript: Full formatted debate transcript.

    Returns:
        List of message dicts.
    """
    user = _JUDGE_PROMPT.format(
        task_context=task_context,
        solution=solution,
        transcript=transcript,
    )
    return [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def _format_transcript(rounds: List[DebateRound]) -> str:
    """Format all debate rounds as a readable transcript string.

    Args:
        rounds: Completed debate rounds.

    Returns:
        Formatted transcript text.
    """
    lines: List[str] = []
    for rnd in rounds:
        lines.append(f"--- Round {rnd.round_number} ---")
        for arg in rnd.arguments:
            side = arg.position.value.upper()
            lines.append(f"[{arg.debater_name} / {side}]")
            lines.append(arg.argument.strip())
            lines.append("")
    return "\n".join(lines)


def _parse_judge_verdict(judge_text: str) -> Tuple[bool, Optional[DebatePosition], float, str]:
    """Parse the judge's response into structured fields.

    Args:
        judge_text: Raw text from the judge LLM.

    Returns:
        Tuple of (passed, winning_position, confidence, rationale).
    """
    lower = judge_text.lower()

    # Verdict
    passed: bool
    if "verdict: pass" in lower or "verdict:pass" in lower:
        passed = True
        winning_position = DebatePosition.FOR
    elif "verdict: fail" in lower or "verdict:fail" in lower:
        passed = False
        winning_position = DebatePosition.AGAINST
    else:
        # Fall back: count positive/negative keywords
        positive_words = ["acceptable", "sound", "correct", "valid", "approved",
                          "suitable", "good", "strong"]
        negative_words = ["unacceptable", "flawed", "incorrect", "invalid",
                          "rejected", "weak", "poor", "insufficient"]
        pos_count = sum(1 for w in positive_words if w in lower)
        neg_count = sum(1 for w in negative_words if w in lower)
        passed = pos_count > neg_count
        winning_position = DebatePosition.FOR if passed else DebatePosition.AGAINST

    # Confidence
    explicit_conf = _extract_confidence(judge_text)
    confidence = explicit_conf if explicit_conf is not None else _heuristic_confidence(judge_text)

    # Rationale: extract the section between RATIONALE: and WEAK POINTS:
    rationale_match = re.search(
        r"RATIONALE[:\s]+(.+?)(?:WEAK\s+POINTS|END|\Z)",
        judge_text,
        re.DOTALL | re.IGNORECASE,
    )
    if rationale_match:
        rationale = rationale_match.group(1).strip()
    else:
        rationale = judge_text.strip()

    return passed, winning_position, confidence, rationale


# ---------------------------------------------------------------------------
# Default debater configurations
# ---------------------------------------------------------------------------

_DEFAULT_DEBATER_CONFIGS: List[DebaterConfig] = [
    DebaterConfig(
        name="StrategicAdvocate",
        position=DebatePosition.FOR,
        persona=(
            "You reason about strategic merit and overall coherence of the solution. "
            "Focus on whether the approach is well-structured, scalable, and aligned "
            "with the stated task requirements."
        ),
    ),
    DebaterConfig(
        name="FactualCritic",
        position=DebatePosition.AGAINST,
        persona=(
            "You are a rigorous FactualVerificationCritic. Question every assumption. "
            "Look for logical gaps, unsupported claims, edge cases the solution misses, "
            "and practical limitations. Be specific—cite the exact wording you dispute."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Main evaluator class
# ---------------------------------------------------------------------------

class DebateEvaluator:
    """Multi-agent debate quality gate for validating proposed solutions.

    Conducts a structured adversarial debate between FOR and AGAINST debaters,
    then uses a judge to render a verdict.  Can serve as a quality gate before
    accepting an evolved agent or prompt configuration.

    Args:
        generate_fn: Callable ``(messages: List[dict]) -> str`` that calls an LLM.
            Messages follow the OpenAI chat format (list of dicts with "role" and
            "content" keys).  This abstraction allows any backend (OpenAI, Anthropic,
            local models) without direct dependency.
        num_rounds: Number of debate rounds.  3 is typically sufficient; more rounds
            increase cost but may surface deeper weaknesses.
        debater_configs: Explicit debater configurations.  If omitted, uses two
            default debaters: a strategic advocate (FOR) and a factual critic (AGAINST).
            Must contain at least one FOR and one AGAINST debater.
        convergence_threshold: Similarity threshold [0, 1] above which consecutive
            rounds are considered degenerate.  Debate ends early if this is exceeded.
            Default 0.85 (high similarity = debaters are just repeating themselves).
        embed_fn: Optional embedding function ``(text: str) -> List[float]`` for
            convergence detection.  Falls back to word-overlap Jaccard when not provided.
        alignment_detector: Optional ``AlignmentDriftDetector`` instance.  When
            provided along with a ``solution_agent_fn`` in ``evaluate()``, runs
            drift detection as an additional verification step.
        pass_threshold: Minimum judge confidence required to treat a PASS verdict as
            a genuine pass.  A confident PASS is more trustworthy than a marginal one.
            Default 0.5 (confident majority).

    Raises:
        ValueError: On invalid configuration.
    """

    def __init__(
        self,
        generate_fn: Callable[[List[Dict[str, str]]], str],
        num_rounds: int = 3,
        debater_configs: Optional[List[DebaterConfig]] = None,
        convergence_threshold: float = 0.85,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        alignment_detector: Optional[Any] = None,
        pass_threshold: float = 0.5,
    ) -> None:
        if num_rounds < 1:
            raise ValueError("num_rounds must be >= 1.")
        if not (0.0 < convergence_threshold <= 1.0):
            raise ValueError("convergence_threshold must be in (0, 1].")
        if not (0.0 <= pass_threshold <= 1.0):
            raise ValueError("pass_threshold must be in [0, 1].")

        self._generate_fn = generate_fn
        self.num_rounds = num_rounds
        self.convergence_threshold = convergence_threshold
        self._embed_fn = embed_fn
        self._alignment_detector = alignment_detector
        self.pass_threshold = pass_threshold

        # Validate and store debater configurations
        if debater_configs is None:
            self._debater_configs = list(_DEFAULT_DEBATER_CONFIGS)
        else:
            self._debater_configs = debater_configs
        self._validate_debater_configs(self._debater_configs)

        logger.debug(
            "DebateEvaluator: %d debaters, %d rounds, convergence_threshold=%.2f",
            len(self._debater_configs),
            self.num_rounds,
            self.convergence_threshold,
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_debater_configs(configs: List[DebaterConfig]) -> None:
        """Ensure at least one FOR and one AGAINST debater is present."""
        if not configs:
            raise ValueError("At least one debater configuration is required.")
        has_for = any(c.position == DebatePosition.FOR for c in configs)
        has_against = any(c.position == DebatePosition.AGAINST for c in configs)
        if not has_for:
            raise ValueError(
                "At least one debater must be assigned DebatePosition.FOR."
            )
        if not has_against:
            raise ValueError(
                "At least one debater must be assigned DebatePosition.AGAINST."
            )

    # ------------------------------------------------------------------
    # Convergence measurement
    # ------------------------------------------------------------------

    def _round_text(self, rnd: DebateRound) -> str:
        """Concatenate all argument texts in a round into one string."""
        return " ".join(arg.argument for arg in rnd.arguments)

    def _measure_convergence(
        self, round_a: DebateRound, round_b: DebateRound
    ) -> float:
        """Measure similarity between two consecutive rounds.

        Uses embedding-based cosine similarity when an embed_fn is available,
        otherwise falls back to word-overlap Jaccard similarity.

        Args:
            round_a: Earlier round.
            round_b: Later round.

        Returns:
            Similarity in [0, 1].
        """
        text_a = self._round_text(round_a)
        text_b = self._round_text(round_b)

        if self._embed_fn is not None:
            try:
                emb_a = self._embed_fn(text_a)
                emb_b = self._embed_fn(text_b)
                return _cosine_similarity(emb_a, emb_b)
            except Exception as exc:
                logger.warning(
                    "DebateEvaluator: embedding failed, falling back to Jaccard — %s", exc
                )

        return _word_overlap_similarity(text_a, text_b)

    # ------------------------------------------------------------------
    # Single debater turn
    # ------------------------------------------------------------------

    def _run_debater_turn(
        self,
        debater: DebaterConfig,
        solution: str,
        task_context: str,
        round_number: int,
        total_rounds: int,
        previous_rounds: List[DebateRound],
    ) -> DebateArgument:
        """Generate one debater argument for the current round.

        Args:
            debater: Configuration for this debater.
            solution: The solution being debated.
            task_context: Task description.
            round_number: Current round (1-based).
            total_rounds: Total planned rounds.
            previous_rounds: All completed rounds for context.

        Returns:
            ``DebateArgument`` with quality score populated.
        """
        transcript = _format_transcript(previous_rounds)
        messages = _build_debater_messages(
            debater=debater,
            solution=solution,
            task_context=task_context,
            round_number=round_number,
            total_rounds=total_rounds,
            transcript=transcript,
        )
        try:
            argument_text = self._generate_fn(messages)
        except Exception as exc:
            logger.warning(
                "DebateEvaluator: debater '%s' generation failed — %s",
                debater.name, exc,
            )
            argument_text = f"[Generation error: {exc}]"

        quality = _estimate_argument_quality(argument_text)
        return DebateArgument(
            debater_name=debater.name,
            position=debater.position,
            round_number=round_number,
            argument=argument_text,
            quality_score=quality,
        )

    # ------------------------------------------------------------------
    # Judge turn
    # ------------------------------------------------------------------

    def _run_judge(
        self,
        solution: str,
        task_context: str,
        rounds: List[DebateRound],
    ) -> Tuple[bool, Optional[DebatePosition], float, str, List[str]]:
        """Run the judge LLM over the full debate transcript.

        Args:
            solution: The solution being evaluated.
            task_context: Task description.
            rounds: Completed debate rounds.

        Returns:
            Tuple of (passed, winning_position, confidence, rationale, weak_points).
        """
        transcript = _format_transcript(rounds)
        messages = _build_judge_messages(
            solution=solution,
            task_context=task_context,
            transcript=transcript,
        )
        try:
            judge_text = self._generate_fn(messages)
        except Exception as exc:
            logger.error("DebateEvaluator: judge generation failed — %s", exc)
            return False, DebatePosition.AGAINST, 0.0, f"Judge error: {exc}", []

        passed, winning_pos, confidence, rationale = _parse_judge_verdict(judge_text)
        weak_points = _extract_weak_points(judge_text)

        logger.debug(
            "DebateEvaluator: judge verdict=%s confidence=%.2f",
            "PASS" if passed else "FAIL",
            confidence,
        )
        return passed, winning_pos, confidence, rationale, weak_points

    # ------------------------------------------------------------------
    # Drift check integration
    # ------------------------------------------------------------------

    def _run_drift_check(
        self,
        solution_agent_fn: Callable,
        probe_examples: Optional[List[Dict[str, Any]]],
        probe_evaluator: Optional[Callable],
        evaluation_inputs: Optional[List[Dict[str, Any]]],
    ) -> Optional[Any]:
        """Run alignment drift detection if configured.

        Args:
            solution_agent_fn: The candidate agent callable.
            probe_examples: Safety probe examples.
            probe_evaluator: Probe scoring function.
            evaluation_inputs: Inputs for semantic/behavioral drift.

        Returns:
            ``DriftReport`` or None if detector not available.
        """
        if self._alignment_detector is None or not _DRIFT_AVAILABLE:
            return None
        if solution_agent_fn is None:
            return None

        try:
            report = self._alignment_detector.check(
                agent_fn=solution_agent_fn,
                probe_examples=probe_examples,
                probe_evaluator=probe_evaluator,
                evaluation_inputs=evaluation_inputs,
            )
            return report
        except Exception as exc:
            logger.warning(
                "DebateEvaluator: drift check failed (non-blocking) — %s", exc
            )
            return None

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
    ) -> DebateResult:
        """Run the multi-agent debate and return a quality verdict.

        The debate proceeds for ``num_rounds`` rounds (or fewer if convergence is
        detected early).  Each debater argues in turn within each round.  After all
        rounds the judge renders a verdict.  If an ``AlignmentDriftDetector`` and
        ``solution_agent_fn`` are provided, a drift check is appended to the result.

        Args:
            solution: Text description of the proposed solution / agent config being
                evaluated.  This is what the debaters argue about.
            task_context: Description of the target task and requirements.  Provides
                the evaluation context the judge uses to assess the solution.
            solution_agent_fn: Optional callable representing the agent being
                evaluated.  Required for alignment drift detection.  Signature:
                ``(example: Dict) -> str``.
            probe_examples: Safety probe examples passed to drift detection.
            probe_evaluator: Scorer for probe examples, passed to drift detection.
            evaluation_inputs: Inputs for semantic/behavioral drift measurement.

        Returns:
            ``DebateResult`` with verdict, rationale, weak points, and optional
            drift report.
        """
        if not solution or not solution.strip():
            raise ValueError("solution must not be empty.")
        if not task_context or not task_context.strip():
            raise ValueError("task_context must not be empty.")

        logger.info(
            "DebateEvaluator: starting debate (%d debaters, up to %d rounds)",
            len(self._debater_configs),
            self.num_rounds,
        )

        rounds: List[DebateRound] = []

        for round_idx in range(1, self.num_rounds + 1):
            current_round = DebateRound(round_number=round_idx)

            # Each debater takes a turn
            for debater in self._debater_configs:
                arg = self._run_debater_turn(
                    debater=debater,
                    solution=solution,
                    task_context=task_context,
                    round_number=round_idx,
                    total_rounds=self.num_rounds,
                    previous_rounds=rounds,
                )
                current_round.arguments.append(arg)
                logger.debug(
                    "DebateEvaluator: round %d, %s (%s), quality=%.2f",
                    round_idx,
                    debater.name,
                    debater.position.value,
                    arg.quality_score,
                )

            # Convergence check (skip first round — nothing to compare)
            if rounds:
                conv_score = self._measure_convergence(rounds[-1], current_round)
                current_round.convergence_score = conv_score
                logger.debug(
                    "DebateEvaluator: round %d convergence=%.3f (threshold=%.2f)",
                    round_idx,
                    conv_score,
                    self.convergence_threshold,
                )
                if conv_score >= self.convergence_threshold:
                    rounds.append(current_round)
                    logger.info(
                        "DebateEvaluator: early stop at round %d — "
                        "convergence score %.3f >= %.2f",
                        round_idx,
                        conv_score,
                        self.convergence_threshold,
                    )
                    break

            rounds.append(current_round)

        # Judge evaluation
        passed, winning_pos, confidence, rationale, weak_points = self._run_judge(
            solution=solution,
            task_context=task_context,
            rounds=rounds,
        )

        # Apply pass_threshold: a low-confidence PASS is treated as failure
        if passed and confidence < self.pass_threshold:
            logger.info(
                "DebateEvaluator: PASS overridden to FAIL — "
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

        # Optional alignment drift check
        drift_report = self._run_drift_check(
            solution_agent_fn=solution_agent_fn,
            probe_examples=probe_examples,
            probe_evaluator=probe_evaluator,
            evaluation_inputs=evaluation_inputs,
        )

        # If drift exceeded threshold, force failure regardless of judge
        if (
            drift_report is not None
            and hasattr(drift_report, "any_exceeded")
            and drift_report.any_exceeded
        ):
            logger.warning(
                "DebateEvaluator: result overridden to FAIL due to alignment drift."
            )
            passed = False
            winning_pos = DebatePosition.AGAINST
            weak_points.insert(
                0, "Alignment drift detected — behavioral regression from baseline."
            )

        result = DebateResult(
            passed=passed,
            winning_position=winning_pos,
            judge_rationale=rationale,
            confidence=confidence,
            rounds=rounds,
            weak_points=weak_points,
            argument_quality=arg_quality,
            num_rounds_run=len(rounds),
            drift_report=drift_report,
        )

        logger.info(
            "DebateEvaluator: verdict=%s confidence=%.2f rounds=%d weak_points=%d",
            "PASS" if result.passed else "FAIL",
            result.confidence,
            result.num_rounds_run,
            len(result.weak_points),
        )
        return result


# ---------------------------------------------------------------------------
# Public helpers for building custom debater configurations
# ---------------------------------------------------------------------------

def make_heterogeneous_debaters(
    num_for: int = 1,
    num_against: int = 2,
) -> List[DebaterConfig]:
    """Build a set of heterogeneous debaters with distinct personas.

    Heterogeneous roles (inspired by A-HMAD) produce richer critiques than
    identical debaters because each role probes a different failure dimension:
    logical consistency, factual accuracy, strategic fit, cost efficiency.

    Args:
        num_for: Number of FOR (advocate) debaters.
        num_against: Number of AGAINST (critic) debaters.

    Returns:
        List of ``DebaterConfig`` instances with pre-defined personas.

    Raises:
        ValueError: If num_for < 1 or num_against < 1.
    """
    if num_for < 1 or num_against < 1:
        raise ValueError("num_for and num_against must each be >= 1.")

    for_personas = [
        ("StrategicAdvocate", "Focus on strategic merit, scalability, and alignment with task requirements."),
        ("ImplementationAdvocate", "Focus on practical feasibility, implementation clarity, and resource usage."),
        ("HistoricalAdvocate", "Reference analogous successful approaches from prior art to support the solution."),
    ]
    against_personas = [
        ("LogicalCritic", "Probe logical consistency: look for circular reasoning, unsupported leaps, and missing edge cases."),
        ("FactualCritic", "Verify factual claims: challenge numbers, benchmarks, and empirical assertions."),
        ("CostCritic", "Assess cost-efficiency: flag solutions that are over-engineered or have excessive inference costs."),
        ("RobustnessCritic", "Stress-test robustness: describe adversarial inputs or distribution shifts that would break the solution."),
    ]

    configs: List[DebaterConfig] = []

    for i in range(num_for):
        name, persona = for_personas[i % len(for_personas)]
        if i >= len(for_personas):
            name = f"{name}_{i}"
        configs.append(DebaterConfig(name=name, position=DebatePosition.FOR, persona=persona))

    for i in range(num_against):
        name, persona = against_personas[i % len(against_personas)]
        if i >= len(against_personas):
            name = f"{name}_{i}"
        configs.append(DebaterConfig(name=name, position=DebatePosition.AGAINST, persona=persona))

    return configs


__all__ = [
    "DebateEvaluator",
    "DebateResult",
    "DebateRound",
    "DebateArgument",
    "DebaterConfig",
    "DebatePosition",
    "make_heterogeneous_debaters",
]

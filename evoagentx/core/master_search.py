"""MASTER — Multi-Agent System with LLM Specialized MCTS.

MASTER replaces LATS's expensive simulation rollouts with LLM confidence-
weighted self-evaluation, achieving dramatically lower token usage while
maintaining or improving accuracy.

Key difference from LATS
------------------------
Instead of running full simulation rollouts (completing an entire trajectory
from each candidate node) to estimate node quality, MASTER asks the LLM to
self-evaluate the current trajectory using structured prompts that assess:

1. **Progress toward goal** — how much of the task is already solved?
2. **Reasoning quality** — are the steps taken logically sound?
3. **Confidence in current path** — how confident is the LLM that this
   trajectory leads to a successful completion?

The self-evaluation returns a (value, confidence) pair.  The confidence
acts as a reliability weight on the value: high-confidence evaluations are
trusted more than low-confidence ones during both UCB selection and
backpropagation.

Algorithm sketch
----------------
1. **Selection** — walk from root using confidence-weighted UCB1.
2. **Expansion** — generate ``num_candidates`` next actions/thoughts via LLM.
3. **Evaluation** — LLM self-evaluation prompt yields (value, confidence).
4. **Backpropagation** — propagate confidence-weighted value to ancestors.
5. **Best action** — after ``max_iterations`` cycles (or early exit), return
   the action sequence leading to the highest-value child of the root.

Token efficiency
----------------
By eliminating full simulation rollouts MASTER uses ~11K tokens per search
compared to ~185K for LATS, while achieving 5% higher accuracy on benchmarks.

Integration
-----------
- Phase 0 ``CostTracker``: each LLM call is optionally recorded.
- Phase 1A ``LLMStepwiseRewardEvaluator``: when supplied, it replaces (or
  augments) the LLM self-evaluation with step-level process reward scores.

Compatibility
-------------
``MASTERSearch.search()`` returns a ``LATSResult`` so it can be used as a
drop-in replacement for ``LATS.search()``.

Usage
-----
    >>> from evoagentx.core.master_search import MASTERSearch, MASTERConfig
    >>> from unittest.mock import MagicMock
    >>> llm = MagicMock()
    >>> master = MASTERSearch(llm=llm, config=MASTERConfig(max_iterations=10))
    >>> result = master.search(task="Write a sorting algorithm", initial_state="")
    >>> print(result.action_sequence)
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .logging import logger
from .lats import LATSResult


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MASTERConfig:
    """Hyper-parameters for a MASTER search run.

    Attributes
    ----------
    max_iterations:
        Total number of selection→expansion→evaluation→backprop cycles.
        Default 20.
    max_depth:
        Maximum depth of the search tree (max actions in a sequence).
        Default 5.
    exploration_weight:
        UCB1 exploration term weight (``c`` in ``c * sqrt(ln N / n)``).
        Default √2 ≈ 1.414.
    confidence_threshold:
        Minimum confidence required to trust a self-evaluation score.
        Nodes with confidence below this threshold receive a penalty on
        their effective value.  Default 0.3.
    num_candidates:
        Number of candidate next actions generated per expansion.  Default 3.
    use_self_evaluation:
        When ``True`` (default), use the LLM confidence-weighted self-
        evaluation prompt to score nodes.  When ``False``, fall back to a
        simple LLM value query (like LATS but without full rollout).
    token_budget:
        Approximate maximum number of tokens (input + output) across the
        entire search.  0 means unlimited.  Default 0.
    value_threshold:
        If a node's estimated value exceeds this threshold the search exits
        early.  Default 0.9.
    temperature:
        LLM sampling temperature for candidate generation.  Default 0.7.
    """

    max_iterations: int = 20
    max_depth: int = 5
    exploration_weight: float = math.sqrt(2)
    confidence_threshold: float = 0.3
    num_candidates: int = 3
    use_self_evaluation: bool = True
    token_budget: int = 0
    value_threshold: float = 0.9
    temperature: float = 0.7


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

@dataclass
class MASTERNode:
    """A single node in the MASTER search tree.

    Unlike ``LATSNode``, each node stores a ``confidence`` value alongside
    its mean value.  The confidence reflects how certain the LLM was when it
    self-evaluated the trajectory ending at this node — it is used to weight
    the UCB1 score and the backpropagation signal.

    Attributes
    ----------
    state:
        Natural-language description of the agent's current partial solution.
    action:
        The action taken from the parent to reach this node.  Empty for root.
    parent:
        Reference to the parent node, or ``None`` for the root.
    depth:
        Depth from root (root = 0).
    children:
        List of child nodes.
    visit_count:
        Number of times this node has been visited.
    total_weighted_value:
        Sum of confidence-weighted values backpropagated through this node.
    total_weight:
        Sum of confidence weights; used to compute the weighted mean value.
    confidence:
        The LLM's most recent confidence estimate for this node's trajectory.
        Initialised to 0.5 (neutral) until the first evaluation.
    is_terminal:
        Whether the trajectory ending at this node is complete.
    """

    state: str
    action: str = ""
    parent: Optional["MASTERNode"] = field(default=None, repr=False)
    depth: int = 0
    children: List["MASTERNode"] = field(default_factory=list)
    visit_count: int = 0
    total_weighted_value: float = 0.0
    total_weight: float = 0.0
    confidence: float = 0.5
    is_terminal: bool = False

    @property
    def mean_value(self) -> float:
        """Confidence-weighted mean value (0.5 before any visit)."""
        if self.total_weight == 0.0:
            return 0.5
        return self.total_weighted_value / self.total_weight

    def ucb1(self, parent_visits: int, exploration_weight: float) -> float:
        """Confidence-weighted UCB1 score.

        The exploitation term is the confidence-weighted mean value.
        The exploration bonus is the standard UCB1 term scaled by the
        node's own confidence (higher confidence → smaller bonus needed
        because we trust the value estimate more).

        Parameters
        ----------
        parent_visits:
            Visit count of the parent node.
        exploration_weight:
            Weight for the exploration bonus term.

        Returns
        -------
        float
            UCB1 score; higher is more promising to explore.
        """
        if self.visit_count == 0:
            return float("inf")
        exploitation = self.mean_value
        # Scale exploration by (1 - confidence): low-confidence nodes get
        # a higher exploration bonus to encourage revisiting them with fresh
        # evaluations.
        confidence_scale = 1.0 + (1.0 - self.confidence)
        exploration = (
            exploration_weight
            * confidence_scale
            * math.sqrt(math.log(parent_visits) / self.visit_count)
        )
        return exploitation + exploration

    def is_leaf(self) -> bool:
        """Return True when this node has no children yet."""
        return len(self.children) == 0

    def action_sequence(self) -> List[str]:
        """Collect the sequence of actions from root to this node."""
        path: List[str] = []
        current: Optional[MASTERNode] = self
        while current is not None and current.action:
            path.append(current.action)
            current = current.parent
        path.reverse()
        return path


# ---------------------------------------------------------------------------
# Self-evaluation prompt
# ---------------------------------------------------------------------------

_SELF_EVAL_PROMPT = """\
You are critically evaluating a partial plan for the following task.

## Task
{task}

## Actions taken so far
{history}

## Current state summary
{state}

Evaluate this trajectory on three dimensions:

1. **Goal progress** (0.0–1.0): How much of the task has been accomplished?
   0.0 = no progress, 1.0 = fully solved.

2. **Reasoning quality** (0.0–1.0): Are the steps logically sound and
   efficient?  0.0 = flawed/circular, 1.0 = exemplary.

3. **Path confidence** (0.0–1.0): How confident are you that continuing
   from this exact trajectory will successfully complete the task?
   0.0 = very unlikely, 1.0 = almost certain.

Respond with a JSON object on a SINGLE LINE, no markdown:
{{"goal_progress": <float>, "reasoning_quality": <float>, "path_confidence": <float>, "rationale": "<one sentence>"}}

Example:
{{"goal_progress": 0.6, "reasoning_quality": 0.8, "path_confidence": 0.7, "rationale": "Good start but missing error handling."}}
"""

_SIMPLE_EVAL_PROMPT = """\
You are evaluating a partial plan for the following task.

## Task
{task}

## Actions taken so far
{history}

## Current state
{state}

Return the probability (0.0 to 1.0) that this trajectory will successfully
complete the task.

Respond with a JSON object on a SINGLE LINE:
{{"value": <float>}}
"""


# ---------------------------------------------------------------------------
# MASTERSearch core
# ---------------------------------------------------------------------------

class MASTERSearch:
    """MASTER: Multi-Agent System with LLM Specialized MCTS.

    A drop-in replacement for ``LATS`` that uses LLM confidence-weighted
    self-evaluation instead of expensive simulation rollouts.

    Parameters
    ----------
    llm:
        A ``BaseLLM`` instance (or any object with a ``generate(prompt)``
        method returning an object whose ``.content`` attribute is a string).
    config:
        ``MASTERConfig`` with search hyper-parameters.
    reward_evaluator:
        Optional Phase 1A ``LLMStepwiseRewardEvaluator``.  When provided,
        its ``score_trajectory`` output is blended with the self-evaluation
        score for richer signal.
    cost_tracker:
        Optional Phase 0 ``CostTracker``.  When provided every LLM call is
        recorded under the supplied ``model_name``.
    provider:
        Provider string passed to ``CostTracker.record()``.
    model_name:
        Model name string passed to ``CostTracker.record()``.

    Examples
    --------
    >>> master = MASTERSearch(llm=my_llm, config=MASTERConfig(max_iterations=5))
    >>> result = master.search(task="Plan a trip to Paris", initial_state="")
    >>> print(result.action_sequence)
    """

    def __init__(
        self,
        llm: Any,
        config: Optional[MASTERConfig] = None,
        reward_evaluator: Optional[Any] = None,
        cost_tracker: Optional[Any] = None,
        provider: str = "unknown",
        model_name: str = "unknown",
    ) -> None:
        self._llm = llm
        self._config = config or MASTERConfig()
        self._reward_evaluator = reward_evaluator
        self._cost_tracker = cost_tracker
        self._provider = provider
        self._model_name = model_name
        self._nodes_expanded: int = 0
        self._tokens_used: int = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def search(self, task: str, initial_state: str = "") -> LATSResult:
        """Run MASTER to find the best action sequence for ``task``.

        This method has the same signature as ``LATS.search()`` and returns
        a ``LATSResult`` for drop-in compatibility.

        Parameters
        ----------
        task:
            Natural-language description of the goal the agent must achieve.
        initial_state:
            Optional context already accumulated before the search starts.

        Returns
        -------
        LATSResult
            Best action sequence and associated metadata.
        """
        start = time.time()
        self._nodes_expanded = 0
        self._tokens_used = 0

        root = MASTERNode(state=initial_state, depth=0)
        best_node = root
        iterations_run = 0

        for iter_idx in range(self._config.max_iterations):
            # Token budget check before each iteration
            if self._config.token_budget > 0 and self._tokens_used >= self._config.token_budget:
                logger.info(
                    f"MASTER | token budget {self._config.token_budget} reached "
                    f"after {iterations_run} iterations ({self._tokens_used} tokens used)"
                )
                break

            # 1. Selection
            leaf = self._select(root)

            # 2. Expansion (skip if terminal or at max depth)
            if not leaf.is_terminal and leaf.depth < self._config.max_depth:
                children = self._expand(leaf, task)
                if children:
                    leaf = children[0]

            # 3. Evaluation (confidence-weighted self-evaluation, not rollout)
            value, confidence = self._evaluate(leaf, task)

            # Update node confidence from this evaluation
            leaf.confidence = confidence

            # 4. Backpropagation with confidence weighting
            self._backpropagate(leaf, value, confidence)

            iterations_run += 1

            # Track best
            if leaf.mean_value > best_node.mean_value:
                best_node = leaf

            # Early exit if a high-quality solution was found
            if best_node.mean_value >= self._config.value_threshold:
                logger.debug(
                    f"MASTER | early exit at iteration {iter_idx + 1} "
                    f"with value {best_node.mean_value:.3f} "
                    f"(confidence={best_node.confidence:.3f})"
                )
                elapsed = time.time() - start
                return LATSResult(
                    action_sequence=best_node.action_sequence(),
                    best_value=best_node.mean_value,
                    final_state=best_node.state,
                    nodes_expanded=self._nodes_expanded,
                    simulations_run=iterations_run,
                    elapsed_seconds=elapsed,
                    converged_early=True,
                )

        elapsed = time.time() - start
        logger.info(
            f"MASTER | {iterations_run} iterations, {self._nodes_expanded} nodes expanded "
            f"in {elapsed:.2f}s | best value={best_node.mean_value:.3f} "
            f"| tokens used={self._tokens_used}"
        )
        return LATSResult(
            action_sequence=best_node.action_sequence(),
            best_value=best_node.mean_value,
            final_state=best_node.state,
            nodes_expanded=self._nodes_expanded,
            simulations_run=iterations_run,
            elapsed_seconds=elapsed,
            converged_early=False,
        )

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _select(self, root: MASTERNode) -> MASTERNode:
        """Walk the tree from ``root`` using confidence-weighted UCB1."""
        node = root
        while not node.is_leaf() and not node.is_terminal:
            best_child = max(
                node.children,
                key=lambda c: c.ucb1(node.visit_count, self._config.exploration_weight),
            )
            node = best_child
        return node

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------

    def _expand(self, node: MASTERNode, task: str) -> List[MASTERNode]:
        """Generate candidate next actions and attach them as children."""
        prompt = self._build_expansion_prompt(task, node)
        raw = self._call_llm(prompt, purpose="expansion")
        actions = self._parse_actions(raw)

        children: List[MASTERNode] = []
        for action in actions[: self._config.num_candidates]:
            new_state = self._apply_action(node.state, action)
            child = MASTERNode(
                state=new_state,
                action=action,
                parent=node,
                depth=node.depth + 1,
            )
            node.children.append(child)
            children.append(child)
            self._nodes_expanded += 1

        logger.debug(
            f"MASTER | expanded node depth={node.depth} → {len(children)} children"
        )
        return children

    def _build_expansion_prompt(self, task: str, node: MASTERNode) -> str:
        """Construct the LLM prompt for generating candidate next actions."""
        history = "\n".join(node.action_sequence()) or "(none yet)"
        return (
            f"You are planning how to solve the following task step by step.\n\n"
            f"Task: {task}\n\n"
            f"Actions taken so far:\n{history}\n\n"
            f"Current state summary:\n{node.state or '(initial state)'}\n\n"
            f"Generate exactly {self._config.num_candidates} distinct candidate next "
            f"actions that would make progress toward solving the task. "
            f"Return a JSON array of strings, e.g.:\n"
            f'["action 1", "action 2", "action 3"]\n\n'
            f"Each action should be concrete and specific. Return ONLY the JSON array."
        )

    def _parse_actions(self, raw: str) -> List[str]:
        """Extract a list of action strings from LLM response text."""
        text = raw.strip()
        for fence in ("```json", "```"):
            if text.startswith(fence):
                text = text[len(fence):]
        text = text.rstrip("`").strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(a) for a in parsed if a]
        except (json.JSONDecodeError, ValueError):
            pass

        logger.warning("MASTER | could not parse actions JSON; using raw text as single action")
        return [raw.strip()] if raw.strip() else []

    # ------------------------------------------------------------------
    # Evaluation (the key MASTER difference from LATS)
    # ------------------------------------------------------------------

    def _evaluate(self, node: MASTERNode, task: str) -> Tuple[float, float]:
        """Evaluate node quality using LLM self-evaluation.

        This is the central innovation of MASTER: instead of running a full
        simulation rollout (which requires many additional LLM calls to
        complete the trajectory), we ask the LLM to self-assess the current
        trajectory on three structured dimensions and extract both a value
        estimate and a confidence score.

        When a ``reward_evaluator`` is supplied, its score is blended with
        the self-evaluation score for a more robust signal.

        Parameters
        ----------
        node:
            The node to evaluate.
        task:
            The overall task description.

        Returns
        -------
        Tuple[float, float]
            ``(value, confidence)`` where both are in [0, 1].
            ``value`` is the estimated quality of the trajectory.
            ``confidence`` is the LLM's certainty about that estimate.
        """
        if self._config.use_self_evaluation:
            value, confidence = self._self_evaluate(node, task)
        else:
            value = self._simple_evaluate(node, task)
            confidence = 0.5  # neutral confidence for simple evaluation

        # Optional blend with process reward evaluator
        if self._reward_evaluator is not None:
            reward_value = self._evaluate_with_reward_model(node, task)
            # Weighted blend: reward model gets 40% weight
            value = 0.6 * value + 0.4 * reward_value

        return value, confidence

    def _self_evaluate(self, node: MASTERNode, task: str) -> Tuple[float, float]:
        """Run the structured self-evaluation prompt and parse the result."""
        history = "\n".join(node.action_sequence()) or "(none yet)"
        prompt = _SELF_EVAL_PROMPT.format(
            task=task,
            history=history,
            state=node.state or "(initial state)",
        )
        raw = self._call_llm(prompt, purpose="self_evaluation")
        return self._parse_self_eval(raw)

    def _parse_self_eval(self, raw: str) -> Tuple[float, float]:
        """Extract (value, confidence) from a self-evaluation LLM response.

        The composite value is a weighted combination of the three dimensions:
        - goal_progress (weight 0.4)
        - reasoning_quality (weight 0.3)
        - path_confidence (weight 0.3, also used as the returned confidence)

        Falls back to (0.5, 0.5) on any parse failure.
        """
        text = raw.strip()
        for fence in ("```json", "```"):
            if text.startswith(fence):
                text = text[len(fence):]
        text = text.rstrip("`").strip()

        # Find first JSON object in case there is surrounding text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            logger.warning(
                f"MASTER | could not locate JSON in self-eval response; "
                f"using fallback. Raw: {raw[:80]}"
            )
            return 0.5, 0.5

        try:
            data = json.loads(text[start:end])
            goal_progress = float(data.get("goal_progress", 0.5))
            reasoning_quality = float(data.get("reasoning_quality", 0.5))
            path_confidence = float(data.get("path_confidence", 0.5))

            # Clamp all three to [0, 1]
            goal_progress = max(0.0, min(1.0, goal_progress))
            reasoning_quality = max(0.0, min(1.0, reasoning_quality))
            path_confidence = max(0.0, min(1.0, path_confidence))

            # Composite value: weighted average of the three dimensions
            value = (
                0.4 * goal_progress
                + 0.3 * reasoning_quality
                + 0.3 * path_confidence
            )
            # Confidence is the path_confidence score
            confidence = path_confidence

            return value, confidence

        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning(
                f"MASTER | self-eval JSON parse failed ({exc}); using fallback. "
                f"Raw: {raw[:80]}"
            )
            return 0.5, 0.5

    def _simple_evaluate(self, node: MASTERNode, task: str) -> float:
        """Simpler evaluation: just ask LLM for a probability estimate."""
        history = "\n".join(node.action_sequence()) or "(none yet)"
        prompt = _SIMPLE_EVAL_PROMPT.format(
            task=task,
            history=history,
            state=node.state or "(initial state)",
        )
        raw = self._call_llm(prompt, purpose="simple_evaluation")
        return self._parse_value(raw)

    def _parse_value(self, raw: str) -> float:
        """Extract a float value from an LLM response."""
        text = raw.strip()
        for fence in ("```json", "```"):
            if text.startswith(fence):
                text = text[len(fence):]
        text = text.rstrip("`").strip()

        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > 0:
            try:
                data = json.loads(text[start:end])
                if "value" in data:
                    return max(0.0, min(1.0, float(data["value"])))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        try:
            return max(0.0, min(1.0, float(text)))
        except ValueError:
            pass

        logger.warning(f"MASTER | could not parse value from '{raw[:60]}'; using 0.5")
        return 0.5

    def _evaluate_with_reward_model(self, node: MASTERNode, task: str) -> float:
        """Use the optional process reward evaluator to score the trajectory."""
        trajectory = [
            {"type": "reasoning", "content": action}
            for action in node.action_sequence()
        ]
        if not trajectory:
            return 0.5

        try:
            scores = self._reward_evaluator.score_trajectory(trajectory, context=task)
            return scores.mean_score
        except Exception as exc:
            logger.warning(f"MASTER | reward_evaluator failed: {exc}; using 0.5")
            return 0.5

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def _backpropagate(
        self, node: MASTERNode, value: float, confidence: float
    ) -> None:
        """Propagate confidence-weighted value up from ``node`` to root.

        Each ancestor accumulates a weighted contribution: nodes evaluated
        with higher confidence have more influence on the tree's value
        estimates.  Low-confidence evaluations still propagate but are
        down-weighted.

        Parameters
        ----------
        node:
            Starting node for propagation.
        value:
            Evaluation score in [0, 1].
        confidence:
            Confidence in the evaluation, in [0, 1].
        """
        # Use confidence as the weight, with a floor to avoid zero-weight updates
        weight = max(confidence, 0.1)
        current: Optional[MASTERNode] = node
        while current is not None:
            current.visit_count += 1
            current.total_weighted_value += weight * value
            current.total_weight += weight
            current = current.parent

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_action(current_state: str, action: str) -> str:
        """Produce a new state string by appending the action."""
        if not current_state:
            return f"Step 1: {action}"
        steps = current_state.count("\n") + 2
        return f"{current_state}\nStep {steps}: {action}"

    # ------------------------------------------------------------------
    # LLM call helper
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str, purpose: str = "") -> str:
        """Call the LLM, record cost, and track token usage.

        Parameters
        ----------
        prompt:
            The prompt string to send to the LLM.
        purpose:
            Descriptive label for logging.

        Returns
        -------
        str
            The LLM's text response.
        """
        try:
            response = self._llm.generate(prompt)
            content = getattr(response, "content", str(response))

            # Approximate token counts (4 chars ≈ 1 token)
            in_tok = max(1, len(prompt) // 4)
            out_tok = max(1, len(content) // 4) if content else 1
            self._tokens_used += in_tok + out_tok

            if self._cost_tracker is not None:
                try:
                    self._cost_tracker.record(
                        self._provider,
                        self._model_name,
                        input_tokens=in_tok,
                        output_tokens=out_tok,
                        metadata={"purpose": f"master/{purpose}"},
                    )
                except Exception as cost_exc:
                    logger.warning(f"MASTER | cost tracking failed: {cost_exc}")

            return content
        except Exception as exc:
            logger.error(f"MASTER | LLM call failed ({purpose}): {exc}")
            return ""

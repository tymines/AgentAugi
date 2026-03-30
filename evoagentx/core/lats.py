"""Language Agent Tree Search (LATS) — runtime MCTS planning for agents.

At each decision point an agent faces, LATS runs a miniature Monte Carlo Tree
Search over possible next-action sequences, using an LLM to both generate
candidate actions (expansion) and estimate node quality (simulation), before
selecting and executing the best action found.

Conceptually inspired by LATS (Zhou et al., 2023) but written from scratch to
fit the EvoAgentX architecture.

Algorithm sketch
----------------
1. **Selection** — walk from the root using UCB1 to pick the most promising
   unexplored leaf.
2. **Expansion** — ask the LLM to generate *width* candidate next actions from
   that leaf's state.
3. **Simulation** — fast rollout from each child: ask the LLM to complete the
   trajectory from that child and return an estimated value in [0, 1].
4. **Backpropagation** — propagate the simulation value up the path from the
   selected leaf to the root, updating visit counts and total value.
5. **Best action** — after *n_simulations* iterations return the action
   sequence leading to the highest-value child of the root.

Integration
-----------
- Phase 0 ``CostTracker``: each LLM call is optionally recorded.
- Phase 1A ``LLMStepwiseRewardEvaluator``: when a reward evaluator is
  supplied it replaces the cheap simulation rollout with a step-level score,
  which is faster and more accurate.
- Phase 2B ``DifficultyRouter``: the caller can supply the difficulty tier so
  LATS uses an appropriate simulation model rather than always the expensive
  one.

Usage
-----
    >>> from evoagentx.core.lats import LATS, LATSConfig
    >>> from unittest.mock import MagicMock
    >>> llm = MagicMock()          # replace with real BaseLLM
    >>> lats = LATS(llm=llm, config=LATSConfig(max_depth=3, n_simulations=10))
    >>> best = lats.search(
    ...     task="Write a function that checks if a number is prime.",
    ...     initial_state="",
    ... )
    >>> print(best.action_sequence)
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .logging import logger


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LATSConfig:
    """Hyper-parameters for a LATS search run.

    Attributes
    ----------
    max_depth:
        Maximum depth of the tree (max actions in a sequence).  Default 5.
    max_width:
        Maximum number of child nodes per expansion.  Default 3.
    n_simulations:
        Total number of selection→expansion→simulation→backprop cycles to run.
        Default 20.
    exploration_constant:
        UCB1 exploration term weight (``c`` in ``c * sqrt(ln N / n)``).
        Higher values encourage exploration; lower values exploit known paths.
        Default √2 ≈ 1.414.
    simulation_depth:
        Maximum number of steps in a fast-rollout simulation.  Default 3.
    value_threshold:
        If a node's estimated value exceeds this threshold the search exits
        early (good-enough solution found).  Default 0.9.
    temperature:
        LLM sampling temperature used when generating candidate actions.
        Default 0.7.
    """

    max_depth: int = 5
    max_width: int = 3
    n_simulations: int = 20
    exploration_constant: float = math.sqrt(2)
    simulation_depth: int = 3
    value_threshold: float = 0.9
    temperature: float = 0.7


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

@dataclass
class LATSNode:
    """A single node in the LATS search tree.

    Attributes
    ----------
    state:
        Natural-language description of the agent's current partial solution
        state — the cumulative context of all actions taken to reach this
        node.
    action:
        The action taken from the parent to reach this node.  Empty string
        for the root.
    parent:
        Reference to the parent node, or ``None`` for the root.
    depth:
        Depth from root (root = 0).
    children:
        List of child nodes created during expansion.
    visit_count:
        Number of times this node has been visited during selection.
    total_value:
        Sum of simulation values backpropagated through this node.
    is_terminal:
        Whether the trajectory ending at this node is complete.
    """

    state: str
    action: str = ""
    parent: Optional["LATSNode"] = field(default=None, repr=False)
    depth: int = 0
    children: List["LATSNode"] = field(default_factory=list)
    visit_count: int = 0
    total_value: float = 0.0
    is_terminal: bool = False

    @property
    def mean_value(self) -> float:
        """Average value of rollouts through this node (0.5 before any visit)."""
        if self.visit_count == 0:
            return 0.5
        return self.total_value / self.visit_count

    def ucb1(self, parent_visits: int, exploration_constant: float) -> float:
        """UCB1 score balancing exploitation and exploration.

        Parameters
        ----------
        parent_visits:
            Visit count of the parent node.
        exploration_constant:
            Weight of the exploration bonus term.

        Returns
        -------
        float
            UCB1 score; higher is more promising to explore.
        """
        if self.visit_count == 0:
            return float("inf")
        exploitation = self.mean_value
        exploration = exploration_constant * math.sqrt(
            math.log(parent_visits) / self.visit_count
        )
        return exploitation + exploration

    def is_leaf(self) -> bool:
        """Return True when this node has no children yet."""
        return len(self.children) == 0

    def action_sequence(self) -> List[str]:
        """Collect the sequence of actions from root to this node."""
        path: List[str] = []
        current: Optional[LATSNode] = self
        while current is not None and current.action:
            path.append(current.action)
            current = current.parent
        path.reverse()
        return path


# ---------------------------------------------------------------------------
# Search result
# ---------------------------------------------------------------------------

@dataclass
class LATSResult:
    """The outcome of a LATS search.

    Attributes
    ----------
    action_sequence:
        Ordered list of actions from root to the best terminal node found.
    best_value:
        Estimated quality of the best solution found (in [0, 1]).
    final_state:
        Accumulated state string at the best node.
    nodes_expanded:
        Total number of tree nodes expanded during the search.
    simulations_run:
        Number of selection→backprop cycles completed.
    elapsed_seconds:
        Wall-clock time for the search.
    converged_early:
        ``True`` if the search stopped early because a high-value solution
        was found (value ≥ ``LATSConfig.value_threshold``).
    """

    action_sequence: List[str]
    best_value: float
    final_state: str
    nodes_expanded: int
    simulations_run: int
    elapsed_seconds: float
    converged_early: bool = False


# ---------------------------------------------------------------------------
# LATS core
# ---------------------------------------------------------------------------

class LATS:
    """Language Agent Tree Search — MCTS planning loop for LLM agents.

    Parameters
    ----------
    llm:
        A ``BaseLLM`` instance (or any object with a ``generate(prompt)``
        method returning an object whose ``.content`` attribute is a string).
    config:
        ``LATSConfig`` with search hyper-parameters.
    reward_evaluator:
        Optional Phase 1A ``LLMStepwiseRewardEvaluator``.  When provided it
        is used to score simulated trajectories instead of the cheap LLM
        rollout, improving value estimation quality.
    cost_tracker:
        Optional Phase 0 ``CostTracker``.  When provided every LLM call is
        recorded under the supplied ``model_name``.
    provider:
        Provider string passed to ``CostTracker.record()`` (e.g.
        ``"anthropic"``).
    model_name:
        Model name string passed to ``CostTracker.record()``.

    Examples
    --------
    >>> lats = LATS(llm=my_llm, config=LATSConfig(n_simulations=5))
    >>> result = lats.search(task="Plan a trip to Paris", initial_state="")
    >>> print(result.action_sequence)
    """

    def __init__(
        self,
        llm: Any,
        config: Optional[LATSConfig] = None,
        reward_evaluator: Optional[Any] = None,
        cost_tracker: Optional[Any] = None,
        provider: str = "unknown",
        model_name: str = "unknown",
    ) -> None:
        self._llm = llm
        self._config = config or LATSConfig()
        self._reward_evaluator = reward_evaluator
        self._cost_tracker = cost_tracker
        self._provider = provider
        self._model_name = model_name
        self._nodes_expanded: int = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def search(self, task: str, initial_state: str = "") -> LATSResult:
        """Run LATS to find the best action sequence for ``task``.

        Parameters
        ----------
        task:
            Natural-language description of the goal the agent must achieve.
        initial_state:
            Optional context already accumulated before the search starts
            (e.g. prior tool outputs, conversation history).

        Returns
        -------
        LATSResult
            Best action sequence and associated metadata.
        """
        start = time.time()
        self._nodes_expanded = 0

        root = LATSNode(state=initial_state, depth=0)
        best_node = root
        sims_run = 0

        for sim_idx in range(self._config.n_simulations):
            # 1. Selection
            leaf = self._select(root)

            # 2. Expansion (skip if already terminal or at max depth)
            if not leaf.is_terminal and leaf.depth < self._config.max_depth:
                children = self._expand(leaf, task)
                if children:
                    leaf = children[0]  # evaluate the first new child

            # 3. Simulation
            value = self._simulate(leaf, task)

            # 4. Backpropagation
            self._backpropagate(leaf, value)

            sims_run += 1

            # Track best
            if leaf.mean_value > best_node.mean_value:
                best_node = leaf

            # Early exit if we found a high-quality solution
            if best_node.mean_value >= self._config.value_threshold:
                logger.debug(
                    f"LATS | early exit at simulation {sim_idx + 1} "
                    f"with value {best_node.mean_value:.3f}"
                )
                elapsed = time.time() - start
                return LATSResult(
                    action_sequence=best_node.action_sequence(),
                    best_value=best_node.mean_value,
                    final_state=best_node.state,
                    nodes_expanded=self._nodes_expanded,
                    simulations_run=sims_run,
                    elapsed_seconds=elapsed,
                    converged_early=True,
                )

        elapsed = time.time() - start
        logger.info(
            f"LATS | {sims_run} simulations, {self._nodes_expanded} nodes expanded "
            f"in {elapsed:.2f}s | best value={best_node.mean_value:.3f}"
        )
        return LATSResult(
            action_sequence=best_node.action_sequence(),
            best_value=best_node.mean_value,
            final_state=best_node.state,
            nodes_expanded=self._nodes_expanded,
            simulations_run=sims_run,
            elapsed_seconds=elapsed,
            converged_early=False,
        )

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _select(self, root: LATSNode) -> LATSNode:
        """Walk the tree from ``root`` using UCB1 until a leaf is reached.

        Parameters
        ----------
        root:
            Starting node of the tree walk.

        Returns
        -------
        LATSNode
            The leaf node selected for expansion.
        """
        node = root
        while not node.is_leaf() and not node.is_terminal:
            best_child = max(
                node.children,
                key=lambda c: c.ucb1(node.visit_count, self._config.exploration_constant),
            )
            node = best_child
        return node

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------

    def _expand(self, node: LATSNode, task: str) -> List[LATSNode]:
        """Generate candidate next actions and attach them as children.

        Parameters
        ----------
        node:
            The leaf node to expand.
        task:
            The overall task description (provides goal context to the LLM).

        Returns
        -------
        List[LATSNode]
            Newly created child nodes (may be empty on LLM failure).
        """
        prompt = self._build_expansion_prompt(task, node)
        raw = self._call_llm(prompt, purpose="expansion")
        actions = self._parse_actions(raw)

        children: List[LATSNode] = []
        for action in actions[: self._config.max_width]:
            new_state = self._apply_action(node.state, action)
            child = LATSNode(
                state=new_state,
                action=action,
                parent=node,
                depth=node.depth + 1,
            )
            node.children.append(child)
            children.append(child)
            self._nodes_expanded += 1

        logger.debug(f"LATS | expanded node depth={node.depth} → {len(children)} children")
        return children

    def _build_expansion_prompt(self, task: str, node: LATSNode) -> str:
        """Construct the LLM prompt for generating candidate next actions."""
        history = "\n".join(node.action_sequence()) or "(none yet)"
        return (
            f"You are planning how to solve the following task step by step.\n\n"
            f"Task: {task}\n\n"
            f"Actions taken so far:\n{history}\n\n"
            f"Current state summary:\n{node.state or '(initial state)'}\n\n"
            f"Generate exactly {self._config.max_width} distinct candidate next actions "
            f"that would make progress toward solving the task. "
            f"Return a JSON array of strings, e.g.:\n"
            f'["action 1", "action 2", "action 3"]\n\n'
            f"Each action should be concrete and specific. Return ONLY the JSON array."
        )

    def _parse_actions(self, raw: str) -> List[str]:
        """Extract a list of action strings from LLM response text.

        Handles markdown fences and falls back to a single-item list on
        parse failures to keep the search alive.
        """
        text = raw.strip()
        # Strip markdown fences
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

        logger.warning("LATS | could not parse actions JSON; using raw text as single action")
        return [raw.strip()] if raw.strip() else []

    # ------------------------------------------------------------------
    # Simulation (fast rollout)
    # ------------------------------------------------------------------

    def _simulate(self, node: LATSNode, task: str) -> float:
        """Estimate the value of the trajectory ending at ``node``.

        If a ``reward_evaluator`` (Phase 1A) was supplied, uses it to score
        the accumulated action sequence as a trajectory.  Otherwise falls
        back to asking the LLM to rate the current state directly.

        Parameters
        ----------
        node:
            The leaf node to evaluate.
        task:
            The overall task description.

        Returns
        -------
        float
            Estimated value in [0, 1].
        """
        if self._reward_evaluator is not None:
            return self._simulate_with_reward_evaluator(node, task)
        return self._simulate_with_rollout(node, task)

    def _simulate_with_reward_evaluator(self, node: LATSNode, task: str) -> float:
        """Use Phase 1A process reward model to value the trajectory."""
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
            logger.warning(f"LATS | reward_evaluator failed: {exc}; falling back to rollout")
            return self._simulate_with_rollout(node, task)

    def _simulate_with_rollout(self, node: LATSNode, task: str) -> float:
        """Fast LLM rollout: estimate task completion probability from node state."""
        prompt = self._build_simulation_prompt(task, node)
        raw = self._call_llm(prompt, purpose="simulation")
        return self._parse_value(raw)

    def _build_simulation_prompt(self, task: str, node: LATSNode) -> str:
        """Construct the simulation valuation prompt."""
        history = "\n".join(node.action_sequence()) or "(none yet)"
        return (
            f"You are evaluating a partial plan for the following task.\n\n"
            f"Task: {task}\n\n"
            f"Actions taken so far:\n{history}\n\n"
            f"Current state:\n{node.state or '(initial state)'}\n\n"
            f"Estimate the probability (0.0 to 1.0) that continuing from this "
            f"state will successfully complete the task. "
            f"Return ONLY a JSON object: {{\"value\": <float>}}"
        )

    def _parse_value(self, raw: str) -> float:
        """Extract a float value from the LLM's simulation response."""
        text = raw.strip()
        for fence in ("```json", "```"):
            if text.startswith(fence):
                text = text[len(fence):]
        text = text.rstrip("`").strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "value" in parsed:
                val = float(parsed["value"])
                return max(0.0, min(1.0, val))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Try bare float
        try:
            val = float(text)
            return max(0.0, min(1.0, val))
        except ValueError:
            pass

        logger.warning(f"LATS | could not parse value from '{raw[:60]}'; using 0.5")
        return 0.5

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------

    def _backpropagate(self, node: LATSNode, value: float) -> None:
        """Propagate ``value`` up from ``node`` to the root.

        Parameters
        ----------
        node:
            The node from which to start propagation.
        value:
            Simulation value to add to each ancestor's total.
        """
        current: Optional[LATSNode] = node
        while current is not None:
            current.visit_count += 1
            current.total_value += value
            current = current.parent

    # ------------------------------------------------------------------
    # Action application
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_action(current_state: str, action: str) -> str:
        """Produce a new state string by appending the action to the context.

        This is deliberately simple — the full agent loop would actually
        *execute* the action (e.g. call a tool).  Here we just concatenate
        for planning purposes.
        """
        if not current_state:
            return f"Step 1: {action}"
        steps = current_state.count("\n") + 2
        return f"{current_state}\nStep {steps}: {action}"

    # ------------------------------------------------------------------
    # LLM call helper
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str, purpose: str = "") -> str:
        """Call the LLM and optionally record cost.

        Parameters
        ----------
        prompt:
            The prompt string to send to the LLM.
        purpose:
            Descriptive label for debug logging.

        Returns
        -------
        str
            The LLM's text response.
        """
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
                        metadata={"purpose": f"lats/{purpose}"},
                    )
                except Exception as cost_exc:
                    logger.warning(f"LATS | cost tracking failed: {cost_exc}")

            return content
        except Exception as exc:
            logger.error(f"LATS | LLM call failed ({purpose}): {exc}")
            return ""

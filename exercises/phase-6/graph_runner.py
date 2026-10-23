"""🧠 Exercise, Steps 6.2 and 6.4: what LangGraph does for us, in plain Python. No LangGraph, no API.

Three small functions are the core of src/shortsmith/agents/graph.py:
1. merge_update: a node returns only what it changed; reducers decide how each field is merged
   (replace by default, append for lists like critic_rounds, sum for usage).
2. route_after_critic: the evaluator-optimizer loop's exit (the conditional edge after the critic).
3. run_graph: the loop that runs nodes, follows edges (fixed or conditional), and stops at END.

The finished versions: RunState's reducers (agents/state.py), route_after_critic() and the
LangGraph wiring (agents/graph.py). Write these first, then compare.
Check them from the repo root:  uv run pytest exercises/phase-6
The states are plain dicts here, so you can check every answer on paper.
"""

from collections.abc import Callable

END = "__end__"

State = dict
Node = Callable[[State], dict]  # gets the whole state, returns a partial update
Reducer = Callable[[object, object], object]  # (old value, new value) -> merged value
Edge = str | Callable[[State], str]  # a fixed next node, or a function of the state that names it


def merge_update(state: State, update: dict, reducers: dict[str, Reducer]) -> State:
    """A NEW state: `state` with `update` merged in. `state` itself must not change.

    Example: state {"score": 5, "rounds": [1]}, update {"score": 7, "rounds": [2]},
             reducers {"rounds": operator.add}
          -> {"score": 7, "rounds": [1, 2]}

    Hints:
    1. Start from a shallow copy of state (dict(state)).
    2. For each key in update: with a reducer, the new value is reducer(old, new); without one,
       the new value simply replaces the old one.
    3. A key that isn't in the state yet: with a reducer, there is no "old" value. Use the
       reducer's empty start: here, just take the new value as it is.
    """
    raise NotImplementedError("your turn: follow the hints above")


def route_after_critic(state: State, pass_score: float = 7.0, max_rounds: int = 2) -> str:
    """"revise" when the last critic report wants changes and rounds are left; otherwise END.

    state["critic_rounds"] is a list of reports, oldest first; each report is a dict with
    "score" (0-10) and "weak" (a list of scene numbers). One report per review, so after the
    first review the list has 1 report, after the first revision's review 2, and so on.

    Rules, in this order:
    1. The last report's score >= pass_score, or it has no weak scenes: END (done).
    2. max_rounds revisions have been made already: END (give up; keep the best effort).
       How many revisions so far? One fewer than the number of reports.
    3. Otherwise: "revise".
    """
    raise NotImplementedError("your turn: follow the rules above")


def run_graph(nodes: dict[str, Node], edges: dict[str, Edge], start: str, state: State,
              reducers: dict[str, Reducer] | None = None, max_steps: int = 25) -> tuple[State, list[str]]:
    """Run nodes from `start` until an edge leads to END. Returns (final state, the nodes in the order they ran).

    Example: nodes {"a": ..., "b": ...}, edges {"a": "b", "b": lambda s: END}
          -> runs a, then b, then stops: path ["a", "b"]

    Hints:
    1. current = start; path = []. Loop: run nodes[current](state), merge its update into the
       state with merge_update, append current to path.
    2. The next node: edges[current] is either a string (a fixed edge) or a function (a
       conditional edge): call it with the NEW state.
    3. Stop when the next node is END.
    4. A graph with a loop that never ends is a bug: raise RuntimeError after max_steps nodes
       (LangGraph calls this the recursion limit: 25 by default too).
    """
    raise NotImplementedError("your turn: follow the hints above")

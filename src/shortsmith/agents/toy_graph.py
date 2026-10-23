"""Step 6.1: LangGraph on a toy graph, before the real one. No LLM, no API: runs in a second.

    START -> write -> check --(too long)------> shorten -> check ...
                         `--(short enough)--> END

Four ideas, all visible here:
- state: one object every node reads. Here a Pydantic model (the real graph uses one too).
- nodes: plain functions state -> partial update (a dict with only the fields they change).
- edges: fixed (write -> check) or conditional (a function of the state names the next node).
- compile() checks the wiring and returns a runnable graph; invoke() runs it to the end.

`history` has a reducer (operator.add): a node returns ["its line"] and LangGraph APPENDS it.
`hook` has none: a node's value REPLACES the old one. That's the whole difference.

Run it from the repo root:
    uv run python -m shortsmith.agents.toy_graph
"""

import operator
from typing import Annotated

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

MAX_WORDS = 8  # the writer's hook rule


class ToyState(BaseModel):
    topic: str
    hook: str = ""
    words: int = 0
    checks: int = 0  # how many times `check` ran: the conditional edge's loop counter
    history: Annotated[list[str], operator.add] = []  # reducer: updates are appended, not replaced


def write(state: ToyState) -> dict:
    """A deliberately wordy first draft (a real writer node would call the LLM here)."""
    hook = f"Here is everything you never knew you needed to know about {state.topic} today"
    return {"hook": hook, "history": [f"write: {len(hook.split())} words"]}


def check(state: ToyState) -> dict:
    words = len(state.hook.split())
    return {"words": words, "checks": state.checks + 1, "history": [f"check: {words} words"]}


def shorten(state: ToyState) -> dict:
    """Drop the first two words: crude, but each pass gets closer to the rule."""
    hook = " ".join(state.hook.split()[2:])
    return {"hook": hook, "history": [f"shorten: -> {hook!r}"]}


def route(state: ToyState) -> str:
    """The conditional edge: it only reads the state and names the next node, it changes nothing."""
    return "shorten" if state.words > MAX_WORDS else END


def build() -> StateGraph:
    graph = StateGraph(ToyState)
    graph.add_node("write", write)
    graph.add_node("check", check)
    graph.add_node("shorten", shorten)
    graph.add_edge(START, "write")
    graph.add_edge("write", "check")
    graph.add_conditional_edges("check", route, ["shorten", END])  # the list = where it may go (for drawing)
    graph.add_edge("shorten", "check")
    return graph


def main() -> None:
    app = build().compile()
    print(app.get_graph().draw_mermaid())  # Mermaid text (paste into mermaid.live); draw_ascii() needs grandalf

    print("\n--- stream(stream_mode='updates'): what each node returned ---")
    for update in app.stream({"topic": "plov"}, stream_mode="updates"):
        print(update)

    print("\n--- stream(stream_mode='values'): the whole state after each step ---")
    for step, values in enumerate(app.stream({"topic": "plov"}, stream_mode="values")):
        print(step, {k: v for k, v in values.items() if k != "history"}, f"history: {len(values['history'])} lines")

    print("\n--- invoke(): the final state ---")
    final = app.invoke({"topic": "plov"})
    print(type(final).__name__, final)


if __name__ == "__main__":
    main()
